"""Correctness regressions for the eval/probe metric path.

Each test pins one of the fixes in the eval-probes-correctness PR:
train-stream leak in evaluate_batches, token-weighted batch means, the
average-rank headline replacing the clamped geomean, needle early-mode
truncation, induction scoring window, associative-recall key uniqueness,
monotone breaking points, memory-ablation no-op detection, EMA composite
weights, and the sliding-window tail.
"""

from __future__ import annotations

import contextlib
import io
import json
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from flower.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from flower.data import token_batches, validation_token_batches
from flower.eval import (
    default_window_stride,
    evaluate_batches,
    sliding_window_loss,
    sliding_window_starts,
)
from flower.probes.composite import (
    _NEEDLE_VALUES,
    _memory_read_ablation,
    _monotone_breaking_point,
    associative_recall_probe,
    attach_average_ranks,
    induction_copy_probe,
    memory_ablation_probe,
    mqar_probe,
    needle_in_text_probe,
    run_composite_eval,
)


def tiny_eval_config(
    *,
    dataset: str = "synthetic",
    eval_seq_len: int = 256,
    vocab_size: int = 128,
    sequence_length: int = 32,
    synthetic_vocab_size: int | None = None,
    tokenizer: str = "byte",
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            d_model=32,
            num_heads=4,
            num_layers=1,
            ffn_dim=64,
            max_seq_len=32,
            local_window=16,
        ),
        data=DataConfig(
            dataset=dataset,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            synthetic_vocab_size=synthetic_vocab_size or vocab_size,
            eval_seq_len=eval_seq_len,
        ),
        training=TrainingConfig(seed=0, batch_size=4),
    )


class CountLossModel(nn.Module):
    """LM stand-in whose `loss` is the batch's supervised-token count.

    Returning a known per-batch scalar (instead of a real CE mean) makes the
    loss-weighting arithmetic of `evaluate_batches` directly checkable. Records
    every (input_ids, labels) it is shown.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        if labels is None:
            labels = input_ids
        self.seen.append((input_ids, labels))
        count = int((labels[:, 1:] != -100).sum())
        return {"logits": None, "loss": torch.tensor(float(count))}


class ShiftedOneHotLM(nn.Module):
    """Real-CE LM stand-in: predicts the previous token, plus an optional
    constant logit boost derived from a read module.

    The boost flows through an attribute named `read_attr`. When that name is
    one of `_ABLATABLE_MODULE_NAMES` ("mem_read"), the memory-ablation probe
    patches it and the loss changes; any other name simulates a variant whose
    memory read the probe cannot patch (phase_memory's bound `_read` method).
    """

    def __init__(self, vocab_size: int, *, read_attr: str | None = None, boost: float = 4.0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.read_attr = read_attr
        self.boost = boost
        self.seen: list[torch.Tensor] = []
        if read_attr is not None:
            constant = nn.Module()

            def constant_forward(x: torch.Tensor) -> torch.Tensor:
                return torch.ones_like(x)

            constant.forward = constant_forward  # type: ignore[method-assign]
            setattr(self, read_attr, constant)

    def _read_scale(self) -> float:
        if self.read_attr is None:
            return 0.0
        return self.boost * float(getattr(self, self.read_attr)(torch.zeros(1)))

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        self.seen.append(input_ids)
        onehot = F.one_hot(input_ids, self.vocab_size).float()
        logits = torch.roll(onehot, shifts=1, dims=1)
        logits[:, 0, :] = 0.0
        logits[..., 3] += self._read_scale()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}


class OracleRecallModel(nn.Module):
    """First-occurrence recall oracle (same rule as the sweep-7 test suite):
    at each position, predict the token that followed the FIRST earlier
    occurrence of the current token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        b, t = input_ids.shape
        logits = input_ids.new_zeros((b, t, self.vocab_size), dtype=torch.float32)
        for bi in range(b):
            first: dict[int, int] = {}
            for ti in range(t):
                tok = int(input_ids[bi, ti])
                if tok in first and first[tok] + 1 < t:
                    logits[bi, ti, int(input_ids[bi, first[tok] + 1])] = 10.0
                if tok not in first:
                    first[tok] = ti
        return {"logits": logits, "loss": None}


class MemorylessModel(nn.Module):
    """All-zero logits: never clears the 0.5 breaking-point bar."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        logits = input_ids.new_zeros((*input_ids.shape, self.vocab_size), dtype=torch.float32)
        return {"logits": logits, "loss": None}


# ---------------------------------------------------------------------------
# Fix 1: evaluate_batches must use the validation stream and weight batch
# losses by supervised-token counts.
# ---------------------------------------------------------------------------


def test_evaluate_batches_uses_validation_stream_not_training_stream() -> None:
    cfg = tiny_eval_config()
    cfg.training.seed = 7  # != DataConfig.validation_seed (4321)
    device = torch.device("cpu")
    model = CountLossModel()

    evaluate_batches(model, cfg, device, batches_count=1)

    train_first = next(token_batches(cfg.data, cfg.training.batch_size, device, seed=7))
    val_first = next(validation_token_batches(cfg.data, cfg.training.batch_size, device))
    seen_ids = model.seen[0][0]
    assert torch.equal(seen_ids, val_first)
    assert not torch.equal(seen_ids, train_first)


def test_evaluate_batches_weights_batch_losses_by_supervised_tokens() -> None:
    # MQAR rows carry variable num_pairs/num_queries, so per-batch supervised
    # counts differ and a mean-of-means is not the token-weighted mean.
    cfg = tiny_eval_config(dataset="mqar", sequence_length=128, synthetic_vocab_size=64, eval_seq_len=128)
    model = CountLossModel()

    metrics = evaluate_batches(model, cfg, torch.device("cpu"), batches_count=8)

    counts = [int((labels[:, 1:] != -100).sum()) for _, labels in model.seen]
    assert len(counts) == 8
    assert len(set(counts)) > 1  # the case a mean-of-means gets wrong
    # CountLossModel's "loss" IS the batch's supervised count, so the correct
    # token-weighted aggregate of these batch means is sum(c^2)/sum(c).
    expected = sum(c * c for c in counts) / sum(counts)
    mean_of_means = sum(counts) / len(counts)
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["loss"] != pytest.approx(mean_of_means)


# ---------------------------------------------------------------------------
# Fix 2: the headline is the average-rank composite, not the clamped geomean.
# ---------------------------------------------------------------------------


def _old_geomean_loss_like(rank_inputs: dict[str, float]) -> float:
    """The removed headline formula, kept here to pin WHY it was removed."""
    return math.exp(sum(math.log(max(v, 1e-9)) for v in rank_inputs.values()) / len(rank_inputs))


def _identical_row(**overrides: float) -> dict[str, float]:
    row = {
        "rank_fineweb_bpb": 0.40,
        "rank_induction_copy_loss": 3.20,
        "rank_assoc_recall_loss": 2.50,
        "rank_mqar_neg_breaking_point": -32.0,
        "rank_text_recall_neg_breaking_point": -16.0,
        "rank_needle_neg_breaking_point": -4.0,
        "rank_blimp_mini_error": 0.0125,
    }
    row.update(overrides)
    return row


def test_average_rank_headline_has_no_perfect_blimp_cliff() -> None:
    # Two runs identical except one has a PERFECT blimp score (error 0.0).
    perfect = _identical_row(rank_blimp_mini_error=0.0)
    near_perfect = _identical_row(rank_blimp_mini_error=0.0125)

    # The old geomean clamped 0.0 to 1e-9 and swung >5x on this difference.
    g_perfect = _old_geomean_loss_like(perfect)
    g_near = _old_geomean_loss_like(near_perfect)
    assert max(g_perfect, g_near) / min(g_perfect, g_near) > 5.0

    rows = [perfect, near_perfect]
    attach_average_ranks(rows)
    # The rank headline is confined to [1, n_rows] and moves by at most one
    # rank step per differing column — no multiplicative cliff exists.
    assert 1.0 <= rows[0]["composite_avg_rank"] <= 2.0
    assert 1.0 <= rows[1]["composite_avg_rank"] <= 2.0
    assert rows[1]["composite_avg_rank"] - rows[0]["composite_avg_rank"] == pytest.approx(1.0)


def test_average_rank_headline_responds_to_breaking_points() -> None:
    weak = _identical_row(rank_mqar_neg_breaking_point=-16.0)
    strong = _identical_row(rank_mqar_neg_breaking_point=-64.0)
    # The negated breaking points are <= 0, so the old geomean clamped BOTH to
    # 1e-9: a 4x capacity difference was invisible. Ranks see it.
    assert _old_geomean_loss_like(weak) == _old_geomean_loss_like(strong)

    # Three runs, distinct on every column. A is best on both CONTROL columns,
    # C is best on the breaking-point column only. The breaking point must
    # pull C's composite up (and push A's down) relative to a controls-only
    # headline — i.e. the capacity metric carries signal.
    control_best = {
        "rank_fineweb_bpb": 1.0,
        "rank_induction_copy_loss": 1.0,
        "rank_mqar_neg_breaking_point": -8.0,
    }
    middle = {
        "rank_fineweb_bpb": 2.0,
        "rank_induction_copy_loss": 2.0,
        "rank_mqar_neg_breaking_point": -16.0,
    }
    capacity_best = {
        "rank_fineweb_bpb": 3.0,
        "rank_induction_copy_loss": 3.0,
        "rank_mqar_neg_breaking_point": -64.0,
    }

    rows = [dict(control_best), dict(middle), dict(capacity_best)]
    attach_average_ranks(rows)
    controls_only = [
        {k: v for k, v in row.items() if k != "rank_mqar_neg_breaking_point"} for row in rows
    ]
    attach_average_ranks(controls_only)

    assert rows[0]["composite_avg_rank"] > controls_only[0]["composite_avg_rank"]  # A pushed down
    assert rows[2]["composite_avg_rank"] < controls_only[2]["composite_avg_rank"]  # C pulled up


def test_average_rank_handles_rows_with_missing_metric_columns() -> None:
    # The aggregate script flattens rank_inputs into `rank_*` columns on rows
    # that may individually MISS metrics (e.g. an unmeasured ablation); those
    # rows must rank only on the columns they carry.
    rows = [
        {"rank_fineweb_bpb": 0.30, "rank_blimp_mini_error": 0.01},
        {"rank_fineweb_bpb": 0.35, "rank_blimp_mini_error": 0.02},
        {"rank_fineweb_bpb": 0.40},  # blimp missing
    ]
    attach_average_ranks(rows)
    assert rows[0]["composite_avg_rank"] == pytest.approx(1.0)
    assert rows[1]["composite_avg_rank"] == pytest.approx(2.0)
    assert rows[2]["composite_avg_rank"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Fix 3: needle probe truncation must keep the queried fact (early mode).
# ---------------------------------------------------------------------------


def _fake_filler_docs(config: DataConfig, limit: int | None = None):
    # One validation "document" made of ~120-char filler sentences, so the
    # needle probe's filler pool needs no network access.
    sentence = "f" * 119 + "."
    yield " ".join([sentence] * 40)


@pytest.mark.parametrize("depth_mode", ["early", "late"])
def test_needle_probe_truncation_keeps_queried_fact(monkeypatch, depth_mode: str) -> None:
    monkeypatch.setattr("flower.probes.composite.fineweb_validation_documents", _fake_filler_docs)
    cfg = tiny_eval_config(dataset="fineweb_edu", eval_seq_len=512, vocab_size=256, sequence_length=64)
    model = ShiftedOneHotLM(cfg.model.vocab_size)

    out = needle_in_text_probe(
        model, cfg, torch.device("cpu"), trials=2, num_pairs_list=(4,), depth_modes=(depth_mode,)
    )
    assert "skipped" not in out  # probe ran (synthetic/mqar datasets are the skip path)
    assert out["capacity_curve"][depth_mode]

    budget = 512 - 8
    for ids in model.seen:
        assert ids.shape[1] <= budget + 16  # cap honoured (prefix + continuation)
        text = bytes(ids[0].tolist()).decode("utf-8", errors="replace")
        assert "secret word for" in text  # the query prompt survived
        # At least one full planted fact survived truncation.
        assert any(f" is {v}." in text for v in _NEEDLE_VALUES)

    if depth_mode == "early":
        # Early mode plants the queried fact FIRST; keeping the tail (the old
        # bug) deleted exactly that fact. The truncated prefix must therefore
        # still START with the queried fact.
        for ids in model.seen:
            text = bytes(ids[0].tolist()).decode("utf-8", errors="replace")
            assert text.startswith(" The secret word for")


# ---------------------------------------------------------------------------
# Fix 4: induction scoring starts at the first REPEATED token.
# ---------------------------------------------------------------------------


class InductionWindowOracle(nn.Module):
    """Correct exactly at the positions the fixed window should score.

    The probe builds [pattern(p), filler(p), pattern(p)]. This model predicts
    every token of the SECOND pattern perfectly (using the repeat structure)
    and nothing before it. Under the fixed window (first scored logit at 2p,
    predicting the token at 2p+1) accuracy is exactly 1.0; under the old window
    (first scored logit at 2p-1, predicting the repeated block's FIRST token —
    unpredictable by induction) accuracy was (p-2)/(p-1).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        b, t = input_ids.shape
        p = t // 3
        rows = torch.arange(b)
        logits = input_ids.new_zeros((b, t, self.vocab_size), dtype=torch.float32)
        for pos in range(2 * p, t - 1):
            # seq[pos + 1] repeats pattern[pos + 1 - 2p] from the first copy.
            # (Index batch rows pairwise: `logits[:, pos, idx]` would write the
            # batch cross-product instead.)
            logits[rows, pos, input_ids[:, pos + 1 - 2 * p]] = 10.0
        return {"logits": logits, "loss": None}


def test_induction_probe_scores_only_repeated_block() -> None:
    cfg = tiny_eval_config(eval_seq_len=128)
    model = InductionWindowOracle(cfg.model.vocab_size)

    result = induction_copy_probe(model, cfg, torch.device("cpu"), batches=2, batch_size=8)

    assert set(result) == {"loss", "accuracy", "tokens"}  # output keys stable
    # Only positions >= 2p are scored: p-1 predictions per row.
    pattern_len = max(4, 128 // 3)
    assert result["tokens"] == 2 * 8 * (pattern_len - 1)
    assert result["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Fix 5: associative recall samples keys without replacement.
# ---------------------------------------------------------------------------


def test_associative_recall_unique_keys_let_perfect_oracle_score_1() -> None:
    # At vocab 128 with 8 pairs, WITH-replacement sampling duplicated a key in
    # ~11% of rows; the first-occurrence oracle then answers with the WRONG
    # duplicate's value and accuracy < 1.0. Unique keys make it exactly 1.0.
    cfg = tiny_eval_config(eval_seq_len=128)
    model = OracleRecallModel(cfg.model.vocab_size)

    result = associative_recall_probe(model, cfg, torch.device("cpu"), batches=8, batch_size=8, pairs=8)

    assert result["accuracy"] == 1.0
    assert result["examples"] == 8 * 8


def test_associative_recall_falls_back_when_vocab_smaller_than_pairs() -> None:
    cfg = tiny_eval_config(eval_seq_len=64, vocab_size=8, synthetic_vocab_size=8)
    model = OracleRecallModel(cfg.model.vocab_size)

    result = associative_recall_probe(model, cfg, torch.device("cpu"), batches=2, batch_size=2, pairs=8)

    assert 0.0 <= result["accuracy"] <= 1.0  # runs; uniqueness impossible at vocab 8


# ---------------------------------------------------------------------------
# Fix 6: breaking points are monotone prefixes.
# ---------------------------------------------------------------------------


def test_monotone_breaking_point_is_a_prefix_not_largest_pass() -> None:
    # Non-monotonic curve: pass 16, fail 32, pass 64 -> the answer is 16 (the
    # old "largest passing level" rule reported 64).
    assert _monotone_breaking_point({"16": 0.9, "32": 0.45, "64": 0.95}) == 16
    assert _monotone_breaking_point({"16": 0.9, "32": 0.91, "64": 0.45}) == 32
    assert _monotone_breaking_point({"16": 0.4, "32": 0.9}) == 0  # fails at the first level
    # Unmeasurable (NaN) levels are gaps, not failures.
    assert _monotone_breaking_point({"16": 0.9, "32": float("nan"), "64": 0.9}) == 16
    assert _monotone_breaking_point({"16": float("nan")}) == 0


def test_mqar_breaking_point_uses_monotone_prefix() -> None:
    cfg = tiny_eval_config(eval_seq_len=256)
    device = torch.device("cpu")

    oracle = mqar_probe(
        OracleRecallModel(cfg.model.vocab_size), cfg, device,
        batches=2, batch_size=4, num_pairs_list=(16, 32),
    )
    assert oracle["breaking_points"]["long"] == 32

    memoryless = mqar_probe(
        MemorylessModel(cfg.model.vocab_size), cfg, device,
        batches=2, batch_size=4, num_pairs_list=(16, 32),
    )
    assert memoryless["breaking_points"]["long"] == 0


# ---------------------------------------------------------------------------
# Fix 7: memory ablation reports whether it actually ablated anything.
# ---------------------------------------------------------------------------


def test_memory_ablation_probe_measures_matched_module() -> None:
    cfg = tiny_eval_config()
    model = ShiftedOneHotLM(cfg.model.vocab_size, read_attr="mem_read", boost=4.0)
    original_forward = model.mem_read.forward

    out = memory_ablation_probe(model, cfg, torch.device("cpu"), doc_limit=2)

    assert out["ablated"] is True
    # The boost made normal bpb worse than the ablated (boost removed) bpb.
    assert out["delta_bpb"] < 0.0
    # Patching is fully undone on exit.
    assert model.mem_read.forward is original_forward


def test_memory_ablation_probe_flags_unpatchable_read_path() -> None:
    # phase_memory-style variant: carries a memory read, but not through a
    # module named in _ABLATABLE_MODULE_NAMES -> nothing is patched and the
    # probe must say so instead of fabricating delta_bpb == 0.0.
    cfg = tiny_eval_config()
    model = ShiftedOneHotLM(cfg.model.vocab_size, read_attr="reader", boost=4.0)

    out = memory_ablation_probe(model, cfg, torch.device("cpu"), doc_limit=2)

    assert out["ablated"] is False
    assert math.isnan(out["delta_bpb"])
    assert math.isnan(out["ablated_bpb"])
    assert out["normal_bpb"] > 0.0


def test_memory_read_ablation_state_lists_patched_modules() -> None:
    cfg = tiny_eval_config()
    model = ShiftedOneHotLM(cfg.model.vocab_size, read_attr="mem_read")
    original_forward = model.mem_read.forward

    with _memory_read_ablation(model) as state:
        assert state.patched  # something matched
        assert state  # truthy while patching
        assert model.mem_read.forward is not original_forward
    assert model.mem_read.forward is original_forward
    assert state.patched  # the record remains readable after exit


def test_run_composite_eval_headline_and_ablation_exclusion() -> None:
    cfg = tiny_eval_config(eval_seq_len=32)
    device = torch.device("cpu")

    unpatched = run_composite_eval(
        ShiftedOneHotLM(cfg.model.vocab_size, read_attr="reader"), cfg, device=device, doc_limit=1
    )
    assert "geomean_loss_like" not in unpatched  # clamped headline removed
    assert "memory_ablation_neg_delta_bpb" not in unpatched["rank_inputs"]
    assert unpatched["metrics"]["memory_ablation"]["ablated"] is False
    assert "memory_ablation_neg_delta_bpb" not in unpatched["lower_is_better"]

    patched = run_composite_eval(
        ShiftedOneHotLM(cfg.model.vocab_size, read_attr="mem_read"), cfg, device=device, doc_limit=1
    )
    assert "memory_ablation_neg_delta_bpb" in patched["rank_inputs"]
    assert patched["metrics"]["memory_ablation"]["ablated"] is True
    # NaN deltas must survive the JSON round trip (train.py writes this dict).
    assert json.loads(json.dumps(unpatched))["metrics"]["memory_ablation"]["delta_bpb"] != 0.0


# ---------------------------------------------------------------------------
# Fix 8: the final composite eval runs on the same weights as the final val
# metrics (EMA when ema_decay > 0) and records which.
# ---------------------------------------------------------------------------


def _write_train_config(tmp_path, *, ema_decay: float) -> str:
    config = f"""
model:
  variant: vanilla_local
  vocab_size: 64
  d_model: 16
  num_heads: 2
  num_layers: 1
  ffn_dim: 32
  max_seq_len: 16
  local_window: 8
data:
  dataset: synthetic
  tokenizer: byte
  sequence_length: 16
  synthetic_vocab_size: 64
  eval_seq_len: 16
training:
  batch_size: 2
  steps: 1
  lr: 0.001
  device: cpu
  log_backend: none
  save_checkpoints: false
  output_dir: {tmp_path / 'out'}
  ema_decay: {ema_decay}
  composite_eval: true
  composite_eval_json: {tmp_path / 'out' / 'composite_ranker.json'}
  metrics_json: {tmp_path / 'metrics.json'}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config)
    return str(path)


@pytest.mark.parametrize("ema_decay,expected", [(0.99, "ema"), (0.0, "raw")])
def test_final_composite_eval_uses_val_metric_weights(tmp_path, ema_decay: float, expected: str) -> None:
    from flower.train import train

    config_path = _write_train_config(tmp_path, ema_decay=ema_decay)
    metrics = train(["--config", config_path, "--steps", "1", "--device", "cpu"])

    assert metrics["composite_eval_weights"] == expected
    composite = json.loads((tmp_path / "out" / "composite_ranker.json").read_text())
    assert composite["eval_weights"] == expected
    assert "geomean_loss_like" not in composite
    # vanilla_local carries no ablatable memory read: the metric is excluded
    # from rank inputs rather than reported as a fabricated 0.0.
    assert "memory_ablation_neg_delta_bpb" not in composite["rank_inputs"]


# ---------------------------------------------------------------------------
# Fix 9: sliding-window eval scores the document tail; stride default matches
# the help text.
# ---------------------------------------------------------------------------


def test_sliding_window_starts_always_cover_the_tail() -> None:
    for total in (1, 5, 31, 32, 33, 64, 100, 257):
        for window in (4, 16, 32, 64):
            for stride in (1, 4, 16, 32):
                starts = sliding_window_starts(total, window, stride)
                effective = min(window, total)
                assert starts[0] == 0
                assert starts[-1] + effective == total  # final window ends at the document end
                assert starts == sorted(set(starts))
                if stride <= window:
                    covered: set[int] = set()
                    for start in starts:
                        covered |= set(range(start, start + effective))
                    assert covered == set(range(total))


def test_sliding_window_loss_scores_every_token() -> None:
    class RecordingLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.windows: list[torch.Tensor] = []

        def forward(self, input_ids, labels=None):
            self.windows.append(input_ids[0])
            return {"logits": None, "loss": torch.tensor(1.0)}

    model = RecordingLM()
    tokens = torch.arange(100, dtype=torch.long)
    starts = sliding_window_starts(100, 32, 16)

    loss = sliding_window_loss(model, tokens, window_size=32, stride=16, device=torch.device("cpu"))

    assert len(model.windows) == len(starts)  # one forward per window, incl. the clamped tail
    covered: set[int] = set()
    for window in model.windows:
        covered |= set(window.tolist())
    assert covered == set(range(100))  # no unscored tail stride
    assert loss == pytest.approx(1.0)  # every window's mean loss is 1.0


@pytest.mark.parametrize(
    "window_size,expected",
    [(3, 1), (8, 2), (32, 8), (256, 64), (2048, 64)],
)
def test_default_window_stride(window_size: int, expected: int) -> None:
    assert default_window_stride(window_size) == expected


def test_stride_help_text_matches_code_default() -> None:
    from flower.eval import evaluate as evaluate_cli

    buf = io.StringIO()
    with pytest.raises(SystemExit):
        with contextlib.redirect_stdout(buf):
            evaluate_cli(["--help"])
    assert "min(64, window_size//4)" in buf.getvalue()
