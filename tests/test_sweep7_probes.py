from __future__ import annotations

import torch
from torch import nn

from flower.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from flower.models.memory import MemoryRead
from flower.probes.composite import (
    _continuation_nll,
    associative_recall_probe,
    induction_copy_probe,
    mqar_probe,
    needle_in_text_probe,
    text_recall_probe,
)


class RecordingModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_lengths: list[int] = []

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        self.seen_lengths.append(int(input_ids.shape[1]))
        logits = input_ids.new_zeros((*input_ids.shape, self.vocab_size), dtype=torch.float32)
        return {"logits": logits, "loss": None}


def tiny_eval_config(eval_seq_len: int = 256) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=128,
            d_model=32,
            num_heads=4,
            num_layers=1,
            ffn_dim=64,
            max_seq_len=32,
            local_window=16,
        ),
        data=DataConfig(dataset="synthetic", sequence_length=32, synthetic_vocab_size=128, eval_seq_len=eval_seq_len),
        training=TrainingConfig(seed=0),
    )


def test_recall_probes_use_eval_sequence_length() -> None:
    cfg = tiny_eval_config(eval_seq_len=192)
    model = RecordingModel(cfg.model.vocab_size)
    device = torch.device("cpu")

    induction_copy_probe(model, cfg, device, batches=1, batch_size=1)
    associative_recall_probe(model, cfg, device, batches=1, batch_size=1, pairs=8)

    assert model.seen_lengths[0] >= 180
    assert model.seen_lengths[1] == 192


def test_mqar_reports_short_and_long_delay_capacity_curves() -> None:
    cfg = tiny_eval_config(eval_seq_len=256)
    model = RecordingModel(cfg.model.vocab_size)

    result = mqar_probe(
        model,
        cfg,
        torch.device("cpu"),
        batches=1,
        batch_size=1,
        num_pairs_list=(16,),
        query_fraction=0.25,
    )

    assert set(result["capacity_curve"]) == {"short", "long"}
    assert set(result["breaking_points"]) == {"short", "long"}
    assert max(model.seen_lengths) == 256


class OracleRecallModel(nn.Module):
    """Perfect associative-recall model: at each position predict the token that
    followed the FIRST earlier occurrence of the current token.

    For an MQAR/text-recall sequence `[k1,v1,...,kN,vN, delay, q,a,...]` the keys
    are unique and planted at the front, so first-occurrence recall puts a one-hot
    on the correct planted value at every query position (robust to keys recurring
    as random delay tokens). A working probe must register a high breaking point;
    this is the true-positive control the probe suite previously lacked.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        b, t = input_ids.shape
        logits = input_ids.new_zeros((b, t, self.vocab_size), dtype=torch.float32)
        for bi in range(b):
            first: dict[int, int] = {}
            for ti in range(t):
                tok = int(input_ids[bi, ti])
                if tok in first and first[tok] + 1 < t:
                    pred = int(input_ids[bi, first[tok] + 1])
                    logits[bi, ti, pred] = 10.0
                if tok not in first:
                    first[tok] = ti
        return {"logits": logits, "loss": None}


def test_oracle_recall_model_scores_high_breaking_point() -> None:
    # True-positive: a perfect-recall model must reach the top of the capacity curve.
    cfg = tiny_eval_config(eval_seq_len=512)
    model = OracleRecallModel(cfg.model.vocab_size)
    pool = torch.arange(2, 120, dtype=torch.long)

    result = text_recall_probe(
        model, cfg, torch.device("cpu"),
        batches=2, batch_size=4, num_pairs_list=(16, 32), token_pool=pool,
    )
    assert result["breaking_points"]["long"] >= 32
    assert result["capacity_curve"]["long"]["16"] > 0.95


def test_memoryless_model_scores_chance_breaking_point() -> None:
    # True-negative: a zero-logit model must NOT pass the breaking-point threshold.
    cfg = tiny_eval_config(eval_seq_len=512)
    model = RecordingModel(cfg.model.vocab_size)  # returns all-zero logits
    pool = torch.arange(2, 120, dtype=torch.long)

    result = text_recall_probe(
        model, cfg, torch.device("cpu"),
        batches=4, batch_size=8, num_pairs_list=(16, 32), token_pool=pool,
    )
    assert result["breaking_points"]["long"] == 0
    assert result["capacity_curve"]["long"]["16"] < 0.5


def test_mqar_oracle_true_positive_and_memoryless_true_negative() -> None:
    # Same controls for the synthetic mqar_probe so both probes are pinned.
    cfg = tiny_eval_config(eval_seq_len=512)
    oracle = mqar_probe(
        OracleRecallModel(cfg.model.vocab_size), cfg, torch.device("cpu"),
        batches=2, batch_size=4, num_pairs_list=(16, 32),
    )
    memoryless = mqar_probe(
        RecordingModel(cfg.model.vocab_size), cfg, torch.device("cpu"),
        batches=2, batch_size=4, num_pairs_list=(16, 32),
    )
    assert oracle["breaking_points"]["long"] >= 32
    assert memoryless["breaking_points"]["long"] == 0


class FixedFavoriteModel(nn.Module):
    """Model whose logits always favor one fixed token id, with a `loss` head.

    Used to verify `_continuation_nll` is lower for the favored continuation than
    for any other — i.e. the needle probe's NLL-based candidate scoring picks the
    token the model actually predicts.
    """

    def __init__(self, vocab_size: int, favorite: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.favorite = favorite

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        b, t = input_ids.shape
        logits = input_ids.new_zeros((b, t, self.vocab_size), dtype=torch.float32)
        logits[..., self.favorite] = 10.0
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return {"logits": logits, "loss": loss}


def test_continuation_nll_lower_for_predicted_token() -> None:
    model = FixedFavoriteModel(vocab_size=32, favorite=7)
    prefix = [1, 2, 3]
    nll_fav = _continuation_nll(model, prefix, [7, 7], torch.device("cpu"))
    nll_other = _continuation_nll(model, prefix, [5, 5], torch.device("cpu"))
    assert nll_fav < nll_other


def test_needle_probe_skips_synthetic_dataset() -> None:
    cfg = tiny_eval_config(eval_seq_len=256)  # dataset="synthetic"
    out = needle_in_text_probe(RecordingModel(cfg.model.vocab_size), cfg, torch.device("cpu"))
    assert out["skipped"] is True
    assert out["breaking_point"] == 0


def test_energy_read_uses_logsumexp_not_softmax_weighted_average() -> None:
    cfg = ModelConfig(
        vocab_size=16,
        d_model=4,
        num_heads=1,
        memory_slots=2,
        short_memory_slots=0,
        energy_read=True,
        energy_beta_init=1.0,
    )
    read = MemoryRead(cfg)
    with torch.no_grad():
        read.q.weight.zero_()
        read.q.bias.zero_()
        read.kv.weight.zero_()
        read.kv.bias.zero_()
        read.kv.weight[:4] = torch.eye(4)
        read.kv.weight[4:] = torch.eye(4)
        read.out.weight.copy_(torch.eye(4))
        read.out.bias.zero_()

    x = torch.zeros(1, 1, 4)
    memory = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]])

    out = read(x, memory)
    softmax_average = torch.full_like(out, 1.0)

    assert not torch.allclose(out, softmax_average)
    assert torch.all(out > softmax_average)
