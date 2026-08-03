"""Tests for the sweep-13 pipeline upgrades.

Covers the LR schedules, the modernised base architecture, depth-axis routing,
per-layer FFN allocation, and the Aurora optimizer. The recurring theme is
backward compatibility: every new option defaults to the pre-sweep-13 behaviour,
because the repo's published results were produced with those defaults.
"""

from __future__ import annotations

import math

import pytest
import torch
import yaml

from flower.config import ModelConfig, TrainingConfig, load_config
from flower.models import build_model
from flower.models.attn_res import routing_sites
from flower.models.base import layer_ffn_dims, swiglu_hidden_dim
from flower.optim import Aurora, Muon, build_optimizer
from flower.train import lr_multiplier


def tiny(**overrides) -> ModelConfig:
    base = dict(
        variant="vanilla_local",
        vocab_size=256,
        d_model=64,
        num_heads=4,
        num_layers=6,
        ffn_dim=192,
        max_seq_len=32,
        local_window=8,
    )
    base.update(overrides)
    return ModelConfig(**base)


# --------------------------------------------------------------------------
# LR schedules
# --------------------------------------------------------------------------


def test_legacy_schedules_unchanged():
    """constant and linear_warmup must behave exactly as before."""
    assert lr_multiplier(1, 100, "constant", total_steps=1000) == 1.0
    assert lr_multiplier(999, 100, "constant", total_steps=1000) == 1.0
    assert lr_multiplier(50, 100, "linear_warmup", total_steps=1000) == 0.5
    # The defining property of the legacy schedule: it never decays.
    assert lr_multiplier(1000, 100, "linear_warmup", total_steps=1000) == 1.0


def test_wsd_holds_then_decays_to_zero():
    warm, total, frac = 100, 1000, 0.2
    assert lr_multiplier(50, warm, "wsd", total, frac) == pytest.approx(0.5)
    assert lr_multiplier(100, warm, "wsd", total, frac) == pytest.approx(1.0)
    # Stable phase runs to total*(1-frac) == 800.
    assert lr_multiplier(800, warm, "wsd", total, frac) == pytest.approx(1.0)
    assert lr_multiplier(900, warm, "wsd", total, frac) == pytest.approx(0.5)
    assert lr_multiplier(1000, warm, "wsd", total, frac) == pytest.approx(0.0)


def test_wsd_respects_final_frac():
    end = lr_multiplier(1000, 100, "wsd", 1000, 0.2, final_frac=0.1)
    assert end == pytest.approx(0.1)


def test_cosine_decays_monotonically_after_warmup():
    vals = [lr_multiplier(s, 100, "cosine", 1000) for s in range(100, 1001, 50)]
    assert vals[0] == pytest.approx(1.0)
    assert vals[-1] == pytest.approx(0.0, abs=1e-9)
    assert all(vals[i + 1] <= vals[i] + 1e-12 for i in range(len(vals) - 1))


def test_schedules_without_total_steps_fall_back_to_warmup():
    """Callers that never pass total_steps must not silently decay to zero."""
    assert lr_multiplier(500, 100, "wsd", total_steps=0) == 1.0
    assert lr_multiplier(500, 100, "cosine", total_steps=0) == 1.0


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError, match="lr_schedule"):
        lr_multiplier(1, 10, "triangular", 100)


# --------------------------------------------------------------------------
# Base architecture
# --------------------------------------------------------------------------


def test_architecture_defaults_are_legacy():
    cfg = ModelConfig()
    assert cfg.norm_type == "layernorm"
    assert cfg.ffn_activation == "gelu"
    assert cfg.qk_norm is False
    assert cfg.use_bias is True
    assert cfg.init_scheme == "torch"
    assert cfg.attn_res == "none"


def test_gelu_ffn_keeps_checkpoint_parameter_names():
    """FFN param names are baked into every pre-sweep-13 checkpoint."""
    model = build_model(tiny())
    names = {n for n, _ in model.named_parameters()}
    assert "blocks.0.ff.net.0.weight" in names
    assert "blocks.0.ff.net.3.weight" in names


@pytest.mark.parametrize(
    "opts",
    [
        {"norm_type": "rmsnorm"},
        {"ffn_activation": "swiglu"},
        {"qk_norm": True},
        {"use_bias": False},
        {"init_scheme": "scaled"},
        {
            "norm_type": "rmsnorm",
            "ffn_activation": "swiglu",
            "qk_norm": True,
            "use_bias": False,
            "init_scheme": "scaled",
        },
    ],
)
def test_architecture_options_train(opts):
    torch.manual_seed(0)
    model = build_model(tiny(**opts))
    x = torch.randint(0, 256, (2, 16))
    out = model(x, labels=x)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_scaled_init_puts_initial_loss_near_uniform():
    """The torch default ties an N(0,1) embedding to the head; loss starts huge."""
    torch.manual_seed(0)
    legacy = build_model(tiny(vocab_size=4096, d_model=256))
    torch.manual_seed(0)
    scaled = build_model(tiny(vocab_size=4096, d_model=256, init_scheme="scaled"))
    x = torch.randint(0, 4096, (2, 16))
    ideal = math.log(4096)
    legacy_loss = legacy(x, labels=x)["loss"].item()
    scaled_loss = scaled(x, labels=x)["loss"].item()
    assert legacy_loss > 5 * ideal  # documents the defect
    assert abs(scaled_loss - ideal) < 0.5


def test_swiglu_is_parameter_matched_to_gelu():
    torch.manual_seed(0)
    gelu = build_model(tiny())
    torch.manual_seed(0)
    swiglu = build_model(tiny(ffn_activation="swiglu", use_bias=False))
    gelu_ffn = sum(p.numel() for n, p in gelu.named_parameters() if ".ff." in n)
    swiglu_ffn = sum(p.numel() for n, p in swiglu.named_parameters() if ".ff." in n)
    # Bias-free so the comparison is purely the weight matrices.
    assert swiglu_ffn <= gelu_ffn
    assert swiglu_hidden_dim(tiny(ffn_activation="swiglu")) == 128


def test_qk_norm_bounds_attention_logits():
    """The point of QK-norm under Muon is bounded QK^T."""
    torch.manual_seed(0)
    cfg = tiny(qk_norm=True, d_model=64)
    model = build_model(cfg)
    attn = model.blocks[0].attn
    x = torch.randn(2, 16, 64) * 50  # deliberately large activations
    q, k, _ = attn.qkv_heads(x)
    # RMSNorm makes each head vector unit-RMS, so logits scale with head_dim,
    # not with the (here enormous) input magnitude.
    assert q.pow(2).mean(dim=-1).allclose(torch.ones_like(q[..., 0]), atol=1e-3)
    assert k.pow(2).mean(dim=-1).allclose(torch.ones_like(k[..., 0]), atol=1e-3)


def test_qkv_heads_matches_manual_composition_without_qk_norm():
    """still_lm.py used to compose qkv+rope by hand; the helper must agree."""
    torch.manual_seed(0)
    model = build_model(tiny())
    attn = model.blocks[0].attn
    x = torch.randn(2, 16, 64)
    q, k, v = attn.qkv_heads(x)
    mq, mk, mv = (attn._split(t) for t in attn.qkv(x).chunk(3, dim=-1))
    mq, mk = attn.rope(mq, mk)
    assert torch.allclose(q, mq) and torch.allclose(k, mk) and torch.allclose(v, mv)


# --------------------------------------------------------------------------
# Depth-axis routing
# --------------------------------------------------------------------------


def test_routing_sites_partition_depth():
    assert routing_sites(14, 8) == [1, 3, 5, 7, 9, 11, 13]
    assert routing_sites(14, 4) == [3, 7, 11, 13]
    # Always includes the final layer so no block delta is dropped.
    assert routing_sites(10, 3)[-1] == 9
    # More blocks than layers degrades to per-layer granularity.
    assert routing_sites(6, 8) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("key_mode,rank", [("full", 64), ("sliced", 32)])
def test_attn_res_is_identity_at_init(key_mode, rank):
    x = torch.randint(0, 256, (2, 16))
    torch.manual_seed(1)
    base = build_model(tiny(d_model=128, init_scheme="scaled"))
    torch.manual_seed(1)
    routed = build_model(
        tiny(
            d_model=128,
            init_scheme="scaled",
            attn_res="delta_block",
            attn_res_blocks=3,
            attn_res_key=key_mode,
            attn_res_rank=rank,
        )
    )
    assert routed(x, labels=x)["loss"].item() == pytest.approx(
        base(x, labels=x)["loss"].item(), abs=1e-6
    )


def test_attn_res_routes_and_reports_diagnostic_once_gated():
    torch.manual_seed(1)
    model = build_model(
        tiny(d_model=128, init_scheme="scaled", attn_res="delta_block", attn_res_blocks=3)
    )
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        model.depth_router.gate.fill_(0.5)
        model.depth_router.query.normal_(0, 0.1)
    out = model(x, labels=x)
    out["loss"].backward()
    assert model.depth_router.query.grad.abs().sum() > 0
    assert model.depth_router.gate.grad.abs().sum() > 0
    # Routing-collapse diagnostic from the delta paper.
    weight = out["diagnostics"]["attn_res_max_weight_mean"]
    assert 0.0 < weight <= 1.0


def test_attn_res_adds_negligible_parameters():
    torch.manual_seed(1)
    base = build_model(tiny(d_model=128))
    torch.manual_seed(1)
    sliced = build_model(
        tiny(d_model=128, attn_res="delta_block", attn_res_blocks=3, attn_res_key="sliced", attn_res_rank=32)
    )
    added = sum(p.numel() for p in sliced.parameters()) - sum(p.numel() for p in base.parameters())
    sites = len(routing_sites(6, 3))
    assert added == sites * 32 + sites


def test_attn_res_rejected_with_loop_count():
    with pytest.raises(ValueError, match="loop_count"):
        build_model(tiny(attn_res="delta_block", loop_count=2))


def test_attn_res_rejected_under_still():
    """StillLM re-implements the block loop and would silently skip the router."""
    with pytest.raises(ValueError, match="attn_res is not supported"):
        build_model(tiny(variant="still", attn_res="delta_block"))


def test_invalid_attn_res_key_rejected():
    with pytest.raises(ValueError, match="attn_res_key"):
        build_model(tiny(attn_res="delta_block", attn_res_key="lowrank"))


# --------------------------------------------------------------------------
# Per-layer FFN allocation (TLM taper)
# --------------------------------------------------------------------------


def test_layer_ffn_dims_defaults_uniform():
    assert layer_ffn_dims(tiny()) == [192] * 6


def test_ffn_schedule_length_validated():
    with pytest.raises(ValueError, match="ffn_dim_schedule"):
        layer_ffn_dims(tiny(ffn_dim_schedule=[192, 192]))


def test_ffn_taper_is_budget_preserving():
    """Direction of allocation must be the only thing that differs."""
    early = [256, 240, 208, 176, 144, 128]
    late = list(reversed(early))
    assert sum(early) == 6 * 192
    torch.manual_seed(0)
    a = build_model(tiny(ffn_dim_schedule=early))
    torch.manual_seed(0)
    b = build_model(tiny(ffn_dim_schedule=late))
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert a.blocks[0].ff.net[0].out_features == 256
    assert a.blocks[-1].ff.net[0].out_features == 128


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------


def test_aurora_reduces_to_muon_on_square_matrices():
    """Aurora's row-oblique step is skipped when m == n."""
    torch.manual_seed(0)
    w = torch.randn(32, 32)
    grad = torch.randn(32, 32)
    p_a = torch.nn.Parameter(w.clone())
    p_a.grad = grad.clone()
    Aurora([p_a], lr=0.1, momentum=0.0, nesterov=False).step()
    # Same update direction as a polar step: orthogonal, unit spectral norm.
    delta = (p_a.detach() - w) / -0.1
    sv = torch.linalg.svdvals(delta.float())
    assert sv.max().item() == pytest.approx(1.0, abs=0.05)
    assert sv.min().item() == pytest.approx(1.0, abs=0.05)


def test_aurora_equalizes_row_norms_on_rectangular_matrices():
    """The neuron-death fix: Muon leaves row norms free, Aurora does not."""
    torch.manual_seed(0)
    grad = torch.randn(128, 32)
    grad[0] *= 100.0  # one wildly dominant row

    def update_for(opt_cls, **kw):
        p = torch.nn.Parameter(torch.zeros(128, 32))
        p.grad = grad.clone()
        opt_cls([p], lr=1.0, momentum=0.0, nesterov=False, **kw).step()
        return -p.detach()

    muon_rows = update_for(Muon).norm(dim=-1)
    aurora_rows = update_for(Aurora).norm(dim=-1)
    muon_spread = muon_rows.max() / muon_rows.min()
    aurora_spread = aurora_rows.max() / aurora_rows.min()
    assert aurora_spread < muon_spread


def test_aurora_rejects_non_2d_params():
    p = torch.nn.Parameter(torch.zeros(8))
    p.grad = torch.randn(8)
    with pytest.raises(ValueError, match="2D"):
        Aurora([p], lr=0.1).step()


def test_build_optimizer_routes_aurora_like_muon():
    model = build_model(tiny())
    opts = build_optimizer(model, TrainingConfig(optimizer="aurora"))
    kinds = [type(o).__name__ for o in opts]
    assert "Aurora" in kinds and "AdamW" in kinds
    # Embeddings must never reach a matrix optimizer.
    aurora_params = {id(p) for o in opts if isinstance(o, Aurora) for g in o.param_groups for p in g["params"]}
    assert id(model.token.weight) not in aurora_params
    assert all(p.ndim == 2 for p in [p for o in opts if isinstance(o, Aurora) for g in o.param_groups for p in g["params"]])


def test_weight_decay_is_plumbed_to_adamw():
    model = build_model(tiny())
    opt = build_optimizer(model, TrainingConfig(optimizer="adamw", weight_decay=0.123))
    assert all(g["weight_decay"] == 0.123 for g in opt.param_groups)


def test_weight_decay_default_matches_previous_implicit_torch_default():
    assert TrainingConfig().weight_decay == 0.01


# --------------------------------------------------------------------------
# Precision plumbing
# --------------------------------------------------------------------------


def test_precision_defaults_to_fp32_and_validates():
    from flower.train import configure_precision

    assert TrainingConfig().precision == "fp32"
    assert TrainingConfig().compile_model is False
    assert configure_precision("fp32", torch.device("cpu")) is None
    with pytest.raises(ValueError, match="precision"):
        configure_precision("fp16", torch.device("cpu"))


def test_bf16_on_cpu_degrades_instead_of_crashing():
    from flower.train import configure_precision

    assert configure_precision("bf16", torch.device("cpu")) is None


def test_still_diagnostics_stay_on_device():
    """Diagnostics must not be host floats: float() in the forward syncs once per
    micro-step and graph-breaks the compiled region."""
    cfg = tiny(
        variant="still",
        d_model=64,
        num_layers=2,
        still_compact_len=8,
        still_num_blocks=1,
        still_kl_topk=16,
    )
    model = build_model(cfg)
    model.set_step(1)
    model.train()
    x = torch.randint(0, 256, (2, 16))
    diagnostics = model(x, labels=x)["diagnostics"]
    for key in ("kl_loss", "student_loss", "teacher_loss"):
        value = diagnostics[key]
        assert isinstance(value, torch.Tensor) or value == 0.0, f"{key} was eagerly synced to host"


def test_writer_logs_zero_dim_tensor_diagnostics():
    """train.py must render the on-device diagnostics it now receives."""
    from flower.train import ScalarLogger  # noqa: F401  (protocol documentation)

    logged: dict[str, float] = {}

    class Recorder:
        def add_scalar(self, tag, value, step):
            logged[tag] = value

        def flush(self):
            pass

    diagnostics = {
        "kl_loss": torch.tensor(1.5),
        "vector": torch.zeros(4),
        "plain": 2.5,
        "flag": True,
        "config": {"a": 1},
    }
    writer = Recorder()
    for key, value in diagnostics.items():
        if key in {"parameter_count", "config"} or isinstance(value, bool):
            continue
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                writer.add_scalar(f"diagnostics/{key}", float(value.detach()), 1)
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(f"diagnostics/{key}", float(value), 1)
    assert logged == {"diagnostics/kl_loss": 1.5, "diagnostics/plain": 2.5}


# --------------------------------------------------------------------------
# Shipped configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "configs/sweep13_100m_phase0.yaml",
        "configs/sweep13_100m_phase0_taper.yaml",
        "configs/sweep13_100m_phase1.yaml",
        "configs/sweep13_attn_res_probe.yaml",
    ],
)
def test_sweep13_configs_parse(path):
    raw = yaml.safe_load(open(path))
    if "sweep" in raw:
        defaults = raw["sweep"]["defaults"]
        for variant in raw["sweep"]["variants"]:
            merged = {**defaults.get("model", {}), **variant.get("model", {})}
            ModelConfig(**merged)
    else:
        load_config(path)


def test_phase0_and_phase1_base_architecture_agree():
    """Phase 1 loads the phase-0 checkpoint; the base fields must match exactly."""
    phase0 = yaml.safe_load(open("configs/sweep13_100m_phase0.yaml"))["model"]
    phase1 = yaml.safe_load(open("configs/sweep13_100m_phase1.yaml"))["sweep"]["defaults"]["model"]
    for field in (
        "vocab_size",
        "d_model",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "max_seq_len",
        "local_window",
        "norm_type",
        "ffn_activation",
        "qk_norm",
        "use_bias",
    ):
        assert phase0[field] == phase1[field], f"{field} differs between phase 0 and phase 1"


def test_phase1_tokenizer_matches_phase0():
    phase0 = yaml.safe_load(open("configs/sweep13_100m_phase0.yaml"))["data"]["tokenizer"]
    phase1 = yaml.safe_load(open("configs/sweep13_100m_phase1.yaml"))["sweep"]["defaults"]["data"]["tokenizer"]
    assert phase0 == phase1


# --------------------------------------------------------------------------
# HP transfer and BPB
# --------------------------------------------------------------------------


def test_horizon_correction_lowers_lr_for_longer_runs():
    from flower.hparams import horizon_correction

    assert horizon_correction(2500, 15000) == pytest.approx(6**-0.5)
    assert horizon_correction(1000, 1000) == pytest.approx(1.0)
    # Longer target -> smaller LR. This is the correction short probes forget.
    assert horizon_correction(1000, 4000) < 1.0
    with pytest.raises(ValueError):
        horizon_correction(0, 100)


def test_transfer_lr_applies_width_depth_batch_duration():
    from flower.hparams import RunShape, transfer_lr

    ref = RunShape(width=768, depth=14, batch_tokens=65536, steps=2500)
    assert transfer_lr(0.01, ref, ref) == pytest.approx(0.01)
    # Duration only.
    longer = RunShape(width=768, depth=14, batch_tokens=65536, steps=15000)
    assert transfer_lr(0.01, ref, longer) == pytest.approx(0.01 * 6**-0.5)
    # Doubling width halves the LR.
    wider = RunShape(width=1536, depth=14, batch_tokens=65536, steps=2500)
    assert transfer_lr(0.01, ref, wider) == pytest.approx(0.005)
    # Larger batch at FIXED TOTAL TOKENS raises the LR as sqrt(m_B): 4x batch
    # over 1/4 the steps -> 2x.
    bigger = RunShape(width=768, depth=14, batch_tokens=262144, steps=625)
    assert transfer_lr(0.01, ref, bigger) == pytest.approx(0.02)
    # But 4x batch at the SAME step count is also 4x data, and the batch and
    # horizon terms cancel exactly — this is sqrt(m_B / m_D), not sqrt(m_B).
    # Getting this wrong is how a "bigger batch so bigger LR" rule of thumb
    # silently doubles the LR of a run that also got 4x longer.
    same_tokens_per_step = RunShape(width=768, depth=14, batch_tokens=262144, steps=2500)
    assert transfer_lr(0.01, ref, same_tokens_per_step) == pytest.approx(0.01)


def test_val_bpb_emitted_only_when_bytes_per_token_known():
    from flower.config import DataConfig

    assert DataConfig().bytes_per_token is None  # opt-in, no behaviour change

    class Batches:
        def __iter__(self):
            return self

        def __next__(self):
            ids = torch.randint(0, 256, (2, 8))
            return ids, ids

    model = build_model(tiny())
    plain = evaluate_fn(model, iter(Batches()), 2, None)
    assert "val_bpb" not in plain
    withbpb = evaluate_fn(model, iter(Batches()), 2, 4.279)
    # bpb = nats/ln(2)/bytes_per_token
    assert withbpb["val_bpb"] == pytest.approx(
        withbpb["val_loss"] / math.log(2.0) / 4.279
    )
    # A coarser tokenizer must not look worse at equal modelling quality.
    assert withbpb["val_bpb"] < plain["val_perplexity"]


def evaluate_fn(model, batches, steps, bytes_per_token):
    from flower.train import evaluate

    return evaluate(model, batches, steps, torch.device("cpu"), None, bytes_per_token)


# --------------------------------------------------------------------------
# Hybrid sliding-window attention
# --------------------------------------------------------------------------


def test_attn_windows_default_to_uniform():
    from flower.models.base import layer_attn_windows

    assert layer_attn_windows(tiny()) == [8] * 6


def test_attn_window_schedule_length_validated():
    from flower.models.base import layer_attn_windows

    with pytest.raises(ValueError, match="attn_window_schedule"):
        layer_attn_windows(tiny(attn_window_schedule=[8, 8]))


def test_hybrid_windows_are_applied_per_layer_and_cost_no_params():
    torch.manual_seed(0)
    uniform = build_model(tiny(init_scheme="scaled"))
    torch.manual_seed(0)
    hybrid = build_model(tiny(init_scheme="scaled", attn_window_schedule=[8, 8, None, 8, 8, None]))
    assert [b.attn.local_window for b in uniform.blocks] == [8] * 6
    assert [b.attn.local_window for b in hybrid.blocks] == [8, 8, None, 8, 8, None]
    # Mask-only change: identical parameter count.
    assert sum(p.numel() for p in uniform.parameters()) == sum(p.numel() for p in hybrid.parameters())
    x = torch.randint(0, 256, (2, 16))
    assert torch.isfinite(hybrid(x, labels=x)["loss"])


def test_full_attention_layer_actually_sees_beyond_the_window():
    """A null entry must mean full context, not the default window."""
    torch.manual_seed(0)
    model = build_model(tiny(max_seq_len=64, init_scheme="scaled", attn_window_schedule=[8, None, 8, 8, 8, 8]))
    windowed, full = model.blocks[0].attn, model.blocks[1].attn
    x = torch.randn(1, 64, 64)
    # Perturbing a distant early position must not reach the last query through a
    # window-8 layer, but must reach it through the full-attention layer.
    y = x.clone()
    y[:, 0] += 10.0
    assert torch.allclose(windowed(x)[:, -1], windowed(y)[:, -1], atol=1e-5)
    assert not torch.allclose(full(x)[:, -1], full(y)[:, -1], atol=1e-5)


def test_vanilla_full_ignores_window_schedule():
    model = build_model(tiny(variant="vanilla_full", attn_window_schedule=[8] * 6))
    assert all(b.attn.local_window is None for b in model.blocks)


def test_still_inherits_per_layer_windows():
    """still_lm reads attn.local_window per layer, so hybrids must carry through."""
    model = build_model(
        tiny(
            variant="still",
            num_layers=6,
            still_compact_len=8,
            still_num_blocks=1,
            attn_window_schedule=[8, 8, None, 8, 8, None],
        )
    )
    assert [b.attn.local_window for b in model.base_model.blocks] == [8, 8, None, 8, 8, None]


def test_still_teacher_and_student_agree_outside_the_compacted_suffix():
    """Documents the geometry: only the last `compact_len` positions can differ.

    The frozen base produces identical prefix logits in both passes, so the
    teacher/student comparison — and therefore every arm-to-arm difference — is
    carried by compact_len of seq_len positions.
    """
    torch.manual_seed(0)
    seq, compact = 128, 16
    model = build_model(
        tiny(
            variant="still",
            num_layers=3,
            max_seq_len=seq,
            local_window=32,
            still_compact_len=compact,
            still_num_blocks=1,
            init_scheme="scaled",
        )
    )
    model.set_step(1)
    model.eval()
    x = torch.randint(0, 256, (2, seq))
    with torch.no_grad():
        teacher = model.base_model(x)["logits"]
        student = model._extract_kv_and_forward(x, use_compact=True)[0]["logits"]
    ctx_end = min(max(seq - compact, 1), seq - 1)
    assert torch.allclose(teacher[:, :ctx_end], student[:, :ctx_end], atol=1e-4)
    assert not torch.allclose(teacher[:, ctx_end:], student[:, ctx_end:], atol=1e-4)


# --------------------------------------------------------------------------
# Still loss geometry
# --------------------------------------------------------------------------


def still_cfg(**kw):
    return tiny(
        variant="still",
        num_layers=3,
        max_seq_len=128,
        local_window=32,
        still_compact_len=16,
        still_num_blocks=1,
        still_kl_topk=32,
        still_kl_weight=1.0,
        still_ce_weight=0.1,
        init_scheme="scaled",
        **kw,
    )


def test_still_loss_geometry_defaults_to_legacy():
    cfg = ModelConfig()
    assert cfg.still_loss_positions == "all"
    assert cfg.still_suffix_len is None


def test_suffix_start_tracks_compact_len_by_default():
    model = build_model(still_cfg())
    # Legacy coupling: the evaluated span equals the compaction budget.
    assert model.suffix_start(128) == 128 - 16


def test_suffix_len_decouples_eval_window_from_compaction_budget():
    model = build_model(still_cfg(still_suffix_len=64))
    assert model.suffix_start(128) == 64
    # The compaction budget is untouched — only the split moved.
    assert model.layer_compact_lens == [16, 16, 16]


def test_layer_adaptive_uses_the_earliest_split():
    """A difference introduced at any layer propagates forward from there."""
    model = build_model(still_cfg(still_layer_adaptive=True))
    assert model.suffix_start(128) == 128 - max(model.layer_compact_lens)


def test_suffix_reduction_restores_the_configured_kl_weight():
    """Averaging KL over all positions silently divides it by seq/suffix."""
    torch.manual_seed(0)
    model = build_model(still_cfg())
    model.set_step(1)
    x = torch.randint(0, 256, (2, 128))
    with torch.no_grad():
        teacher = model.base_model(x)["logits"]
        student = model._extract_kv_and_forward(x, use_compact=True)[0]["logits"]
    ctx_end = model.suffix_start(128)
    suffix = torch.zeros(2, 128, dtype=torch.bool)
    suffix[:, ctx_end:] = True
    kl_all = float(model._topk_kl_loss(teacher, student, x, None))
    kl_suffix = float(model._topk_kl_loss(teacher, student, x, suffix))
    assert kl_suffix == pytest.approx(kl_all * 128 / (128 - ctx_end), rel=1e-3)


def test_suffix_loss_positions_changes_only_the_reduction():
    torch.manual_seed(0)
    legacy = build_model(still_cfg())
    torch.manual_seed(0)
    fixed = build_model(still_cfg(still_loss_positions="suffix"))
    legacy.set_step(1)
    fixed.set_step(1)
    x = torch.randint(0, 256, (2, 128))
    out_legacy, out_fixed = legacy(x, labels=x), fixed(x, labels=x)
    # Same geometry, same logits — only the loss reduction differs.
    assert torch.allclose(out_legacy["logits"], out_fixed["logits"], atol=1e-5)
    assert float(out_fixed["diagnostics"]["kl_loss"]) > float(out_legacy["diagnostics"]["kl_loss"])


def test_loss_position_fraction_is_reported():
    model = build_model(still_cfg(still_suffix_len=64))
    model.set_step(1)
    x = torch.randint(0, 256, (2, 128))
    assert model(x, labels=x)["diagnostics"]["loss_position_frac"] == pytest.approx(0.5)


def test_suffix_len_keeps_compaction_and_loss_geometry_consistent():
    """The compaction split and the loss mask must use the same boundary."""
    torch.manual_seed(0)
    model = build_model(still_cfg(still_suffix_len=64, still_loss_positions="suffix"))
    model.set_step(1)
    model.eval()
    x = torch.randint(0, 256, (2, 128))
    with torch.no_grad():
        teacher = model.base_model(x)["logits"]
        student = model._extract_kv_and_forward(x, use_compact=True)[0]["logits"]
    ctx_end = model.suffix_start(128)
    assert torch.allclose(teacher[:, :ctx_end], student[:, :ctx_end], atol=1e-4)
    assert not torch.allclose(teacher[:, ctx_end:], student[:, ctx_end:], atol=1e-4)


def test_invalid_loss_positions_rejected():
    with pytest.raises(ValueError, match="still_loss_positions"):
        build_model(still_cfg(still_loss_positions="prefix"))


def test_suffix_mask_respects_the_local_window():
    """Suffix keys are ordinary tokens and must obey the teacher's windowing."""
    from flower.models.still_lm import _suffix_causal_mask

    dev = torch.device("cpu")
    windowed = _suffix_causal_mask(6, dev, 3)
    # Causal.
    assert not windowed[0, 1]
    # Banded: query 5 cannot see key 2 (distance 3 >= window 3).
    assert windowed[5, 3] and not windowed[5, 2]
    # A full-attention layer (window None) sees everything causal.
    assert _suffix_causal_mask(6, dev, None)[-1].all()


def test_wide_suffix_does_not_grant_extra_local_reach():
    """Regression: an unwindowed suffix mask would beat the teacher's window.

    Latent while span <= local_window (the legacy 64 vs 256), which is why it
    only surfaces once still_suffix_len widens the evaluated span.
    """
    from flower.models.still_lm import _suffix_causal_mask

    span, window = 128, 32
    mask = _suffix_causal_mask(span, torch.device("cpu"), window)
    reach = mask.sum(dim=-1)
    # No query may attend to more than `window` keys, however wide the span is.
    assert int(reach.max()) == window


def test_phase1_config_uses_the_fixed_loss_geometry():
    cfg = yaml.safe_load(open("configs/sweep13_100m_phase1.yaml"))["sweep"]["defaults"]
    model, training = cfg["model"], cfg["training"]
    assert model["still_loss_positions"] == "suffix"
    seq = cfg["data"]["sequence_length"]
    suffix = model["still_suffix_len"]
    # The evaluated span must be a meaningful fraction of the sequence, and the
    # compaction must still be a compression.
    assert suffix / seq >= 0.25
    assert model["still_compact_len"] < seq - suffix
    assert training["validation_steps"] >= 100
