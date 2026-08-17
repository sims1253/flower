"""Probes must not leave the model reconfigured.

compaction_kl_curve overrides each compactor's compact_len and latents to hit
the requested compression ratios. Before the try/finally restore this was a
permanent mutation: anything evaluated after the probe ran against the
last-ratio (most aggressive) compaction setting. These tests pin that the
full still eval suite leaves every state_dict entry bit-identical and every
compactor holding its original Parameter objects.
"""

from __future__ import annotations

import torch

from flower.config import DataConfig, ExperimentConfig, ModelConfig
from flower.models import build_model
from flower.probes.still_eval import compaction_kl_curve, run_still_composite_eval


def tiny_still_config() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            variant="still",
            vocab_size=256,
            d_model=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            max_seq_len=64,
            local_window=32,
            still_compact_len=16,
            still_num_blocks=1,
            still_d_latent=32,
            still_kl_topk=50,
            still_kl_weight=1.0,
            still_ce_weight=0.0,
        ),
        data=DataConfig(dataset="synthetic", sequence_length=64, synthetic_vocab_size=256),
    )


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def assert_bit_identical(before: dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    after = model.state_dict()
    assert set(before) == set(after), "state_dict keys changed"
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} was mutated by a probe"


def test_kl_curve_restores_compactor_state():
    torch.manual_seed(0)
    cfg = tiny_still_config()
    model = build_model(cfg.model)
    device = torch.device("cpu")

    before_state = snapshot(model)
    before_compact = [(comp.compact_len, comp.latents) for comp in model.compactors]

    out = compaction_kl_curve(
        model, cfg, device, batches=1, batch_size=2, compression_ratios=(2, 8)
    )

    assert set(out) == {"2", "8"}
    assert out["8"]["compact_len"] == 8  # 64 // 8
    assert_bit_identical(before_state, model)
    for (compact_len, latents), comp in zip(before_compact, model.compactors):
        assert comp.compact_len == compact_len, "compact_len not restored"
        assert comp.latents is latents, "latents Parameter not restored by identity"


def test_full_still_eval_leaves_model_bit_identical():
    torch.manual_seed(0)
    cfg = tiny_still_config()
    model = build_model(cfg.model)
    device = torch.device("cpu")

    before_state = snapshot(model)
    before_compact = [(comp.compact_len, comp.latents) for comp in model.compactors]

    out = run_still_composite_eval(model, cfg, device=device)

    # Dropped decorative metrics stay dropped.
    assert "iterative_compaction" not in out
    assert "compression_utility" not in out
    assert set(out) == {"variant", "compaction_kl_curve", "needle_through_compaction"}
    assert_bit_identical(before_state, model)
    for (compact_len, latents), comp in zip(before_compact, model.compactors):
        assert comp.compact_len == compact_len, "compact_len not restored"
        assert comp.latents is latents, "latents Parameter not restored by identity"
