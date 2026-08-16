"""Tests for the SDP cross-attention that replaced ``nn.MultiheadAttention`` in
``summary_memory`` and ``bloom_memory`` (NEXT_IDEAS.md section 4 blocker fix).

Two properties under test:

1. ``SDPCrossAttention`` is a numerical drop-in for the ``nn.MultiheadAttention``
   cross-attention it replaces: with the same projection weights and the same
   inputs, the two produce identical outputs. This pins the refactor against a
   silent behavioural change — the whole point was to fix the graph break / OOM,
   not to change the math.

2. ``remap_legacy_mha_state_dict`` rewrites a pre-fix checkpoint's
   ``nn.MultiheadAttention`` parameters (``in_proj_weight`` / ``in_proj_bias`` /
   ``out_proj.*``) into the new ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj``
   layout so the existing summary_memory / bloom_memory checkpoints
   (sweep7/13) keep loading. Idempotent on new-format state_dicts.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flower.config import ModelConfig
from flower.models import build_model
from flower.models.memory import SDPCrossAttention, remap_legacy_mha_state_dict


def _cfg(variant: str, **over) -> ModelConfig:
    base = dict(
        variant=variant,
        vocab_size=128,
        d_model=32,
        num_heads=4,
        num_layers=2,
        ffn_dim=64,
        max_seq_len=64,
        local_window=16,
        memory_slots=8,
        use_bias=True,
        # summary needs perceiver; bloom needs its hash params to construct.
        summary_style="perceiver",
        bloom_num_hashes=4,
        bloom_summary_points=4,
    )
    base.update(over)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# SDPCrossAttention numerical equivalence to nn.MultiheadAttention
# ---------------------------------------------------------------------------


def test_sdpa_cross_attention_matches_multihead_attention():
    """With identical projection weights and inputs, SDPCrossAttention reproduces
    nn.MultiheadAttention's cross-attention output exactly.

    nn.MultiheadAttention stores in_proj_weight = [Wq; Wk; Wv] (row-stacked,
    shape (3D, D)) and in_proj_bias = [bq; bk; bv]; out_proj is a plain Linear.
    SDPCrossAttention holds these as four separate Linears. Copying the stacked
    weights into the three projections and the out_proj across makes the two
    compute the same matmuls + the same SDPA (no mask = cross-attention), so the
    result must match to float precision. The match is exact in practice (0.0
    abs diff on CPU) because both paths reduce to the identical SDPA kernel.
    """
    D, H, P, T, B = 192, 8, 16, 256, 2
    cfg = _cfg("summary_memory", d_model=D, num_heads=H)
    mha = nn.MultiheadAttention(D, H, batch_first=True)
    sdp = SDPCrossAttention(cfg)

    # Copy MHA's stacked in_proj_weight [Wq; Wk; Wv] into the three projections.
    wq, wk, wv = mha.in_proj_weight.tensor_split(3, dim=0)
    bq, bk, bv = mha.in_proj_bias.tensor_split(3, dim=0)
    with torch.no_grad():
        sdp.q_proj.weight.copy_(wq)
        sdp.q_proj.bias.copy_(bq)
        sdp.k_proj.weight.copy_(wk)
        sdp.k_proj.bias.copy_(bk)
        sdp.v_proj.weight.copy_(wv)
        sdp.v_proj.bias.copy_(bv)
        sdp.out_proj.weight.copy_(mha.out_proj.weight)
        sdp.out_proj.bias.copy_(mha.out_proj.bias)

    q_input = torch.randn(B, P, D)
    kv_input = torch.randn(B, T, D)

    out_ref, _ = mha(q_input, kv_input, kv_input, need_weights=False)
    out_new = sdp(q_input, kv_input)

    assert out_new.shape == out_ref.shape == (B, P, D)
    assert torch.equal(out_new, out_ref)  # exact on CPU; tight fallback below
    assert torch.allclose(out_new, out_ref, atol=1e-6)


def test_sdpa_cross_attention_param_count_matches_multihead_attention():
    """The refactor must not change the param count — summary/bloom are part of a
    param-matched bake-off (NEXT_IDEAS.md section 4), so any capacity shift
    confounds the comparison. nn.MultiheadAttention and SDPCrossAttention both
    hold 4*D*D + 4*D params."""
    D, H = 192, 8
    cfg = _cfg("summary_memory", d_model=D, num_heads=H)
    mha = nn.MultiheadAttention(D, H, batch_first=True)
    sdp = SDPCrossAttention(cfg)
    mha_params = sum(p.numel() for p in mha.parameters())
    sdp_params = sum(p.numel() for p in sdp.parameters())
    assert mha_params == sdp_params == 4 * D * D + 4 * D


def test_sdpa_cross_attention_backward_reaches_all_projections():
    """All four projections receive a gradient — guards against a wiring typo
    that would silently leave one projection frozen."""
    D, H, P, T, B = 64, 4, 8, 32, 2
    cfg = _cfg("summary_memory", d_model=D, num_heads=H)
    sdp = SDPCrossAttention(cfg)
    q_input = torch.randn(B, P, D)
    kv_input = torch.randn(B, T, D)
    sdp(q_input, kv_input).float().sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        proj = getattr(sdp, name)
        assert proj.weight.grad is not None, f"{name}.weight got no grad"
        assert torch.isfinite(proj.weight.grad).all()


# ---------------------------------------------------------------------------
# Checkpoint compatibility shim — remap_legacy_mha_state_dict
# ---------------------------------------------------------------------------


def _legacy_mha_state_dict(prefix_attr: str, d_model: int) -> dict[str, torch.Tensor]:
    """Build a synthetic legacy nn.MultiheadAttention state_dict for one block,
    under the attribute name ``prefix_attr`` (e.g. 'perceiver' or 'summary_attn')."""
    mha = nn.MultiheadAttention(d_model, 8, batch_first=True)
    base = f"blocks.0.{prefix_attr}"
    return {
        f"{base}.in_proj_weight": mha.in_proj_weight.clone(),
        f"{base}.in_proj_bias": mha.in_proj_bias.clone(),
        f"{base}.out_proj.weight": mha.out_proj.weight.clone(),
        f"{base}.out_proj.bias": mha.out_proj.bias.clone(),
    }


def test_remap_legacy_mha_splits_in_proj_into_qkv():
    """in_proj_weight [Wq;Wk;Wv] -> q_proj/k_proj/v_proj; biases likewise."""
    D = 192
    legacy = _legacy_mha_state_dict("perceiver", D)
    remapped = remap_legacy_mha_state_dict(dict(legacy))

    base = "blocks.0.perceiver"
    # New keys present, old keys gone.
    for k in ("q_proj", "k_proj", "v_proj"):
        assert f"{base}.{k}.weight" in remapped
        assert f"{base}.{k}.bias" in remapped
    assert f"{base}.out_proj.weight" in remapped
    assert f"{base}.out_proj.bias" in remapped
    assert f"{base}.in_proj_weight" not in remapped
    assert f"{base}.in_proj_bias" not in remapped

    # Values: the stacked weight split into three contiguous slices.
    wq, wk, wv = legacy[f"{base}.in_proj_weight"].tensor_split(3, dim=0)
    bq, bk, bv = legacy[f"{base}.in_proj_bias"].tensor_split(3, dim=0)
    assert torch.equal(remapped[f"{base}.q_proj.weight"], wq)
    assert torch.equal(remapped[f"{base}.k_proj.weight"], wk)
    assert torch.equal(remapped[f"{base}.v_proj.weight"], wv)
    assert torch.equal(remapped[f"{base}.q_proj.bias"], bq)
    assert torch.equal(remapped[f"{base}.k_proj.bias"], bk)
    assert torch.equal(remapped[f"{base}.v_proj.bias"], bv)
    assert torch.equal(remapped[f"{base}.out_proj.weight"], legacy[f"{base}.out_proj.weight"])
    assert torch.equal(remapped[f"{base}.out_proj.bias"], legacy[f"{base}.out_proj.bias"])


def test_remap_legacy_mha_is_idempotent_on_new_format():
    """A state_dict already in the new format (q_proj/k_proj/v_proj) passes through
    untouched — no in_proj_weight to split, so the remap is a no-op."""
    cfg = _cfg("summary_memory", d_model=64)
    block = build_model(cfg).blocks[0]
    sd = {f"blocks.0.perceiver.{k}": v.clone() for k, v in block.perceiver.state_dict().items()}
    remapped = remap_legacy_mha_state_dict(dict(sd))
    assert set(remapped.keys()) == set(sd.keys())
    for k in sd:
        assert torch.equal(remapped[k], sd[k])


def test_remap_legacy_mha_noop_for_unrelated_state_dict():
    """A state_dict with neither perceiver.* nor summary_attn.* MHA keys is
    returned unchanged (e.g. a vanilla checkpoint)."""
    sd = {"blocks.0.local.qkv.weight": torch.randn(96, 32), "tok_emb.weight": torch.randn(128, 32)}
    remapped = remap_legacy_mha_state_dict(dict(sd))
    assert set(remapped.keys()) == set(sd.keys())


def test_remap_legacy_mha_drops_bias_for_use_bias_false_target():
    """nn.MultiheadAttention always carries bias, but SDPCrossAttention respects
    config.use_bias. A use_bias=False checkpoint (e.g. sweep13) has MHA bias
    tensors that the new bias-free module does not expect; bias=False must drop
    them so the strict load matches. Catches the real sweep13-bloom load failure."""
    D = 192
    legacy = _legacy_mha_state_dict("summary_attn", D)
    remapped = remap_legacy_mha_state_dict(dict(legacy), bias=False)
    base = "blocks.0.summary_attn"
    # Weights present (q/k/v/out), biases gone.
    for k in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert f"{base}.{k}.weight" in remapped
    assert not any(".bias" in k and k.startswith(base) for k in remapped)


@pytest.mark.parametrize("bias", [True, False])
def test_remap_legacy_mha_loads_strictly_at_either_bias_setting(bias: bool):
    """End-to-end: a legacy MHA state_dict remapped with the matching `bias`
    flag must load strictly into a model built at that use_bias setting."""
    D = 64
    cfg = _cfg("summary_memory", d_model=D, num_heads=8, use_bias=bias)
    model = build_model(cfg)
    # Fabricate a legacy state_dict from the new model: collapse q/k/v back to
    # the stacked in_proj layout, keep out_proj.
    full_sd = {k: v.clone() for k, v in model.state_dict().items()}
    for i in range(cfg.num_layers):
        base = f"blocks.{i}.perceiver"
        wq = full_sd.pop(f"{base}.q_proj.weight")
        wk = full_sd.pop(f"{base}.k_proj.weight")
        wv = full_sd.pop(f"{base}.v_proj.weight")
        full_sd[f"{base}.in_proj_weight"] = torch.cat([wq, wk, wv], dim=0)
        if bias:
            bq = full_sd.pop(f"{base}.q_proj.bias")
            bk = full_sd.pop(f"{base}.k_proj.bias")
            bv = full_sd.pop(f"{base}.v_proj.bias")
            full_sd[f"{base}.in_proj_bias"] = torch.cat([bq, bk, bv], dim=0)
    remapped = remap_legacy_mha_state_dict(dict(full_sd), bias=bias)
    fresh = build_model(cfg)
    fresh.load_state_dict(remapped)  # strict=True — no missing/unexpected keys


@pytest.mark.parametrize("variant,attr", [("summary_memory", "perceiver"), ("bloom_memory", "summary_attn")])
def test_legacy_checkpoint_loads_into_new_model_end_to_end(variant: str, attr: str):
    """Simulate resuming a pre-fix checkpoint: fabricate a full legacy state_dict
    by replacing the new module's q/k/v/out_proj with the old MHA in_proj/out_proj
    layout, remap it, and load strictly into a freshly-built new model."""
    D = 64
    cfg = _cfg(variant, d_model=D, num_heads=8)
    model = build_model(cfg)
    full_sd = {k: v.clone() for k, v in model.state_dict().items()}

    # Rewrite the per-block cross-attention params back to the legacy MHA layout.
    base = f"blocks.0.{attr}"
    wq = full_sd.pop(f"{base}.q_proj.weight")
    wk = full_sd.pop(f"{base}.k_proj.weight")
    wv = full_sd.pop(f"{base}.v_proj.weight")
    bq = full_sd.pop(f"{base}.q_proj.bias")
    bk = full_sd.pop(f"{base}.k_proj.bias")
    bv = full_sd.pop(f"{base}.v_proj.bias")
    full_sd[f"{base}.in_proj_weight"] = torch.cat([wq, wk, wv], dim=0)
    full_sd[f"{base}.in_proj_bias"] = torch.cat([bq, bk, bv], dim=0)
    # out_proj.* keep the same key names in both layouts; leave them.

    # Do the same for block 1 if it exists, so the strict load has no missing keys.
    if cfg.num_layers > 1:
        for i in range(1, cfg.num_layers):
            b = f"blocks.{i}.{attr}"
            wq_i = full_sd.pop(f"{b}.q_proj.weight")
            wk_i = full_sd.pop(f"{b}.k_proj.weight")
            wv_i = full_sd.pop(f"{b}.v_proj.weight")
            bq_i = full_sd.pop(f"{b}.q_proj.bias")
            bk_i = full_sd.pop(f"{b}.k_proj.bias")
            bv_i = full_sd.pop(f"{b}.v_proj.bias")
            full_sd[f"{b}.in_proj_weight"] = torch.cat([wq_i, wk_i, wv_i], dim=0)
            full_sd[f"{b}.in_proj_bias"] = torch.cat([bq_i, bk_i, bv_i], dim=0)

    remapped = remap_legacy_mha_state_dict(dict(full_sd))
    fresh = build_model(cfg)
    fresh.load_state_dict(remapped)  # strict=True by default
    # The cross-attention weights round-trip exactly.
    for i in range(cfg.num_layers):
        proj = getattr(fresh.blocks[i], attr)
        orig = getattr(model.blocks[i], attr)
        assert torch.equal(proj.q_proj.weight, orig.q_proj.weight)
        assert torch.equal(proj.k_proj.weight, orig.k_proj.weight)
        assert torch.equal(proj.v_proj.weight, orig.v_proj.weight)


# ---------------------------------------------------------------------------
# Forward / shape validation for the refactored arms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["summary_memory", "bloom_memory"])
def test_refactored_arms_forward_shape_and_finite_loss(variant: str):
    """Both refactored arms must forward correctly: right output shape, finite
    loss. Pins that the SDPCrossAttention swap didn't break the forward path."""
    cfg = _cfg(variant, d_model=64, vocab_size=128, max_seq_len=64, local_window=16, memory_slots=8)
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 64))
    out = model(tokens, labels=tokens)
    assert out["logits"].shape == (2, 64, cfg.vocab_size)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
