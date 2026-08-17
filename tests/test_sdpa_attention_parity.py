"""Dense-path SDPA switch for flow/hamiltonian attention: numerical parity.

FlowSelfAttention and HamiltonianSelfAttention used to run their dense
(non-flex) path through `scaled_dot_attention`, which materializes the full
(B, H, T, T) score tensor — the OOM/spill-prone path CausalSelfAttention
documents abandoning in favour of F.scaled_dot_product_attention. The swap
must be numerically equivalent, not merely close: published runs of these
variants have to reproduce. These tests pin that by recomputing the old
dense path from the module's own internals and comparing.

Also pins the bool-mask cache: same object on a key hit, correct rebuild on
seq_len/window change, and content equal to a freshly built causal_mask.
"""

from __future__ import annotations

import pytest
import torch

from flower.config import ModelConfig
from flower.models.base import causal_mask, scaled_dot_attention
from flower.models.flow_attention import FlowSelfAttention
from flower.models.hamiltonian_attention import HamiltonianSelfAttention


def tiny_config(**overrides) -> ModelConfig:
    defaults = dict(
        vocab_size=128,
        d_model=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=64,
        local_window=8,
        flow_steps=1,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def legacy_dense_forward(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The pre-SDPA dense path, recomputed from the module's own internals.

    Shares qkv/split/flows with the module so the comparison isolates exactly
    the attention-kernel swap.
    """
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    q, k, v = attn._split(q), attn._split(k), attn._split(v)
    q = attn.q_flow(q)
    k = attn.k_flow(k)
    mask = causal_mask(x.shape[1], x.device, attn.local_window).view(1, 1, x.shape[1], x.shape[1])
    out = scaled_dot_attention(q, k, v, mask)
    out = out.transpose(1, 2).contiguous().view(x.shape)
    return attn.out(out)


ATTENTION_MODULES = [
    ("flow", FlowSelfAttention),
    ("hamiltonian", HamiltonianSelfAttention),
]


@pytest.mark.parametrize("module_name,attn_cls", ATTENTION_MODULES)
@pytest.mark.parametrize("local_window", [None, 4])
def test_dense_path_matches_legacy_scaled_dot_attention(module_name, attn_cls, local_window):
    torch.manual_seed(0)
    cfg = tiny_config(local_window=local_window)
    attn = attn_cls(cfg, local_window).eval()
    x = torch.randn(2, 16, cfg.d_model)
    with torch.no_grad():
        new_out = attn(x)
        ref_out = legacy_dense_forward(attn, x)
    assert torch.allclose(new_out, ref_out, atol=1e-5, rtol=1e-5), (
        f"{module_name} dense path diverged from legacy scaled_dot_attention: "
        f"max abs diff {(new_out - ref_out).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("module_name,attn_cls", ATTENTION_MODULES)
def test_bool_mask_cache_hits_and_rebuilds(module_name, attn_cls):
    torch.manual_seed(0)
    cfg = tiny_config(local_window=8)
    attn = attn_cls(cfg, cfg.local_window).eval()
    device = torch.device("cpu")

    # Cold: builds and stores.
    first = attn._get_causal_bool_mask(16, device)
    assert first.shape == (16, 16) and first.dtype == torch.bool
    assert torch.equal(first, causal_mask(16, device, attn.local_window))
    # Same key: same object, not a rebuild.
    assert attn._get_causal_bool_mask(16, device) is first
    # Different seq_len: rebuild at the new shape.
    longer = attn._get_causal_bool_mask(32, device)
    assert longer.shape == (32, 32)
    assert torch.equal(longer, causal_mask(32, device, attn.local_window))
    # Different window: rebuild with the local constraint.
    attn.local_window = 4
    narrower = attn._get_causal_bool_mask(16, device)
    assert torch.equal(narrower, causal_mask(16, device, 4))
    # Full forward leaves the cache warm for the next call.
    x = torch.randn(2, 16, cfg.d_model)
    with torch.no_grad():
        attn(x)
    assert attn._get_causal_bool_mask(16, device) is attn._cached_attn_mask
