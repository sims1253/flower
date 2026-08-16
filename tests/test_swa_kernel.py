"""Correctness tests for the specialized sliding-window attention kernels.

Neither kernel is wired into the model (both lost to flex — see their module
docstrings). They are kept as verified reference implementations and as the
evidence closing the "beat flex on attention" direction. These tests keep them
honest: a reference implementation that has silently rotted is worse than none,
because the next person re-derives the wrong conclusion from it.

CUDA + Triton only; skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda_only
@pytest.mark.parametrize("B,H,T,D,W", [(1, 2, 256, 64, 128), (2, 3, 512, 64, 256), (2, 2, 512, 32, 512)])
def test_swa_forward_matches_dense_reference(B, H, T, D, W):
    from flower.kernels.fp8_swa_attention import reference_swa_attention
    from flower.kernels.swa_attention import swa_attention

    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = swa_attention(q, k, v, W)
    ref = reference_swa_attention(q.float(), k.float(), v.float(), W)
    rel = (out.float() - ref).abs().mean() / ref.abs().mean()
    # bf16 rounding level; the kernel accumulates in fp32 like flex does.
    assert rel < 0.01, f"forward relative error {rel:.5f}"


@cuda_only
@pytest.mark.parametrize("B,H,T,D,W", [(1, 2, 256, 64, 128), (2, 2, 512, 64, 256)])
def test_swa_gradients_match_dense_reference(B, H, T, D, W):
    """All three gradients, not just the output — a wrong dK/dV would still
    produce a plausible-looking forward."""
    from flower.kernels.fp8_swa_attention import reference_swa_attention
    from flower.kernels.swa_attention import swa_attention

    torch.manual_seed(0)
    base = [torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q, k, v = (t.clone().requires_grad_(True) for t in base)
    qr, kr, vr = (t.clone().float().requires_grad_(True) for t in base)
    g = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    swa_attention(q, k, v, W).backward(g)
    reference_swa_attention(qr, kr, vr, W).backward(g.float())

    for name, got, want in [("dq", q.grad, qr.grad), ("dk", k.grad, kr.grad), ("dv", v.grad, vr.grad)]:
        rel = (got.float() - want).abs().mean() / (want.abs().mean() + 1e-12)
        assert torch.isfinite(got).all(), f"{name} has non-finite entries"
        assert rel < 0.02, f"{name} relative error {rel:.5f}"


@cuda_only
def test_swa_rejects_full_context_window():
    """window<=0 must route to flex rather than silently computing nothing."""
    from flower.kernels.swa_attention import swa_attention

    q, k, v = (torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(ValueError, match="window must be positive"):
        swa_attention(q, k, v, 0)


@cuda_only
def test_fp8_swa_forward_refuses_grad_tensors():
    """It is forward-only; silently detaching would be a training-corrupting bug."""
    from flower.kernels.fp8_swa_attention import fp8_swa_attention_forward

    q, k, v = (
        torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        for _ in range(3)
    )
    with pytest.raises(RuntimeError, match="forward-only"):
        fp8_swa_attention_forward(q, k, v, 64)
