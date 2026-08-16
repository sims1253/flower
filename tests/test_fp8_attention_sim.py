"""Tests for the FP8-attention quality probe (`model.fp8_attention_sim`).

This flag exists to answer "does FP8-precision attention hurt trained quality?"
without first writing an FP8 attention backward. Its whole value depends on two
properties that are easy to break silently:

  1. the forward must be *genuinely* e4m3-rounded (a no-op would make the probe
     report "FP8 is free" while measuring nothing), and
  2. the gradient must pass straight through (a wrong gradient would make the
     arm diverge for reasons that have nothing to do with FP8 precision).

Both are asserted here. CPU-only.
"""

from __future__ import annotations

import pytest
import torch

from flower.models.base import _fake_quant_e4m3


def test_forward_is_actually_quantized():
    """A no-op would silently turn the probe into a null experiment."""
    torch.manual_seed(0)
    x = torch.randn(2, 2, 32, 16) * 0.5
    y = _fake_quant_e4m3(x)
    assert not torch.equal(y, x), "fake-quant returned its input unchanged"
    assert (y - x).abs().mean() > 0, "no quantization error introduced"


def test_forward_matches_explicit_quantize_dequantize():
    torch.manual_seed(0)
    x = torch.randn(2, 2, 32, 16) * 0.5
    scale = x.abs().amax().clamp(min=1e-12) / 448.0
    expected = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype) * scale
    assert torch.equal(_fake_quant_e4m3(x), expected)


def test_gradient_passes_straight_through():
    """Straight-through estimator: d(out)/d(in) must be exactly identity.

    If this regresses, the probe arm's loss curve would reflect a broken
    gradient rather than FP8 precision, and would be misread as "FP8 attention
    diverges".
    """
    torch.manual_seed(0)
    x = torch.randn(2, 2, 8, 16, requires_grad=True)
    y = _fake_quant_e4m3(x)
    g = torch.randn_like(y)
    y.backward(g)
    assert torch.equal(x.grad, g)


def test_quantization_error_is_far_above_bf16():
    """Guards the premise: e4m3 must be materially coarser than bf16.

    The measured motivation for the probe is that e4m3 attention carries ~18x
    bf16's output error. If this assertion ever fails, the probe is no longer
    testing what it claims to.
    """
    torch.manual_seed(0)
    x = torch.randn(4096, 64)
    e_fp8 = (_fake_quant_e4m3(x) - x).abs().mean()
    e_bf16 = (x.to(torch.bfloat16).float() - x).abs().mean()
    assert e_fp8 > 5 * e_bf16, f"e4m3 err {e_fp8:.5f} not materially worse than bf16 {e_bf16:.5f}"


def test_handles_all_zero_input_without_nan():
    """amax==0 must not divide by zero — a fully-masked or dead head is real."""
    x = torch.zeros(2, 2, 8, 16)
    y = _fake_quant_e4m3(x)
    assert torch.isfinite(y).all()
    assert torch.equal(y, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_preserves_shape_and_dtype(dtype):
    x = torch.randn(2, 3, 8, 16, dtype=dtype)
    y = _fake_quant_e4m3(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
