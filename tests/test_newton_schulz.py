"""Tests for the Newton-Schulz orthogonalisation primitive in flower.optim.

Covers both paths:
  - `_zeropower_via_newtonschulz5`      — the per-matrix (legacy) NS iteration
  - `_zeropower_newtonschulz5_batched`  — the batched `bmm` path (NEXT_IDEAS.md §5)

The batched path is required to match the legacy path numerically (a `bmm` over a
same-shape stack reduces slice-for-slice to looping the legacy `mm`), and both
must produce a valid approximate polar factor (largest singular value ≈ 1). All
three schedules (quintic5 / cubic5 / hybrid_v4) must keep working.
"""

from __future__ import annotations

import pytest
import torch

from flower.optim import (
    _ns_schedule,
    _zeropower_newtonschulz5_batched,
    _zeropower_via_newtonschulz5,
)

CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICES = [CUDA] if CUDA.type == "cuda" else [torch.device("cpu")]
# Shapes from the spec: a mix of wide, tall, rectangular-FFN, and square.
SHAPES = [(128, 256), (256, 128), (768, 3072), (1024, 1024)]
DTYPES = [torch.bfloat16, torch.float32]


def _legacy(g: torch.Tensor, steps: int, schedule: str) -> torch.Tensor:
    return _zeropower_via_newtonschulz5(g, steps, schedule)


def _batched_equiv(g: torch.Tensor, steps: int, schedule: str) -> torch.Tensor:
    """Run the batched primitive on a single matrix, mirroring Muon.step's orientation.

    The batched primitive expects the smaller side on the left (same as the legacy
    path's internal transpose); we reproduce that orientation here and undo it so
    the comparison is apples-to-apples.
    """
    transposed = g.size(0) > g.size(1)
    gi = g.T.contiguous() if transposed else g
    out = _zeropower_newtonschulz5_batched(gi.unsqueeze(0), _ns_schedule(schedule, steps))[0]
    out = out.to(g.dtype)
    return out.T.contiguous() if transposed else out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_matches_legacy_per_matrix(device, m, n, dtype):
    """The batched primitive must reproduce the legacy per-matrix path numerically.

    bf16 runs in the native bf16 kernel path (looser, 1e-2); fp32 casts to bf16
    internally per the NS design (the polynomial runs in bf16 for speed), so it's
    also bounded by bf16 precision — 1e-3 relative is the floor there.
    """
    torch.manual_seed(0)
    g = torch.randn(m, n, device=device, dtype=dtype)
    ref = _legacy(g, 5, "quintic5")
    out = _batched_equiv(g, 5, "quintic5")
    tol = 1e-2 if dtype is torch.bfloat16 else 1e-3
    rel = (out.float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
    assert rel < tol, f"{m}x{n} {dtype}: batched vs legacy rel error {rel:.2e} >= {tol:.0e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m,n", SHAPES)
def test_output_is_approximately_orthogonal(device, m, n):
    """The NS output approximates the polar factor: the spectrum is compressed hard.

    The defining contract of a Newton-Schulz step is that it shrinks the spectral
    range toward 1 — the input gradient has a large top singular value (spectral
    norm of a random matrix scales with sqrt(max(m,n))), and the output's top
    singular value lands near 1. Five quintic steps overshoot slightly (~1.2) but
    never diverge. Both paths must satisfy this.
    """
    torch.manual_seed(1)
    g = torch.randn(m, n, device=device, dtype=torch.float32)
    in_top = torch.linalg.svdvals(g)[0].item()
    for label, fn in (("legacy", _legacy), ("batched", _batched_equiv)):
        out = fn(g, 5, "quintic5").float()
        out_top = torch.linalg.svdvals(out)[0].item()
        # Compression: out_top must be a small fraction of in_top, and bounded near 1.
        assert out_top < 0.1 * in_top, f"{label} {m}x{n}: top {out_top:.3f} vs in {in_top:.3f}, no compression"
        assert out_top < 1.5, f"{label} {m}x{n}: top singular {out_top:.3f} diverged past 1.5"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("schedule", ["quintic5", "cubic5", "hybrid_v4"])
def test_all_three_schedules_compress_spectrum_and_match(device, schedule):
    """Every supported schedule must still compress the spectrum, and the batched
    path must match the legacy path for it.

    Guards against silently dropping a schedule when wiring the batched path. The
    honest per-schedule contract: each compresses the input spectrum dramatically
    (top singular ~30 -> <2), but they land at different points — cubic5 is a
    deliberately softer orthogonaliser (~0.76) than quintic5 (~1.2), and hybrid_v4
    pins singular values hard to 1 via its stabilize tail. We assert the shared
    property (compression) and the exact path-equivalence, not a fixed band.
    """
    torch.manual_seed(2)
    g = torch.randn(256, 256, device=device, dtype=torch.float32)
    in_top = torch.linalg.svdvals(g)[0].item()
    legacy_out = _legacy(g, 5, schedule).float()
    batched_out = _batched_equiv(g, 5, schedule).float()
    for label, out in (("legacy", legacy_out), ("batched", batched_out)):
        out_top = torch.linalg.svdvals(out)[0].item()
        assert out_top < 2.0, f"{label} {schedule}: top singular {out_top:.3f} diverged"
        assert out_top < 0.1 * in_top, f"{label} {schedule}: no spectral compression"
    # The paths must agree to bf16 precision for every schedule.
    rel = (batched_out - legacy_out).norm().item() / (legacy_out.norm().item() + 1e-12)
    assert rel < 1e-2, f"{schedule}: batched vs legacy rel {rel:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_hybrid_v4_pins_singular_values_to_one(device):
    """hybrid_v4's defining feature: its stabilize tail pins singular values to 1.

    quintic5 leaves a soft spectrum (range ~[0.04, 1.2]); hybrid_v4 (8x quintic +
    2x stabilize) collapses it to a tight band around 1. This is the most
    informative single assertion about schedule correctness.
    """
    torch.manual_seed(4)
    g = torch.randn(256, 256, device=device, dtype=torch.float32)
    for label, fn in (("legacy", _legacy), ("batched", _batched_equiv)):
        out = fn(g, 5, "hybrid_v4").float()
        s = torch.linalg.svdvals(out)
        assert 0.95 < s[0].item() < 1.05, f"{label} hybrid_v4 top {s[0].item():.4f}"
        assert 0.95 < s[-1].item() < 1.05, f"{label} hybrid_v4 bottom {s[-1].item():.4f}"


@pytest.mark.parametrize("device", DEVICES)
def test_batched_stack_matches_per_matrix_loop(device):
    """A real stack of same-shape matrices must match looping the legacy path.

    This is the property Muon.step relies on: batching N params is equivalent to
    running NS on each, not an approximation.
    """
    torch.manual_seed(3)
    stack = torch.randn(8, 512, 512, device=device, dtype=torch.bfloat16)
    refs = torch.stack([_legacy(stack[i], 5, "quintic5") for i in range(8)])
    outs = _zeropower_newtonschulz5_batched(stack, _ns_schedule("quintic5", 5)).to(refs.dtype)
    rel = (outs.float() - refs.float()).norm().item() / (refs.float().norm().item() + 1e-12)
    assert rel < 1e-2, f"stack vs loop rel error {rel:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_unknown_schedule_raises(device):
    """A typo in the schedule name must raise, not silently fall back."""
    g = torch.randn(8, 8, device=device)
    with pytest.raises(ValueError):
        _legacy(g, 5, "quintic")  # missing the 5
    with pytest.raises(ValueError):
        _ns_schedule("typo", 5)
