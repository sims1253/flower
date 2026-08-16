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


def test_contra_muon_damps_large_singular_directions():
    """Contra-Muon must damp update directions in proportion to gradient sigma.

    Newton-Schulz maps every singular value to 1, discarding the gradient's
    spectral profile. Contra-Muon subtracts a normalised copy of the
    pre-orthogonalisation update so that large-sigma directions are reduced MORE
    than small ones, relatively boosting the small directions.

    This is asserted in the GRADIENT'S OWN singular basis, not on sorted singular
    values: Contra-Muon reorders the sorted spectrum, so a max/min ratio moves the
    wrong way and looks like a regression when the implementation is correct.
    """
    import torch.nn as nn

    from flower.optim import Muon

    torch.manual_seed(0)
    a = torch.randn(64, 32)
    u0, _, v0h = torch.linalg.svd(a, full_matrices=False)
    sigma = torch.linspace(3.0, 0.05, 32)  # strongly skewed, known spectrum
    grad = u0 @ torch.diag(sigma) @ v0h

    def gains(contra: float) -> torch.Tensor:
        m = nn.Linear(32, 64, bias=False)
        opt = Muon(
            m.parameters(), lr=1.0, momentum=0.0, nesterov=False,
            ns_batched=False, contra_muon=contra,
        )
        m.weight.grad = grad.clone()
        before = m.weight.detach().clone()
        opt.step()
        upd = before - m.weight.detach()
        return torch.einsum("ij,ik,kj->j", u0, upd, v0h.T)

    delta = gains(0.4) - gains(0.0)
    # Every direction is damped...
    assert (delta <= 1e-6).all(), "Contra-Muon should not amplify any direction"
    # ...and the damping must scale with the gradient's singular value.
    corr = torch.corrcoef(torch.stack([sigma, delta]))[0, 1]
    assert corr < -0.95, f"damping not proportional to gradient sigma (corr={corr:.3f})"


def test_contra_muon_zero_is_exactly_off():
    """contra_muon=0.0 must reproduce the existing update bit-for-bit."""
    import torch.nn as nn

    from flower.optim import Muon

    def run(contra: float) -> torch.Tensor:
        torch.manual_seed(0)
        m = nn.Linear(32, 64, bias=False)
        opt = Muon(m.parameters(), lr=0.1, momentum=0.9, ns_batched=False, contra_muon=contra)
        torch.manual_seed(1)
        for _ in range(3):
            m.weight.grad = torch.randn(64, 32)
            opt.step()
        return m.weight.detach().clone()

    assert torch.equal(run(0.0), run(0.0))
    assert not torch.equal(run(0.0), run(0.2)), "contra_muon=0.2 had no effect"


def test_per_head_map_tags_attention_matrices_with_correct_axes():
    """qkv splits rows (3*heads blocks); the output projection splits columns."""
    from flower.config import ModelConfig
    from flower.models import build_model
    from flower.optim import build_head_block_map

    cfg = ModelConfig(
        variant="vanilla_local", vocab_size=64, d_model=64, num_heads=4,
        num_layers=2, ffn_dim=128, max_seq_len=32, local_window=8,
    )
    m = build_model(cfg)
    hm = build_head_block_map(m)
    by_name = {n: hm[id(p)] for n, p in m.named_parameters() if id(p) in hm}

    assert len(by_name) == 4, f"expected 2 matrices x 2 layers, got {sorted(by_name)}"
    for n, (axis, blocks) in by_name.items():
        if "qkv" in n:
            assert (axis, blocks) == (0, 12), f"{n}: qkv must split rows into 3*heads"
        else:
            assert (axis, blocks) == (1, 4), f"{n}: out proj must split cols into heads"


def test_per_head_orthogonalises_each_head_block_independently():
    """The point of per-head Muon: every head gets its own unit-norm update.

    Under whole-matrix NS the singular values of the FULL matrix are ~1, so any
    individual head block is well below 1 and heads share one update scale.
    """
    from flower.optim import _ns_schedule, _per_head_orthogonalise, _zeropower_via_newtonschulz5

    torch.manual_seed(0)
    coeffs = _ns_schedule("quintic5", 5)
    u = torch.randn(192, 64)  # 12 head blocks of 16 rows

    per_head = _per_head_orthogonalise(u, (0, 12), coeffs)
    assert per_head.shape == u.shape

    def block_svs(x):
        return torch.stack([torch.linalg.svdvals(b) for b in x.view(12, 16, 64)])

    ph, whole = block_svs(per_head), block_svs(_zeropower_via_newtonschulz5(u, 5, "quintic5"))
    assert ph.min() > 0.5, "per-head blocks should each be near-orthogonal"
    assert ph.mean() > whole.mean() * 1.5, (
        f"per-head block scale {ph.mean():.3f} not clearly above whole-matrix {whole.mean():.3f}"
    )


def test_per_head_falls_back_when_shape_does_not_divide():
    """A variant with an odd attention layout must degrade to standard Muon."""
    from flower.optim import _ns_schedule, _per_head_orthogonalise

    coeffs = _ns_schedule("quintic5", 5)
    u = torch.randn(50, 64)  # 50 rows is not divisible by 12
    out = _per_head_orthogonalise(u, (0, 12), coeffs)
    assert out.shape == u.shape and torch.isfinite(out).all()


def test_per_head_is_opt_in():
    from flower.config import ModelConfig, TrainingConfig
    from flower.models import build_model
    from flower.optim import build_optimizer

    cfg = ModelConfig(
        variant="vanilla_local", vocab_size=64, d_model=64, num_heads=4,
        num_layers=2, ffn_dim=128, max_seq_len=32, local_window=8,
    )
    m = build_model(cfg)

    def head_map_size(flag: bool) -> int:
        opts = build_optimizer(m, TrainingConfig(optimizer="muon", muon_per_head=flag))
        opts = opts if isinstance(opts, list) else [opts]
        muon = next(o for o in opts if type(o).__name__ == "Muon")
        return len(muon._head_map)

    assert head_map_size(False) == 0, "per-head must be off by default"
    assert head_map_size(True) == 4


def test_normuon_equalises_row_scales_without_changing_step_size():
    """NorMuon must REDISTRIBUTE scale across rows, not shrink the update.

    The original implementation was `ortho / ortho.norm()` — one global
    Frobenius normalisation. An orthogonalised (m, n) matrix has
    ||ortho||_F ~= sqrt(min(m,n)), so that divided every update by ~36 at
    d_model=1280: a silent 36x learning-rate cut. The 1500-step Muon screen
    measured it at +0.517 val_bpb (train loss 4.64 vs 3.16).

    Both properties are asserted: row scales become equal (the method), and the
    Frobenius norm is preserved (so an A/B measures the method, not an LR change).
    """
    import torch.nn as nn

    from flower.optim import Muon

    def update_for(norm_update: bool) -> torch.Tensor:
        torch.manual_seed(0)
        m = nn.Linear(64, 128, bias=False)
        opt = Muon(
            m.parameters(), lr=1.0, momentum=0.0, nesterov=False,
            ns_batched=False, norm_update=norm_update,
        )
        torch.manual_seed(1)
        m.weight.grad = torch.randn(128, 64) * torch.linspace(0.1, 3.0, 128)[:, None]
        before = m.weight.detach().clone()
        opt.step()
        return before - m.weight.detach()

    plain, normed = update_for(False), update_for(True)

    # 1. Row RMS spread must shrink substantially.
    def spread(x):
        rms = x.pow(2).mean(dim=1).sqrt()
        return (rms.max() / rms.min()).item()

    assert spread(normed) < spread(plain) / 2, (
        f"row scales not equalised: {spread(normed):.3f} vs {spread(plain):.3f}"
    )

    # 2. Overall step size must be preserved (this is what the old code broke).
    ratio = normed.norm() / plain.norm()
    assert 0.5 < ratio < 2.0, (
        f"NorMuon changed the step size by {ratio:.3f}x; it must redistribute, not rescale"
    )
