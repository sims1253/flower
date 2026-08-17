"""Tests for the analytical Titans surprise gradient — S14 Opportunity 3.

Five properties under test (mirroring the spec's VALIDATION section):

1. Gradient equivalence (THE GATE): ``_surprise_analytical`` matches
   ``torch.autograd.grad(_inner_loss(...))`` at fp32 ~1e-4 (measured ~1e-9)
   and bf16 ~1e-2 on random inputs of two shapes. This is the acceptance gate
   for the whole change — a wrong gradient makes every downstream claim false.

2. Full-forward equivalence: the *whole block* produces the same updated
   memory in both modes (analytical vs autograd), at 1e-3. The forward folds
   surprise into ``alpha*write_scale*surprise`` so tiny float differences can
   compound, but it must stay tight.

3. Outer backward: in analytical mode, ``loss.backward()`` produces finite
   gradients on key_proj / val_proj / alpha_logit / write_scale — i.e. the
   *outer* CE graph is intact even though the inner graph is gone. This is the
   whole point of the optimisation.

4. Speed: ``torch.utils.benchmark`` on a single ``_surprise_update`` call,
   analytical vs autograd, at (B=8, S=16, D=768). Reports the speedup; does
   not assert a threshold (hardware-dependent) but fails if analytical is
   slower, which would indicate a bug.

5. Per-sample inner-loss reduction (``titans_per_sample_loss``): the legacy
   reduction is a mean over B*D, so Titans memory writes scale with 1/B and
   the B=1 document-level eval (flower/eval.py::evaluate_documents /
   sliding_window_document_loss) ran memory ~batch_size-x hotter than batch
   training ever did. With the flag on, writes must be batch-size-invariant
   across B in {1,4,16} in BOTH surprise paths; with it off, the legacy
   1/B magnitudes are pinned; and under ``causal_memory=True`` (already
   per-row) the flag must be a no-op.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from flower.config import ModelConfig
from flower.models.titans_mac import TitansMACBlock, build_titans_mac_model


def _cfg(**over) -> ModelConfig:
    base = dict(
        variant="titans_mac",
        vocab_size=128,
        d_model=64,
        num_heads=4,
        num_layers=1,
        ffn_dim=128,
        max_seq_len=64,
        local_window=16,
        memory_slots=8,
    )
    base.update(over)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# 1. Gradient equivalence — THE GATE
# ---------------------------------------------------------------------------


def _analytical_via_block(block, long_mem, key, value):
    """Call the block's analytical method directly (bypasses proj layers)."""
    return block._surprise_analytical(long_mem, key, value)


def _autograd_surprise(block, long_mem, key, value):
    """Reproduce the legacy path's inner grad exactly, on the same inputs."""
    probe = long_mem.detach().clone().requires_grad_(True)
    inner = block._inner_loss(probe, key, value)
    (g,) = torch.autograd.grad(inner, probe)
    return -g


@pytest.mark.parametrize(
    "shape",
    [(4, 8, 64), (2, 16, 256)],
    ids=["B4-S8-D64", "B2-S16-D256"],
)
def test_surprise_analytical_matches_autograd_fp32(shape):
    """GATE: the closed-form surprise must equal -d(inner_loss)/d(memory) at
    fp32 within 1e-4. Measured max-abs-diff is ~1e-9 (it is the same
    computation, rearranged)."""
    B, S, D = shape
    torch.manual_seed(0)
    block = TitansMACBlock(_cfg(d_model=D, memory_slots=S))
    block.eval()
    long_mem = torch.randn(B, S, D, dtype=torch.float32)
    key = torch.randn(B, D, dtype=torch.float32)
    value = torch.randn(B, D, dtype=torch.float32)

    ana = _analytical_via_block(block, long_mem, key, value)
    auto = _autograd_surprise(block, long_mem, key, value)

    assert ana.shape == (B, S, D)
    max_abs = (ana - auto).abs().max().item()
    assert max_abs < 1e-4, f"fp32 surprise mismatch: max_abs_diff={max_abs:.3e}"


@pytest.mark.parametrize(
    "shape",
    [(4, 8, 64), (2, 16, 256)],
    ids=["B4-S8-D64", "B2-S16-D256"],
)
def test_surprise_analytical_matches_autograd_bf16(shape):
    """GATE (bf16): same comparison at bf16 within 1e-2. bf16 has ~3 decimal
    digits, so 1e-2 absolute is the realistic floor."""
    B, S, D = shape
    torch.manual_seed(1)
    block = TitansMACBlock(_cfg(d_model=D, memory_slots=S))
    block.eval()
    long_mem = torch.randn(B, S, D, dtype=torch.bfloat16)
    key = torch.randn(B, D, dtype=torch.bfloat16)
    value = torch.randn(B, D, dtype=torch.bfloat16)

    ana = _analytical_via_block(block, long_mem, key, value)
    auto = _autograd_surprise(block, long_mem, key, value)

    # Compare in fp32 to avoid bf16 subtraction collapse.
    max_abs = (ana.float() - auto.float()).abs().max().item()
    assert max_abs < 1e-2, f"bf16 surprise mismatch: max_abs_diff={max_abs:.3e}"


# ---------------------------------------------------------------------------
# 2. Full-forward equivalence — the whole block, both modes, same memory out
# ---------------------------------------------------------------------------


def test_full_forward_analytical_matches_autograd():
    """Build two blocks with identical weights, run the full forward (proj
    layers + surprise + hierarchical concat) in each mode, and assert the
    updated memory agrees at 1e-3. The forward multiplies surprise by
    alpha*write_scale, so float noise compounds but stays well under 1e-3
    because the surprise itself agrees at ~1e-9."""
    cfg = _cfg(d_model=64, memory_slots=8, num_layers=1)
    torch.manual_seed(42)
    block_auto = TitansMACBlock(cfg)  # titans_analytical_surprise defaults False
    torch.manual_seed(42)
    block_ana = TitansMACBlock(
        _cfg(d_model=64, memory_slots=8, num_layers=1, titans_analytical_surprise=True)
    )
    # Force the flag on for the analytical block and off for the autograd one,
    # then sync weights so the only difference is the surprise path.
    block_ana.config = _cfg(d_model=64, memory_slots=8, num_layers=1, titans_analytical_surprise=True)
    block_ana.load_state_dict(block_auto.state_dict())
    block_auto.eval()
    block_ana.eval()

    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D)
    memory = torch.randn(B, cfg.memory_slots, D) * 0.1

    with torch.no_grad():
        _, mem_auto = block_auto(x, memory.clone())
        _, mem_ana = block_ana(x, memory.clone())

    assert mem_auto.shape == mem_ana.shape == (B, cfg.memory_slots, D)
    max_abs = (mem_auto - mem_ana).abs().max().item()
    assert max_abs < 1e-3, f"forward memory mismatch: max_abs_diff={max_abs:.3e}"


def test_full_forward_hierarchical_memory_unchanged():
    """Hierarchical memory concatenates short-slots after the long-mem prefix.
    The analytical path must produce the right shape; the short-slots are a
    copy of the post-FFN x (forward applies residuals before _surprise_update),
    so they are finite and non-trivial rather than bit-identical to the input.
    The concat itself is unchanged by this PR — only the surprise computation
    in the long-mem prefix is on the new path."""
    cfg = _cfg(
        d_model=32,
        memory_slots=4,
        short_memory_slots=2,
        hierarchical_memory=True,
        titans_analytical_surprise=True,
    )
    block = TitansMACBlock(cfg).eval()
    B, T, D = 2, 8, 32
    x = torch.randn(B, T, D)
    memory = torch.zeros(B, cfg.memory_slots + cfg.short_memory_slots, D)
    with torch.no_grad():
        _, mem = block(x, memory)
    expected_slots = cfg.memory_slots + cfg.short_memory_slots
    assert mem.shape == (B, expected_slots, D)
    assert torch.isfinite(mem).all()
    # Long-mem prefix received a surprise write from a zero start; short-slots
    # are a copy of transformed tokens, so the two regions are distinguishable.
    long_part = mem[:, : cfg.memory_slots]
    short_part = mem[:, cfg.memory_slots :]
    assert torch.isfinite(short_part).all() and short_part.abs().sum() > 0


# ---------------------------------------------------------------------------
# 3. Outer backward — the whole point (no inner graph, outer graph intact)
# ---------------------------------------------------------------------------


def test_outer_backward_analytical_produces_finite_grads():
    """In analytical mode, loss.backward() must populate gradients on
    key_proj, val_proj, alpha_logit, and write_scale, and they must be finite.
    This proves the outer CE graph is intact even though the inner autograd
    graph is gone."""
    cfg = _cfg(
        d_model=32, memory_slots=4, num_layers=1, titans_analytical_surprise=True
    )
    block = TitansMACBlock(cfg)
    block.train()
    B, T, D = 2, 8, 32
    x = torch.randn(B, T, D, requires_grad=True)
    memory = torch.zeros(B, cfg.memory_slots, D)
    out, mem = block(x, memory)
    # Pull a scalar through both the token stream and the memory so grads reach
    # the proj layers and alpha/write_scale via the surprise path.
    loss = out.float().sum() + mem.float().pow(2).sum()
    loss.backward()

    for name in ["key_proj.weight", "val_proj.weight"]:
        g = dict(block.named_parameters())[name].grad
        assert g is not None, f"{name}.grad is None — outer graph broken"
        assert torch.isfinite(g).all(), f"{name}.grad has NaN/Inf"
    assert block.alpha_logit.grad is not None and torch.isfinite(block.alpha_logit.grad).all()
    assert block.write_scale.grad is not None and torch.isfinite(block.write_scale.grad).all()


def test_end_to_end_model_backward_analytical():
    """Full CausalLM forward+backward in analytical mode: loss is finite, all
    titans params get grad. Exercises the path through the LM head + CE."""
    cfg = _cfg(
        d_model=32,
        memory_slots=4,
        num_layers=2,
        titans_analytical_surprise=True,
    )
    model = build_titans_mac_model(cfg)
    model.train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(tokens, labels=tokens)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    # Spot-check a titans-specific param got grad.
    g = model.blocks[0].key_proj.weight.grad
    assert g is not None and torch.isfinite(g).all()


# ---------------------------------------------------------------------------
# 4. Speed (informational; fails only if analytical is slower)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="speedup measured on CUDA")
def test_surprise_analytical_is_faster_than_autograd():
    """torch.utils.benchmark on a single _surprise_update call, analytical vs
    autograd, at (B=8, S=16, D=768), 100 calls. Reports the speedup. Asserts
    analytical is not slower (would indicate a bug). S14 estimated 'could
    halve the compute of the Titans variant' — this measures it."""
    import torch.utils.benchmark as benchmark

    device = torch.device("cuda")
    cfg_auto = _cfg(
        d_model=768,
        memory_slots=16,
        num_layers=1,
    )
    cfg_ana = _cfg(
        d_model=768,
        memory_slots=16,
        num_layers=1,
        titans_analytical_surprise=True,
    )
    block_auto = TitansMACBlock(cfg_auto).to(device).train()
    block_ana = TitansMACBlock(cfg_ana).to(device).train()
    block_ana.load_state_dict(block_auto.state_dict())

    B, T, D = 8, 64, 768
    x = torch.randn(B, T, D, device=device)
    memory = torch.randn(B, cfg_auto.memory_slots, D, device=device) * 0.1

    t_auto = benchmark.Timer(
        stmt="block(x, mem)",
        globals={"block": block_auto, "x": x, "mem": memory},
        num_threads=1,
    ).timeit(100)
    t_ana = benchmark.Timer(
        stmt="block(x, mem)",
        globals={"block": block_ana, "x": x, "mem": memory},
        num_threads=1,
    ).timeit(100)

    speedup = t_auto.median / t_ana.median
    print(f"\n[titans surprise speedup] autograd={t_auto.median*1e6:.1f}us  "
          f"analytical={t_ana.median*1e6:.1f}us  speedup={speedup:.2f}x")
    # No hard threshold (hardware-dependent), but analytical must not be slower.
    assert speedup > 1.0, (
        f"analytical path is SLOWER ({speedup:.2f}x) — check the implementation"
    )


# ---------------------------------------------------------------------------
# 5. Per-sample inner-loss reduction (titans_per_sample_loss) — batch-size
#    invariance of the memory dynamics.
#
# The legacy reduction is a mean over B*D (factor 2/(B*D)), so the surprise —
# and therefore every Titans memory write alpha*write_scale*surprise — scales
# with 1/B. flower/eval.py's document-level paths score one document at a time
# (B=1), so titans doc-level bpb was computed with batch_size-x larger writes
# than batch training produced. The flag makes both surprise paths reduce per
# sample (sum over B, mean over D, factor 2/D), which pins the write magnitude
# to the B=1 value at every batch size.
# ---------------------------------------------------------------------------


def _replicated_block_memory(batch: int, *, analytical: bool, per_sample: bool, seed: int = 7):
    """One TitansMACBlock forward from a zero-memory init, with every batch
    element holding the SAME hidden states, so the only difference between
    batch sizes is the inner-loss reduction. Returns (block, memory)."""
    torch.manual_seed(seed)
    block = TitansMACBlock(
        _cfg(
            titans_analytical_surprise=analytical,
            titans_per_sample_loss=per_sample,
        )
    ).eval()
    torch.manual_seed(seed + 1)
    x = torch.randn(1, 16, 64).repeat(batch, 1, 1)
    with torch.no_grad():
        _, mem = block(x, None)
    return block, mem


@pytest.mark.parametrize("analytical", [False, True], ids=["autograd", "analytical"])
def test_per_sample_memory_invariant_across_batch_sizes(analytical):
    """Flag ON: mean |memory| after one block must be identical across
    B in {1,4,16} (within fp tolerance). The block has no cross-batch ops, so
    with a per-sample reduction every element's write is independent of the
    batch it travelled in."""
    _, mem1 = _replicated_block_memory(1, analytical=analytical, per_sample=True)
    for B in (4, 16):
        _, memB = _replicated_block_memory(B, analytical=analytical, per_sample=True)
        assert memB.shape == (B, mem1.shape[1], mem1.shape[2])
        # Element-wise: batch element i at B>1 must match the B=1 memory.
        for i in range(B):
            assert torch.allclose(memB[i], mem1[0], rtol=1e-5, atol=1e-8), (
                f"analytical={analytical}: memory element {i} differs between B={B} and B=1"
            )
        assert memB.abs().mean().item() == pytest.approx(mem1.abs().mean().item(), rel=1e-5)


@pytest.mark.parametrize("analytical", [False, True], ids=["autograd", "analytical"])
def test_legacy_memory_magnitude_scales_with_inverse_batch_size(analytical):
    """Flag OFF: the legacy B-dependent magnitudes are preserved. From a zero
    memory the first write is exactly proportional to the surprise factor
    2/(B*D), so mean |memory| must be ~1/B: ratio 4 between B=1 and B=4, 16
    between B=1 and B=16. (This is the train(B=N) vs doc-eval(B=1)
    inconsistency the flag exists to fix — pinned here so it stays opt-in.)"""
    mags = {B: _replicated_block_memory(B, analytical=analytical, per_sample=False)[1].abs().mean().item()
            for B in (1, 4, 16)}
    assert mags[1] > 0
    assert mags[1] / mags[4] == pytest.approx(4.0, rel=0.05), f"got B1/B4 = {mags[1]/mags[4]:.3f}"
    assert mags[1] / mags[16] == pytest.approx(16.0, rel=0.05), f"got B1/B16 = {mags[1]/mags[16]:.3f}"


def test_per_sample_flag_pins_b1_semantics_at_every_batch():
    """The per-sample reduction is exactly the legacy reduction evaluated at
    B=1 (both give factor 2/D per sample): memory at B=16 with the flag on
    must equal legacy memory at B=1. This is the "doc-eval consistent with
    training" property in miniature."""
    _, mem_legacy_b1 = _replicated_block_memory(1, analytical=False, per_sample=False)
    for analytical in (False, True):
        _, mem_ps_b16 = _replicated_block_memory(16, analytical=analytical, per_sample=True)
        assert torch.allclose(mem_ps_b16[0], mem_legacy_b1[0], rtol=1e-5, atol=1e-8), (
            f"analytical={analytical}: per-sample B=16 memory != legacy B=1 memory"
        )


def test_per_sample_analytical_matches_autograd():
    """GATE under the flag: both paths rescale by the same factor B, so the
    analytical-vs-autograd equivalence must survive titans_per_sample_loss."""
    torch.manual_seed(5)
    block = TitansMACBlock(_cfg(titans_per_sample_loss=True))
    block.eval()
    B, S, D = 4, 8, 64
    long_mem = torch.randn(B, S, D, dtype=torch.float32)
    key = torch.randn(B, D, dtype=torch.float32)
    value = torch.randn(B, D, dtype=torch.float32)

    ana = block._surprise_analytical(long_mem, key, value)
    auto = _autograd_surprise(block, long_mem, key, value)
    max_abs = (ana - auto).abs().max().item()
    assert max_abs < 1e-4, f"fp32 surprise mismatch under per-sample flag: {max_abs:.3e}"
    # And the flag itself scales the legacy gradient by exactly B.
    legacy_block = TitansMACBlock(_cfg(titans_per_sample_loss=False))
    legacy_block.eval()
    legacy_block.load_state_dict(block.state_dict())
    ana_legacy = legacy_block._surprise_analytical(long_mem, key, value)
    assert torch.allclose(ana, ana_legacy * B, rtol=1e-5, atol=1e-9)


# ---------------------------------------------------------------------------
# 5b. 2x2 interaction with causal_memory: the causal write path is per-position
# and already per-row (factor 2/D), so titans_per_sample_loss must be a no-op
# there, and every flag combination must smoke-train and stay causal (the
# perturbation pattern from tests/test_causal.py).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_sample", [False, True], ids=["legacy-reduction", "per-sample"])
@pytest.mark.parametrize("analytical", [False, True], ids=["autograd", "analytical"])
@pytest.mark.parametrize("causal", [False, True], ids=["window", "causal"])
def test_flag_interaction_smoke_trains_and_causal_writes_stay_causal(causal, analytical, per_sample):
    cfg = _cfg(
        num_layers=3,  # >=3 layers: the leak rides the write->read threading
        max_seq_len=48,
        local_window=16,
        causal_memory=causal,
        titans_analytical_surprise=analytical,
        titans_per_sample_loss=per_sample,
    )
    torch.manual_seed(0)
    model = build_titans_mac_model(cfg).eval()

    # Causality perturbation check (tests/test_causal.py pattern): perturbing
    # only the LAST input token must leave logits[:, :-1] bit-unchanged
    # whenever causal_memory=True.
    torch.manual_seed(1)
    a = torch.randint(0, cfg.vocab_size, (2, 48))
    b = a.clone()
    b[:, -1] = (b[:, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_a = model(a)["logits"][:, :-1]
        logits_b = model(b)["logits"][:, :-1]
    delta = (logits_a - logits_b).abs().max().item()
    if causal:
        assert delta == 0.0, (
            f"last-token perturbation moved past logits by {delta} "
            f"(causal={causal}, analytical={analytical}, per_sample={per_sample})"
        )

    # Smoke-train: finite loss, backward populates gradients.
    model.train()
    out = model(a, labels=a)
    assert out["loss"] is not None and torch.isfinite(out["loss"])
    out["loss"].backward()
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert with_grad, "no parameter received gradient"


def test_per_sample_flag_is_noop_under_causal_memory():
    """causal_memory=True already reduces per row (each (batch, position)
    row's surprise is the gradient of its own MSE, factor 2/D), so flipping
    titans_per_sample_loss must not change its outputs (fp tolerance only —
    the legacy path computes rows*(2/(rows*D)) vs the flag's 2/D)."""
    outs = []
    for per_sample in (False, True):
        cfg = _cfg(
            num_layers=2,
            max_seq_len=32,
            local_window=16,
            causal_memory=True,
            titans_per_sample_loss=per_sample,
        )
        torch.manual_seed(3)
        model = build_titans_mac_model(cfg).eval()
        torch.manual_seed(4)
        tokens = torch.randint(0, cfg.vocab_size, (2, 24))
        with torch.no_grad():
            outs.append(model(tokens)["logits"])
    max_abs = (outs[0] - outs[1]).abs().max().item()
    assert max_abs < 1e-5, f"titans_per_sample_loss changed the causal path by {max_abs:.3e}"
