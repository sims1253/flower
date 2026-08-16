# Compositional Muon + Aurora — measured results (450M, RTX 5090)

**Verdict: CM does not work here. Per-head Muon remains the recommendation.**

Screen: `configs/comp_muon_screen_450m.yaml`, `runs/comp_muon_screen_450m/`.
7 arms x 1500 steps, seed 0, FP8 stack fixed across all arms, 2026-08-12.

| arm | val_bpb | vs base | tok/s | vs base |
|---|---:|---:|---:|---:|
| `cm_per_head_ref` | **1.10016** | **-0.02768** | 63,400 | 1.009x |
| `cm_isotropic` | 1.11631 | -0.01153 | 63,293 | 1.008x |
| `cm_full` | 1.11675 | -0.01109 | 56,458 | 0.899x |
| `aurora` | 1.11770 | -0.01013 | 57,846 | 0.921x |
| `cm_baseline` | 1.12783 | — | 62,809 | 1.000x |
| `cm_full_mp_0p5` | 1.15201 | +0.02417 | 55,662 | 0.886x |

(`aurora_wd` was still running when this was written; add it when it lands.)

## Both controls reproduced, so the cross-screen comparison is valid

| quantity | this screen | `muon_screen_450m` | delta |
|---|---:|---:|---:|
| baseline | 1.12783 | 1.12785 | -0.00002 |
| per-head | 1.10016 | 1.10071 | -0.00055 |

Two independent screens, days apart, across a reboot. Per-head Muon's -0.027 is
real and replicated. This is what makes the rest of the table interpretable.

## CM captures ~40% of per-head's gain — the whitening is the problem

CM beats the baseline (-0.0115), so it is not broken; it does something. But it
loses to plain per-head Muon by **+0.016**, and per-head is *free* while
`cm_full` costs 10% throughput.

**The step-size explanation is ruled out.** `cm_full_mp_0p5` existed precisely to
separate "the rule is wrong for us" from "the effective LR was off", and it went
the **wrong way**: halving the step made it worse than the baseline (+0.024). So
CM's effective step was not too large. If anything it is already too small. The
rule underperforms, not its scaling.

**Mechanism.** Per-head Muon equalises update scale across heads — every head
gets its own unit-spectral-norm update, which is the entire content of the
method. CM's isotropic rule is

    delta_Q[h] = s_K[h] * msign(G_Q[h])

where `s_K[h] = (||W_K[h]||_F^2 / head_dim + lam)^{-1/2}` **varies per head**. So
CM orthogonalises per head and then immediately multiplies each head by a
different scalar, re-introducing exactly the cross-head scale variation that
per-head orthogonalisation removes. The full matrix-whitened path does the same
thing with `C^{-1}` in place of a scalar and measures the same (1.11675 vs
1.11631 — a 0.0004 gap, i.e. nothing).

That the isotropic and full paths land within 0.0005 of each other is itself
informative: the *granularity* of the whitening is irrelevant here. What matters
is that there is whitening at all. Reading the two CM arms as "the approximation
is as good as the real thing" would be the wrong conclusion — they agree because
both are dominated by the same defect.

**This is a claim about this model, not about CM in general.** CM's published
gains are at 340M/1B with separate Q/K/V projections and no QK-norm. Flower has
`qk_norm: true` (parameter-free `HeadRMSNorm`), which already makes attention
logits invariant to W_Q/W_K spectral norm — plausibly it has *already* captured
what partner whitening is for, leaving only the scale damage. That is a
hypothesis, not a measurement.

## Aurora: works, but not enough to pay for itself

-0.010 val_bpb at **-7.9% throughput** (57,846 vs 62,809). The throughput cost
closely matches the -9% predicted from the isolated optimizer bench
(`bench_optimizer_step.txt`, 4.65x optimizer step at ~2.8% of step). It beats the
baseline and loses to per-head while costing throughput per-head does not.

Note the earlier CPU-measured 2.4x optimizer cost **understated** the GPU figure
(4.65x). CPU optimizer timings do not transfer; use the GPU bench.

## What to do

1. **Keep `muon_per_head: true`.** Best quality in both screens, free.
2. **Do not deploy CM or Aurora.**
3. **CM is closed** unless someone tests the one remaining hypothesis below.

### The one open follow-up, and why it was not run

The reference ships a `per_mat_renorm` knob, deliberately not vendored (see the
CM header in `flower/optim.py`), which rescales each leg back to the Frobenius
norm of its orthogonalized factor *before* partner whitening — so the whitening
"only redistributes weight across heads, not step size". Given the mechanism
above, that is the obvious thing to try: it is the knob that would most directly
limit the scale damage.

It was not run because it is a new implementation justified by a
post-hoc hypothesis, and the window screen (a measured 1.268x throughput lever)
is a better use of the card. If CM is ever revisited, start there — and note it
still would not beat per-head on *cost*, since per-head is free.

## Predictions vs measurement (for calibrating future estimates)

| quantity | predicted | measured |
|---|---:|---:|
| `cm_full` throughput | ~0.85x | 0.899x |
| `aurora` throughput | ~0.91x | 0.921x |
| `cm_isotropic` throughput | ~0.97x | **1.008x** |
| CM beats per-head | expected | **no, lost by 0.016** |

The throughput model (optimizer ms/step converted to % of step at accum 16) held
to within ~5% on the two expensive arms. `cm_isotropic` came in *free* rather
than at the predicted ~3% cost, so the isolated bench overstates cost for cheap
optimizer variants — presumably the extra work overlaps with something in situ.

The quality prediction was simply wrong, and the reasoning behind it is worth
recording: CM was expected to win because per-head won and CM is "per-head plus
principle". The screen's `cm_per_head_ref` arm is the only reason that error was
caught rather than being written up as "CM improves on baseline by 0.011".
Always screen against the thing you claim to improve on, not only against the
control.
