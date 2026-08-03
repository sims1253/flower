# Sweep 10 — Results: Does flow-matching help KV compaction beyond raw capacity?

## TL;DR

The original `sweep_still_novel` "flow wins" result (train ppl 2.68 vs 2.84) was a
**parameter-count confound**: the flow arm had 3.6× more *trainable* compactor
params (7.1M vs 2.0M) because `count_parameters()` only counts trainable params
and the base model is frozen. Sweep 10 controls for this. The honest answer:

- **At matched parameters, the flow mechanism trends better at both budgets,
  but the advantage is not lockable outside seed noise at n=3.**
- The flow arm is **higher-variance** across seeds (sd ~0.028-0.032 vs standard
  ~0.017), which is itself an interesting and under-reported property.
- The original "flow wins by 0.16" was ~80% capacity, ~20% possibly-real-but-
  noisy mechanism.

This is a "huh, that's interesting, but not a slam dunk" result — exactly the
kind of thing worth knowing before investing further in flow-as-compactor.

## Setup

Five arms, all sharing the same frozen 12.2M base model (d_model 384, 6 layers,
6 heads). Only the **trainable compactor** differs, and arms are param-matched
within ±2.5%:

| arm | config | trainable compactor |
|-----|--------|---------------------|
| A std_2.0M | standard, d_latent=128, blocks=2 | 1,972,224 |
| B flow_7.1M | flow, steps=5 (original) | 7,110,144 |
| C std_7.1M | standard, d_latent=192, blocks=4 | 7,096,320 |
| D flow_2.4M | flow, steps=5, velocity_hidden=8 | 2,446,752 |
| E std_2.4M | standard, d_latent=144, blocks=2 | 2,384,640 |

3 seeds each, 6500 steps, held-out `val_perplexity` (the original sweep only
logged train perplexity — a second confound in the same direction).

**Decisive comparisons:** B vs C (flow vs standard at 7.1M) and D vs E (flow vs
standard at 2.4M).

## Results

| arm | n | val_ppl per seed | mean | sd | range |
|-----|---|-----------------|------|------|-------|
| A std_2.0M | 3 | 2.828, 2.869, 2.847 | 2.848 | 0.017 | [2.828, 2.869] |
| **B flow_7.1M** | 3 | 2.646, 2.714, 2.670 | **2.677** | 0.028 | [2.646, 2.714] |
| **C std_7.1M** | 2* | 2.727, 2.675 | **2.701** | 0.026 | [2.675, 2.727] |
| **D flow_2.4M** | 3 | 2.781, 2.849, 2.783 | **2.805** | 0.032 | [2.781, 2.849] |
| **E std_2.4M** | 3 | 2.859, 2.856, 2.820 | **2.845** | 0.017 | [2.820, 2.859] |

*C_seed1 crashed repeatedly (compaction-phase memory spike on the d_latent=192
config) and is being retried; the B-vs-C comparison is n=3 vs n=2 until it lands.

### Decisive comparison 1: B flow vs C std (high budget, 7.1M)

- B mean 2.677 vs C mean 2.701 — **flow trends 0.024 better**
- Ranges **overlap**: B's max (2.714) > C's min (2.675)
- Verdict: **suggestive but not significant at n≤3.** Flow doesn't clearly help
  when the compactor is capacity-rich.

### Decisive comparison 2: D flow vs E std (low budget, 2.4M)

- D mean 2.805 vs E mean 2.845 — **flow trends 0.040 better**
- Ranges **overlap narrowly**: D's max (2.849, a high outlier) > E's min (2.820)
- Verdict: **flow trends clearly better by mean, but its own high variance
  prevents a clean non-overlapping call.** With n=2 on E this looked locked
  (gap 0.053, no overlap); E_seed2 (2.820) came in low and closed the gap.

## The interesting findings

1. **Flow helps by mean at both budgets, but it's noisy.** The consistent
   direction (flow mean < standard mean at *both* 7.1M and 2.4M) is the most
   compelling signal — it's unlikely to be coincidence that flow wins on mean
   in both paired comparisons. But the effect size (~0.024-0.040) is comparable
   to the flow arm's own seed standard deviation (~0.030), so it doesn't clear
   the bar for "clearly real" at n=3.

2. **Flow is higher-variance than standard.** This is the under-reported finding.
   The flow compactor's val_ppl sd is ~0.028-0.032 across seeds, nearly double
   the standard compactor's ~0.017. The flow's training dynamics (Euler
   integration of a learned velocity field) appear more seed-sensitive — some
   seeds find a good trajectory, others don't. If flow-matching compaction is to
   be practical, **reducing this variance** (e.g. trajectory regularization,
   better velocity-net init, more Euler steps) may matter more than the mean
   improvement.

3. **The original "flow wins by 0.16" was mostly capacity.** Comparing B
   (7.1M, 2.677) to A (2.0M, 2.848): the 0.171 gap shrinks to ~0.024 once you
   control for params (B vs C). So ~85% of the apparent flow advantage was just
   "more trainable parameters," confirming the confound hypothesis that
   motivated this sweep.

## Caveats

- **C is n=2** (seed1 retry pending). If C_seed1 lands high (>2.701), B's mean
  advantage holds; if low (<2.675), B-vs-C becomes a tie.
- **Mixed batch sizes across arms** (A: batch 64×1; B: 32×2; C/D/E: 16×4) — all
  have effective batch 64 with identical expected gradients, but the physical
  batch differs. This is a minor inconsistency; the seed-noise floor (±0.04)
  dominates any batch-size effect at this scale.
- **One scale only** (12.2M frozen base, 6500 steps). Whether flow's mean
  advantage grows or vanishes at larger scale is open.
- **train ≈ val** on this task (A: train 2.853 vs val 2.848), so the original
  train-only perplexities were roughly trustworthy for *relative* comparison —
  the param confound was the real issue, not the train/val gap.

## What this means for next steps

Per `NEXT_IDEAS.md`'s decision rule: the flow direction is **promising but not
proven** — not a clear "go all-in" (B<C was non-significant) nor a clear "dead
end" (D<E trended real). The two most interesting follow-ups:

1. **Variance reduction for flow compaction** (the novel finding). If the flow
   arm's seed variance can be brought down to the standard's level, the mean
   advantage would become significant. Worth a small targeted sweep varying
   flow_steps, velocity-net init, and trajectory regularization.

2. **Pivot to tapered compactor budgets** (the cheap orthogonal idea from the
   TLMs paper). Flow is a "maybe"; tapering is a likely free win and is
   independent of the flow question.

## Code changes (this sweep)

- `StillCompactorFlow` gained a `velocity_hidden` kwarg (lets the velocity nets
  shrink for param-matched comparisons); backward-compatible, default unchanged.
- `still_velocity_hidden` config field added.
- `configs/sweep_still_flow_matched.yaml` — the five-arm param-matched sweep.
- `scripts/sweep10_analyze.py` — auto-aggregates val_ppl per arm + verdicts.
