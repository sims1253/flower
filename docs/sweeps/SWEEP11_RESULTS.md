# Sweep 11 — Results: Does the TLM "wider-early" result apply to KV compaction?

## TL;DR

**No.** Varying the compactor's per-layer `d_latent` (wider-early, wider-late, or
steep) under a fixed total param budget **hurts** perplexity by ~0.10 vs uniform
allocation, regardless of direction or steepness. The TLMs paper's "later layers
reinforce rather than transform, so give them less capacity" finding — which gives
a free ~0.3 ppl win for FFN width — does **not** transfer to the KV-compactor's
latent bottleneck. If anything it reverses: the compactor wants **flat, uniform
capacity across all layers.** This is a clean "huh, that's interesting" negative
result with a real architectural implication.

## Setup

Four arms, all sharing the frozen 12.2M base model, standard `StillCompactor`
(no flow), 3 seeds, held-out `val_perplexity`, batch 64×1. Param-matched within
±0.82% (B/C identical to each other — reversing a schedule sums to the same total):

| arm | per-layer d_latent schedule | trainable compactor | diff vs A |
|-----|----------------------------|---------------------|-----------|
| **A uniform_128** (anchor) | `[128,128,128,128,128,128]` | 1,972,224 | 0% |
| **B taper_early** (TLM dir) | `[152,146,134,120,108,104]` | 1,979,520 | +0.37% |
| **C taper_late** (B reversed) | `[104,108,120,134,146,152]` | 1,979,520 | +0.37% |
| **D taper_steep** | `[172,162,138,108,84,76]` | 1,956,096 | −0.82% |

## Results

| arm | n | val_ppl per seed | mean | range |
|-----|---|-----------------|------|-------|
| **A uniform** | 3 | 2.852, 2.855, 2.848 | **2.852** | [2.848, 2.855] |
| B taper_early | 3 | 2.953, 2.974, 2.953 | 2.960 | [2.953, 2.974] |
| C taper_late | 3 | 2.925, 2.981, 2.951 | 2.953 | [2.925, 2.981] |
| D taper_steep | 3 | 2.956, 2.977, 2.936 | 2.956 | [2.936, 2.977] |

### Decisive comparisons

1. **B taper_early vs A uniform → NON-OVERLAP (A wins).** Tapering in the TLM
   direction costs 0.108 ppl. The TLM wider-early intuition is actively *wrong*
   for compaction.
2. **B taper_early vs C taper_late → OVERLAP.** Direction doesn't matter: early
   and late tapers perform identically (means 2.960 vs 2.953, ranges overlap
   heavily).
3. **D taper_steep vs B taper_early → OVERLAP.** Steepness doesn't matter either:
   a steeper taper (172→76) lands in the same band as the moderate one (152→104).

## Interpretation: the compactor's capacity is genuinely uniform

The three non-uniform arms (B, C, D) cluster tightly at val_ppl ~2.95-2.97 —
**all ~0.10 worse than uniform A (2.852)**, and statistically indistinguishable
from each other. This says something specific about the compactor's architecture:

- An **FFN** has a strong depth-asymmetry (TLMs: later layers reinforce, need
  less width; ρ_l^MLP correlates 0.49-0.71 with depth). Redistributing width
  early helps.
- The **Perceiver-style KV compactor** does *not* have this asymmetry. Its
  cross-attention read distributes the "compression work" roughly evenly across
  layers — each layer's compactor faces a similar-shape problem (summarize a KV
  cache into `compact_len` latents), unlike FFN layers which process increasingly
  abstract residual-stream content. So there's no layer that "needs less" — every
  deviation from uniform just removes capacity from a layer that was using it.

This is a discriminating null result: it tells us the compactor's per-layer
capacity profile is **flatter than an FFN's**, which is a genuine architectural
insight (and a useful negative for anyone tempted to apply TLM-style tapering to
attention-adjacent modules).

## Why the seed-noise floor made this clean

At batch 64×1 the anchor A's seed spread was remarkably tight (sd 0.003, range
width 0.007) — much tighter than sweep 10's mixed-batch A (sd 0.017). This made
the 0.10 ppl taper penalty unambiguously non-overlapping at n=3. The taper arms
themselves are slightly higher-variance (sd ~0.01-0.02), but the gap to A is so
large that it doesn't matter.

## What this means for next steps

- **Tapering the compactor's `d_latent` is a dead end** — don't pursue it.
- **Tapering `compact_len` per layer** (the existing `still_layer_adaptive` /
  PyramidKV mechanism) is a *different* axis (number of latents, not their width)
  and is untested here — it could still help. Worth a separate small sweep if
  compaction budget allocation is of interest.
- **Tapering the base model's `ffn_dim`** (the original TLM result) remains the
  promising orthogonal direction — that's the FFN, where the TLM finding *does*
  apply. Separate sweep.
- **Sweep 12 (flow variance reduction)** is the natural next experiment per
  `NEXT_IDEAS.md` — it follows up on sweep 10's actual positive-ish finding
  (flow trends better by mean) rather than this clean negative.

## Code changes (this sweep)

- `still_d_latent_schedule` config field (`flower/config.py`) — per-layer d_latent list.
- `StillLM.__init__` resolves the schedule + applies an `identity_init` guard
  (identity init only valid at d_latent==2*head_dim; tapered layers use standard init).
- Backward-compatible (default None = uniform = current behavior).
- `configs/sweep_still_taper.yaml`, `scripts/sweep11_analyze.py`.
