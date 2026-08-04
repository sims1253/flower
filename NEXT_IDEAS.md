# Next experiment ideas (queue, post-Sweep 10)

Updated from an internal research-notes scan on 2026-06-30. These are candidates for the
*next* "huh, that's interesting" experiment after the param-matched flow ablation
resolves. Ordered by cheapness × interestingness.

## 1. Tapered compactor budget (TLMs) — cheap, orthogonal, likely a free win

**Source:** `entities/tapered-language-models.md` (arXiv:2606.23670, Bayat).

TLMs show that under a fixed total param budget, **wider-early / narrower-late
MLP allocation** beats uniform by ~0.3 ppl, and the *direction* matters more
than the amount (wider-late is ~1 ppl *worse*). The mechanism: later layers
reinforce rather than transform, so they need less capacity.

**Flower angle:** the codebase already has `still_layer_adaptive` (PyramidKV-style
layer-adaptive compactor budgets). Sweep 10's arm E shows how to vary per-layer
compactor width cleanly. Two cheap variants to A/B vs uniform at matched total
params:
- **Taper the compactor's `d_latent` / `compact_len` across layers** (cosine
  schedule, d_start/d_end = 1.5/0.5). Directly tests whether the
  reinforce-not-transform finding applies to KV compaction.
- **Taper the base model's `ffn_dim` across layers** (the original TLM result).
  Cheapest possible win; just vary per-layer FFN width under fixed total.

Cost: ~config + small code change for per-layer width. Expected: free win if the
mechanism transfers; null result is also informative.

## 2. Discrete-flow / diffusion compaction — natural extension if flow helps

**Source:** `entities/illada.md` (8B bidirectional masked diffusion LM, competitive
with Qwen2.5-7B), `concepts/discrete-flow-maps.md`, `concepts/flow-matching.md`.

If Sweep 10's B-vs-C shows the flow *mechanism* helps, the obvious next step is
to swap the continuous-time Euler velocity field for a **masked-diffusion /
discrete-flow** compaction step: the compact KV slots are "denoised" from a
masked state conditioned on the full cache. This connects directly to the wiki's
flow-LM debate cluster (iLLaDA, DFMs, coupling-models, s-flm).

Cost: larger code change (new compactor class). Only worth it if Sweep 10 is
positive — gate on the B-vs-C result.

## 3. KV-cache spectral partitioning — already prototyped, underexplored

**Source:** `concepts/kv-cache-spectral-compression.md`. The novel sweep's
`novel_spectral` arm (StillCompactorSpectral) was middle-of-the-pack but never
param-matched against the standard compactor either — same confound as flow.
Worth a param-matched retest if spectral is to be taken seriously.

Cost: reuse the Sweep 10 infrastructure (matched-budget variants).

## Decision rule
- If Sweep 10 B<C (flow mechanism wins): go to #2.
- If Sweep 10 B≥C (flow win was capacity): the flow direction is dead at this
  scale; pivot to #1 (tapering) as the next cheap, orthogonal "free win" probe.

## 4. Long-context memory bake-off — bloom shows signal; needs param-matched control

**Source:** `docs/training-speedups.md` Section 13 + `docs/sweeps/SWEEP13_PIPELINE.md`
section 13. The long-context direction enabled by FlexAttention.

At seq=8192 / window=2048 (4x ratio, the floor where memory can matter), a 2-seed
directional pass found **bloom_memory beats vanilla_local by -0.0054 bpb** (val_bpb
1.0996 vs 1.1050), consistently across both seeds. This is the first positive
long-context memory signal in the project.

**The confound:** bloom (54.5M) had ~64% more params than vanilla (33.3M), so
the win could be capacity, not mechanism. The powered follow-up
(`configs/sweep13_longctx_memory_powered.yaml`) adds a **param-matched vanilla
control** (d640/L10, 49.6M non-emb, within 8% of bloom's 46.1M). If bloom still
beats the matched vanilla, the mechanism is real; if not, it was capacity.

**summary_memory note:** the third arm hit a compile perf issue — its perceiver
cross-attention (`nn.MultiheadAttention`) graph-breaks under `torch.compile`,
dropping GPU util to ~25%. It also OOMed at batch 8 from fragmentation (fixed by
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Before re-running
summary_memory at scale, replace the `nn.MultiheadAttention` perceiver with a
plain scaled-dot-product cross-attention (S14 Opportunity — same class of fix as
the bloom diagnostics graph-break).
