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

## 4. Long-context memory bake-off — bloom "win" was capacity, not mechanism (resolved)

**Source:** `docs/training-speedups.md` Section 13 + `docs/sweeps/SWEEP13_PIPELINE.md`
section 13. The long-context direction enabled by FlexAttention.

At seq=8192 / window=2048 (4x ratio), a 2-seed directional pass found bloom_memory
beat a smaller vanilla by -0.0054 bpb — but bloom had ~64% more params. The
powered follow-up resolved the confound with a **param-matched vanilla control**
(d640/L10, 49.6M non-emb vs bloom's 46.1M), both at 10000 steps:

- **vanilla_matched54M: val_bpb 1.0385 ± 0.0023** (2 seeds)
- **bloom_memory: val_bpb 1.0728** (1 seed)

**At matched params and steps, plain sliding-window attention beats bloom by
+0.034 bpb** (~15x the seed noise). The memory mechanism is *hurting* at this
scale: params spent on hash/summary/routing machinery return less than spending
them on a wider vanilla transformer. This is the expected null at 33-54M and
matches Section 13's prediction that memory mechanisms need ~500M+ to show real
signal. **Do not chase memory-mechanism wins below ~500M — they are capacity
artefacts.** The long-context infra is now in place and validated; the next step
that could flip this is the 600M / seq 32K scale-up (Phase 3, needs rented
8xGPU hardware).

**summary_memory note:** the third arm's compile perf blocker is **resolved**.
Its perceiver cross-attention (`nn.MultiheadAttention`) graph-broke under
`torch.compile` and OOMed at batch 8 / seq 8192 — both fixed by replacing MHA
with a compile-clean `F.scaled_dot_product_attention` cross-attention
(`SDPCrossAttention` in `flower/models/memory.py`, shared by `summary_memory`
and `bloom_memory`, which had the identical pattern). The exact failing config
(summary_memory, batch 8, seq 8192) now runs to completion at **14.99 GB peak**
(32 GB 5090) with **zero new graph breaks** vs vanilla (the 3 residual breaks
live in the shared flex block-mask path, present in all variants). Param-count
delta is **0** (`4*D*D + 4*D` either way), so the param-matched comparison is
unconfounded. Legacy checkpoints (sweep7/13 `perceiver.*` / `summary_attn.*`
`in_proj_weight`/`out_proj`) remap automatically at every load site via
`remap_legacy_mha_state_dict`. The bake-off's third arm can now be re-run at
scale.

**Follow-up (not in scope for this fix):** `flow_ot_memory.py` (`source_attn`)
and `flow_pma.py` (`pma`) also use `nn.MultiheadAttention` but are not part of
this bake-off and were left untouched; convert them to `SDPCrossAttention` if a
long-context / compile-perf need arises there.

## 5. Profiling finding: the Muon optimizer step is the dominant step cost (not the model)

**Source:** full-step profile via `scripts/profile_bloom_step.py` (RTX 5090,
bloom_memory d=384 L=4 seq=512 B=8, bf16, Muon).

A full training step at this config is **93.5 ms/step**, split:
- **forward+backward: 39.4 ms (42%)**
- **Muon optimizer step: 54.2 ms (58%)**

The optimizer dominates. Inside it, `_zeropower_via_newtonschulz5` issues
**15 matmuls × 45 two-D params = 675 `aten::mm` calls** per step (the profiler
shows 696 incl. the model's fwd/bwd). `Muon.step` alone is 65.3 ms of the
103.9 ms profiler CUDA total (63%). The per-param Python loop dispatches each
2D weight through NS individually.

The 45 Muon params have only **6 distinct shapes**, and 24/45 are the identical
`(384,384)` — so the NS iterations are highly batchable in principle (a single
batched `bmm` over all same-shape params per NS line, instead of 45 separate
`mm`s). But the spectral-norm normalization (`x / x.norm()`) is per-matrix, so
batching needs a grouped/batched norm first.

**S14 Opportunity 2 (bloom routing) is done and not worth more time:** the hash
einsum + softmax + diagnostics are **0.5% of profiled CUDA** (0.57 ms of
103.9 ms). The single BloomMemoryBlock fwd+bwd is 2.49 ms CUDA, of which
attention (cutlass fmha) is 27% and cuBLAS sgemm (FFN/linears) is 56% — both
already optimal. No further bloom-path change will move the wall-clock needle.

**Next real win = S14 Opportunity 1 / 5a (Turbo-Muon Triton kernel, or batched
NS).** This is the highest-leverage speedup left: a 2-3x faster optimizer step
(~54 ms -> ~20 ms) would cut total step time by ~36%. The `cubic5` schedule
(`muon_ns_schedule: cubic5`) is already wired and cuts NS matmuls 15 -> 10
(-33% optimizer compute) at a documented ~1e-3 val-loss cost (arXiv:2606.00371)
— a config-level experiment worth running before writing any kernel.
