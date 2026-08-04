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

**Update (2026-08-04): batched NS is implemented and measured. The win is
batching, not the Triton kernel.** S14-5a proposed vendoring
`tboissin/newton_schulz_triton` to fuse the three per-iteration matmuls into
Triton (the ~2.8× figure is on NS *compute*). Before building it, profiling
showed the bottleneck was not compute but launch overhead. At the
`summary_memory` d512/L8 seq8192 shape (the committed `profile_longctx_step.py`
target):

  - legacy (per-param `mm`): 166.6 ms/step, optimizer 114.4 ms (69% of step),
    GPU self-CUDA ~61 ms/step → **wall-clock 2.7× the GPU time**. The GPU sat
    idle waiting on the per-param Python dispatch loop (7038 `aten::mm` calls,
    ~13 µs of CUDA work each, paying full launch overhead every time).
  - batched (`bmm` over same-shape param groups, `muon_ns_batched: true`):
    **64.0 ms/step, optimizer 13.4 ms (21%)**, 255k tok/s (+160%), `mm` calls
    7038 → 1008 (-86%), replaced by 270 `bmm`. GPU self-CUDA (~54 ms) now
    matches wall-clock — the GPU is no longer starved.

Net: **full step 2.6× faster (166 → 64 ms), optimizer step 8.5× faster (114 →
13 ms)** — well past the ~36% step-cut target above. This is why batching beat
the Triton path here: a fused single-matrix kernel still launches once per param,
so it would not have moved a launch-bound wall-clock. Batching collapses the
launches, which is the actual bottleneck.

Batching is also numerically free: a `bmm` over a same-shape stack reduces
slice-for-slice to looping the legacy `mm` (bit-identical at these shapes —
`tests/test_newton_schulz.py` asserts rel error 0.00 on the spec shapes and the
50-step smoke-train loss curves agree to 4 decimals). So there is no coefficient-
matching problem and no new dependency; the `cublas` fallback
(`muon_ns_batched: false`) reproduces the exact legacy dispatch for any run that
predates the flag. Aurora stays on the per-matrix `_polar_simple_quintic` path
(its row-oblique rebalancing loop recomputes a per-matrix-scaled NS input each
pass, so there is no stable stack to batch) — see `optim.py` Aurora docstring.

S14 Opportunity 5a (Triton kernel) remains open as a *compute-bound* win for a
later regime (much wider matrices where the matmuls themselves, not launches,
dominate); at the current d512–d1024 shapes it would not have helped. Marking
the launch-overhead half of S14-1/5a done; the compute-fusion half is deferred
until profiling shows it is the bottleneck.

## 6. S14-5b: Liger FusedLinearCrossEntropy (fused lm_head + CE) — implemented, modest win at the bake-off shape

**Status: DONE (implemented + validated; memory win measured and smaller than spec hoped).**

Adopted `LigerFusedLinearCrossEntropyLoss` (link-org/Liger-Kernel) so the tied
`lm_head` projection + CE loss never materializes the full `(B*T, vocab)` logits
tensor during training. `liger-kernel>=0.8.1` was already a dependency. The
flag is `model.fused_linear_ce` (default False; old runs reproduce).

**What runs fused / what stays eager.** When `fused_linear_ce` is on, the model
is training, and CUDA is available, `CausalLM.forward` skips `_compute_logits`
entirely: the Liger kernel takes the pre-normed hidden state and the tied
embedding weight *by reference* and returns only the scalar loss (+ input/weight
grads), so the `(B*T, vocab)` logits tensor is never allocated. The shift is
applied to the activations and labels *before* the fused call (the kernel
projects internally), generalised to `offset` so the same helper serves the main
head (offset=1) and each untied MTP head (offset=i+2). Eval / inference /
logprob consumers are untouched: they run in eval mode or without labels, so the
eager `_compute_logits` path (including the FP8-head eval path) still runs and
`out["logits"]` is a real tensor there. `train.py` reads only `out["loss"]` in
its training step, so `logits=None` under fused training breaks nothing. On CPU
the Triton kernel cannot run, so the forward falls back to eager automatically.
Liger accumulates the softmax/CE internally in fp32 regardless of input dtype,
so `bf16_cross_entropy` (S4) is a no-op under the fused path — strictly *more*
precise than eager bf16 CE.

**Numerical equivalence (RTX 5090, CUDA).** fp32 eager vs fused loss diff
`4.8e-7`; tied-embedding weight-grad max diff `2.2e-8`. Well inside the spec's
1e-4 (loss) / 1e-3 (grad) tolerance — essentially exact. Covered by
`tests/test_training_speedups.py` (S14-5b section, CUDA-gated like the FP8 and
Flex tests). Full suite (240 tests) green.

**Measured memory reduction (the spec asked for >30%; the real number is
smaller, and the reason is informative).** At the 450M `vanilla_matched`
config (d1280/L20, vocab=16K, bf16, SDPA), `torch.cuda.max_memory_allocated`
for one fwd+bwd+step:

| shape            | eager   | fused   | Δ         |
|------------------|---------|---------|-----------|
| seq=8192, B=1    | 20.18 GB| 19.35 GB| **−0.83 GB (−4.1%)** |

That is a real, reproducible saving but far below the >30% the spec projected.
The reason: at this model/seq ratio the logits tensor that gets eliminated is
`8192 × 16384 × 2 B (bf16) ≈ 256 MB`, plus its backward grad ≈ 256 MB, plus
CE's internal fp32 log-sum-exp ≈ 512 MB — i.e. ~1 GB out of a 20 GB budget. The
413M params + optimizer states + attention activations dominate. The >30%
figure would hold where logits *do* dominate: much larger `B*T` (big batch or
seq) and/or much larger vocab, or once the attention path stops competing for
the budget (see the seq=32K note below). The saving scales linearly with
`B*T*vocab`, so it grows at the bake-off's effective batch (B=2 → ~1.7 GB saved)
and would be the dominant term at vocab=50K+.

**seq=32768 (S14-5b's "does it fit now?" claim).** The fused CE path itself
runs cleanly at seq=32768 — verified on a tiny d128/L2 model (13.37 GB peak,
loss finite, `logits is None`). But the 413M model at seq=32768 still OOMs on
this box, and the blocker is **not** the head: it is the SDPA path's dense
`(1,1,32768,32768)` causal mask (~4 GB fp32), which is exactly the problem
FlexAttention (S1) was added to solve. The measurement had to use the SDPA path
because FlexAttention's unfused math fallback hit a WSL2 driver glitch under
`torch.compile`-less eager; under the real bake-off config (`compile_model:
true`, `flex_attention: true`) FlexAttention compiles the mask away and the
fused CE's −0.83 GB becomes the marginal win that matters. **Net: fused CE is
necessary-but-not-sufficient for seq=32K; it must be paired with the compiled
FlexAttention path, which is already the bake-off default.** Re-measure both
flags on together (and at B=2) once the WSL2 driver is stable or on rented
H100/B200 hardware where the driver doesn't degrade under repeated ~20 GB
allocations — the per-run B=2 numbers were not captured this session because the
WSL2 GPU driver entered a degraded "device not ready" state after several
near-capacity allocations.

**Scope.** `CausalLM` only. The four `CausalLM` subclasses that override
`forward` (`flow_pma`, `flow_meanflow`, `frequency_decay_memory`, `engram_lite`)
and `StillLM` are untouched — they have their own head/CE paths. The bake-off's
`vanilla_local` and `bloom_memory` both inherit `CausalLM.forward`, so both get
the fused path automatically.

**torch.compile interaction (operational caveat).** The bake-off runs
`compile_model: true`. Under compile, the fused path adds **one graph break
inside the Liger library** (`liger_kernel/ops/fused_linear_cross_entropy.py`,
caused by a `.item()` sync on the non-ignore token count) on top of the
pre-existing graph break the baseline already has at `base.py:871` (the
diagnostic-walker `for module in self.modules()` loop, which deopts the forward
frame to eager in *both* fused-on and fused-off configs). Net: enabling
`fused_linear_ce` does not change the number of *Flower*-owned graph breaks —
the new break is library-internal and cannot be removed without forking Liger.
The compiled fused forward+backward still produces a loss equivalent to compiled
eager (verified at rtol 5e-2 on bf16, `test_fused_linear_ce_matches_eager_under_compile`),
which is the property that matters for a training run. The lazy Liger init is
primed before the first compiled call (the bake-off's `train.py` runs one
warmup forward), so the import path is not traced. A user who sets
`fused_linear_ce=True` on a non-CUDA model gets a one-time warning that the
forward is falling back to eager (no silent fallback).


