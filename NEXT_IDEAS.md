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

## 7. S14 Opportunity 3: analytical Titans surprise without autograd (research contribution, implemented)

**Status (2026-08-05): implemented, gated behind `model.titans_analytical_surprise`
(default off). Legacy autograd path preserved bit-for-bit for reproducibility.**

`docs/training-speedups.md` §14 Opportunity 3 flags the TitansMACBlock surprise
computation as the hot path of the `titans_mac` variant: every forward step it
detaches the memory, creates a fresh `requires_grad` leaf, builds an inner
autograd graph for the associative-retrieval MSE, calls `torch.autograd.grad` to
get the surprise signal, and (in training) retains that inner graph so the outer
CE can backprop through it. Building and destroying that inner graph every step
is pure overhead.

The contribution: the inner surprise gradient has a **closed form**. Given the
inner loss

    s_s = <M_s, k>/√D ;  w_s = softmax(s)_s ;  p = Σ_s w_s·M_s ;  a = p − v
    loss = ‖a‖²/(B·D)   (F.mse_loss mean over batch AND feature)

the gradient w.r.t. each memory slot M_s has two paths — direct (M_s appears in
p) and through-softmax (M_s appears in s_s, which feeds every weight via the
softmax Jacobian) — and they sum to

    d(loss)/d(M_s) = (2/(B·D))·w_s·[ a + (<a, M_s − p>/√D)·k ]

so the surprise (negative gradient) is a handful of batched einsums and
element-wise ops. **No inner graph is constructed.** The *outer* CE gradient is
unaffected — every op in the closed form is a standard differentiable PyTorch
op, so autograd still flows through `key_proj`/`val_proj`/`alpha_logit`/
`write_scale` and through cross-layer memory. Only the *inner* graph build is
removed.

**One subtlety worth flagging because the S14 doc gets it wrong.** The S14
derivation drops the `2/(B·D)` constant ("absorbed into alpha·write_scale").
That is incorrect: `F.mse_loss` with default mean reduction divides by `B·D`,
not `D`, and dropping the constant (a) breaks the gradient-equivalence gate by
a factor of `2/(B·D)`, and (b) breaks checkpoint compatibility — `alpha_logit`
is initialised to `−2.0` and `write_scale` to `1.0`, so the constant is *not*
free to absorb without shifting the effective write rate. The implementation
keeps the exact `2/(B·D)` factor.

**Validation (the acceptance gate):** `tests/test_titans_surprise.py` compares
the closed form against `torch.autograd.grad(_inner_loss(...))` on random
inputs at two shapes.

| shape         | dtype | max-abs-diff | gate | status |
|---------------|-------|--------------|------|--------|
| (B=4,S=8,D=64)   | fp32 | ~1.9e-9 | 1e-4 | PASS |
| (B=2,S=16,D=256) | fp32 | ~2.3e-9 | 1e-4 | PASS |
| (B=4,S=8,D=64)   | bf16 | ~1.2e-4 | 1e-2 | PASS |
| (B=2,S=16,D=256) | bf16 | ~1.2e-4 | 1e-2 | PASS |

The fp32 residual is at the ULP floor — it is the same computation, rearranged.
Full-block forward equivalence (both modes, identical weights): memory output
agrees at <1e-3. Outer-backward sanity: `loss.backward()` in analytical mode
produces finite grads on `key_proj.weight`, `val_proj.weight`, `alpha_logit`,
`write_scale` (the outer graph is intact). `pytest tests/` stays green (251
passed).

**Measured speedup** (`scripts/bench_titans_surprise.py`, RTX 5090, isolated
`_surprise_update` at B=8/S=16/D=768, 100 calls):

| mode                              | autograd | analytical | speedup | saves / layer / step |
|-----------------------------------|----------|------------|---------|----------------------|
| eval (no_grad, `create_graph=False`) | 5249 µs  | 3397 µs    | **1.55×** | 1.85 ms |
| train (`create_graph=True`)       | 4178 µs  | 3407 µs    | **1.23×** | 0.77 ms |

The S14 estimate ("could halve the compute of the Titans variant") was
optimistic. The analytical path is a few cheap einsums either way; the win is
the eliminated graph construction, which is most visible at eval (1.55×) and
shrinks in training (1.23×) because `create_graph=True` keeps a graph
regardless. For a 28-layer Titans model that is ≈21 ms saved per training step,
≈52 ms per eval step — real but modest, since the surprise is a small fraction
of the full block (attention + FFN dominate). The win scales with layer count.

**Why it is paper-worthy regardless of the modest speedup.** Computing
Titans-style surprise without autograd is, as far as we know, a novel result —
the original Titans paper (Behrouz et al. 2024) and follow-ups compute the
surprise signal via `torch.autograd.grad` and treat the graph overhead as
intrinsic. The closed form shows it isn't: the surprise of an
associative-retrieval MSE under softmax attention has an exact, cheap
analytical expression that preserves end-to-end differentiability. That is a
clean, self-contained contribution. If a convergence study at the bake-off
scale (600M / seq 32K) shows the analytical path matches the autograd path in
final BPB — which the 1e-9 fp32 equivalence strongly implies it must — this is
a citable optimisation for any Titans-style memory module.

**What is NOT done here (open for the convergence study).** This task delivers
equivalence + the speedup measurement. The remaining research question — does
training-to-convergence with `titans_analytical_surprise=True` reproduce the
autograd path's loss curve at the bake-off scale? — needs a real training run
and is out of scope for this change. The flag is off by default so the
question can be answered by a single config flip on the next titans sweep.

## 8. S14 CUDA graphs (reduce-overhead): works, but shape-dependent; the real blocker was grad accumulation

**Source:** the overnight task doc (Task 1). Goal: flip `compile_mode` from
`default` to `reduce-overhead` in production configs for a "free" 10-30%
throughput win from CUDA-graph launch-overhead elimination.

**The prompt assumed this was a 1-line config flip. It was not.** The real
blocker, which the prompt flagged as a risk ("verify the optimizer doesn't
reallocate momentum buffers") but understated, is a hard PyTorch bug:
`torch.compile(mode='reduce-overhead')` + `gradient_accumulation_steps > 1`
crashes with `RuntimeError: accessing gradient tensor output of CUDAGraphs that
has been overwritten by a subsequent run`. This is pytorch/pytorch#169545
(open, unfixed as of torch 2.13). The standard workarounds the error message
suggests — `torch.compiler.cudagraph_mark_step_begin()` between micro-steps,
cloning the loss output — do **not** fix it (verified). Only one thing does:
**pre-allocate persistent `.grad` buffers before CUDAGraph capture**, so the
graph pins stable `.grad` addresses instead of allocating fresh ones during
capture (which the next replay then overwrites).

**The fix (implemented in `flower/train.py`).** When
`compile_mode == "reduce-overhead"` AND `gradient_accumulation_steps > 1` AND
CUDA, train.py now does `p.grad = torch.zeros_like(p)` for every trainable
parameter after compile, and switches the per-step `zero_grad(set_to_none=True)`
to `set_to_none=False` so the buffers stay allocated. The `.grad` memory cost
(1× params) is already the steady-state cost of any backward, so there is no
new memory penalty. Default mode and accum==1 are untouched (the standard
`set_to_none=True` low-memory path). Correctness verified: loss matches
`default` mode within 2e-5 at the same seed across 5 steps (the tiny delta is
bf16 kernel-selection noise between the two compile modes, not a numerics bug).

**Measured speedup (real `flower.train`, real fineweb_edu, 80 steps, RTX 5090):**

| config | shape | default tok/s | RO tok/s | speedup |
|--------|-------|---------------|----------|---------|
| 100m longctx | vanilla d768/L14, seq8192, b4/accum2, flex+local | 77,228 | 83,159 | **+7.7%** |
| 100m phase0 | vanilla d768/L14, seq1024, b16/accum4, SDPA | 90,071 | 83,741 | **-7.0%** |
| bloom bake-off | bloom d512/L8, seq8192, b4/accum2, flex+bloom | 211,837 | 207,673 | **-2.0%** |
| 450m longctx | vanilla d1280/L20, seq8192, b2/accum16, flex | 52,428 | **OOM** | n/a |

**Verdict: reduce-overhead is shape-dependent, not a universal win.** It helps
only on the long-context compute-bound shape (seq8192, long per-step GPU work,
where launch overhead is a small fraction that CUDA graphs still trim). On
short-sequence / high-launch-rate shapes (seq1024/b16) and on the small bloom
variant it is neutral-to-slower (capture overhead exceeds launch savings). At
450M it does not fit at all on the 32GB 5090 — CUDA graphs reserve private
memory pools (~18.6 GB) on top of the 25 GB default-mode peak, exceeding the
VRAM cap. So reduce-overhead is a small-config / long-sequence tool only.

**Config flips applied:**
- `sweep13_100m_longctx_phase0.yaml`: flipped to `reduce-overhead` (+7.7%,
  validated end-to-end through real train.py).
- `sweep13_100m_phase0.yaml` (seq1024): kept `default`, updated the stale TODO
  comment to record the -7% measurement.
- `sweep13_450m_longctx_memory.yaml`, `sweep13_longctx_memory_bakeoff.yaml`:
  kept `default` (450M unmeasured; the comparable bloom shape was -2%). A
  measured A/B at 450M is the gating experiment before flipping those.

**Why this matters beyond the speedup.** The grad-accum fix is a
correctness/enabling fix independent of the throughput question: it makes
`reduce-overhead` usable at all on any config with accumulation, which is most
of them. Anyone who later flips a long-context config to reduce-overhead gets
working CUDA graphs without re-discovering the bug.

## 9. S14 bugfix: attention-window warmup exhausted the torch.compile recompile limit

**Source:** Task 2. The runs log (`runs_sweep13_longctx_bakeoff.log`) showed a
`recompile_limit` hit. The task prompt attributed it to the warmup ramp, but
the logged stack trace was actually from `summary_memory` on *stale pre-refactor
code* (the `_get_or_build_block_mask` compile-safe guard added in the S1 work
already fixed that path). So the first step was reproducing the *actual* warmup
trigger, which is real and separate.

**Reproduced.** With `attn_warmup_steps > 0` + `flex_attention` + compile, a
20-step ramp (window 64→1024) caused `create_block_mask` to recompile **8
times** then hit `recompile_limit` (default 8). Each distinct window value
recompiles because flex's `mask_mod` closes over the window integer, and
`create_block_mask` is itself dynamo-traced (guard:
`mask_mod.__closure__[0].cell_contents == <window>`). After hitting the limit
flex falls back to its eager dense `math_attention` path, which materializes the
full (B,H,T,T) scores tensor — throughput collapsed to 16k tok/s.

**Fix (Option 1 from the prompt: quantize).** Added
`ModelConfig.attn_warmup_quantize: int = 0` (step-stride). When > 1, the ramp
only advances the window every `quantize` steps (held constant in between),
bounding distinct windows to `ceil(warmup_steps / quantize)`. With
`quantize=8` over the 20-step ramp: recompiles dropped 8→**3**, **0** limit
hits, flex stays on the fused kernel. The final window still reaches
`local_window` exactly; `quantize=0` reproduces the legacy per-step ramp (old
runs unaffected); `warmup_steps=0` is still a no-op. The ramp's training-
dynamics benefit (start narrow, widen) is preserved — it jumps in `quantize`-
step plateaus instead of 1-position steps. Test added
(`test_window_warmup_quantize_bounds_distinct_windows`).

**Note:** all production long-context configs currently have
`attn_warmup_steps=0` (warmup off), so this is a latent-bug fix that unblocks
warmup+compile coexistence for any future run that turns warmup on. No config
flip needed yet; the field defaults to the legacy behaviour.

## 10. S14 activation checkpointing: the seq=32K enabler (flex+checkpoint incompatibility fixed)

**Source:** Task 3. Goal: wrap each transformer block in
`torch.utils.checkpoint` so seq=32K fits on the 32GB 5090 (Section 13's binding
constraint is activation memory, not compute). There was no checkpointing
anywhere in the repo.

**Implemented.** `ModelConfig.activation_checkpoint: bool = False`. When on
(and training, and `loop_count==1`), `CausalLM.forward` wraps each block in
`checkpoint(block, x, memory, use_reentrant=False)`. Off in eval (no backward,
so checkpointing only wastes a recompute) and when false (old runs reproduce).
AttnRes (`depth_router`) is left uncheckpointed — it reads inter-block deltas,
and is not used in the long-context configs this targets.

**The non-obvious blocker: checkpoint + FlexAttention are incompatible.**
`use_reentrant=False` recomputes the block forward during backward in a context
the outer `torch.compile` graph does **not** cover, so flex falls back to its
eager `math_attention` path, which materializes the full (B,H,T,T) scores
tensor and OOMs at long context (~3 GB just for scores at seq8192; ~50 GB at
seq32K). This is documented in pytorch/pytorch#147879. The working workaround
(from that issue): **compile `flex_attention` as a standalone callable** so the
fused kernel is available in the recompute too. Implemented as
`_load_flex_attention_compiled()` in `base.py`, used by `_forward_flex` when
`activation_checkpoint` is on. 4 tests added (defaults-off, loss-identity with
dropout RNG, eval no-op, CUDA memory reduction).

**Correctness (the strict gate).** With dropout > 0, the loss is **bit-
identical** (delta = 0.0) with vs without checkpointing at the same seed —
proving `use_reentrant=False` correctly saves/restores RNG state across the two
forward passes, and the differentiable memory tensor carried between blocks
flows through checkpointing correctly.

**Measured memory (the headline result):**

| shape | no-checkpoint | checkpoint | reduction |
|-------|--------------|-----------|-----------|
| 100M, seq8192, b4 (fwd+bwd peak) | 16.45 GB | 6.97 GB | **-58%** |
| 100M, seq2048, b2 (smaller, dense-flex fits) | 3.14 GB | 2.13 GB | -32% |
| 100M, seq32768, b1 | **OOM (29.8 GB fwd, OOM bwd)** | **11.09 GB** | **fits** |
| 100M, seq32768, b2 | OOM | 13.75 GB | fits with headroom |

**seq=32K now fits on the 5090 — the Section 13 gate is open.** Without
checkpointing, seq=32K OOMs on the forward alone (29.8 GB for activations at
b1). With checkpointing it peaks at 11.1 GB (b1) / 13.8 GB (b2) — comfortable
headroom on the 32 GB card, enough to even raise batch. This directly determines
that the Section 13 scale-up (memory mechanisms at seq=32K) is possible on local
hardware, not just rented 8×GPU. Production throughput measured end-to-end
through `flower.train` at seq=32K/b2/accum4 with checkpointing on:
**~77k tok/s, 13.1 GB peak** (configs/sweep13_100m_longctx32k_checkpoint.yaml).
The no-checkpoint comparison OOMs as documented, so the "throughput cost" is
moot at this scale — checkpointing is what makes the run possible, not slower.

**Throughput cost where it is optional (seq≤8K that already fits):** ~25-33%
slower (one extra forward per backward). The trade is favourable exactly when
memory is the binding constraint. The flag is off by default so
throughput-sensitive runs at seq≤8K that already fit are unaffected.

### 10a. Selective activation checkpointing — implemented (recompute only the big activations)

**Source:** Task 3b. Full checkpointing (§10) cuts activation memory O(num_layers)
→ O(1) but recomputes **every** activation in the block during backward,
costing ~25-33% throughput. The dominant activation in a SwiGLU/GELU block is
the FFN intermediate `(B, T, ffn_dim)` — at 100M/seq8192/b4/bf16 that is
~134 MiB/layer, vs ~6 MiB for the residual stream `(B, T, d_model)`. PyTorch
2.13's `torch.utils.checkpoint.create_selective_checkpoint_contexts` lets a
policy decide per-op whether to recompute (`MUST_RECOMPUTE`) or save
(`MUST_SAVE`). The win it targets: keep cheap activations materialized (no
recompute cost for them) while still dropping the big ones.

**Implemented.** `activation_checkpoint` now accepts `False | True |
"selective"` (was `bool`). `True`/`False` are unchanged. When `"selective"`,
`CausalLM.forward` wraps each block in
`checkpoint(block, x, memory, use_reentrant=False, context_fn=...)` where the
context_fn is `_selective_checkpoint_context_fn` (base.py): a byte-threshold
policy that returns `MUST_RECOMPUTE` for any op output ≥ 8 MiB and `MUST_SAVE`
otherwise — targeting the FFN intermediate while keeping the residual stream
and norm outputs materialized. The same gates as full checkpointing apply
(training, `loop_count==1`, `depth_router is None`), and the
`_flex_needs_compile` workaround is active (selective recomputes flex too, so
the standalone-compiled-flex kernel is needed in the recompute). 4 tests added
mirroring the full-checkpoint tests: config validation (rejects typos),
loss-identity-with-dropout (the RNG gate — selective, full, and none must be
bit-identical at the same seed in eager mode), eval no-op, and a CUDA
memory-reduction test.

**Correctness (strict).** In eager mode with dropout>0, the loss is
**bit-identical** (delta = 0.0) across none / full / selective at the same seed
— proving the byte-threshold policy + `use_reentrant=False` preserves RNG state
across the forward and recompute passes, exactly like full checkpointing.
Under `torch.compile` the three differ by ~6e-4, but that is compile
reduction-order noise (the same kernel-rounding delta documented for the fused-
CE test, `test_fused_linear_ce_matches_eager_under_compile`), not an RNG bug —
full and selective stay bit-identical to each other under compile too.

**Measured (RTX 5090, bf16, torch.compile default, flex, 100M d768/L14):**

| shape | mode | peak (fwd+bwd) | tok/s | ms/step |
|-------|------|----------------|-------|---------|
| seq8192 b4 | none | 13.74 GB | 164k | 200 |
| seq8192 b4 | full | 3.44 GB | 80k | 408 |
| seq8192 b4 | **selective** | **5.28 GB** | **~85k** | **~385** |
| seq32768 b1 | none | 13.85 GB | 178k | 184 |
| seq32768 b1 | full | 3.29 GB | 81k | 403 |
| seq32768 b1 | **selective** | **5.64 GB** | **80k** | **411** |

**Reading.** Selective lands between none and full on memory (5.28 GB at 8k is
-61% vs none, +53% vs full) and recovers a sliver of throughput over full at
8k (~85k vs ~80k tok/s, ~5-9%). The throughput recovery is **small**, not the
"huge win" the framing hoped for, and the reason is instructive: the FFN
intermediate is both the dominant activation **and** the dominant recompute
cost (the gate/up/down matmuls are the heavy ops), so dropping exactly that
tensor means selective recomputes the expensive ops just like full does. The
only recompute it avoids is for the cheap residual/norm ops, which are cheap
to begin with. Selective's value here is therefore **memory-granularity**, not
throughput: it is the right knob when you have a bit more memory headroom than
full requires and want a slightly less aggressive recompute, but it does not
beat full on throughput at these shapes.

**When to use which.**
- `false` — default; seq≤8K that already fits, throughput-sensitive.
- `true` — maximum memory saving (the seq=32K enabler); accept the full
  recompute cost.
- `"selective"` — middle ground when memory is tight but you have ~50% more
  headroom than full needs and want to keep the small activations live; not a
  throughput win over full at these shapes.

## 11. FP8 training via Transformer Engine — blocked on this box (no sm_120 wheel)

**Source:** §13 precision routing. FP8 FFN/attn matmuls are the biggest
remaining throughput lever on Blackwell (~2× the BF16 throughput on the FFN,
which is ~60% of compute). The eval-only `torch._scaled_mm` FP8 head (§3) cannot
extend to training: `_scaled_mm` has **no backward kernel** in torch 2.13
(verified: `derivative for aten::_scaled_mm is not implemented`). FP8 *training*
needs Transformer Engine (TE), which handles the autocast + backward.

**Blocked.** `transformer-engine[pytorch]` v2.17 has no prebuilt wheel for
torch 2.13+cu130 (very new; TE's wheels lag) — `pip install` triggers a C++
source build ("Error compiling objects for extension"), and the meta-package
stays empty. A full source build (CMake + ninja) is what OOM'd the entire WSL2
machine on the first attempt (pegged all cores + thrashed the WDDM driver) and
required a WSL restart. So FP8-via-TE is not viable on this WSL2+sm_120 box
without either (a) a controlled, ninja-job-capped source build in a fresh
session, or (b) a different host with a prebuilt wheel / more RAM.

**Recommendation.** Revisit on a non-WSL host or when TE ships a cu130/sm_120
wheel. The §13 precision-routing config scaffolding (`ffn_precision`,
`attn_precision`) is already in place; only the TE-backed matmul casting is
missing. When TE is available, the work is: swap `nn.Linear`→`te.Linear` in
`FeedForward`/`CausalSelfAttention` behind `ffn_precision="fp8"`, keep memory
modules bf16 (§13 rule), validate TE+`torch.compile`+Muon coexist, gate loss
delta <0.02. Estimated ~1 day once the dependency installs.

## 12. Liger FusedRMSNorm + FusedSwiGLU: 2× SLOWER under compile — do not adopt

**Source:** §5b (Liger-Kernel), which ranked FusedRMSNorm/SwiGLU/RoPE as P1
"drop-in" wins. This is the measurement that refutes that ranking for the
norm/SwiGLU kernels specifically (FusedLinearCrossEntropy — §6 — was the
exception and IS adopted; it's a memory win, not a kernel-speed win).

**Measured (RTX 5090, bf16, torch.compile mode=default — the production
setting), d768/L14/seq8192/b4 (the 100M longctx shape):**

| component (compiled) | current (nn.Linear+F.silu) | Liger | speedup |
|----------------------|---------------------------|-------|---------|
| RMSNorm | 0.54 ms | 0.69 ms | **0.78× (slower)** |
| SwiGLU FFN | 4.89 ms | 9.72 ms | **0.50× (2× slower)** |
| **full step (fwd+bwd+opt, 14 layers)** | **73.8 ms** | **166.4 ms** | **0.44× (-56%)** |

In eager mode (no compile) Liger is 1.33-1.38× faster, as advertised — its
fused Triton kernels beat unfused eager. **But the production path compiles**,
and `torch.compile` already fuses the `nn.Linear`+`F.silu`+elementwise-mul chain
into equivalent-or-better Triton kernels. Liger's standalone fused kernels then
compose *worse* with the surrounding compiled graph (likely a materialization
boundary at Liger's custom autograd `Function`), so they cost rather than save.
No memory benefit either (7.85 GB both ways at this shape).

**Verdict: do not adopt LigerRMSNorm / LigerSwiGLUMLP.** This confirms the §5a
thesis (the prior optimizer-profiling finding that the step was launch-bound and
compile-fusion beats hand-fused kernels at these shapes) and extends it: at
flower's shapes, `torch.compile` subsumes the per-op fused-kernel libraries.
The exception is ops that change the *memory* profile (FusedLinearCE eliminates
the logits materialization) — those remain worth adopting regardless of kernel
speed. Benchmark: `scripts/bench_liger_kernels.py`.


