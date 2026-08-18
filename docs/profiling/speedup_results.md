# Training speedups — measured results (450M, RTX 5090)

Companion to `baseline_profile.md`, which established *where* step time goes.
This one records what was tried against it, what worked, and — equally load-bearing —
what did not, so the dead ends are not re-explored.

**Target config:** `configs/sweep13_450m_longctx_memory.yaml`, `vanilla_matched`
(413M params, d1280/L20, seq 8192, window 2048, batch 2 x accum 16, bf16 +
FlexAttention + `torch.compile` + Muon).

**Reference baseline** — the two finished seeds in `runs/sweep13_450m_longctx/`:

| seed | val_bpb | tok/s (whole-run avg) |
|---|---:|---:|
| 0 | 0.90234 | 44,810 |
| 1 | 0.90270 | 41,918 |

The **seed band is 0.0004 BPB**. That is the resolution at which any
numerics-changing optimization has to be judged.

## Measurement discipline (read before quoting any number here)

Three different throughput numbers exist for this config and they are **not**
comparable:

1. **52,207 tok/s** — `baseline_profile.md`, steady state, synthetic tokens, 30 steps.
2. **41,918–44,810 tok/s** — `metrics.json`, whole-run average over 10k steps,
   including compile warmup, 10 validation passes and 5 checkpoint writes.
3. **~43,776 tok/s** — `scripts/bench_arms.py`, steady state, 4-step blocks.

The ~17% between (1) and (2) was initially assumed to be recoverable input-pipeline
overhead. **It is not.** Measured with `scripts/bench_data_workers.py`, the
FineWeb-Edu loader sustains **1,079,869 tok/s at the default 2 workers** — roughly
**20x** the rate the model consumes. Data loading was never the constraint. The gap
is measurement scope plus the lower sustained clock of a 17-hour run versus a
2-minute profile.

**Consequence:** compare arms only against a control measured the same way in the
same invocation. Run-to-run drift between invocations is ~2% (the `bf16_baseline`
arm measured 41,562 and 43,776 tok/s on two different days), which is larger than
several of the individual effects below.

## Headline: the stack

All measured in one `scripts/bench_arms.py` invocation (batch 2 x accum 4 x seq
8192, 3 repeats), so these are directly comparable to each other:

| arm | tok/s | vs bf16 | peak GB |
|---|---:|---:|---:|
| `bf16_baseline` | 43,776 | 1.000x | 24.52 |
| `fp8_tensorwise` | 55,397 | 1.265x | 20.31 |
| `fp8_fusedce` | 55,757 | 1.274x | 19.18 |
| `fp8_autotune` | 56,653 | 1.294x | 20.31 |
| **`fp8_stack`** (fp8 + fused CE + autotune) | **57,145** | **1.305x** | **19.18** |
| `fp8_stack_smooth` (stack + smooth_swiglu) | 55,443 | 1.267x | 20.02 |
| `fp8_no_margin` (fp8 alone, no bf16 margin) | 57,845 | 1.321x | 19.84 |

**~1.31x throughput and -5.3 GB peak VRAM**, from three composable changes.
Per-arm spread was 0.2–1.5%, so every gap above is outside noise.

Two rows price a safety option rather than deliver a speedup, and both decisions
belong to the quality screen, not to this table:

- **The bf16 margin costs ~4%** (`fp8_tensorwise` 55,397 vs `fp8_no_margin`
  57,845 — otherwise identical). `fp8_stack_no_margin` exists in the screen
  config to decide whether converting block 0 and block 19 moves the loss.
- **`smooth_swiglu` costs ~3%** (`fp8_stack` 57,145 vs `fp8_stack_smooth`
  55,443). S10 exists to stop FP8 SwiGLU divergence; if tensorwise FP8 is stable
  here without it, that 3% is free.

Enable neither by default. Turn one on only if the loss trace asks for it.

## What worked

### FP8 tensorwise linear training — the one large win

`flower/precision.py`, enabled by `training.fp8_linear`. Converts the FFN
gate/up/down and the Q/K/V/O projections to FP8 via torchao, targeting the cutlass
GEMM bucket that `baseline_profile.md` measures at **62.1% of kernel time**.

Isolated 450M block, full fwd+bwd, compiled:

| recipe | ms/block | vs bf16 |
|---|---:|---:|
| bf16 | 18.87 | — |
| **fp8 tensorwise** | **13.61** | **1.39x** |
| fp8 rowwise | 18.72 | 1.01x |

**Rowwise is pointless on this GPU.** It is the numerically safer recipe
everywhere else, but on sm_120 the per-row scale computation costs back exactly
what the FP8 GEMM wins. Tensorwise is the only recipe that buys anything, so the
speed and the numerical risk are bought together — hence the guardrails
(first/last block kept in bf16, tied LM head never converted, pair with
`smooth_swiglu`).

torchao 0.18 offers only `dynamic` and `disabled` scaling — delayed scaling
(which would amortize the amax pass across steps) is gone, so there is no further
recipe tuning available.

### Liger fused linear+CE — was broken, now fixed

`fused_linear_ce: true` combined with `compile_model: true` — the only
configuration S14-5b was written for — crashed during the first compile on
torch 2.13.0+cu130:

```
InductorError: LoweringException: TypeError:
tuned_addmm() takes 3 positional arguments but 4 were given
target: aten.addmm.dtype
```

Liger calls `torch.addmm(..., out_dtype=...)`, hitting an overload that
inductor's `tuned_addmm` lowering does not accept. Reproduced with **and
without** FP8, so it is a plain Liger + `torch.compile` incompatibility, not an
FP8 interaction. Fixed by routing through `_call_liger_fce`
(`@torch._dynamo.disable`) in `flower/models/base.py` — one graph break at the
very end of the forward, guarding a hand-written Triton kernel that gains
nothing from inductor lowering. Guarded by `tests/test_fused_ce_compile.py`.

Effect: **-1.13 GB peak** (20.31 → 19.18 GB) at roughly neutral throughput.

### EMA update — multi-tensor, bit-exact

The eval EMA ran a Python loop of `mul_`/`add_` over 142 parameters, i.e. 284
kernel launches per step for pure-bandwidth work. Replaced with
`torch._foreach_mul_` + `torch._foreach_add_`, which is **bit-exact** against the
old form (verified).

Deliberately **not** `torch._foreach_lerp_`: algebraically identical, but rounds
differently (~5e-7 drift in fp32). Runs here are compared at 0.0004 BPB
resolution, so a free bit-exact form beats a marginally faster inexact one.

## Quality screen (4 arms x 600 steps, `runs/speedup_screen_450m/`)

Throughput first — these are **whole-run** numbers from a real training run, not
a microbenchmark, and they confirm the bench holds in practice:

| arm | val_bpb | Δ vs bf16 | tok/s (whole run) |
|---|---:|---:|---:|
| `bf16_baseline` | 1.65724 | — | 44,836 |
| `fp8_stack` | 1.65831 | +0.00107 | **57,146 (1.275x)** |
| `fp8_stack_no_margin` | 1.65875 | +0.00151 | 54,099 |
| `fp8_stack_smooth` | 1.65842 | +0.00117 | 55,594 |

Two decisions fall out immediately, independent of how the quality question
resolves:

- **Drop `smooth_swiglu`.** S10 exists to stabilise FP8 SwiGLU, but it does not
  reduce the FP8 offset here (+0.00117 vs +0.00107 without it) — because
  tensorwise FP8 is not diverging in the first place. It costs ~3% throughput
  for nothing. Leave it off.
- **Keep the bf16 margin.** `fp8_stack_no_margin` is worse on *both* axes: worse
  quality (+0.00151) *and* slower in the real run (54,099 vs 57,146). Note this
  **inverts** the isolated benchmark, where dropping the margin was faster
  (57,845 vs 55,397) — but that pair was measured without fused CE or
  autotuning. Stacked with `max-autotune`, the margin is free. A microbenchmark
  ranking does not survive a change of what it is stacked with.

  Caveat on that inversion: the screen runs each arm **once**, and run-to-run
  drift on this box is ~2%. The 5.6% gap is larger than drift so it is probably
  real, but it rests on n=1. If the margin ever needs re-litigating, re-measure
  the stacked pair with `scripts/bench_arms.py --repeats 3` rather than trusting
  this row.

### Is +0.001 val_bpb real? No — it is 12x below reseed noise

The tempting comparison is against the finished runs' 0.0004 seed band. **That
comparison is invalid and would have rejected a 1.275x speedup.** 0.0004 is
measured at 10k steps, fully converged. This screen is 600 steps with
`warmup_steps: 500` — 83% LR ramp, where two seeds have had far less opportunity
to converge toward each other.

Measured directly (`runs/speedup_screen_450m_seed1/`, same arms at seed 1), the
**600-step seed band is 0.01282** — **32x wider** than the 10k-step band:

| quantity | val_bpb |
|---|---:|
| bf16 reseed delta @600 steps (the correct band) | **0.01282** |
| fp8_stack reseed delta @600 steps | 0.01339 |
| fp8_stack offset vs bf16 (same seed) | 0.00107 |
| 10k-step converged seed band (**wrong regime**) | 0.0004 |

The FP8 offset is **an order of magnitude smaller than the noise floor at this
training length**. All three FP8 arms pass comfortably. FP8 also does not amplify
seed sensitivity — its own reseed delta (0.01339) is essentially the bf16 one.

The per-step trace agrees: the gap peaks around step 375–400 and then narrows
rather than widening, which is an offset signature, not divergence.

**Method note worth keeping.** A seed band is a property of a *regime*, not of a
model. Importing one across run lengths is how a harmless precision offset gets
written up as a regression. `scripts/analyze_speedup_screen.py` now measures the
band from a same-length reseed and refuses to run if given neither that nor an
explicit band, rather than defaulting to a plausible-looking number.

This screen establishes only that FP8 does not *diverge* and sits under reseed
noise at 600 steps. Converged quality is a separate claim and needs the 10k run.

## Where the time goes *after* FP8

Re-profiled the `fp8_stack` arm (`docs/profiling/traces_fp8/`). The bf16 category
breakdown in `baseline_profile.md` no longer describes this workload.

**Caveat first:** that script's category table is unusable for this run — the
profiler attributed only 42 ms of a 1206 ms step, because the
`## Call CompiledFxGraph` wrapper swallows the child kernel time. The script
warns about exactly this. The **Top-N CUDA table and the phase split are the
ground truth**, and only those are quoted here.

Top CUDA ops, self device time:

| op | % CUDA | note |
|---|---:|---|
| `aten::_scaled_mm` | 28.1% | the FP8 GEMMs — already the optimized path |
| `flex_attention_backward` | 20.8% | single largest kernel; not tunable (see below) |
| `nvjet_sm120 ... mma` (bf16) | 23.2% | the remaining bf16 GEMMs |
| `Optimizer.step#Muon.step` | 9.9% | |
| `flex_attention` (fwd) | 6.5% | |
| `aten::bmm` / `aten::mm` | 6.8% / 5.6% | NS iteration and Liger's chunked CE |

Phase split (wall-clock between synchronised boundaries, at accum=4):

| phase | ms | % of step |
|---|---:|---:|
| forward | 324.89 | 26.9% |
| backward | 686.07 | 56.9% |
| optimizer + clip + gap | 134.37 | 11.1% |

**This corrects `baseline_profile.md`.** That document measured the optimizer at
"~0 ms, <1% of step" and concluded S14 Opportunity 1 (batched Newton-Schulz)
"will not move the wall-clock needle". The ~0 ms was a measurement artifact — it
was derived as `full step − fwd − bwd − clip` from separately timed runs, and the
note in that document that vanilla's `fwd+bwd` came to 101% of the step is the
tell. Measured directly here, the optimizer is **134 ms/step**.

That said, the conclusion still holds, for a better reason: 134 ms is 11.1% at
accum=4 but only **~2.8% at the production accum=16** (where fwd/bwd is 4x
larger and the optimizer is unchanged), and **the batching is already optimal**.
Verified directly: the 450M model's 100 Muon parameters fall into exactly 4
oriented shape groups with **zero** singleton fallbacks, giving 60 `bmm`/step —
which matches the profiled 300 `bmm` calls over 5 steps exactly. `ns_batched`
already defaults to `True`. The only remaining lever is pre-stacking the momentum
buffers to eliminate the per-step stack/transpose copies, worth well under 1% of
step time and breaking checkpoint compatibility. Not worth it.

## What did not work

### FP8 fast-accum on the backward GEMMs — a no-op on sm_120

`use_fast_accum` (cuBLASLt FAST_ACCUM) is the flag that separated full-rate
from half-rate FP8 on Ada. torchao's recipes already fast-accum the *forward*
GEMM (`Float8LinearConfig` class default — pinned by `tests/test_precision.py`
so a torchao upgrade that changes it fails loudly); the new
`training.fp8_use_fast_accum` flag flips the two backward GEMMs (dgrad/wgrad,
~2/3 of the `_scaled_mm` bucket). Benched 2026-08-15
(`docs/profiling/bench_fast_accum.txt`, 3 repeats, accum 4):

| arm | tok/s | spread | peak GB |
|---|---:|---:|---:|
| perf_control | 55,215 | 0.6% | 19.18 |
| fp8_fast_accum | 55,658 | 0.6% | 19.18 |

**1.008x — inside the drift band.** Consumer Blackwell's default FP8
accumulation is already full-rate; the flag buys nothing here and the quality
screen arm is moot. Re-bench only if a torch/torchao upgrade swaps the FP8
GEMM backend.

**The same bench re-baselined the control at the 400 W power cap.** 55,215
vs the 57,145 documented above (−3.4%): the reference predates the deliberate
600 W → 400 W limit change (€/perf trade, 2026-08). 55,215 at 0.6% spread is
the current-era control; future arms compare against this, not 57,145.
Sustained clock under load at 400 W observed at ~2092 MHz.

### NVFP4 / MXFP4 — dead on this hardware

| format | vs bf16 | rel. error |
|---|---:|---:|
| nvfp4 | 1.02x | 13.9% |
| mxfp4 / mxfp8 | 0.49x | 3.7% |

Consumer Blackwell exposes FP4 tensor cores, but there is no fast sm_120 GEMM
behind them in torch 2.13 / torchao 0.18 — nvfp4 is no faster than bf16 and
mxfp4 is **half the speed**. torchao 0.18 also has no MX/NVFP4 *training* path at
all (`mx_linear` is gone; only `inference_workflow` remains). There is nothing to
build a training recipe on. Revisit only when a sm_120 FP4 GEMM lands.

Incidentally, torchao's NVFP4 path hard-requires **MSLK**
(`meta-pytorch/MSLK`) as its quantization backend. That is the only role MSLK has
here: its CUTLASS kernels (`f8f8bf16_rowwise`, `cutlass_blackwell_fmha`) all fail
to initialize on sm_120 — "Blackwell" in that repo means sm_100a (B200), which has
tcgen05 instructions consumer Blackwell lacks. Its Triton FP8 GEMM does run, but
is slower than PyTorch's native `torch._scaled_mm` at every shape in this model.

### Beating FlexAttention with a hand-written kernel — both precisions lost

Attention is the largest remaining bucket after FP8 (~27.3% of CUDA time:
backward 20.8%, forward 6.5%). Two specialized Triton kernels were written and
verified correct. **Neither is wired in; flex wins.**

`flower/kernels/fp8_swa_attention.py` — **FP8, rejected on numerics.**
Feasibility looked good (FP8 measures 1.88x/1.64x bf16 at attention's inner tile
shapes) and the tuned forward hit 1.31x flex. But e4m3 attention carries **18.4x**
bf16's output error, and per-head scaling changes nothing (18.4x either way),
which identifies the error as **mantissa-limited**: e4m3 has 3 mantissa bits and
attention reduces over only head_dim=64, so noise does not average down the way
it does in the linear layers (K = 1280–3392). No scaling scheme fixes that. No
partial variant helps either — putting even one tensor in e4m3 costs 9.4x.

`flower/kernels/swa_attention.py` — **bf16, rejected on speed.** The
FP8 result prompted the obvious control: how much of the win was *precision* vs
*specialisation*? Answer: all of it was specialisation, and bf16 is both faster
than FP8 here and numerically free.

| kernel | fwd | fwd+bwd | error vs flex |
|---|---:|---:|---:|
| flex (compiled) | 0.580 ms | 2.047 ms | — |
| FP8 specialized | 0.431 ms (1.31x) | not implemented | 18.4x bf16 |
| bf16 specialized, **no LSE** | 0.380 ms (**1.49x**) | not differentiable | 0.0009 |
| bf16 specialized, full fwd+bwd | 0.614 ms (0.94x) | 2.196 ms (**0.93x**) | 0.0005 |

The 1.49x is real but **unusable**, and that is the trap worth recording: it was
measured on a forward with no logsumexp output. Any backward needs
`L = m + log(l)` saved, and adding it takes the raw forward 0.380 → ~0.71 ms. A
forward that cannot back-propagate is not a training speedup. On top of that the
backward — where the prize actually is — does not match flex's
`flex_attention_backward_split_transpose`, an inductor-autotuned split-reduction
template.

Tuning mattered more than anything else and in non-obvious ways: an untuned
launch geometry was **9x slower** than flex; one config shared across the three
kernels gave 0.90x; per-kernel configs gave 0.93x. Anyone revisiting should know
the remaining 7% needs warp specialisation / persistent kernels, not more
parameter search.

Both kernels are kept in-tree with correctness tests
(`tests/test_swa_kernel.py`) as verified reference implementations and as the
evidence closing this direction. **flex is not beatable here by either precision
or specialisation.**

An open question this could not answer: whether FP8-precision attention would
actually cost trained quality. `model.fp8_attention_sim` exists to settle it —
straight-through e4m3 rounding of Q/K/V with flex still computing the attention,
so it needs no FP8 backward. It simulates Q/K/V only, not the softmax
probabilities (an independent 9.4x contributor), so it is a **lower bound** on
the damage. Arm `fp8_attn_sim` in the screen config. Moot unless a future
PyTorch ships a fast FP8 attention backward.

### FlashAttention-4 backend for FlexAttention — unavailable on sm_120

`research/low_precision_kernel_research_2026-08.md` §3.1 calls FlexAttention with
the FA4 (CuTeDSL) backend "**THIS IS YOUR PRODUCTION PATH**", citing 1.2–3.2x over
the Triton implementation. torch 2.13 does expose it:
`kernel_options={"BACKEND": "FLASH"}`, gated on `flash_attn.cute` being importable.

Tested directly (isolated venv with `flash-attn-4==4.0.0b25[cu13]`, so the running
job's env was untouched):

| case | TRITON | FLASH |
|---|---:|---:|
| dense, no mask | 1.068 ms | 1.119 ms |
| causal `block_mask` | 0.913 ms | **`AssertionError: Block sparsity not supported on SM 12.0`** |
| sliding-window `block_mask` (ours) | 0.542 ms | same assertion |

Two independent blockers, both fatal:

1. **Any `block_mask` fails.** Sliding-window attention *is* a block mask, so this
   configuration cannot use FA4 at all. The sm_120 FA4 work is community PR #2634
   and is not in the shipped beta.
2. **Even where FLASH runs it is slower** (1.119 vs 1.068 ms dense), so there is
   no fallback value in restructuring to avoid block masks.

With this, attention is closed on **three independent fronts**: block-size/kernel
tuning is already optimal (1.028x, in noise), hand-written kernels lose (0.93x
best, both FP8 and bf16), and the FA4 backend is unavailable. FlexAttention's
Triton path is the right implementation on this GPU.

### FlexAttention block-size / kernel-option tuning

`scripts/bench_flex_config.py` sweeps 45 (BLOCK_SIZE x kernel_options)
combinations at the exact shape (B2 H20 T8192 D64 window 2048), with a
correctness check on every candidate.

| config | ms (fwd+bwd) | vs default |
|---|---:|---:|
| **default** (BLOCK_SIZE 128, no kernel_options) | **3.69** | 1.000x |
| best correct alternative (BLOCK_SIZE 64, BLOCK_M/N 64, warps 4, stages 2) | 3.59 | 1.028x |
| worst measured | 16.85 | 0.219x |

The best alternative is worth ~0.6% of total kernel time — **inside** the
benchmark's own noise. FlexAttention's defaults are already tuned for this shape.
Most non-default configurations are dramatically worse. Not wired in.

This matters because attention is the largest remaining bucket after FP8
(~32% of kernel time, and the single biggest kernel in the step is
`flex_attention_backward` at 998 ms). It is not improvable by tuning; making it
faster would require an FP8 attention kernel, which flex does not support.

### Larger microbatch

| arm | tok/s | vs batch 2 | peak GB |
|---|---:|---:|---:|
| batch 2 | 55,281 | 1.000x | 19.18 |
| batch 3 | 53,975 | 0.976x | 25.98 |
| batch 4 | OOM (needs >30.25 GB) | — | — |

`baseline_profile.md` found vanilla "launch/dispatch bound, GPU starved between
kernels" and suggested wider batches would raise utilization. **That reading was
taken in bf16 and does not survive FP8**: once the GEMMs shorten, the step is no
longer launch-bound, so batch 3 is *slower* and costs 6.8 GB. Do not re-try.

### Data loader worker count

Not a speedup, but the plumbing bug behind the hypothesis was real and is fixed:
`token_batches` accepted `num_workers`/`prefetch_factor` and then dropped them on
the training path, so training always ran at `_fineweb_loader`'s 2-worker default
no matter what was configured. Now plumbed through with `data.num_workers` /
`data.prefetch_factor` and a regression test. Raising it changes data *order*
(each worker shards documents by `islice(worker_id, None, num_workers)`), so runs
at different worker counts are seed-comparable, not bit-comparable.

### FP8 weight-quantization caching — built, measured, net loss

The profile shows FP8's own quantization overhead is ~4.6% of CUDA time
(137 ms quantize + 132 ms amax per step). With `gradient_accumulation_steps: 16`
every `Float8Linear` re-computes its weight's amax and re-casts it **16 times per
optimizer step** though the weight only changes once — obviously wasteful.

Implemented as `flower.precision.fp8_weight_cache` (a context manager wrapping
the accumulation loop) and measured compiled, at d_model 1280 / hidden 3392:

| activations per microstep | no cache | cached | result |
|---|---:|---:|---:|
| N = 4096 (unrepresentative) | 227.4 ms | 193.7 ms | **1.174x** |
| **N = 16384 (the real shape)** | 374.0 ms | 384.2 ms | **0.973x** |

The real config is batch 2 x seq 8192 = 16384 rows per microstep. At that ratio
activation-side quantization so dominates the weight side that the saving
vanishes, while the cached path's overhead (holding the cast weight live, plus a
second compiled graph inductor fuses less well) costs ~2.7%. **Not wired in.**

The N=4096 row is kept as a warning: identical code, +14.8%, and shipping on it
would have been a regression. This optimization's value is entirely a function of
the weight:activation ratio, so a benchmark at the wrong batch/sequence size
inverts the answer.

**A trap worth recording separately.** The natural implementation is to cache and
invalidate when the weight changes, via `tensor._version`. That is broken here:

    p.data -= x    ->  _version UNCHANGED
    p.add_(x)      ->  _version incremented

`flower/optim.py`'s Muon uses *both* — `p.data -=` for cautious weight decay and
`p.add_` for the main update. A version-based cache would therefore serve stale
quantized weights after some updates and not others, with no error, surfacing
only as a mysterious quality regression. Hence the explicit context manager plus
a sampled-fingerprint check that raises on exit if any weight moved while the
cache was live. `tests/test_fp8_weight_cache.py` pins bit-exactness, the
fail-loud behaviour, and the `_version` fact itself.

torchao 0.18 offers no delayed scaling (only `dynamic`/`disabled`), so the amax
pass cannot be amortized the supported way either.

## How to enable the stack

Everything is off by default so published runs reproduce bit-for-bit. To opt in,
add to a config's `training:` block (and `model:` for the fused CE):

```yaml
model:
  fused_linear_ce: true          # -1.1 GB; requires the dynamo fix in models/base.py
training:
  precision: bf16                # required — FP8 layers consume bf16 activations
  compile_model: true            # required
  compile_mode: max-autotune-no-cudagraphs
  fp8_linear: true
  fp8_recipe: tensorwise         # rowwise measures 1.01x here; do not bother
  fp8_keep_bf16_blocks: 1        # measured better on both quality AND speed
  # smooth_swiglu: deliberately NOT set — measured to cost ~3% and help nothing
```

Do **not** set `compile_mode: reduce-overhead` (CUDA graph pools do not fit at
450M even with FP8's saving) and do **not** raise `batch_size` (batch 3 is slower
and costs 6.8 GB; batch 4 OOMs).

`training.fp8_linear` is validated at startup: it raises if the device is not
CUDA or `precision` is not `bf16`, rather than silently running bf16 under an
FP8-labelled config. Every run records `fp8_recipe`,`fp8_keep_bf16_blocks` and
`fp8_converted_linears` into its `metrics.json`, so a run's precision layout is
recoverable from its own artifacts.

## Tooling added

| script | purpose |
|---|---|
| `scripts/bench_arms.py` | A/B any set of sweep variants through the real train.py wiring; one subprocess per arm so peak-memory readings are independent |
| `scripts/bench_flex_config.py` | Sweep FlexAttention block sizes / kernel options with correctness checks |
| `scripts/bench_data_workers.py` | Loader-only token throughput vs worker count |
| `configs/speedup_screen_450m.yaml` | The screening sweep; every arm above is reproducible from it |

`flower/precision.py::maybe_convert_fp8` is the single FP8 entry point, used by
both `flower/train.py` and `scripts/profile_step.py`, so a benchmark cannot
silently measure a different precision layout than a real run.

## The 10k confirmation run: MISSED its pre-registered band — decision open

`configs/sweep13_450m_longctx_fp8.yaml` pre-registered the success criterion
before launching: **val_bpb within 0.0004 of the finished bf16 seeds**
(0.90234 / 0.90270). The run completed 2026-08-16
(`runs/sweep13_450m_longctx_fp8/`, `runs_450m_fp8_confirm.log`):

| quantity | value |
|---|---:|
| val_bpb (FP8 stack, seed 0, 10k steps, EMA) | **0.90428** |
| offset vs seed 0 / seed 1 | +0.00194 / +0.00158 |
| pre-registered band | 0.0004 |
| throughput (whole-run) | 58,102 tok/s (1.30x vs 44,810) |
| peak VRAM | 21.0 GB |

**By its own criterion the run FAILED**: the converged offset is 4–5x the
band. The 600-step screen's conclusion ("offset an order of magnitude under
reseed noise, no divergence") remains true — this is an offset, not
divergence — but the converged quality cost is real and larger than the band
the config committed to.

Caveats on the band itself: it is measured from **n=2 seeds** of the bf16
config; the true 10k-step seed distribution is unknown, and 0.0016 vs 0.0004
could partly be band noise. A second FP8 seed (or a third bf16 seed) would
bound that — the existing seed-1 screen infrastructure
(`runs/speedup_screen_450m_seed1/`) shows how.

**RESOLVED by re-measurement (2026-08-18, option 3): the offset is REAL.**

A second FP8 seed (10k steps, same config, `runs/sweep13_450m_longctx_fp8_seed1/`,
59,211 tok/s) settled the n=2 band question:

| arm | seed 0 | seed 1 | band | mean |
|---|---:|---:|---:|---:|
| bf16 | 0.90234 | 0.90270 | 0.00036 | 0.90252 |
| FP8 stack | 0.90428 | 0.90414 | **0.00014** | 0.90421 |

The FP8 seeds agree with each other to 0.00014 — **12x tighter than the
offset itself** — and the closest FP8-to-bf16 gap is +0.00144 with zero
overlap between the two n=2 distributions. The pre-registered criterion
(within 0.0004) fails decisively: this is not band noise. The converged
quality cost of the FP8 stack is real and stable at **≈ +0.0017 bpb**
(mean-to-mean +0.00169) in exchange for **1.30x throughput and −5 GB VRAM**.

The remaining choice (accept that trade for exploration runs and keep bf16
for final numbers, or reject FP8 for research comparisons outright) is a
policy call, not a measurement call — the measurement above is complete.
