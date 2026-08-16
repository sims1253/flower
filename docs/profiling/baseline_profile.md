# Baseline training-step profile — 450M long-context bake-off

**Config:** `configs/sweep13_450m_longctx_memory.yaml` — the current scale-up target.
**Hardware:** RTX 5090 (32 GB), WSL2, torch 2.13.0+cu130.
**Variants:** the param-matched bake-off pair:
  - `vanilla_matched` — `vanilla_local`, d1280 / L20 / 20 heads = **413M** params (the control)
  - `bloom_memory` — `bloom_memory`, d1024 / L20 / 16 heads = **481M** params (+17% from the memory bank)

**Shape:** seq=8192, local_window=2048, batch=2, **gradient_accumulation_steps=16**
→ effective batch **262,144 tokens/step** (16 microsteps of fwd/bwd + 1 optimizer step).
bf16 autocast + FlexAttention + `torch.compile(mode="default")` + Muon, exactly as `flower/train.py` runs them.

This is a **measurement** writeup, not an optimisation. It replaces the S14 kernel-size
guesses ("Muon NS is 5–10% of step time", "bloom routing is 0.5%") with measured numbers
at the real shape, and ranks where step time actually goes so the other S14 prompts can be
ordered by ROI.

## How to reproduce

```bash
# Both arms, 20 warmup steps (compile + cudagraph capture), then 10 profiled steps.
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/profile_step.py --warmup 20 --profile-steps 10
# Re-derive the tables below from the exported traces:
uv run python scripts/analyze_profile.py
```

Outputs (in `docs/profiling/traces/`): `{variant}.kernel.trace.json` (Chrome/Perfetto
traces, GPU-kernel timeline), `summary.json` (wall-clock, launch counts, peak memory).

### Methodology / what the numbers mean

- **Step = 16 microsteps + 1 optimiser step.** The earlier "optimiser dominates 58 %"
  finding (NEXT_IDEAS.md §5) was measured at `accum=1`; at the real `accum=16` the
  fwd/bwd work is 16× larger while the optimiser step is unchanged, so that claim must
  (and below, does not) hold. Every percentage here is **of the full accumulated step**.
- **Wall-clock is the budget.** Phase splits (fwd / bwd / opt) use wall-clock between
  `torch.cuda.synchronize()`d boundaries — unambiguous, includes launch + dispatch overhead.
- **Kernel-time % is relative compute weight, not wall share.** Under `torch.compile` +
  cudagraph, `torch.profiler`'s `key_averages()` double-counts (the `## Call CompiledFxGraph`
  wrapper reports the sum of its child kernels as its own self-time). The category table
  below is therefore computed from the **raw trace kernel events**, which do not
  double-count. Kernel time sums to *more* than wall (5.7 s of kernels in a 5.0 s step for
  vanilla) because 16 microsteps' kernels queue/overlap; the **percentages** are the
  reliable signal for "which op class dominates compute," anchored to the wall-clock phases.
- Synthetic tokens stand in for fineweb_edu so no dataset download is needed; kernel time
  depends on shape/dtype, not token content.

## Headline numbers

| variant | params | ms/step | tok/s | peak GB | kernels/step |
|---|---:|---:|---:|---:|---:|
| `vanilla_matched` | 413M | **5021** | 52,207 | 25.06 | 7,000 |
| `bloom_memory`   | 481M | **5027** | 52,152 | 28.60 | 19,114 |

The two arms land at **the same wall-clock** despite bloom doing **2.1× the kernel work**
(11.8 s vs 5.7 s of GPU-kernel time per step). That means vanilla is more
launch/dispatch-bound (GPU starved between kernels) while bloom keeps the GPU busier — the
memory machinery raises GPU utilisation, not wall-time, at this batch size. Bloom's cost
shows up as **+3.5 GB peak** (89 % vs 78 % of the 32 GB card) and **2.7× the launch count**.

## Where the time goes — category breakdown

CUDA kernel time per profiled accum=16 step, from trace kernel events (`scripts/analyze_profile.py`).
Percentages are of that variant's total kernel time (the relative-compute-weight signal).

| category | vanilla | % | bloom | % |
|---|---:|---:|---:|---:|
| matmul: cutlass GEMM (FFN / projections / opt NS) | 3509 ms | **62.1%** | 6291 ms | **53.5%** |
| attention: flex (local self-attn, window=2048)    | 1299 ms | **23.0%** | 3487 ms | **29.6%** |
| norm (RMSNorm, fused triton reduction)            |  117 ms |   2.1% |  774 ms |   6.6% |
| attention: flash (bloom perceiver cross-attn)     |    0 ms |     —  |  633 ms |   5.4% |
| elementwise / copy / fused-misc                   |  495 ms |   8.8% |  385 ms |   3.3% |
| activation (swiglu)                               |  196 ms |   3.5% |  146 ms |   1.2% |
| softmax / cross-entropy                           |   19 ms |   0.3% |   21 ms |   0.2% |
| optimizer: Newton-Schulz / AdamW (eager, see §S14-5a) | <1 ms | <0.1% | <1 ms | <0.1% |
| embedding / gather                                |    4 ms |   0.1% |    0 ms |     —  |
| **total kernel time**                             | **5653 ms** | | **11,763 ms** | |
| wall-clock step (ms)                              | 5021 | | 5027 | |

**Matmul (cutlass GEMM) is the dominant compute class in both arms (54–62%).** These are
the FFN swiglu up/gate/down, the Q/K/V/O projections, and the Muon Newton-Schulz `mm`s —
the profiler cannot separate them by name (they share one kernel), which is exactly why the
phase timing below is the cleaner signal. Flex (local window) attention is the #2 class
(23–30%); bloom additionally spends **5.4% in flash cross-attention** (the perceiver
summary path) and **6.6% in norm** (its extra LayerNorms).

## Phase split (wall-clock, synchronised) — the clean fwd/bwd/opt view

Measured directly as wall-clock between `cuda.synchronize()`d phase boundaries, at accum=16.
The optimiser row is the S14-5a target (Muon NS + AdamW + grad-clip + inter-phase dispatch).

| phase | vanilla | % of step | bloom | % of step |
|---|---:|---:|---:|---:|
| forward (16 microsteps) | 1612 ms | 32% | 1473 ms | 29% |
| backward (16 microsteps)| 3457 ms | 69% | 3231 ms | 65% |
| **fwd+bwd total** | 5069 ms | (101%)* | 4704 ms | 94% |
| **optimizer step** (Muon NS + AdamW) | **~0 ms** | **<1%** | **119 ms** | **2.4%** |

\* vanilla's fwd+bwd slightly exceeds the full step because the two are measured in separate
timed runs (noise); the optimiser is in the noise for vanilla. The optimizer figure is from
a direct diff measurement (full step − fwd+bwd+clip), see `scripts/profile_step.py`.

**Backward dominates forward ~2:1** in both arms — expected for a transformer (bwd stores +
recompute are heavier than fwd). The optimizer is a **rounding error at accum=16** (≤2.4%).

## Top CUDA kernels by self time

Per profiled step (accum=16), from the trace. `triton_tem_fused__...flex_attention...` is
the FlexAttention forward/backward/score kernel; `cutlass_80_tensorop_bf16_s16816gemm` is
the cuBLAS/cutlass BF16 GEMM used by every Linear and the NS iteration.

**vanilla_matched** (top 10):

| ms/step | calls | kernel |
|---:|---:|---|
| 998 | 3200 | `triton_tem_fused_..._flex_attention_backward_split_transpose` |
| 852 | 12960 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` (FFN/proj GEMM) |
| 816 | 9760 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 551 | 6560 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 362 | 6400 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 288 | 3200 | `triton_tem_fused_..._flex_attention_split_transpose` |
| 277 | 3200 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 247 | 3200 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 242 | 3200 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 126 | 3200 | `triton_poi_fused_..._mul_silu_silu_backward` (swiglu) |

**bloom_memory** (top 10):

| ms/step | calls | kernel |
|---:|---:|---|
| 2571 | 3200 | `triton_tem_fused_..._flex_attention_backward_split_transpose` |
| 1494 | 9600 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 1474 | 25280 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 1035 | 15680 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 963 | 9600 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 911 | 3200 | `triton_tem_fused_..._flex_attention_split_transpose` |
| 520 | 6240 | `pytorch_flash::flash_bwd_dq_dk_dv_loop` **(bloom perceiver cross-attn)** |
| 417 | 1440 | `triton_per_fused_..._layer_norm...` (bloom extra norms) |
| 339 | 9600 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |
| 290 | 3200 | `cutlass_80_tensorop_bf16_s16816gemm_relu...` |

The bloom-specific `pytorch_flash::flash_bwd_dq_dk_dv_loop` (520 ms/step, 4.4% of bloom's
kernel time) is the perceiver cross-attention backward — the single largest bloom-only cost.

## Kernel launch counts

| variant | total/step | forward | backward | optimizer |
|---|---:|---:|---:|---:|
| `vanilla_matched` | 7,000 | 1,636 | 2,236 | 3,128 |
| `bloom_memory`    | 19,114 | 4,178 | 8,155 | 6,781 |

Per microstep (÷16): vanilla fwd ≈ 102, bwd ≈ 140; bloom fwd ≈ 261, bwd ≈ 510.
**Bloom launches 2.7× more kernels** — the memory read/write/perceiver path is
launch-heavy. The optimizer (eager Muon + AdamW) contributes 3,128–6,781 launches/step
despite being ≤2.4% of wall-time: it is **launch-bound, not compute-bound**, which is the
premise of the batched-NS work (S14 Opportunity 1/5a) — but at accum=16 the absolute win is
small because the optimizer is already a rounding error in the step.

## Idle gaps / host-side overhead

Between backward and the optimizer step there is **no long idle gap**: the wall-clock
fwd+bwd accounts for 94–101% of the step, leaving ≤6% for the optimizer + all inter-phase
dispatch combined. There is no host-side stall serialising the pipeline at this config — the
step is GPU-throughput-limited, not launch-latency-limited at the step level (though
*within* the optimizer the per-param Python dispatch is still launch-bound, see S14-5a).
Between steps the only overhead is the data iterator (synthetic here, so ~0).

## Direct answers to the S14 questions

- **S14-5a — Muon Newton-Schulz fraction of step time.** The "5–10% of total step time"
  claim is **refuted at the real config**. At accum=16 the optimizer is **~0% of the vanilla
  step and 2.4% (119 ms) of the bloom step.** The claim only held at accum=1, where fwd/bwd
  was 1/16th the work. The batched-NS work (S14 Opportunity 1/5a) targets a ≤2.4% cost —
  worth doing for *launch-count* hygiene (3k–7k launches/step) but it will not move the
  wall-clock needle at this shape.

- **S14-5b — lm_head + cross-entropy fraction.** softmax/CE is **0.2–0.3% of kernel time**
  (19–21 ms/step). At seq=8192 / vocab=16K the (B·T, vocab) logits tensor is large but the
  matmul + CE is not on the hot path. The Liger fused-linear-CE win here is **memory**
  (not materialising the logits tensor) more than time — relevant for reaching seq=32K, not
  for speeding up this config.

- **S14 Opportunity 2 — bloom routing fraction.** Bloom routing (the K-hash matmul + softmax
  + perceiver cross-attention) is, combined, the **flash cross-attn (5.4%) + part of the
  matmul bucket**. The dedicated perceiver `flash_bwd` kernel is 520 ms/step (4.4% of
  bloom's kernel time); the hash/route/softmax themselves are negligible (<0.5%, consistent
  with the earlier small-config finding). The bloom overhead is real (~2× the kernel work of
  vanilla) but does **not** raise wall-clock at this batch size — it raises peak VRAM
  (+3.5 GB) and launch count (2.7×). Further bloom-path kernel work is low-ROI for
  wall-clock.

- **Kernel launches/step.** 7,000 (vanilla) / 19,114 (bloom). The optimizer contributes
  3,128 / 6,781 of those despite being ≤2.4% of wall-time — the highest
  launches-per-wall-second ratio, confirming the optimizer is dispatch-bound. Forward is
  1,636 / 4,178; backward 2,236 / 8,155.

- **Idle gap between backward and optimizer / between steps.** None significant: fwd+bwd is
  94–101% of the step; ≤6% covers the optimizer + grad-clip + all dispatch. No host-side
  stall to fix.

## What to optimise first (3 bullets, grounded in the numbers)

- **The matmul bucket (54–62% of compute) is where any wall-clock win lives** — and it is
  already running optimal cutlass BF16 GEMMs. The lever is *precision* (FP8 GEMMs on the FFN
  / projections, the S13 `ffn_precision`/`attn_precision` scaffolding), not a kernel
  rewrite. A 2× matmul speedup at 58% of compute is a ~30% step speedup; this is the
  highest-ROI prompt on the list (S13 precision routing / FP8).

- **FlexAttention backward is the single largest kernel in both arms (998 ms vanilla,
  2571 ms bloom per step)** and is 23–30% of compute. It is already a fused triton kernel,
  so the lever is *reducing the work* (smaller effective window, or the attention-window
  schedule S2) rather than a faster kernel. Worth measuring whether window=2048 is necessary
  vs a smaller ratio before chasing FP8 attention.

- **Do NOT spend more time on the optimizer (S14-5a) or on bloom routing micro-kernels
  (S14 Opportunity 2) for wall-clock.** Both are ≤2.4% of the step at accum=16. The
  optimizer work remains justified for *launch-count* reduction (3k–7k launches/step is real
  dispatch overhead inside that 2.4%), and fused-linear-CE (S14-5b) remains justified for
  *memory* (seq=32K enabling), but neither is a wall-clock lever at this shape. Rank the
  precision-routing and attention-window prompts above them.
