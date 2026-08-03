# NVIDIA Nemotron Training Techniques — Generalizability Analysis

**Purpose**: Audit NVIDIA's Nemotron training pipeline (all major versions) for kernel-level and systems-level optimizations applicable to Flower (350-600M dense model, single 5090 or 8×H100/B200 node). Cross-reference against the existing `training-speedups.md` to identify what's NEW vs already covered.

**Date**: Aug 2026
**Sources**: Nemotron-4 340B report (arXiv:2406.11704), Nemotron-H 8B/56B/47B (arXiv:2504.03624), Nemotron 3 Ultra 550B (tech report 2026), Llama-Nemotron Ultra 253B (arXiv:2505.00949), Transformer Engine docs/code, NeMo-Megatron configs.

---

## TL;DR

**Nemotron does NOT use custom handwritten CUDA kernels for training throughput.** NVIDIA's entire training-efficiency stack is **Transformer Engine (TE)** — a library of fused, FP8/FP4-aware PyTorch modules (`te.Linear`, `te.TransformerLayer`) backed by optimized cuBLAS/cuDNN GEMMs. The generalizable takeaway for Flower is: **adopt Transformer Engine as a library**, not "write custom kernels." TE is a drop-in for `nn.Linear`, works with plain PyTorch (not Megatron-only), and runs on the RTX 5090.

The single biggest GAP in Flower's current `training-speedups.md`: **it hand-rolls FP8 via `torch._scaled_mm` (Section 3) but never mentions Transformer Engine**, which handles FP8 scaling-factor bookkeeping, amax history, fused kernels, and FP4/MXFP8 automatically. TE would replace Sections 3, 4, and much of 10 with a cleaner, NVIDIA-maintained implementation.

The most novel *precision* finding from Nemotron (not in Flower's doc): **NVFP4 training with Random Hadamard Transforms + stochastic gradient rounding** (Nemotron 3 Ultra). This is the cutting edge for FP4 and is directly relevant to Flower's FP4 plans.

---

## 1. Nemotron Version Map

| Version | Date | Size | Precision trained in | Key technique |
|---|---|---|---|---|
| Nemotron-4 340B | Jun 2024 | 340B dense | BF16 | TP+PP+DP, distributed optimizer, ~42% MFU |
| Nemotron-4 15B | Feb 2024 | 15B dense | BF16 | Same infra, smaller scale |
| Nemotron-H 8B/56B | Apr 2025 | 8B/56B hybrid Mamba | **FP8 per-tensor current scaling** | First major FP8 pretraining recipe |
| Llama-Nemotron Ultra 253B | May 2025 | 253B (NAS from Llama 405B) | BF16 train, FP8 generation | FFN Fusion, NAS compression |
| **Nemotron 3 Ultra 550B-A55B** | 2026 | 550B MoE hybrid | **NVFP4** | FP4 pretraining at 20T-token scale |

The "most recent" Nemotron is **Nemotron 3 Ultra** (NVFP4, 2026). The most relevant to Flower's scale is **Nemotron-H 8B** (FP8 recipe validated at 8B).

---

## 2. Kernel-Level Optimizations

### 2.1 Transformer Engine (THE key finding) — GENERALIZABLE

NVIDIA does not write custom training kernels per-model. Everything flows through **Transformer Engine** (`pip install transformer_engine[pytorch]`):

- **`te.Linear`** — drop-in replacement for `nn.Linear`. Internally manages FP8 scaling factors, amax history, and dispatches to cuBLAS FP8 GEMMs (E4M3 fprop, E5M2 backward). No manual `torch._scaled_mm` or scale-factor bookkeeping needed.
- **`te.TransformerLayer`** — fully fused layer (LN + attention + residual + LN + FFN + residual). Uses cuDNN/FlashAttention fused attention backend.
- **`te.LayerNorm` / `te.RMSNorm`** — fused norm kernels.
- **FP8 autocast API**: `with te.autocast(recipe=DelayedScaling(...)): out = model(x)`. Handles forward+backward quantization, amax sync across GPUs, recipe management.
- **Fused operations** baked into TE modules: bias+activation fusion, dropout+residual fusion, fused softmax (for attention masks). Equivalent to Megatron's `bias_activation_fusion`, `masked_softmax_fusion` flags but maintained and portable.
- **RoPE fusion**: TE applies RoPE inside fused kernels where applicable.

**Works with plain PyTorch** — confirmed via TE quickstart and `te.Linear` API docs. No Megatron dependency. Integrates with FSDP. Integrates with `torch.compile`.

**Blackwell consumer (RTX 5090, SM120) support**: TE officially supports Blackwell including MXFP8 and NVFP4. ⚠️ **SM120 caveat**: SM120 is architecturally distinct from SM100 (B200) — no TMEM, no UMMA/tcgen05, uses SM80-era `mma.sync` instructions, 99KB shared memory vs 228KB. TE dispatches correctly, but some bleeding-edge community kernels (DeepGEMM, trtllm-gen FlashAttention) are SM100-only. TE's cuBLAS-backed path is the safe route on the 5090. Reference: `lna-lab/blackwell-geforce-nvfp4-gemm`, CUTLASS example 87.

**Community validation on 5090**: nanochat FP8/NVFP4 work (github.com/karpathy/nanochat discussions #382) reports **20-30% speedup with FP8 via TE on Blackwell consumer GPUs**, +20% more with NVFP4 on some configs (slower on others). Numerical accuracy maintained. Matches Flower's target scale (~560M params, FineWeb). This is the strongest direct evidence that TE benefits Flower's exact hardware/model-size.

### 2.2 NVFP4 GEMM kernels (Nemotron 3 Ultra) — GENERALIZABLE (Blackwell)

Nemotron 3 Ultra's FP4 training uses **Transformer Engine's cuBLAS NVFP4 GEMM kernels** for fprop, dgrad, and wgrad. These are not custom Nemotron kernels — they ship in TE. Key recipe details:

- **E2M1 datatype** (4-bit float, 2-bit exponent, 1-bit mantissa)
- **2D block quantization on weights** (per-block scale factors)
- **Random Hadamard Transform (RHT) on inputs to wgrad** — reduces outlier activation magnitudes before the FP4 matmul, dramatically improving stability. This is the FP4 analog of Smooth-SwiGLU (Flower Section 10). It's implemented inside TE's NVFP4 path, not as a separate op.
- **Stochastic rounding on gradients** — FP4 has so few bits that nearest-neighbor rounding of gradients introduces systematic bias. Stochastic rounding removes it. TE handles this internally.
- **Recipe**: `NVFP4BlockScaling()` in `transformer_engine.common.recipe`.

**Flower relevance**: This is the production-grade path for Flower's FP4 plans (Section 13 precision routing). TE's `NVFP4BlockScaling` + RHT + stochastic rounding replaces the hand-rolled Smooth-SwiGLU approach with a battle-tested implementation.

### 2.3 No custom FlashAttention variant

Nemotron uses TE's `DotProductAttention`, which dispatches to FlashAttention (for older GPUs) or cuDNN fused attention (Hopper+) or trtllm-gen attention (B200). No proprietary attention kernel. Flower's FlexAttention plan (Section 1) is the right call — it's more flexible than TE's attention for Flower's custom masks (RBF bias, memory mechanisms).

### 2.4 Verdict on custom CUDA kernels for Flower

**Not worth it for standard ops.** NVIDIA — with infinite kernel-engineering resources — routes training through TE/cuBLAS, not custom kernels. The user's Starling custom-kernel experience is valuable for **Flower-specific ops** (memory write/read paths, custom attention biases) but not for matmuls, layer norms, or attention. The ROI ranking:
1. Use TE for all standard Linear/FFN/norm ops (free, maintained, FP8/FP4-ready).
2. Keep FlexAttention for the attention layer (Flower's masks are non-standard).
3. Custom CUDA only for Flower-unique memory-mechanism ops if profiling shows they're bottlenecks.

---

## 3. Systems-Level Training Optimizations

### 3.1 Distributed optimizer (ZeRO-style state sharding) — ALREADY COVERED

Nemotron-4 340B uses `distributed_fused_adam` / `mcore_distributed_optimizer`: shards optimizer state across DP replicas (ZeRO-1/2). Flower's Section 7 covers this via FSDP. ✅ No new info.

Nemotron config also sets `overlap_grad_sync: True`, `overlap_param_sync: true`, `contiguous_grad_buffer: True`, `grad_div_ar_fusion: true` — all of which FSDP provides automatically. ✅ Covered.

### 3.2 Asynchronous checkpointing — PARTIALLY NEW

Nemotron-H uses **NVRx** (`nvidia-resiliency-ext`) for **non-blocking checkpoint saves** — training continues while checkpoints are written to storage in a background stream. This matters for long training runs where blocking checkpoint saves waste GPU-minutes.

**Generalizable**: PyTorch 2.3+ has `torch.distributed.checkpoint` with **async save** support (`torch.distributed.checkpoint.state_dict_saver.async_save`). This is the open equivalent. For Flower's multi-day 5090 runs or rented 8×GPU runs, async checkpointing avoids the ~30-60s stall per checkpoint. Worth adding to Flower's training loop but low priority unless checkpointing frequency is high.

### 3.3 Sequence parallelism — NEW (relevant for Flower's 32K target)

Nemotron-H uses **8-way tensor parallelism + sequence parallelism**. Sequence parallelism (Korthikanti et al., 2022) splits the activation tensor along the **sequence dimension** (not the feature/batch dimension) during the LayerNorm/dropout regions, then gathers before the matmuls. This reduces activation memory by the TP degree.

**Flower relevance**: At seq=32K (Flower's target per Section 13), activation memory dominates. Sequence parallelism is relevant on the 8×GPU node. However, it requires tensor parallelism to be meaningful (it's coupled to TP). At 600M params, the model fits on a single GPU, so TP is unnecessary — and thus sequence parallelism adds complexity without benefit unless Flower goes to much longer contexts. **Low priority for Flower's current scale.**

Alternative that's simpler and already in Flower's plan: **context parallelism / ring attention** for very long sequences, or just FlexAttention's memory-efficient kernel (Section 1) which avoids materializing the full mask.

### 3.4 Pipeline parallelism with interleaving — NOT GENERALIZABLE

Nemotron-4 uses 12-way interleaved pipeline parallelism (Megatron's VP). Only relevant when the model doesn't fit on one GPU. At 600M params, irrelevant. ❌ Datacenter-scale only.

### 3.5 Resiliency / failure attribution — NOT GENERALIZABLE

Nemotron-H's DGX Cloud Resilience service (3.3× MTBF improvement) is proprietary infrastructure for 6144-GPU clusters. ❌ Not applicable to single-node training.

### 3.6 Gradient accumulation fusion — MEGATRON-SPECIFIC

`gradient_accumulation_fusion`, `gradient_as_bucket_view` — Megatron internal optimizations. `torch.compile` + FSDP achieve equivalent fusion. ✅ Covered conceptually.

---

## 4. Precision / Mixed-Precision Strategies (RICHEST FINDING)

### 4.1 FP8 per-tensor current scaling (Nemotron-H) — NEW, HIGHLY RELEVANT

Nemotron-H-56B is the **first major NVIDIA model fully pretrained in FP8**. The recipe:

1. **Hybrid format**: E4M3 for weights + activations (forward), E5M2 for gradients (backward). This is standard and what Flower's Section 3 implies.
2. **Per-tensor current scaling**: one FP32 scale factor per tensor, computed as `max_representable / max_abs(tensor)`. Values too small to fit are flushed to zero. This is **simpler** than delayed scaling (no amax history needed) — it's "current" (just-in-time per step).
3. **Both forward AND backward passes quantized** for all linear layers.
4. **CRITICAL — keep first 4 and last 4 GEMMs in BF16.** This is the stability insight. The first/last layers are most sensitive to quantization noise. Flower's Section 13 already routes LM head to BF16, but this suggests also keeping the first few transformer blocks in BF16.
5. **Results**: <0.1% relative loss gap vs BF16. Downstream accuracy **equal or better** than BF16 on code/math tasks. FP8 never required overtraining.

**⚠️ Scale caveat from Nemotron-H**: *"We found it very important to do verification on a minimum of 8B parameters when constructing our FP8 recipe, as results with smaller models did not generalize."* — This is a direct warning for Flower: at 350-600M params, FP8 training is **riskier** than at 8B+. The recipe should be validated carefully, and BF16 may remain the safer default at Flower's scale. FP8 for the LM head only (Section 3) is lower-risk than full-model FP8.

This maps to TE's `Float8CurrentScaling` recipe — `te.autocast(recipe=Float8CurrentScaling(...))`.

### 4.2 Delayed scaling (Nemotron-4 340B FP8 inference, NeMo default) — COVERED CONCEPTUALLY

The Nemotron-4 NeMo config uses `fp8_hybrid: True`, `fp8_amax_history_len: 1024`, `fp8_amax_compute_algo: max` — this is **delayed scaling** (TE's `DelayedScaling` recipe). It stores 1024 steps of per-tensor max-abs values and uses their max as the scale factor. More conservative than current scaling; the historical production default.

Flower's Section 3 hand-implements a per-tensor scale (current-scaling-like). TE's `DelayedScaling` or `Float8CurrentScaling` would replace this.

### 4.3 MXFP8 (Blackwell) — NEW

Blackwell adds **MXFP8** (Microscaling FP8): per-32-value block scaling (vs per-tensor). All values use E4M3 (no need for E5M2's range because block scaling handles dynamic range locally). Scale factors stored as 8-bit E8M0 (power of 2). This is more precise than per-tensor FP8 and avoids the "flush small values to zero" problem.

TE recipe: `MXFP8BlockScaling(fp_format=Format.E4M3)`.

**Flower relevance**: On the 5090 (Blackwell), MXFP8 is available and may give better accuracy than per-tensor FP8 at Flower's small scale. Worth testing. Only on Blackwell (H100 does not support MXFP8).

### 4.4 NVFP4 pretraining (Nemotron 3 Ultra) — NEW, CUTTING EDGE

The latest. Full FP4 training at 20T tokens. Recipe (from TE + Nemotron 3 Ultra report):

- `NVFP4BlockScaling()` recipe in TE
- E2M1 weights with 2D block quantization
- **Random Hadamard Transform on wgrad inputs** (built into TE's NVFP4 path)
- **Stochastic rounding on gradients** (built into TE)
- **Keep in higher precision**: final 15% of layers (16 of ~118), all attention projections (QKV/output), Mamba output projections, embedding, MTP layers.
- Result: relative loss gap <0.4% vs BF16 (decreasing to 0.03% near end of training).

**Stability lesson (important)**: Nemotron 3 Ultra hit a divergence at ~8T tokens when they tried to reduce the output-layer gradient accumulation precision from FP32 to BF16 (to save communication bandwidth). The MTP loss contribution was lost in BF16's 7 mantissa bits. **Fix: keep FP32 gradient reductions for the output/shared layers even under FP4/FP8 elsewhere.** This directly validates Flower's Section 13 plan to keep the LM head and memory path in higher precision.

**Flower relevance**: This is the production recipe for Flower's FP4 ambitions (Section 10/13). Use TE's `NVFP4BlockScaling` rather than hand-rolling. On the 5090, TE's NVFP4 path should work (SM120 supported, though verify with a smoke test).

### 4.5 Per-layer precision routing — CONFIRMS Flower's Section 13

Nemotron's approach across all versions validates Flower's Section 13 precision-routing philosophy: **precision is per-component, not global.** NVIDIA's consistent pattern:
- First/last layers → BF16 (Nemotron-H: first/last 4; Nemotron 3 Ultra: last 15%)
- Attention projections → higher precision (BF16 or FP8, never FP4)
- FFN → lowest precision (FP8 or FP4)
- Embeddings/output → BF16
- Gradients for output layer → FP32 (hard-won lesson from Nemotron 3 Ultra divergence)

Flower's Section 13 table matches this. ✅ Validated.

---

## 5. What's NEW vs Flower's `training-speedups.md`

| Technique | In Flower doc? | Nemotron source | Recommendation |
|---|---|---|---|
| **Transformer Engine as FP8/FP4 library** | ❌ NO (doc hand-rolls `torch._scaled_mm`) | All Nemotron versions | **ADD**. Replace Sections 3/4/10's manual FP8 with `te.Linear` + autocast. Cleaner, maintained, FP4-ready. |
| **FP8 current scaling (per-tensor, no amax history)** | ❌ NO | Nemotron-H 56B | **ADD** as a precision option. TE `Float8CurrentScaling`. Simpler than delayed scaling. |
| **MXFP8 (block-scaled FP8)** | ❌ NO | Blackwell / TE | **ADD**. On 5090 only. Better accuracy than per-tensor FP8 at small scale. |
| **NVFP4 training recipe (RHT + stochastic rounding)** | ❌ Partially (Section 10 Smooth-SwiGLU is a different approach) | Nemotron 3 Ultra | **ADD**. TE `NVFP4BlockScaling` is the production FP4 path. Supersedes hand-rolled Smooth-SwiGLU. |
| **Keep first N + last N layers in BF16 under FP8** | ❌ Partially (Section 13 keeps head BF16) | Nemotron-H, Nemotron 3 Ultra | **ADD**. Extend Section 13: also keep first 4 blocks BF16, not just the head. |
| **FP32 gradient reductions for output layer** | ❌ NO | Nemotron 3 Ultra divergence | **ADD note**. Critical if using MTP (Section 8) + low precision. |
| **Async checkpointing** | ❌ NO | Nemotron-H (NVRx) | **LOW priority**. PyTorch 2.3+ `async_save`. Matters for multi-day runs. |
| **Sequence parallelism** | ❌ NO | Nemotron-H | **SKIP** for now. Only useful with TP, which 600M doesn't need. Revisit if seq >64K. |
| FSDP / distributed optimizer | ✅ Section 7 | Nemotron-4 | Already covered. |
| FlexAttention | ✅ Section 1 | (not Nemotron) | Already covered; correct choice over TE attention for Flower's masks. |
| Smooth-SwiGLU | ✅ Section 10 | (Peng et al., not Nemotron) | Already covered; TE's NVFP4 RHT is the NVIDIA alternative. |
| Pipeline/tensor parallelism | ❌ (correctly omitted) | Nemotron-4 | Correctly skipped — datacenter-scale, 600M doesn't need it. |

---

## 6. What's Specific to NVIDIA's Datacenter Scale (NOT generalizable)

- **8-way tensor parallelism + 12-way interleaved pipeline parallelism** — only for models >single-GPU memory.
- **768-way / 6144-way data parallelism** — needs a full cluster with InfiniBand fabric.
- **DGX Cloud Resilience service** — proprietary failure attribution for multi-thousand-GPU clusters.
- **NVLink/NVSwitch topology-aware collectives** — the 900 GB/s intra-node fabric drives their TP/PP efficiency. An 8×H100 rented node has this, but Flower won't need TP at 600M.
- **LatentMoE, MTP (inference), hybrid Mamba architecture** — architectural choices, not training-efficiency techniques (and MTP is already in Flower Section 8).
- **Megatron-LM-specific fused kernels** (`bias_dropout_add_fusion`, `masked_softmax_fusion`, `grad_div_ar_fusion`) — these are conceptually equivalent to what TE provides portably, or what `torch.compile` fuses.

---

## 7. Bottom Line for Flower

1. **Adopt Transformer Engine.** It's the single highest-leverage change not in the current doc. Replaces hand-rolled FP8 (Section 3), enables clean FP4 (Sections 10/13), and is exactly what NVIDIA uses internally. `pip install transformer_engine[pytorch]`, swap `nn.Linear` → `te.Linear`, wrap forward in `te.autocast`. Works on the 5090 and rented H100/B200 nodes.

2. **The precision progression for Flower**: BF16 (safe, current default) → FP8 via TE for FFN+attention linears on H100/B200 (validate at 600M — Nemotron warns FP8 recipes may not generalize below 8B) → NVFP4 via TE for FFN only on Blackwell (5090/B200), keeping attention + first/last layers + memory path in BF16/FP8.

3. **Don't write custom CUDA kernels for standard ops.** NVIDIA doesn't. Reserve custom kernels (the user's Starling skills) for Flower-unique memory-mechanism operations, and only after profiling confirms they're the bottleneck.

4. **Validate FP8 at Flower's scale carefully.** Nemotron-H explicitly states FP8 recipes didn't generalize below 8B params. At 350-600M, keep BF16 as the default and treat FP8/FP4 as opt-in experiments with loss-curve monitoring. FP8 LM head only (current Section 3) is the lowest-risk entry point.

5. **No new systems-level technique is worth adding** beyond what Section 7 (FSDP) already covers, except possibly async checkpointing for multi-day runs. Sequence parallelism and pipeline parallelism are premature at 600M.
