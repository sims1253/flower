# State-of-the-Art: CUDA/Kernel Optimizations & Low-Precision Training for LLMs (August 2026)
## Focus: Consumer Blackwell (RTX 5090, sm_120, 32GB)

---

## THE FOUNDATIONAL INSIGHT: SM120 is a "Chimera", NOT Datacenter Blackwell

Before everything else, this is the critical fact that determines what works on a 5090:

**SM120 is a hybrid architecture that borrows from three generations but matches none:**

| Feature | Source | SM120 Implementation |
|---------|--------|---------------------|
| MMA Instructions | SM80 (Ampere) | `mma.sync.aligned.m16n8k32` — warp-level, register-to-register |
| TMA (Tensor Memory Access) | SM90 (Hopper) | Async bulk GMEM→SMEM loads |
| FP4/FP8 Block Scaling | SM100 (B200) | `mxf8f6f4.block_scale` in MMA instruction |
| Tensor Memory (TMEM) | SM100 only | **DOES NOT EXIST** |
| UMMA / tcgen05 | SM100 only | **DOES NOT EXIST** |
| Shared Memory | Unique | **99 KB/SM** (vs 228 KB on SM100) |
| Cluster Multicast | SM100 | **Not supported** (1×1×1 only) |

**Source:** lna-lab/blackwell-geforce-nvfp4-gemm (GitHub, 2026-04), confirmed by CUTLASS examples 79a-d and FA4 PR #2634.

**Critical implication:** Any kernel written for SM100 (DeepGEMM, upstream CUTLASS SM100 collectives, WGMMA-based Flash Attention, tcgen05 paths) will fail to compile or crash on SM120. SM120 has block-scaled MMA (`mma.sync.aligned.block_scale.kind::mxf4nvf4`) but uses the OLD warp-level MMA model, not the new async UMMA/tcgen05 model. This means FP4 IS hardware-supported on 5090 — but via a completely different instruction path than datacenter Blackwell.

---

## 1. NVFP4 / FP4 ON CONSUMER BLACKWELL (sm_120)

### 1.1 A4Q Kernel (Attention with 4-bit Q) — THE CRITICAL ONE
**Source:** Jetha Chan (jetha-chan/jethac on GitHub/X/LinkedIn), open-sourced ~July 4, 2026
**Status:** Kernel released, being upstreamed to FlashInfer/vLLM

**What it is:** A native NVFP4 attention kernel for consumer/SoC Blackwell (RTX 50 series, RTX PRO 6000, DGX Spark). Uses `mma.sync.kind::mxf4nvf4.block_scale` — the SM120-native block-scaled MMA instruction — to consume the 4-bit KV cache directly in the tensor core with **zero dequantization**.

**The problem it solves:** NVIDIA ships thousands of hand-tuned attention kernels, but the 4-bit ones target datacenter silicon (tcgen05/UMMA). On a 5090/DGX Spark, the 4-bit KV cache was being unpacked through a slow conversion chain → 4-bit attention was *slower* than 16-bit. A4Q reads the 4-bit cache straight into the tensor core.

**Key design:** Native `mma.sync.kind::mxf4nvf4.block_scale` QKᵀ that consumes the NVFP4 KV cache directly. The block_scale MMA instruction handles per-16-element UE4M3 scaling natively.

**Measured results (from independent reproductions):**
- **Nemotron-3-Omni-30B:** TTFT −39% at 256K context (drkeys, 4× DGX Spark cluster A/B study)
- **Qwen3.6-27B:** TTFT −23%, decode +4%
- **mr_r0b0t production serving (DGX Spark):** 248 tok/s aggregate decode, 135× concurrency, +67% KV-cache capacity vs FP8 at equal context
- **Passkey retrieval:** 100% correct through full 256K context, quality identical to FP16
- Speculative decoding stackable (76% acceptance)

**When it wins vs doesn't:**
- WINS where attention is the bottleneck: long context, wide-head models (Gemma-4 with head_dim=512)
- WINS on prefill/TTFT for ANY model doing paged attention — win grows with context length
- NEUTRAL where attention isn't the bottleneck
- Mamba/attention hybrids break even on decode but see TTFT drop 23-39%

**Inference relevance:** YES for Flower if it ever does long-context inference. The capacity win (+67% KV vs FP8) is huge for 32GB.

**Training relevance:** ⚠️ A4Q is an **inference** kernel (KV cache). It does NOT provide a fast FP4 GEMM for the forward/backward linear layers. For training, the FP4 GEMM problem remains separate (see 1.2).

**Limitations:**
- Inference-only (KV cache path)
- Requires FlashInfer fork patches for SM120 dispatch
- Packed GQA had crashes on some SM120 GQA shapes (being fixed in FA4 SM120 PR #2634)

### 1.2 SF Tensor's tcgen05.mma Reverse-Engineering
**Source:** SF Tensor (sf-tensor.com/engineering/bitwise-tcgen05), July 11, 2026
**Repo:** sf-tensor/tcgen05 (GitHub) — open source, dependency-free

**What it is:** A bit-exact software model of NVIDIA Blackwell's `tcgen05.mma` instruction (the datacenter tensor core), matching real B200 hardware bit-for-bit across **46 hardware-validated datapaths**.

**Key findings — the fixed-point accumulator model:**
The tcgen05 datapath is NOT IEEE FMA. It's a **fixed-point accumulator**:
1. **Truncate inputs** to container (A/B: `bits & 0xffffe000` for TF32; classify special values AFTER truncation)
2. **Decode to integers** (not floats) — signed mantissa + exponent
3. **Multiply integer mantissas** exactly (no per-product rounding)
4. **Shared fixed-point window**: `quantum_exp = max(nonzero product raw exponents, C exponent) - 25`. All products + C shifted into this single quantum, truncated toward zero, summed as exact integers. **No per-term rounding** — the only "rounding" is alignment into the window.
5. **Emit F32** by truncating magnitude to 23 fraction bits

**Why this matters for FP4:** The NVFP4 path (`scale_vec::4X`) uses UE4M3 block scales. Understanding the exact accumulator behavior (truncation toward zero, no final RNE round) is essential for:
- Formal kernel verification
- Predicting when FP4 accumulation will diverge from BF16
- Understanding "shrinkage bias" (see FP4 training papers)

**sm_120 relevance:** This models tcgen05 (SM100/B200), which **does NOT exist on SM120**. However, SM120's `mma.sync.block_scale` likely shares a similar accumulator design. The model is a reference for understanding block-scaled MMA behavior generally.

**Formal verification angle:** SF Tensor pairs this with a Z3/SMT-based kernel optimizer that formally verifies optimized candidates against the bit-exact model. This is the frontier of "proving your kernel is correct."

### 1.3 FP4 GEMM State on SM120 (PyTorch/torchao/CUDA 13.x)

**CRITICAL STATUS: FP4 GEMM kernels exist for inference on SM120, but there is NO production training path.**

**What EXISTS (inference):**
- **CUTLASS examples 79a-79d** (official NVIDIA, SM120 GeForce GEMM):
  - 79a: NVFP4 × NVFP4 → BF16 (forward, warp-specialized persistent, block_scale MMA)
  - 79b: NVFP4 × NVFP4 → NVFP4 (chained GEMM with block-scale-factor epilogue)
  - 79c: MXFP8 × MXFP6 → BF16 (mixed precision)
  - 79d: Grouped GEMM (for MoE)
  - Tile size: ThreadBlockShape `<128,128,128>`, ClusterShape `<1,1,1>`
  - NVFP4 MMA throughput: **2× vs MXFP8, 4× vs Ada FP8**
  - CUDA 12.8+ required for SM120, 12.9+ for SM121
- **torchao 0.18**: MXTensor supports FP4 conversion, `to_mx(x, torch.float4_e2m1fn_x2)` — but NO training GEMM path
- **FlashInfer**: NVFP4 KV cache support (PR #2820), SM120 patches merged

**What DOES NOT EXIST (training):**
- No FP4 backward GEMM in torchao/PyTorch for SM120
- torchao's `Float8LinearConfig` only supports FP8, not FP4
- No `torch._scaled_mm` path for FP4 with autograd on SM120
- Transformer Engine NVFP4 training: **crashes on SM120** (see below)

**When will fast FP4 training GEMM land for SM120?**
- Community work (Harry-Chen/fp4_sm120, lna-lab) has polyfilled stochastic rounding and RHT for SM120
- The CUTLASS 79a kernel demonstrates the forward GEMM works; a backward would need implementing
- Estimate: late 2026/early 2027 for a usable training path, likely via community patches first

### 1.4 MSLK (Meta Superintelligence Labs Kernels)
**Source:** github.com/meta-pytorch/MSLK (formerly FBGEMM GenAI)

**What it is:** High-performance kernels for GenAI training/inference: FP8 row-wise quantization, collective comms.

**SM120 support:** **YES** — compatibility table lists `12.0a` as supported architecture in MSLK 1.0.0+ (PyTorch 2.10+). C++ kernels compiled for sm_120a; Triton JIT kernels work more broadly.

**For Flower:** MSLK's FP8 rowwise quantization kernels may offer an alternative to torchao's (which measured 1.01× on SM120 — "pointless"). Worth benchmarking MSLK's rowwise path specifically.

### 1.5 Transformer Engine NVFP4 on SM120
**Status: BROKEN on SM120 as of TE 2.16/2.17**

- TE issue #3062: "NVFP4 default recipe (RHT + stochastic rounding) crashes on sm_120: fused kernel exceeds 101376-byte shared-memory opt-in cap"
- TE issue #2255: "Is NVFP4 not supported for RTX 50 series?" — confirmed not working
- TE PR #2833: Enabling/guarding sm120 support (non-attention) — in progress

**Harry-Chen/fp4_sm120** polyfills the two missing pieces:
1. **Stochastic rounding:** `cvt.rs.satfinite.e2m1x4.f32` is missing on SM120 (exists on SM100). Polyfilled using `cvt.rn.satfinite.e2m1x2.f32` + software SR noise injection. **Bit-exact match vs SM100 hardware** (validated on B300).
2. **RHT GEMM:** TE's `rht_gemm_ntt_w_sfc` uses SM100 UMMA/tcgen05. Polyfilled with WMMA. **0% mismatch vs SM100 reference** (validated on GB300).

**Bottom line:** TE NVFP4 training COULD work on SM120 with these patches, but the shared-memory cap (99 KB vs 228 KB) remains a hard constraint — fused kernels must be re-tiled.

---

## 2. FP8 TRAINING ADVANCES

### 2.1 torchao FP8 Training (Current State)
**Source:** docs.pytorch.org/ao/stable/workflows/training.html (Updated March 2026)

**API:** `convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name("tensorwise"))`

**Three recipes:**
1. **`tensorwise`** — fastest, single scale per tensor. This is what Flower measured at 1.39× per-block.
2. **`rowwise`** — better outlier handling. **Flower measured 1.01× on SM120 — pointless.**
3. **`rowwise_with_gw_hp`** — most accurate, weight gradient in high precision

**Key change (2026):** Delayed and static scaling have been **DEPRECATED and removed** from torchao.float8 (PR #1753, issue #1680). Only **dynamic scaling** remains going forward. This simplifies the API.

**Performance:** Up to 1.5× at 512 GPU / 405B scale, 1.25× at 8 GPU / 8B scale. Speedups increase with larger GEMM shapes (M, K, N).

**SM120 note:** FP8 tensorwise works via cuBLAS FP8 GEMM. The 1.01× rowwise result suggests SM120's cuBLAS rowwise FP8 path is not optimized — the overhead of per-row scaling exceeds the GEMM speedup at small-medium shapes.

### 2.2 torch._scaled_mm Backward Support
**Status: Fully supported**

`torch._scaled_mm` is the underlying CUDA op for FP8/MX matmuls. Supports:
- Forward: FP8 × FP8 → BF16/FP8
- Backward: Both gradient GEMMs (grad_input, grad_weight)
- Scale_result parameter for output quantization

Recent fixes (PR #190452): fixed `_scaled_mm` dropping `scale_result` on FP8 output path.

### 2.3 MXFP8 (OCP MX format) vs NVFP8
**torchao prototype:** `torchao.prototype.mx_formats` — MXFP8 training for `nn.Linear`, "on its way to stable"

**MXFP8 GEMM on B200:** cuBLAS MXFP8 GEMM via `torch._scaled_mm`, ~2× vs BF16 on common shapes.

**SM120 status:** MXFP8 uses `mxf8f6f4.block_scale` MMA which IS available on SM120 (CUTLASS 79c demonstrates MXFP8×MXFP6). However, torchao's MXFP8 training path is not yet validated on SM120 — it targets B200.

**Key difference:** MXFP8 uses 32-element blocks with UE8M0 (power-of-2) scales. NVFP8 (conceptual) would use 16-element blocks with E4M3 scales. MXFP8 is the OCP standard; NVFP4/NVFP8 are NVIDIA's enhanced versions.

### 2.4 FP8 Attention (e4m3 mantissa limitation)
**The head_dim=64 problem:** FP8 e4m3 has 3 mantissa bits → only 8 representable mantissa values. For attention QKᵀ, this severely limits precision when head_dim=64 (the dot product sums 64 terms in FP8 accumulation).

**Current solutions:**
- FA4 SM120 PR #2634 adds **fp8 KV-cache decode** (e4m3/e5m2): ~1.6-1.9× faster than BF16 at GQA ratio ≤ 4, half the KV bandwidth, within ~2e-3 of FP8-quantized reference
- **Inference only** — FP8 attention during training is not standard; FlexAttention runs in BF16/FP16

### 2.5 DeepSeek-V4: MXFP4 Weights + FP4 QAT
**Source:** arXiv:2606.19348 (DeepSeek-V4 paper, April 2026), Section 5.2.1

**Architecture:** DeepSeek-V4-Pro (1.6T params, 49B activated) and V4-Flash (284B, 13B activated). Supports 1M token context.

**FP4 QAT (Section 5.2.1):** Applied during **post-training** (not pre-training):
- MXFP4 QAT for MoE expert weights
- Part of comprehensive post-training pipeline
- Pre-training uses BF16 + Muon optimizer

**Key architectural innovations (relevant to mixed precision):**
- **CSA (Compressed Sparse Attention):** fine-grained, top-k over compressed entries
- **HCA (Heavily Compressed Attention):** very heavy compression, full sequence
- **mHC (Manifold-constrained Hyper-Connections):** enhanced residual connections
- **Muon optimizer:** faster convergence, greater stability
- **FP4 KV Cache:** 50× KV cache reduction at 1M tokens vs V3.2

**Training framework:** TileLang for flexible kernel development. Batch-invariant deterministic kernel libraries.

**For Flower (sm_120 relevance):** DeepSeek-V4's FP4 is for **inference** (weights + KV cache). The QAT approach is post-training, not from-scratch FP4 pretraining. The hybrid CSA/HCA attention is inference-time efficiency, not training speedup.

---

## 3. KERNEL OPTIMIZATION TECHNIQUES

### 3.1 FlexAttention + FlashAttention-4 Backend
**Source:** pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible (March 2026)

**THE BIG WIN for SM120:** FlexAttention now has a FlashAttention-4 backend via CuTeDSL.

**Performance:** **1.2× to 3.2× over existing Triton implementation** on compute-bound workloads (Hopper/Blackwell).

**How it works:**
- FA4 is written in CuTeDSL (Python DSL for CUTLASS abstractions)
- Inductor generates CuTeDSL code from `score_mod`/`mask_mod` functions
- FA4 provides extension points for FlexAttention's score modifications + block sparsity
- Same async pipeline infrastructure shared between FA4 and Flex

**SM120 specifics:** FA4 SM120 integration is PR #2634 (thad0ctor, open as of July 2026):
- Uses SM80-base kernels (cp.async + TMA + warp-specialized forward) — NOT tcgen05
- Forward: head dims 64/96/128/192/256
- Backward: head dims 64/96/128/256 (+ d=192,dv=128; equal-dims D192 backward exceeds 99KB smem cap)
- fp8 KV-cache decode: ~1.6-1.9× faster than BF16
- torch.compile compatible (bit-identical to eager)
- Performance vs FA2 on 5090: ~parity to +11% on D256 GQA forward, parity on backward

**For Flower:** **THIS IS YOUR PRODUCTION PATH.** FlexAttention with FA4 backend is the correct attention implementation for SM120. The score_mod API handles your local-window causal attention and any custom modifications.

### 3.2 FlashAttention-4 Status
**Source:** github.com/Dao-AILab/flash-attention

**Status:** Active beta (`fa4-v4.0.0.beta25` as of July 29, 2026). Written in CuTeDSL, optimized for Hopper + Blackwell.

**Paper:** "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design" (arXiv:2603.05451)

**Key innovations:**
- Deeply pipelined, warp-specialized kernel
- TMEM (Tensor Memory) for accumulators (SM100 only)
- Ping-pong scheduling: overlap one tile's matmuls with another's exp() — because SFU didn't keep pace with tensor cores
- CuTeDSL for Python authoring + JIT compilation

**SM120 support:** Via PR #2634 (community contribution). Not in main FA4 yet — needs merging. The 99KB smem limit forces different tiling than SM100.

### 3.3 CUTLASS 3.x / CuTe for Custom Kernels
**Source:** CUTLASS 3.9.0+ (examples 79a-79d for SM120)

**SM120 support:** Official via examples 79_blackwell_geforce_gemm:
- Block-scaled datatypes (NVFP4, MXFP8, MXFP6)
- Warp-Specialized persistent kernel (cooperative + ping-pong schedules from Hopper)
- Cluster Launch Control (dynamic SW scheduler) — but cluster shape must be 1×1×1
- TMA loads supported, multicast NOT supported

**CuTeDSL:** Python DSL for CUTLASS, enables JIT workflows. Used by FA4 and CODA. Growing rapidly as the authoring layer for Blackwell kernels.

### 3.4 Triton Compiler for SM120
**Source:** Multiple Triton PRs (2026)

**Status:** Working but with caveats.

**Key fixes for SM120:**
- PR #9734: Fix PTX codegen segfaults (non-deterministic SIGSEGV on RTX 5070 Ti/5080/5090)
- PR #10002: Default global_scratch allocator fallback for Blackwell SM12.0
- Commit 1a453a3: Enable batched TMA GEMM tests on sm120
- XLA PR #39253: Triton GEMM default configs for consumer Blackwell

**Liger Kernel on SM120:** v0.8.1 adds CuTe DSL scaffolding for Blackwell, CUTLASS CuTe DSL RMSNorm, dtype-aware num_warps (Blackwell-gated). Triton-based kernels work but may need arch-specific tuning.

### 3.5 Megakernel Design (FlashKDA)
**Source:** MoonshotAI/FlashKDA (1065 stars), docs/20260420-flashkda-v1-deep-dive.md

**What it is:** Fused kernel for Kimi Delta Attention (KDA) — the attention mechanism in Kimi K3. Built on CUTLASS. Deep-dive doc covers chunk-size selection, state management, and fused gate/conv1d/norm operations.

**Relevance:** Megakernel design pattern (fusing multiple sequential operations into one kernel to avoid global memory round-trips) is the frontier. FlashKDA is specific to KDA, but the **design principles** apply:
- Chunk size selection balancing compute density vs. state I/O
- Fused gating + activation + normalization
- Persistent kernel with state management

**SM120 support:** Requires SM90+, CUDA 12.9+. Should compile for SM120 but no specific testing documented.

### 3.6 CODA: GEMM-Epilogue Programs
**Source:** arXiv:2605.19269 (May 2026), github.com/open-lm-engine/coda-kernels (235 stars)
**Authors:** Han Guo (MIT), Tri Dao (Princeton/Together AI), et al.

**What it is:** GPU kernel abstraction that rewrites Transformer operators as GEMM-plus-epilogue programs. Fuses normalization, activations, residual updates, and reductions into the GEMM output tile **before it's written to global memory**.

**The problem it solves:** Non-GEMM ops (RMSNorm, SwiGLU, residuals, RoPE, CE loss) are memory-bound. They repeatedly move large tensors through global memory doing little arithmetic. As FP8/FP4 GEMMs get faster, this bottleneck becomes proportionally larger.

**Key insight:** Many framework-level operators can be algebraically **reparameterized** to execute while the GEMM output tile is still on-chip. CODA fixes the GEMM mainloop and exposes composable epilogue primitives: scaling, reductions, pairwise transformations, accumulation.

**Implemented kernels (all as GEMM + epilogue):**
- `linear_swiglu`, `linear_cross_entropy`, `linear_qknorm_rope`
- Forward: GEMM + residual + partial RMSNorm + weight scaling, GEMM + RMSNorm + SwiGLU, GEMM + RMSNorm + RoPE, GEMM + RMSNorm + partial CE
- Backward: Full backward kernels for each forward pattern

**Built on:** CUTLASS CuTeDSL. Targets Hopper (H100). v0.2 released July 19, 2026.

**LLM-authored kernels:** CODA is designed so that both humans AND LLMs can author kernels using its primitives — tested with AI-generated kernels achieving high performance.

**SM120 relevance:** Built for H100 (SM90). CuTeDSL kernels may be adaptable to SM120 since SM120 supports TMA + warp-specialized schedules. The epilogue fusion concept is directly applicable — Flower's "Liger RMSNorm/SwiGLU 2× SLOWER under compile" problem is exactly what CODA's fused GEMM+RMSNorm+SwiGLU kernel solves.

**For Flower:** **HIGH PRIORITY** — CODA's `linear_swiglu` and GEMM+RMSNorm fusion could recover the performance loss from Liger's standalone kernels being slower under compile. The epilogue fusion eliminates the memory round-trip entirely.

### 3.7 Formal Kernel Verification
**Source:** SF Tensor (sf-tensor.com/kernel-optimizer), Gimlet Labs, triton-verify

**SF Tensor approach:**
1. Bit-exact hardware model of tcgen05.mma (46 datapaths)
2. Z3 SMT solver proves optimized kernel candidates match the model
3. Symbolic execution over all input patterns
4. Only candidates that formally verify are kept

**triton-verify (amshah1022):** Formal verification of pointer safety in Triton GPU kernels using Z3. Proves for ALL parameter values that pointer accesses are safe.

**Gimlet Labs:** "Formally Verifying AI-Generated GPU Kernels" — building trust in AI-generated kernels through formal methods.

**For Flower:** Research-grade. Not directly applicable yet, but signals the direction: AI-generated kernels (KernelBench) + formal verification = reliable automated kernel optimization.

### 3.8 KernelBench / JAXBench
**Sources:**
- KernelBench (ScalingIntelligence/KernelBench): Benchmark for "Can LLMs Write Efficient GPU Kernels?" — Torch → CUDA
- JAXBench (arXiv:2607.20466): 50 curated JAX/TPU kernel workloads, production-ready evaluation harness

**Status:** Both are benchmarks for evaluating AI agents that generate kernels. KernelBench is GPU-focused, JAXBench is TPU-focused.

**Relevance:** Tooling for the future where AI agents write custom kernels for your specific shapes. Not production-ready for Flower today.

---

## 4. MIXED PRECISION TRAINING STRATEGIES

### 4.1 FP4 Training: The State of the Art

**NVIDIA's NVFP4 Training (arXiv:2509.25149, "Pretraining LLMs with NVFP4"):**
- Trained 12B model on **10T tokens** in NVFP4 — longest documented 4-bit training run
- Achieves MMLU-pro 62.58% vs FP8's 62.62% — **near-parity**
- **Recipe:** RHT (Random Hadamard Transform) + stochastic rounding + 2D block scaling + selective high-precision layers
- NVFP4 > MXFP4 due to smaller blocks (16 vs 32) and E4M3 scales (vs UE8M0)
- Run on **B200/GB200** (datacenter) — NOT validated on SM120

**"Shrinkage Bias" paper (arXiv:2606.20381, June 2026 — UFP4 Recipe):**
- **Critical finding:** E2M1 format (used by NVFP4/MXFP4) has inherent **Shrinkage Bias** — systematic negative rounding error from geometric asymmetry of representable bins
- This bias **accumulates multiplicatively across layers** and is **amplified by RHT**
- **Solution: UFP4** — use uniform E1M2/INT4-style grids instead of E2M1
- UFP4 with full-RHT (all 3 GEMMs) outperforms E2M1-based baselines on 124B MoE
- Implication: **Future accelerators should support E1M2/INT4 as first-class training primitives**

**"Training LLMs with MXFP4" (Tseng et al., MLIR 2025):**
- Establishes the MXFP4 training recipe: RHT + stochastic rounding
- Foundation for NVIDIA's NVFP4 work

**"Elucidating the Design Space of FP4 Training" (arXiv:2509.17791):**
- Systematic study of FP4 training design choices
- Scaling, rounding, outlier handling

### 4.2 Random Hadamard Transform (RHT)
**What it does:** Norm-preserving rotation that disperses outlier energy across all coordinates before quantization. Applied along the shared GEMM reduction dimension.

**Math:** `Y = XWᵀ = (XH')(WH')ᵀ` where `H' = SH` (random sign matrix × Hadamard matrix). Since H' is orthogonal, the full-precision result is preserved; only the quantized operands change.

**For SM120:** Harry-Chen/fp4_sm120 provides WMMA-based RHT GEMM polyfill (replaces tcgen05). 0% mismatch vs SM100 reference.

### 4.3 Stochastic Rounding
**Purpose:** Unbiased gradient estimation in low-precision. `Pr[q=gb] = (x/s - ga)/(gb - ga)`. Preserves values in expectation.

**SM120 problem:** `cvt.rs.satfinite.e2m1x4.f32` (native FP4 stochastic rounding) is **missing on SM120** (exists on SM100).

**Polyfill (Harry-Chen/fp4_sm120):** Uses `cvt.rn.satfinite.e2m1x2.f32` + software SR noise injection. **Bit-exact match vs SM100 hardware.**

### 4.4 Per-Layer Precision Routing
**Concept:** FP4 for FFN GEMMs (compute-heavy), FP8 for attention (precision-sensitive), BF16 for memory path (embeddings, norms).

**NVFP4 paper approach:** "selective high-precision layers" — preserve numerically sensitive layers in higher precision.

**DeepSeek-V4:** MXFP4 for MoE expert weights (QAT, post-training), not for attention.

**For Flower:** Currently FP8 tensorwise for all linears. A routing approach (FP8 attention path stays, experiment with FP4 FFN when kernels exist) is the natural evolution — **but requires FP4 backward GEMM which doesn't exist on SM120 yet.**

### 4.5 SmoothQuant / Smooth-SwiGLU
**SmoothQuant:** Migrate quantization difficulty from activations to weights by scaling. Reduces activation outliers.

**For FP8 stability:** Helps with rowwise FP8 (which measured 1.01× on SM120). But since tensorwise FP8 already works well (1.39×), the stability techniques matter more for FP4.

---

## 5. MEMORY/THROUGHPUT OPTIMIZATIONS

### 5.1 Activation Checkpointing
**Source:** pytorch.org/blog/activation-checkpointing-techniques

**Selective/FFN-only checkpointing:** Recompute only FFN blocks (which have large activations) during backward. Keep attention activations (smaller) in memory.

**PyTorch AC advances (2026):**
- Per-layer AC policies
- Composable with torch.compile, DTensor, FSDP2, FP8
- "Adacc" (arXiv:2508.00806): adaptive framework unifying compression + recomputation

**For Flower (32GB, seq=32K):** FFN-only checkpointing is the standard approach. At 67% MFU with BF16, activation memory is likely the constraint at 450M params.

### 5.2 CUDA Graphs + Gradient Accumulation
**Status: STILL PROBLEMATIC (2026)**

**Known issues:**
- Issue #169545: torch.compile + CUDAGraph + gradient accumulation fails
- Issue #181723: `make_graphed_callables` can silently corrupt parameter gradient accumulation (static grad buffers)
- PR #187732: Fix `static_graph=True + no_sync()` gradient accumulation regression (open, June 2026)
- Commit ea7e147: Improved CUDAGraph grad accumulation error messages

**NVIDIA guidance (CUDA Graph Best Practices):** Deferred gradient hooks with `make_graphed_callables` eliminate computation-communication overlap.

**For Flower:** Flower already found CUDA Graphs OOM at 450M and are shape-dependent. This is a **known limitation** with active fixes in progress but not resolved. The `max-autotune-no-cudagraphs` compile mode remains the safer choice.

### 5.3 torch.compile Best Practices for Blackwell
**Mode:** `max-autotune-no-cudagraphs` (what Flower uses — correct choice)

**Key considerations:**
- Inductor generates Triton kernels; for Blackwell-specific patterns, CuTeDSL backend is emerging (FlexAttention FA4 path)
- Graph breaks are the enemy — each `print`, data-dependent control flow, or unsupported op breaks the graph
- Shape specialization: dynamic shapes prevent kernel fusion; pad to fixed shapes where possible

### 5.4 Liger Kernel Updates
**Source:** Liger-Kernel v0.8.1 (2026)

**New on Blackwell:**
- CuTe DSL cross-entropy scaffolding for Blackwell/B200 (#1279)
- CUTLASS CuTe DSL RMSNorm (#1299)
- `infer_device_arch()` for arch-aware dispatch
- dtype-aware num_warps (Blackwell-gated) (#1267)

**For Flower:**
- ✅ FusedLinearCE: adopted (-1.1GB) — correct
- ❌ RMSNorm/SwiGLU: 2× SLOWER under compile — **CODA's GEMM+RMSNorm+SwiGLU fusion is the fix**

### 5.5 FSDP2 Advances (2026)
**Source:** Multiple PyTorch PRs

**Relevant for single-GPU:** FSDP2 is multi-GPU, but its memory-efficient all-gather patterns influenced single-GPU design.

**Key 2026 advances:**
- Separate NCCL communicator for reduce-scatter (AG/RS overlap) — PR #177015
- `set_separate_reduce_scatter_group` (opt-in AG/RS overlap) — PR #186335
- MORI SDMA all-gather backend + zero-copy output — PR #188276

**For Flower (single GPU):** Not directly applicable. But the FP8 + FSDP2 integration (50% throughput speedup at scale) validates the FP8 training path.

### 5.6 CPU Offloading of Optimizer States
**Source:** ZeRO-Offload (DeepSpeed), torchao PR #584

**Concept:** Offload optimizer states (Adam: 2× model params in FP32) to CPU memory. Train on GPU, optimize on CPU.

**For single GPU (32GB):** At 450M params, Adam states = ~3.6GB FP32. Not the bottleneck. At 1B+ params, offloading becomes relevant.

**torchao PR #584:** "Optimizer CPU offload for single GPU training" — provides simple API for single-GPU optimizer offloading (exists for FSDP, not for single GPU in main PyTorch).

---

## 6. TRAINING DATA PIPELINE & ALGORITHMIC EFFICIENCY

### 6.1 Token Superposition Training (TST)
**Source:** arXiv:2605.06546 (May 2026), Bowen Peng + Nous Research
**Repo:** Available (paper references code)

**What it is:** A drop-in method that improves data throughput per FLOP during pretraining **without modifying parallelism, optimizer, tokenizer, data, or model architecture.**

**Two phases:**
1. **Superposition phase:** Combine many contiguous tokens into one "bag" and train using multi-hot cross-entropy (MCE) objective
2. **Recovery phase:** Revert to standard next-token prediction training

**Results:**
- Validated at 270M, 600M, 3B, and 10B-A1B MoE scales
- Consistently outperforms baseline loss and downstream evals
- At equal-loss: **up to 2.5× reduction in total pre-training time** at 10B-A1B scale
- Highly robust across settings

**How it works:** By combining tokens into bags, each forward pass processes fewer "superposition tokens" but each represents multiple original tokens. The MCE objective trains the model to predict the bag of tokens. This increases training throughput (fewer tokens per step) while maintaining learning signal.

**For Flower:** **HIGH PRIORITY.** This is a direct training efficiency win with zero architecture change. The superposition phase speeds up pretraining; the recovery phase ensures the model learns standard autoregressive prediction. 2.5× speedup is enormous.

### 6.2 Lighthouse Attention
**Source:** arXiv:2605.06554 (May 2026), nousresearch.com/lighthouse-attention
**Authors:** Bowen Peng, Subho Ghosh, Jeffrey Quesnelle (Nous Research)

**What it is:** Training-only selection-based hierarchical attention. Runs the same forward+backward pass **~17× faster** than standard attention at 512K context on B200. 1.4-1.7× end-to-end pretraining speedup at 98K context.

**Design:**
- **Symmetric pooling:** Q, K, V ALL pooled across L-level pyramid (unlike prior work that only pools K/V)
- **Parameter-free scoring:** Per-head ℓ2 norms of Q and K projections select top-K entries. No learned scorer, no auxiliary loss, no STE.
- **Selection outside the kernel:** Gather chosen entries → contiguous causal sub-sequence → **stock FlashAttention** → scatter back. No custom sparse attention kernel.
- **Gradient-free top-K:** Non-differentiable selection; gradients flow through scatter/FA/gather into WQ,WK,WV

**Two-stage training:**
1. Stage 1 (Lighthouse): Train majority of budget with selection enabled. Throughput: 84-126k tok/s/GPU vs ~46k for dense SDPA.
2. Stage 2 (SDPA-resume): Brief standard attention training. Recovers dense-attention competence.

**Key result:** After SDPA resumption, Lighthouse-trained models **match or beat dense-from-scratch baseline** at same token budget. Sparse training doesn't hollow out dense attention ability.

**For Flower:** **HIGH PRIORITY for long-context training.** If Flower trains at seq=32K+, Lighthouse Attention can dramatically reduce attention compute during the main training phase. The implementation is "two new files + ~600 lines on top of torchtitan." Uses stock FlashAttention (which Flower already has via FlexAttention).

### 6.3 Best Datasets for Small Model Pretraining (2026)
**FineWeb-Edu:** High-quality educational web text. Standard for small model pretraining. Used in SmolLM2, OLMo, etc.

**SmolLM2 corpus:** HuggingFaceTB/smollm-corpus — curated educational + synthetic data. SmolLM2 (1.7B) achieves SOTA through data-centric training. Components:
- FineWeb-Edu (educational web)
- Cosmopedia (synthetic textbooks/tutorials)
- Python-Edu, etc.

**DeepSeek-V4 data:** 32T+ tokens, diverse + high-quality. Proprietary data construction pipeline (not publicly available).

**For Flower (small models):** FineWeb-Edu + SmolLM-corpus remain the best open datasets. TST (6.1) can be applied on top to improve throughput.

---

## PRIORITY RECOMMENDATIONS FOR FLOWER (RTX 5090, sm_120, 32GB)

Based on all research, here are the highest-impact opportunities, ranked:

### Tier 1: Implement Now (High impact, achievable)
1. **Token Superposition Training (TST)** — 2.5× pretraining speedup, zero architecture change. Phase 1 (superposition) + Phase 2 (recovery). Drop-in.
2. **Lighthouse Attention** — 1.4-1.7× speedup at long context (98K+). For 32K context, still meaningful attention compute reduction. Two-stage approach.
3. **CODA kernels (GEMM-epilogue fusion)** — Fixes the "Liger RMSNorm/SwiGLU 2× slower under compile" problem by fusing into GEMM epilogue. `linear_swiglu`, GEMM+RMSNorm+RoPE, GEMM+CE.

### Tier 2: Monitor / Adopt When Stable
4. **FA4 SM120 backend (PR #2634)** — Once merged, FlexAttention gets FA4 backend for SM120 with 1.2-3.2× over Triton. Watch for merge.
5. **A4Q kernel** — For long-context inference with NVFP4 KV cache. +67% KV capacity, -23-39% TTFT. Upstreaming to FlashInfer/vLLM.
6. **torchao MXFP8 training** — Prototype now, may become stable. Could offer ~2× GEMM speedup if SM120 MXFP8 GEMM is optimized.
7. **MSLK FP8 rowwise** — Alternative to torchao's rowwise (which measured 1.01×). May be better optimized for SM120.

### Tier 3: Research / Future
8. **FP4 training on SM120** — NVFP4 GEMM kernels exist (CUTLASS 79a-d) but no training path (backward GEMM). Harry-Chen/fp4_sm120 polyfills the conversion + RHT. The shared-memory cap (99KB) requires kernel re-tiling. Estimated: late 2026/early 2027 for usable path.
9. **UFP4 (uniform E1M2 grid)** — Better than E2M1 for FP4 training (no shrinkage bias). Requires hardware support for E1M2 (future accelerators).
10. **Formal kernel verification** — SF Tensor's Z3 approach. Research-grade but signals the future of reliable kernel optimization.

### What WON'T Help (confirmed dead on SM120)
- **tcgen05/UMMA paths** — Hardware doesn't exist on SM120
- **DeepGEMM** — SM100-only
- **Transformer Engine NVFP4 default recipe** — Crashes (99KB smem cap)
- **FP8 rowwise (torchao)** — 1.01× (overhead exceeds GEMM gain)
- **CUDA Graphs at 450M** — OOM, shape-dependent
- **Liger RMSNorm/SwiGLU standalone under compile** — 2× slower

---

## KEY ARXIV PAPERS (2025-2026)

| Paper | arXiv | Key Finding |
|-------|-------|-------------|
| NVFP4 Pretraining (NVIDIA) | 2509.25149 | 12B/10T tokens in NVFP4, near-FP8 accuracy. RHT + SR + 2D scaling |
| UFP4 (Shrinkage Bias) | 2606.20381 | E2M1 has inherent shrinkage bias; E1M2/INT4 better. UFP4 recipe |
| Elucidating FP4 Design Space | 2509.17791 | Systematic FP4 training design choices |
| Training LLMs with MXFP4 | (Tseng 2025, PMLR) | MXFP4 recipe: RHT + stochastic rounding |
| TST (Token Superposition) | 2605.06546 | 2.5× pretraining speedup, bag-of-tokens + MCE |
| Lighthouse Attention | 2605.06554 | 17× attention speedup, hierarchical selection, 1.4-1.7× e2e |
| CODA | 2605.19269 | GEMM-epilogue fusion for all Transformer ops |
| FlashAttention-4 | 2603.05451 | Algorithm + kernel pipelining co-design for Blackwell |
| DeepSeek-V4 | 2606.19348 | CSA+HCA, mHC, Muon, FP4 QAT, 1M context |
| Kimi K3 | 2607.24653 | 2.8T MoE, KDA attention, Stable LatentMoE |
| Quartet II (NVFP4 gradients) | 2601.22813 | Improved unbiased gradient estimation for NVFP4 |
| MXFP4 pretraining divergence | 2605.09825 | Why FP4 training diverges (gradient accumulation) |
| JAXBench | 2607.20466 | TPU kernel optimization benchmark |

## KEY REPOS

| Repo | Stars | What |
|------|-------|------|
| lna-lab/blackwell-geforce-nvfp4-gemm | 21 | SM120 NVFP4 patches for vLLM+FlashInfer+CUTLASS |
| Harry-Chen/fp4_sm120 | 17 | TE NVFP4 polyfills for SM120 (stochastic rounding, RHT) |
| jethac (a4q-kernel) | — | NVFP4 attention kernel for consumer Blackwell |
| meta-pytorch/MSLK | 118 | Meta's GenAI kernels (FP8, supports sm_120a) |
| open-lm-engine/coda-kernels | 235 | GEMM-epilogue fusion kernels |
| MoonshotAI/FlashKDA | 1065 | KDA megakernel (CUTLASS) |
| sf-tensor/tcgen05 | — | Bit-exact tcgen05.mma model (46 datapaths) |
| ighoshsubho/lighthouse-attention | — | Lighthouse Attention reference impl |
| ScalingIntelligence/KernelBench | — | LLM GPU kernel generation benchmark |
| Dao-AILab/flash-attention | — | FA4 (CuTeDSL), SM120 PR #2634 |

---

*Research compiled August 9, 2026. All sources verified via web search/extraction.*
