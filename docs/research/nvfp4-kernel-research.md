# arXiv:2509.25149 — Kernel-Level Training Optimizations Analysis

**Paper**: "Pretraining Large Language Models with NVFP4" (NVIDIA, Sept 2025)
**Target context**: Flower's 350–600M memory-augmented transformer on RTX 5090 (sm_120, Blackwell)

---

## 1. What the paper actually does

**Important framing correction**: arXiv:2509.25149 is **not a custom-kernel paper**. It is a
**numerical methodology paper** for stable FP4 pretraining (12B Nemotron-H model, 10T tokens).
The authors explicitly state:

> *"This report is primarily concerned with the underlying algorithms and methodology rather
> than with runtime efficiency or system-level optimizations."*

They delegate all kernel work to **NVIDIA Transformer Engine (TE)** — the kernels already exist
in TE PR #2177 and ship as part of the Blackwell NVFP4 stack. The paper's contributions are
**recipes on top of existing hardware/TE kernels**, not novel kernel code.

### Operations the methodology targets (all via TE kernels, not hand-written):

| Operation | What it does | Kernel carrier |
|---|---|---|
| **NVFP4 GEMMs** (Fprop, Dgrad, Wgrad) | 4-bit matmul with E4M3 per-block scales + FP32 per-tensor scale | Blackwell Tensor Cores (native, via TE) |
| **2D block scaling** | 16×16 weight blocks for fwd/bwd consistency (vs 1×16 for acts/grads) | TE quantization kernels |
| **Random Hadamard Transform (RHT)** | 16×16 orthogonal rotation on Wgrad inputs to suppress outliers | TE RHT kernel (batched matmul, memory-bound) |
| **Stochastic rounding** | Unbiased gradient rounding in FP4 conversion | Blackwell Tensor Core instruction (native) |

### What they do NOT fuse or custom-kernel:
- Attention (softmax, QK, SV) stays in BF16/FP32
- Embeddings, norm layers, non-linearities stay BF16/FP32
- Optimizer states, master weights stay FP32
- They note RHT *could* be fused with adjacent layers but leave that as future work

---

## 2. Speedup achieved

The paper reports **no kernel-level speedups** — only accuracy/loss parity.

The hardware-level speedup is **architectural, not algorithmic**:
- NVFP4 Tensor Cores deliver **4× (GB200) to 6× (GB300)** arithmetic throughput vs BF16
- Memory usage ~halved vs FP8
- On the RTX 5090 (consumer Blackwell, sm_120), expect roughly **2× the BF16 FLOPS ceiling**
  for FP4 GEMMs (the 5090 is a cut-down die; not GB200/GB300 tier).

**Implication for Flower**: The 2× FLOPS and 2× memory savings are real and significant at
350–600M scale, but they are unlocked by **using TE's existing NVFP4 path**, not by writing
kernels. The engineering work is integration (TE's `te.Linear` with fp4_recipe), not CUDA.

---

## 3. Generalizable vs architecture-specific

| Component | Generalizable? | To Flower? |
|---|---|---|
| NVFP4 GEMMs | ✅ Yes — any linear layer | ✅ Direct (FFN, qkv, head) |
| 2D weight scaling | ✅ Yes — any weight matrix | ✅ Direct |
| RHT on Wgrad | ⚠️ Partially — needs Wgrad outlier problem to exist | ⚠️ Only if you adopt FP4 |
| Stochastic rounding on gradients | ✅ Yes — any backward pass | ✅ Native HW support |
| Keep final ~15% layers in BF16 | ✅ General stability heuristic | ✅ Adopt if FP4 |
| 12B Nemotron-H specifics (Mamba+Transformer mix) | ❌ Architecture-specific | N/A |

**Bottom line**: Every technique is generalizable to Flower's architecture. None are tied to
the hybrid Mamba-Transformer structure. The only prerequisite is FP4 hardware (Blackwell).

---

## 4. CUDA/Triton kernel patterns to reuse

The paper itself ships **no kernel code**. But the supporting ecosystem (referenced and
adjacent) provides reusable patterns:

### (a) TE NVFP4 kernels (what the paper uses)
- **Repo**: `NVIDIA/TransformerEngine` PR #2177
- **Reusable**: `te.Linear` with `fp4_format="nvfp4"`, recipe objects
- **Pattern**: `te.Linear` auto-routes Fprop/Dgrad/Wgrad to FP4 Tensor Cores with 2D scaling
- **Blackwell gate**: Requires sm_120+; RTX 5090 qualifies

### (b) RHT as a fused tiled matmul (from the paper's Appendix C)
The paper notes RHT is "implemented as batched matrix multiplications, memory-bound, and
**can be fused with other layers to reduce round-trips to device memory**." This is an
explicit invitation to write a fused RHT+quantization Triton kernel — but the paper doesn't
do it. **This is an open opportunity**, not a reusable artifact.

### (c) No custom backward kernels
The paper relies entirely on TE's existing backward GEMMs. No fused-forward-backward patterns.

---

## 5. Related work — fused training kernels ACTUALLY relevant to Flower

This is where the actionable kernel opportunities are. The NVFP4 paper is orthogonal to these.

### Tier 1: Directly applicable to Flower NOW (no architecture change)

#### Liger-Kernel (arXiv:2410.10989) — **highest leverage**
Open-source Triton kernel library for LLM training. ~20% throughput, ~60% memory reduction.
- **FusedLinearCrossEntropy**: chunks the lm_head projection + CE loss to avoid materializing
  full logits. At vocab=50K, seq=2048, this saves **gigabytes**. Directly relevant to Flower's
  tied-embedding head.
- **Fused RMSNorm** (fwd+bwd with cached rms): ~7× faster, ~3× less memory
- **Fused SwiGLU/GeGLU**: recomputes activation in backward, ~1.6× memory reduction
- **Fused RoPE**: ~8× faster
- **Repo**: `link-org/Liger-Kernel` — drop-in, Triton-only, PyTorch-compatible, FSDP-compatible

#### Turbo-Muon Newton-Schulz Triton kernels (Boissin et al. 2025)
**Directly targets Flower's `flower/optim.py` Muon**. Three fused Triton kernels:
- `ns_line_1(X, out=A)`: fused `A = X @ X.T` (Gram matrix)
- `ns_line_2(A, alpha, beta, out=B)`: fused `B = b*A + c*A@A` (4th-order polynomial)
- `ns_line_3(B, X, a, out=C)`: fused `C = a*X + B@X` (the NS update step)
- **AOL preconditioning** fused into first iteration → saves one NS iteration (4 vs 5)
- **Reported**: 2.8× faster than baseline Muon NS, same convergence
- **Repo**: `thib-s/flash-newton-schulz`, also on HuggingFace Kernels (`tboissin/newton_schulz_triton`)
- **Integration**: `from kernels import get_kernel; kern = get_kernel("tboissin/newton_schulz_triton")`
- **Drop-in for Flower**: Replace `_zeropower_via_newtonschulz5()` in `optim.py:130`

#### NVIDIA Emerging Optimizers (Muon with Triton SYRK)
- `nemo.emerging_optimizers` — production Muon with optional `use_syrk=True` Triton kernel
- Explicitly checks `sm_version in ((8,0), (9,0), (10,0), (10,3))` — note: **sm_120 is NOT
  in the tested list** (10.0=Hopper, 10.3=Blackwell-datacenter). The 5090 is sm_120 — may need
  a guard override or correctness validation.
- Includes `coefficient_type` options: "quintic", "polar_express", "cans", "aol", "deepseekv4"

#### Muonium (pip package)
- Fused Triton Gram-matrix kernel + batched `foreach` driver for Muon
- Optional 4-bit/8-bit/fp8 optimizer state (`MuonLP`)
- `Polar Express` orthogonalization strategy (precomputed coefficients, faster convergence)

#### Fused AdamW Triton kernel
- **3.45× faster** than PyTorch AdamW at 50M params (anviit/triton-llm-kernels)
- Fuses weight update + first moment + second moment + weight decay in one kernel
- Relevant to Flower's AdamW path (embeddings, 1D params, head)

### Tier 2: Architecture-specific (Flower's custom memory mechanisms)

These are **the highest-leverage custom kernels for Flower specifically** — no off-the-shelf
solution exists because the operations are novel.

#### Fused Bloom hash routing (`bloom_memory.py:_bloom_route`)
Current code: K separate `nn.Linear` + `softmax` calls, stacked and averaged.
```
K hashes × [Linear(D→S) + softmax + stack + mean]
```
**Fusion opportunity**: A single Triton kernel that:
1. Loads summary items once into SRAM
2. Computes K hash projections in-register
3. Applies temperature-scaled softmax (online softmax, FlashAttention-style)
4. Averages across K
5. Writes the (B, P, S) plan once

At S=16–1024 slots, P=16, K=4, D=384 — this is **bandwidth-bound** (small matmuls).
Fusing eliminates K×2 HBM round-trips. Expected: **2–4× on the routing step**.

#### Fused perceiver summary compression (`bloom_memory.py:_update_memory`)
Current: `nn.MultiheadAttention` (P queries × T tokens) then separate value projection
then `plan.T @ values`.
**Fusion**: Cross-attention + value projection + scatter-write in one tiled kernel.
The `plan.transpose(1,2) @ values` is a batched small matmul — fusable.

#### Fused memory cross-attention read (`memory.py:MemoryRead`)
Tokens cross-attending to memory slots. At S=1024 slots, seq=2048, this is a real attention.
**Use FlexAttention with a custom `score_mod`** (already planned in training-speedups.md §1)
rather than a custom kernel — FlexAttention handles the tiling.

### Tier 3: FP4/FP8 matmul (beyond TE)

#### NVFP4 via TE (the paper's path)
- For Flower at 350–600M: FP4 linear layers give ~2× FLOPS on the 5090
- **Caveat**: The paper's methodology (RHT, 2D scaling, stochastic rounding, keep-last-15%-BF16)
  is needed for stability at 10T tokens. At Flower's scale (shorter runs, smaller model),
  the full recipe may be overkill — MXFP4 or simpler NVFP4 may suffice for experimentation.
- **Risk**: FP4 training stability for memory-augmented architectures is **unstudied**.
  Bloom routing softmax + FP4 could interact badly. Recommend BF16 for memory mechanism
  comparison runs, FP4 only for final scaling runs.

#### Flash-Sparse-Attention (HKUSTDial)
- Sparse + gated + local-window attention with Flex Local Window (per-head window sizes)
- Supports backward, varlen, GQA, fused quant
- **Relevant if** Flower needs per-head sliding windows or sparse softmax thresholds
- The "gated attention" variant maps to Flower's surprise gates

---

## 6. Prioritized recommendations for Flower (RTX 5090, sm_120)

Ranked by (expected speedup × ease × risk-reversibility):

| Priority | Kernel | Source | Effort | Expected gain | Risk |
|---|---|---|---|---|---|
| **P0** | Turbo-Muon NS Triton kernels | `tboissin/newton_schulz_triton` | 1 day | 2–3× on optimizer step | Low (drop-in) |
| **P0** | Liger FusedLinearCrossEntropy | Liger-Kernel | 1 day | Major memory save on head | Low |
| **P1** | Liger FusedRMSNorm | Liger-Kernel | hours | ~7× on norm | Low |
| **P1** | Liger FusedSwiGLU | Liger-Kernel | hours | ~1.6× memory | Low |
| **P1** | Fused AdamW (for embedding/head params) | triton-llm-kernels | 1 day | ~3× on Adam step | Low |
| **P2** | Fused Bloom hash routing | **Custom** (write it) | 3–5 days | 2–4× on routing | Medium |
| **P2** | TE NVFP4 linear layers | Transformer Engine | 2–3 days | ~2× FLOPS | **High** (stability unstudied for memory nets) |
| **P3** | Fused perceiver summary + scatter write | **Custom** | 1 week | 1.5–2× | Medium |
| **P3** | Flash-Sparse-Attention for memory read | HKUSTDial | 2 days | Context-dependent | Low |

---

## 7. Bottom line for the parent question

> *"Does arXiv:2509.25149 propose kernel-level training optimizations?"*

**No.** It's a numerical recipe paper that consumes existing TE kernels. Its value to Flower is:
1. **Confirmation** that NVFP4 training is viable on Blackwell (relevant if Flower goes FP4)
2. **A recipe** (2D scaling + RHT on Wgrad + stochastic rounding + mixed precision) to follow
   *if* FP4 is adopted
3. **No reusable kernel code** — all kernels are in Transformer Engine

The **real kernel opportunities for Flower** are in the adjacent ecosystem (Liger-Kernel,
Turbo-Muon, fused AdamW) and in **custom kernels for Flower's unique memory mechanisms**
(Bloom routing fusion, perceiver summary fusion). Those are where a kernel-experienced
developer gets 2–4× returns beyond what `torch.compile + FlashAttention` already provide.
