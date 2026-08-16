# Binary & Ternary Weight Training for LLMs (From Scratch) — State of the Art, August 2026

> **Scope:** Training transformers from scratch (100M–1B) on a single RTX 5090.
> Covers native ternary/binary training, FP4/low-bit alternatives, kernel support,
> and the optimizer question (Muon vs ternary). Focus on 2024–2026 publications.

---

## TL;DR / Executive Summary

- **Ternary (1.58-bit) native training is mature and works.** BitNet b1.58 and
  TriLM prove ternary models match FP16 perplexity at **≥3B params**, and the gap
  narrows with scale. At 100M–600M, ternary still incurs a measurable perplexity gap
  vs FP16, but downstream benchmarks are surprisingly competitive.
- **The training mechanism is simple:** maintain FP16 "latent/master" weights, ternarize
  on-the-fly in the forward pass (absmean scaling → round to {-1,0,1}), backpropagate
  via the **straight-through estimator (STE)**. Standard AdamW works. No special
  optimizer needed.
- **Muon is a real problem for ternary.** Muon does Newton-Schulz orthogonalization of
  the gradient momentum → produces a dense FP update matrix. Applying this to update
  latent weights that are immediately ternarized is mathematically fine (the latent
  weights are dense floats), but the *philosophy* of Muon (spectral control of updates)
  interacts oddly with the extreme quantization. Recent work (MuonQ, 4-bit-Muon-GRASP)
  shows Muon's optimizer state can be quantized, but nobody has published "Muon + ternary
  weights from scratch." This is an open research question — and potentially novel.
- **No fast GPU ternary training kernels exist.** All fast ternary GEMM kernels
  (T-MAC, bitnet.cpp ELUT, TriRun) target **inference** (decode, batch ≤ ~32). During
  training, the forward pass is simulated: dequant ternary weights to FP16 → standard
  matmul → re-quantize. This means **no training speedup** from ternary weights today
  on GPUs. The win is **memory** (weights stored as 2-bit), not speed.
- **FP4 (NVFP4/MXFP4) is the better path for training speedup on Blackwell.** Native
  FP4 tensor cores exist on RTX 5090 (sm_120), delivering ~3× BF16 throughput. Several
  2025 papers (Quartet, Metis, TetraJet-v2, FP4-All-the-Way) show near-lossless FP4
  training. This is where the hardware + software momentum is.
- **Recommendation for 100M–600M on RTX 5090:** Ternary is a valid *research* direction
  (especially Muon compatibility). For *practical* speed, FP4 training via Quartet-style
  MXFP4 kernels is the better bet. A hybrid (ternary weights + FP8 activations + FP4
  training kernels) is an interesting unexplored middle ground.

---

## 1. BITNET / TERNARY TRAINING FAMILY

### 1.1 BitNet b1.58 — The Foundation
**Sources:**
- Wang et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv:2402.17764 (Feb 2024)
- Full JMLR version: Wang et al., "BitNet: 1-bit Pre-training for Large Language Models," JMLR vol. 26, 24-2050 (2025) — [jmlr.org/beta/papers/v26/24-2050.html](https://jmlr.org/beta/papers/v26/24-2050.html)

**Core mechanism:**
- Replace every `nn.Linear` with `BitLinear`.
- **Weight quantization (absmean):** `γ = mean(|W|)`, then `W_q = RoundClip(W/γ + ε, -1, 1)` → ternary {-1, 0, +1}
- **Activation quantization (absmax):** LayerNorm → absmax quantize to INT8 (256 levels)
- **Forward:** `y = (x_int8 × W_ternary) × (xscale × wscale)` — multiplication becomes add/subtract/skip
- **Training:** Latent FP16 weights maintained. STE for backward pass. AdamW optimizer.
- SubLN (parameter-free LayerNorm) before quantization.

**Quality vs dense (key results):**
- **≥3B params:** Ternary matches FP16 in perplexity AND downstream tasks (zero-shot)
- **<3B:** Ternary is competitive but slightly behind FP16 (gap narrows with scale)
- BitNet b1.58 3B trained on 2T tokens **beats** StableLM-3B (FP16) on all end tasks
- Scaling law: validation loss gap between ternary and FP16 decreases from 0.5 (125M) to 0.09 (100B)

**Hyperparameters (from JMLR appendix, 700M–3.9B models):**
- Adam β = (0.9, 0.95)
- LR: 3e-3 → 1e-3 (cosine), warmup 100K steps
- Batch size: 512 sequences (1M tokens), seq len 2048
- Weight decay: phased schedule

**Key insight:** The ternary representation is learned *during* pretraining, not applied
after. The 0 weight value provides "explicit feature filtering" — pruning by design.
At inference, weights pack 4 ternary values per INT8 (1.58 bits/weight effective).

**Fast kernels?** Inference only (bitnet.cpp, T-MAC). Training = simulated quantization.

---

### 1.2 BitNet b1.58-2B4T — First Open Native 1-bit at Scale
**Sources:**
- "BitNet b1.58 2B4T Technical Report," arXiv:2504.12285 (Apr 2025)
- [huggingface.co/microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

**Key facts:**
- **2B parameters, trained on 4T tokens** — entirely from scratch
- Architecture: standard Transformer with `BitLinear` replacing all linears
- W1.58A8: ternary weights + INT8 activations (absmax, per-token)
- **Three training stages:** Pre-training → SFT → DPO (no RL/PPO/GRPO)
- Two-stage LR schedule during pre-training
- Tokenizer: LLaMA 3 (vocab 128,256)

**Quality (vs FP16 peers at similar size):**
| Benchmark | BitNet 2B | Llama 3.2 1B | Gemma-3 1B | Qwen2.5 1.5B |
|---|---|---|---|---|
| ARC-Challenge | 49.91 | 37.80 | 38.40 | 46.67 |
| Memory (non-emb) | **0.4 GB** | 2 GB | 1.4 GB | 2.6 GB |
| Energy/token | **0.028 J** | 0.258 J | 0.186 J | 0.347 J |

- BitNet 2B uses **10× less memory** and **9× less energy** than FP16 peers

**Inference kernels (open-sourced):**
- Custom CUDA W1.58A8 kernel: pack 4 ternary values → 1 INT8, unpack in SRAM before matmul
- bitnet.cpp for CPU (Ladder framework)
- GGUF format for llama.cpp

**Key insight:** First proof that native ternary at 2B+ scale matches leading FP16 models
of similar size. The released BF16 "master weights" allow continued training.

---

### 1.3 TriLM / Spectra Suite — Open Scaling Study (99M–3.9B)
**Sources:**
- Kaushal et al., "Surprising Effectiveness of Pretraining Ternary Language Models at Scale," ICLR 2025 — [openreview.net/forum?id=TJo6aQb7mK](https://openreview.net/forum?id=TJo6aQb7mK)
- [github.com/NolanoOrg/SpectraSuite](https://github.com/NolanoOrg/SpectraSuite) — 54 models released
- [blog.nolano.ai/Spectra-suite/](https://blog.nolano.ai/Spectra-suite/)

**What it is:** First systematic study of ternary LLMs across scales.
- 9 model sizes: 99M, 190M, 390M, 560M, 830M, 1.1B, 1.5B, 2.4B, 3.9B
- All trained on **same 300B tokens** (SlimPajama) — overtrained regime
- 54 models total (FP16, ternary, + quantized 3/4/6/8-bit versions)

**Training mechanism (identical to BitNet, with refinements):**
- FP16 latent/master weights
- Ternarize on-the-fly in forward pass: scale = absmean, round to {-1,0,1}
- STE for backward pass
- Embeddings + LM head stay FP16 (not ternarized)
- RMSNorm (with parameter, like LLaMA), SwiGLU, RoPE, no bias

**Critical training schedule findings:**
1. **Halve the peak LR at ~50% of training** → sharp loss drop
2. **Remove weight decay at ~66% of training** (ternarization provides sufficient regularization)
- Without both interventions, smaller TriLMs (<1B) start to *diverge* after the halfway mark

**Scaling laws (fitted power law, same exponent α=0.26 for both):**
```
TriLM:   L(N) = 185 × N^(-0.26) + 1.76
FloatLM: L(N) = 159 × N^(-0.26) + 1.67
```
- Same scaling *rate*, but TriLM has higher offset (ε = 1.76 vs 1.67)
- Gap closes with scale; at 3.9B, TriLM matches FloatLM on most downstream tasks
- **Key: at >1B params, TriLM consistently beats QuantLM (post-hoc) and FloatLM of same *bit-size***

**Quality at relevant scales (downstream, not perplexity):**
- TriLM 3.9B matches FloatLM 3.9B on commonsense reasoning + knowledge, despite being **5.9× smaller in bits**
- TriLM 3.9B validation perplexity matches FloatLM 1.1B
- Perplexity gap persists on web corpora (Dolma, RefinedWeb) but not on clean datasets (PTB, LAMBADA)
- **Toxicity concern:** TriLM 3.9B is as toxic as FloatLM 3.9B (larger effective capacity)

**Key insight:** This is the most directly relevant paper for 100M–600M work. It shows
ternary *works* at these scales but with a perplexity gap that closes with scale. The
training schedule (LR drop + weight decay removal) is critical and non-obvious.

---

### 1.4 TriTera — Scaling Ternary to 1.2T Tokens
**Source:** Vaidhya et al., "Scaling Laws and Efficient Inference for Ternary Language Models," ACL 2025 — [aclanthology.org/2025.acl-long.1294](https://aclanthology.org/2025.acl-long.1294/)

**Key contributions:**
- TriTera suite: TriLMs trained on up to **1.2T tokens** — sustained gains at scale
- Finding: **TriLMs benefit more from data scaling than parameter scaling**
- **TriRun GPU kernel:** 2-bit packing for ternary weights → up to **5× faster** inference vs FP16
  - Uses FP16 × INT2 matmul with tensor core mma instructions
  - Dequantizes INT2 weights via bit masks + lookup in registers
  - Best speedup at batch sizes ≤ 32 (memory-bound regime)
- 1.6-bit (TQ1) and 2-bit (TQ2) packing schemes for CPU inference
- Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-5), cosine LR, weight decay 0.1, gradient clip 1.0
- Single LR with warmup (replacing Spectra's dual-LR approach)

**Key insight:** Confirms ternary scales with data. The TriRun kernel is inference-only
but shows the GPU approach (FP16 × INT2 via tensor cores) is viable and fast.

---

### 1.5 Other Ternary Training Work

**"When to transition from 16-bit to 1.58-bit pre-training"** (ACL Findings 2025):
- [aclanthology.org/2025.findings-acl.694](https://aclanthology.org/2025.findings-acl.694.pdf)
- **Surprising finding:** Training FP16 first, then switching to 1.58-bit QAT later,
  produces **better** final 1.58-bit models than training 1.58-bit from scratch
- There exists an optimal transition point t* during training
- Investigates optimizer state importance and gradual quantization phasing

**"When Are 1.58 Bits Enough?"** (scitepress 2025):
- [scitepress.org/Papers/2025/133824](https://www.scitepress.org/Papers/2025/133824/133824.pdf)
- Bottom-up exploration: ternary works for MLPs, GNNs, not just transformers
- Median quantization ≥ mean quantization in most cases
- "BitNet scaling law" holds for encoder-only (hidden ×2 to match FP16) but NOT encoder-decoder

**Signed-Zero Ternary (SZT)** (arXiv:2508.05905, Aug 2025):
- Uses the **4th unused state** in 2-bit ternary encoding to encode sign of sub-threshold weights
- Provides gradient signal in the ternary "dead zone" (|w| ≤ Δ where gradient vanishes)
- Deterministic (no stochastic rounding needed)
- Same storage cost (2 bits/weight), better convergence properties
- **Relevant for from-scratch training** — addresses a real STE weakness

---

## 2. FERMION RESEARCH / NEUTRINO-1

**Sources:**
- [fermionresearch.com/research/neutrino-8b/](https://www.fermionresearch.com/research/neutrino-8b/) (primary)
- [huggingface.co/FermionResearch/Neutrino-8B](https://huggingface.co/FermionResearch/Neutrino-8B)
- [pypi.org/project/fermion-research/](https://pypi.org/project/fermion-research/)
- Third-party: [explainx.ai/blog/fermion-neutrino-1-8b-ternary-july-2026](https://explainx.ai/blog/fermion-neutrino-1-8b-ternary-july-2026)

### What it is
- **Neutrino-1 8B**: 8.19B params, shipped July 27, 2026
- Derived from **Qwen3-8B** (Apache 2.0 base) via **ternary QAT + staged post-training**
- Ships in proprietary "TRTC v4" container: **3.88 GB on disk, 2.56 GB download**
- Three models: 8B (capable), 0.6B (draft for speculative decode), 0.6B-Chat

### Selective ternarization (key architectural choice)
- **252 transformer projection matrices** (7 per layer × 36 layers): ALL ternary
- **Embeddings: int8** (untied, 622 MB each) — NOT ternary
- **Normalization vectors: FP32** — NOT ternary
- **LM head: FP16/FP32** — NOT ternary
- Result: 6.95B ternary weights (2.60 GB) + 1.24 GB int8 embeddings

### Ternary weight statistics
- **62.6% exactly zero**, 18.7% positive, 18.7% negative
- Signs balanced to within 0.02% across 6.95B weights (not enforced by training)
- Early feed-forward layers silence hardest: down-proj layer 2 = 72.5% zeros
- Attention projections maintain flat code density across all depths (~1% band)

### Native format training ("the rounding cliff")
> "Neutrino-1 8B was trained natively in its shipping format. There is no full-precision
> product model that was rounded afterward: the ternary representation is the medium the
> weights learned in."

**The rounding cliff concept (from companion research post "Intelligence at one-eighth the bits"):**
- Native-format training holds frontier capability in ternary at 1/8 the bits of FP16
- **Rounding a trained FP16 model to the same depth lands near chance**
- The training methods are "the lab's unpublished work"

### "Behavioral axis conservation"
From the training section:
- Training pressure on one behavioral axis (e.g., tool calling) **does not degrade the model uniformly** — it degrades *specific other axes*
- Which axes degrade is an empirical property of each training diet
- Example: one tool-calling stage moved tool-use score +20pts while instruction-following bled -9pts
- **Protection is axis-specific:** folding an axis's own training signal into a stage's batches held that axis through the stage (and ONLY that axis)
- Dose-response differs per axis: tool-calling gains complete in ~25 steps, instruction-following needs ~50, knowledge peaks mid-course then decays
- Every stage graded against ALL installed axes; stages accepted only when all axes held above preset floors

### Quality
| Metric | Neutrino-1 8B | Gemma-4-E4B | Llama-3.1-8B |
|---|---|---|---|
| MMLU (5-shot) | **72.1** | 70.57 | 68.3 |
| IFEval (prompt-strict) | 77.2 | 88.26 | 80.4 |
| BFCL v3 | 68.9 | — | 76.1 |
| Download | **2.56 GB** | 16.02 GB | 16.06 GB |

- MMLU 72.1 is notable — beats FP16 Llama-3.1-8B and Gemma-4-E4B
- GSM8K 53.4 (flexible) / 51.73 (strict), zero-shot no CoT

### Serving / engine
- Three runtimes: native C binary (macOS arm64, Linux x86-64), GGUF (llama.cpp fork), MLX (Metal)
- Speculative decode with 0.6B draft: up to 763 tok/s on H100
- CPU decode: 24.9 tok/s on Apple M5
- Native runtimes are macOS/Linux only; Windows and CUDA use torch reference path (1-2 orders of magnitude slower)

### Caveats / skepticism
- Training is **QAT from an existing FP16 model** (Qwen3-8B), NOT from scratch — despite "native format" language
- The "proprietary native ternary format" may be incremental over BitNet-style recipes
- "Native ternary" claim is soft — the explainx.ai analysis notes the prose looked AI-generated
- Format is proprietary/packed; tooling lag vs standard formats
- Independent replication pending
- PrismML Ternary-Bonsai-8B is a competitor that may outperform on some metrics

### Key insight for this project
The selective ternarization pattern (linears ternary, embeddings/head int8/FP16, norms FP32)
is the consensus architecture across ALL ternary work — not just Fermion. The "behavioral axis"
training methodology is novel but specific to post-training alignment, less relevant for from-scratch pretraining.

---

## 3. CAT-Q (Intel, ICML 2026 Spotlight)

**Sources:**
- "CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs," arXiv:2606.26650
- [icml.cc/virtual/2026/oral/71111](https://icml.cc/virtual/2026/oral/71111) (Oral, top 2.2%)
- Code: [github.com/IntelChina-AI/BitTern](https://github.com/IntelChina-AI/BitTern)
- Author page: [jwfandl.github.io](https://jwfandl.github.io/)

### What it is
**Post-training** ternary quantization (NOT from-scratch training) that matches QAT results
using only **512 calibration samples** (~1M tokens) instead of 100B+ training tokens.

### How it works (two components)
1. **Learnable Modulation (LM):** Three learnable factors modulate the distribution of
   pre-trained weights AND the ternary threshold
   - Suppresses outliers
   - Makes FP weights less sensitive to ternarization
   - Reduces distributional misalignment between ternary and FP weights

2. **Softened Ternarization (ST):** Novel differentiable transition function
   - Two-stage relay: differentiable ternarization → hard ternarization
   - Guides convergence (vs. always-hard ternarization which doesn't converge well in PTQ)

### Results
- **Matches BitNet 1.58-bit v1/v2** (trained on 100B tokens) with 512 samples
- **~100,000× reduction in training tokens** vs QAT
- Scales to **235B parameters** (first ternary PTQ at this scale)
- 8 to 60 hours on 8× A100-80GB
- Works on dense AND MoE architectures
- Ternary weights + 8-bit activations (W1.58A8) perform comparably to ternary-weight-only

### Why this matters for the project
- CAT-Q proves the **"rounding cliff" can be narrowed** with smart optimization
- If you train FP16 from scratch then apply CAT-Q, you might match native ternary quality
- But native ternary training (BitNet/TriLM) is still simpler for from-scratch work
- The two-stage relay (differentiable → hard) is a useful technique that could improve STE-based training too

---

## 4. PRACTICAL CONSIDERATIONS FOR FROM-SCRATCH TRAINING

### 4.1 What optimizer works with ternary weights?

**The key realization: you don't optimize the ternary weights directly.**
You maintain **FP16 latent/master weights** and ternarize them on-the-fly. The optimizer
operates on the FP16 latent weights, which are dense and continuous. Any standard optimizer works.

**Published choices:**
| Paper | Optimizer | Notes |
|---|---|---|
| BitNet b1.58 | AdamW (β=0.9, 0.95) | Standard |
| TriLM/Spectra | AdamW (β=0.9, 0.95, ε=1e-5) | + LR drop at 50%, weight decay removal at 66% |
| TriTera | AdamW (β=0.9, 0.95) | Single LR cosine schedule |
| BitNet 2B4T | AdamW (implied) | Pre-train → SFT → DPO |
| FBI-LLM (binary) | Adam (β=0.9, 0.98) | + autoregressive distillation |
| Fermion Neutrino | "Unpublished" | QAT from Qwen3-8B |

**Muon specifically:**
- Muon operates on 2D weight matrices: maintains gradient momentum → Newton-Schulz orthogonalization → polar factor update
- The polar factor is a **dense FP matrix** — it updates the FP16 latent weights, not the ternary weights
- **Mathematically, Muon should work** with ternary latent weights (the STE passes gradients to latent weights regardless of optimizer)
- BUT: the spectral philosophy of Muon (controlling singular value distribution of updates) interacts oddly with extreme quantization — the ternarization destroys the spectral structure Muon preserves
- **Nobody has published Muon + ternary weights.** This is an open research question.
- Related: Full-Stack FP4 (arXiv:2607.04422) shows Newton-Schulz can run in NVFP4 — and MuonQ/Muon-GRASP quantize Muon's optimizer state to 4-bit successfully
- **Verdict:** Try it. It might work fine (the latent weights are dense). It might also reveal interesting interactions. Potentially novel.

### 4.2 Backpropagation through sign/round — the STE

**The Straight-Through Estimator (STE) is the universal answer.**
- Forward: quantize latent weights to {-1,0,1} (or {-1,+1} for binary)
- Backward: pretend quantization didn't happen; pass gradient through unchanged
- Mathematically: `∂L/∂W_latent ≈ ∂L/∂W_quant` (identity substitution for the quantizer Jacobian)

**Key paper insight (arXiv:2405.05171, "Custom Gradient Estimators are STE in Disguise"):**
- Proves that for **Adam** (and other adaptive optimizers), ALL nonzero gradient estimators
  are approximately equivalent to STE — no need for fancy gradient estimators
- For SGD, equivalence holds after adjusting LR + weight init
- **Practical conclusion: just use STE. Don't overthink the gradient estimator.**

**LOTION (arXiv, 2025) — alternative to STE:**
- Smooths the *loss itself* via stochastic rounding noise expectation
- Differentiable almost everywhere → any standard optimizer with convergence guarantees
- Equivalent to curvature-aware regularization (diagonal Gauss-Newton)
- Preserves all global minima of the quantized problem
- Pretrained 150M and 300M models at INT4/INT8/FP4 — lower loss than QAT/PTQ baselines

### 4.3 Batch size and learning rate

From published ternary training:
- **Batch size:** 512 sequences (~1M tokens) at scale; for 100M–600M on single GPU, micro-batch + gradient accumulation
- **LR:** Peak ~3e-3 for ~700M models, cosine decay
- **Critical schedule (TriLM):** Halve LR at 50%, remove weight decay at 66%
- **Warmup:** ~100K steps for large models; shorter for small models
- **Gradient clipping:** 1.0 (standard)
- The LR needs to be high enough that latent weights can actually cross ternary thresholds
- **FBI-LLM (binary):** Uses LR 3e-4, cosine to 3e-5, warmup 2000 steps, gradient clip 1.0

### 4.4 Does ternary training speed up the MATMUL?

**No, not during training on current GPUs.**

During training:
1. Forward pass: ternarize FP16 latent weights → {-1,0,1} → **dequantize back to FP16 for matmul** → standard FP16/TF32 GEMM
2. This is "simulated quantization" — no actual speedup, slight overhead from quantization step
3. The ternary weights are never stored in packed form during training (they're recomputed each forward pass)

Fast ternary GEMM kernels exist but are **inference-only:**
- T-MAC (CPU/NPU, LUT-based): up to 6.6× vs llama.cpp
- bitnet.cpp (CPU/GPU): ELUT kernels, pack-and-unpack for lossless INT8 activation
- TriRun (GPU): FP16 × INT2 via tensor cores, up to 5× vs FP16 at batch ≤ 32

**Why no training kernels?** The backward pass needs FP gradients anyway. And the training
batch sizes are large enough that the GPU becomes compute-bound (not memory-bound), eliminating
the bandwidth advantage of packed ternary weights.

**Exception:** A custom Triton kernel could theoretically do ternary forward + FP backward
faster, but nobody has published this for training. The ternary-engine project (github) shows
1.34× forward-only speedup with C++ GEMM, backward still in PyTorch.

### 4.5 Memory savings during training

**Weights:** 1.58 bits vs 16 bits = **~10× reduction** in weight storage
- 600M params: 1.2 GB (FP16) → 0.12 GB (ternary) — but latent weights still need FP16 during training!

**The catch: latent weights.** During training, you store BOTH:
- FP16 latent weights: full size (e.g., 1.2 GB for 600M)
- Ternary quantized weights: recomputed each step (not stored persistently)
- Optimizer state (AdamW): 2× weight size in FP32 = 4.8 GB for 600M
- Gradients: 1× weight size in FP16 = 1.2 GB

**Actual training memory for ternary 600M:**
- Latent weights (FP16): 1.2 GB
- AdamW states (2× FP32): 4.8 GB
- Gradients (FP16): 1.2 GB
- Activations: depends on batch/seq len
- **Total optimizer+weights+grads: ~7.2 GB** (same as FP16 training!)

**Where ternary saves memory:** Only the final checkpoint (drop latent weights, keep ternary).
During training, the savings come from the optimizer, not the weights.

**To actually save training memory, you need:**
- 8-bit optimizer (bitsandbytes): halves optimizer state → ~2.4 GB saved
- Muon (only first moment, no second moment): ~50% optimizer state reduction vs AdamW
- Gradient checkpointing: trades compute for activation memory

### 4.6 Quality impact: ternary vs dense at same param count

| Scale | Perplexity Gap | Downstream Gap | Notes |
|---|---|---|---|
| 100M | Significant | Moderate | TriLM still learns; gap visible |
| 300M–600M | Moderate | Small | Competitive on some benchmarks |
| 1B+ | Small | Negligible | TriLM ≈ FloatLM downstream |
| 3B+ | ~0 | ~0 | Matches on most benchmarks |
| 3.9B (TriLM) | Small (web corpora) | ~0 (clean tasks) | Matches FloatLM 3.9B |

**Key nuance:** The gap is in *perplexity*, not always in *downstream tasks*.
TriLM 3.9B matches FloatLM 3.9B on commonsense reasoning despite higher perplexity.
At 100M–600M, expect a perplexity gap but potentially acceptable downstream performance.

---

## 5. ALTERNATIVE LOW-BIT FORMATS

### 5.1 Binary (1-bit, {-1,+1}) weights

**BitNet b1 (original):** Binary weights {-1,+1}, zero-mean centralization before binarization.
- Gap to FP16 narrows from 0.5 (125M) to 0.09 (100B) in validation loss
- Requires larger models than b1.58 to match FP16

**FBI-LLM (arXiv:2407.07093):** Fully binarized LLM from scratch (130M, 1.3B, 7B)
- **Key technique: Autoregressive Distillation (AD)** — distill from FP16 teacher
- Matches teacher probabilities at each token location
- W1A16 (binary weights, FP16 activations)
- Finding: pretrained FP16 weights are NOT necessary for binary from-scratch training
- Adam (β=0.9, 0.98), LR 3e-4, cosine decay

**QuEST (arXiv:2502.05003):** Stable training down to **1-bit weights AND activations**
- Hadamard normalization + MSE-optimal quantization fitting
- "Trust gradient estimator" — minimizes error between quantized and true gradients
- INT4 is Pareto-optimal; 2-bit weights Pareto-dominant in weight-only setting
- 1-bit is "surprisingly competitive with 3-bit weights"

**kobimusic/bitnet-1bitllm (HF):** Practical strict 1-bit (±1) LM
- 75M trained on 15B tokens (FineWeb-Edu), val BPC 6.16
- 300M in-flight on single RTX 5090 (~7 days, 24K tok/s)
- Gumbel-hard pointer attention (each query attends to exactly one key)
- Weights: ±1 via sign-STE on latent float, per-channel float scale
- C inference kernel with XNOR-popcount + AVX512

**Key insight:** Binary (1-bit) is harder than ternary (1.58-bit) but works with distillation.
For from-scratch without a teacher, ternary is the safer choice. The zero in ternary is crucial
for feature filtering.

### 5.2 INT4/INT8 weight training

- INT4 QAT is well-established; Kumar et al. (2024) identified 8-bit as Pareto-optimal for QAT
- QuEST pushes Pareto-optimal to **4-bit weights + activations**
- INT8 training is trivially supported on all modern GPUs (INT8 tensor cores)

### 5.3 FP4 weight training (MXFP4 / NVFP4) — THE FRONTIER

This is where hardware and research momentum is concentrated in 2025-2026.

**Key formats:**
- **MXFP4:** E2M1 (4-bit float), block size 32, E8M0 scale per block
- **NVFP4:** E2M1, block size 16, E4M3 scale per block — NVIDIA's preferred format
- Both supported natively on Blackwell (sm_100/sm_120)

**FP4 All the Way (arXiv:2505.19115):**
- First fully quantized FP4 training (weights + activations + gradients)
- NVFP4 (block 16, E4M3 scale) is optimal
- Stochastic rounding for backward/update, round-to-nearest for forward
- **Critical threshold:** when gradient norm < √3 × quantization noise, training stalls
- Llama2-7B on 256 Intel Gaudi2 accelerators, 200B tokens
- Short QAF phase (FP4 forward, BF16 backward) closes the gap to BF16 completely

**Quartet (arXiv:2505.14669):**
- MXFP4 native training on **Blackwell RTX 5090**
- ~2× speedup vs FP8 for linear layers
- QuEST forward (Hadamard + MSE-optimal) + stochastic rounding backward
- Scaling law analysis: MXFP4 is "near-optimal" on accuracy-efficiency trade-off
- At fixed compute budget, MXFP4 accuracy loss is fully compensated by higher efficiency

**Metis (arXiv:2509.00404):**
- Spectral-domain FP4 quantization
- Identifies anisotropy in singular value spectra as fundamental barrier
- Partitions spectra into narrower sub-distributions
- LLaMA-3 8B, 100B tokens: **0.4% training loss gap, 0.1% downstream degradation** vs BF16
- Surpasses NVIDIA's (unpublished) FP4 recipe

**TetraJet-v2 (arXiv:2510.27527):**
- NVFP4 for weights + activations + gradients
- OsciReset (weight oscillation suppression) + OutControl (outlier accuracy)
- Reduces FP4-to-FP gap by 51.3% vs prior SOTA
- 70M–370M models, 50B–200B tokens

**Full-Stack FP4 (arXiv:2607.04422):**
- First end-to-end NVFP4 covering linear + optimizer + attention
- LoRA-SVD for linear layers (dilutes quantization noise)
- NVFP4-native Newton-Schulz for Muon optimizer (!)
- 3B/64B tokens: only 1.47% loss gap to BF16
- **Shows Muon can work in NVFP4** — relevant to the optimizer question

### 5.4 Mixed approaches (ternary weights + FP8 activations)

This is essentially the **BitNet W1.58A8** recipe and is the standard for ternary:
- Ternary {-1,0,1} weights
- INT8 (or FP8) activations via absmax/per-token quantization
- FP16 latent weights + STE during training
- This is proven and stable

### 5.5 Logarithmic Number System (LNS)

**LNS-Madam (arXiv:2106.13914, NVIDIA 2022):**
- Co-designed LNS + multiplicative weight update (Madam optimizer)
- 8-bit LNS matches FP32 accuracy on CV + NLP tasks
- 90% energy reduction vs FP32, 55% vs FP8
- Key: Madam optimizer updates base-2 exponents directly in log space
- No need for LNS-to-integer conversion during weight update

**Earlier LNS training (arXiv:1910.09876):**
- Full log-domain training with 16-bit fixed-point
- LUT-based approximate log-domain addition (eliminates multipliers)
- ~1% accuracy loss vs FP

**Relevance:** LNS is elegant but has **no GPU hardware support**. It's a custom-hardware
story. Not practical for RTX 5090 training.

### 5.6 Fixed-point training

- Well-studied historically; 16-bit fixed-point is sufficient for ~1% accuracy loss
- INT8 training is production-ready (NVIDIA, Google both support it)
- INT4 training requires techniques like QuEST/LOTION to be stable
- Less expressive than floating-point for the same bit budget (no dynamic range scaling)

---

## 6. KERNEL SUPPORT

### 6.1 Fast ternary GEMM kernels — ALL INFERENCE-ONLY

| Kernel | Platform | Method | Speedup | Training? |
|---|---|---|---|---|
| T-MAC (Microsoft) | CPU/NPU | LUT-based, bit-wise | Up to 6.6× vs llama.cpp | ❌ Inference |
| bitnet.cpp (Microsoft) | CPU/GPU | ELUT, pack-and-unpack | Lossless for W1.58A8 | ❌ Inference |
| TriRun (TriTera) | GPU (NVIDIA) | FP16×INT2, tensor cores | Up to 5× vs FP16 (batch≤32) | ❌ Inference |
| Tritium (Rust) | CPU/CUDA | Reference + backend trait | Conformance-gated | Has VJP trait method (training-ready) |
| ternary-engine | CPU (C++) | Dense243 packing | 1.34× (forward only) | Partial (backward in PyTorch) |
| BitNet CUDA kernel | GPU | Pack-store-load-unpack-compute | Custom for 2B4T | ❌ Inference |

**No published fast GPU ternary kernel for the training forward+backward pass.**
Training uses simulated quantization (dequant to FP16 → standard GEMM).

### 6.2 LUT-based approaches (T-MAC detail)

T-MAC's key idea:
1. Decompose n-bit weight matrix into n one-bit matrices
2. For groups of g bits, precompute all 2^g partial sums with activation → store in LUT
3. GEMM becomes table lookup + addition (no multiplication)
4. LUT resides in registers (TBL on ARM, PSHUF on x86)
5. Mirror consolidation + table quantization compress LUT to ¼ size
6. Scales linearly with bit-width (unlike dequant-based methods)

**Limitation:** CPU/NPU focused. No GPU implementation published.

### 6.3 MXFP4/NVFP4 in production training (Kimi K3, DeepSeek V4)

**Kimi K3 (Moonshot AI, July 2026):**
- 2.8T params (MoE, 16/896 experts), 1M context
- **QAT from SFT stage onward:** MXFP4 weights + MXFP8 activations
- NOT post-training quantization — model learns to compensate during training
- Per-Head Muon optimizer (each attention head optimized independently)
- "2.5× scaling efficiency improvement over Kimi K2"
- Native Blackwell + AMD MI400 support for MXFP4

**DeepSeek V4:**
- Routed experts stored in MXFP4 natively
- Converted to NVFP4 for Blackwell inference (lossless cast with `--cast_mxfp4_to_nvfp4`)
- Attention projections, router, shared experts, embeddings, LM head stay in original format

**Key insight from K3/V4:** QAT from SFT onward (not from pretraining) is the production
pattern. Pretraining in higher precision, then QAT during alignment.

### 6.4 What works on consumer Blackwell (RTX 5090, sm_120)?

**NVFP4 (E2M1) tensor cores: YES, natively supported**
- sm_120 has round-to-nearest E2M1 cast + FP4 tensor-core GEMM
- **~1100 TFLOPS** at 16384³ GEMM (~3× BF16, ~55% of dense FP4 peak)
- The gap to peak is "sm_120 cuBLAS maturity"

**BUT: Transformer Engine NVFP4 recipe CRASHES on sm_120**
- Issue [NVIDIA/TransformerEngine#3062](https://github.com/NVIDIA/TransformerEngine/issues/3062)
- TE's fused RHT/stochastic-rounding kernel requests ~232KB shared memory
- sm_120 opt-in cap is **101,376 bytes** (vs sm_100's ~232KB)
- `cudaFuncSetAttribute` → "invalid argument"
- Also: sm_120 lacks hardware `cvt.rs.*.e2m1` (stochastic rounding cast)

**Workarounds (from github.com/Infatoshi/nvfp4-sm120):**
1. **Own the quantization in software** (software SR + RHT in custom kernels)
2. Send quantized operands to native FP4 GEMM via `torch._scaled_mm` (BlockWise1x16 + SWIZZLE_32_4_4)
3. Through torchao's `_addmm_nvfp4_dispatch`
4. Results: **NVFP4 full recipe (SR+RHT) matches BF16** on 3.55M-param model (100% held-out acc vs 1.9% without SR/RHT)

**vLLM NVFP4 on sm_120:**
- Dense NVFP4 GEMMs: working (PR #21309, #41738)
- MoE NVFP4 GEMMs: working (PR #29242)
- KV cache NVFP4: in development (PR #46329)
- Build issue: must use `12.0a` or `12.0f` (not `12.0+PTX`) in `TORCH_CUDA_ARCH_LIST`

**Ternary on sm_120:** No native support. Ternary weights must be dequantized to FP16/INT8
for GEMM. The INT8 tensor core path (for W1.58A8 inference) may work but isn't optimized.

**Practical for RTX 5090 training:**
- **FP4 training:** ✅ Works with custom kernels (nvfp4-sm120 repo, Quartet)
- **FP8 training:** ✅ Works (cuBLAS, TE with workarounds)
- **INT8 training:** ✅ Works (tensor cores)
- **Ternary training:** ✅ Works but no speed benefit (simulated quantization)
- **Binary training:** ✅ Works but no speed benefit (simulated quantization)

---

## 7. SYNTHESIS: Recommendations for 100M–600M on RTX 5090

### Option A: Ternary (BitNet/TriLM-style) — Research-focused
**Pros:**
- Proven at scale, simple implementation (replace Linear with BitLinear)
- Interesting research questions (Muon compatibility, training dynamics)
- Tiny checkpoints (0.12 GB for 600M ternary vs 1.2 GB FP16)
- Clear precedent: TriLM 99M–390M models are directly comparable

**Cons:**
- No training speedup (simulated quantization)
- No memory savings during training (latent weights + optimizer state dominate)
- Perplexity gap at small scales (100M–600M)
- Need custom training schedule (LR drop + weight decay removal per TriLM)

**Implementation:**
```python
# BitLinear layer (simplified)
class BitLinear(nn.Linear):
    def forward(self, x):
        # Weight quantization (absmean → ternary)
        w_scale = self.weight.abs().mean()
        w_q = torch.round(self.weight / (w_scale + eps)).clamp(-1, 1)
        # Activation quantization (absmax → INT8)
        x_norm = RMSNorm(x)
        x_scale = x_norm.abs().max(dim=-1, keepdim=True).values / 127
        x_q = torch.round(x_norm / (x_scale + eps)).clamp(-128, 127)
        # Matmul + dequantize
        return F.linear(x_q, w_q) * x_scale * w_scale
    # STE is automatic via autograd (round has zero grad, but we don't need it
    # because gradients flow to self.weight which is the latent FP weight)
```

### Option B: FP4 (NVFP4/MXFP4) — Speed-focused
**Pros:**
- **~3× speedup** vs BF16 on RTX 5090 (native FP4 tensor cores)
- Near-lossless quality (0.4% loss gap per Metis)
- Active research area with working kernels (Quartet, nvfp4-sm120)
- Production-validated (Kimi K3, DeepSeek V4)

**Cons:**
- More complex implementation (Hadamard transforms, stochastic rounding)
- TransformerEngine broken on sm_120 (need custom kernels)
- Less memory savings than ternary for weights (4 bits vs 1.58 bits)
- Newer, less battle-tested at small scales

### Option C: Hybrid (ternary weights + FP4 training kernels)
**Unexplored territory.** Could combine:
- Ternary weights for minimal weight storage
- FP4/INT8 activations (as in BitNet W1.58A8)
- Fast FP4 tensor core GEMMs with on-the-fly ternary→FP4 dequant
- Potentially novel and publishable

### Option D: Dense FP16/BF16 with Muon — Baseline
- The current approach. Well-understood, fast convergence with Muon.
- Use as the baseline to compare against.

### The Muon question
- **Muon + ternary latent weights:** Should work (latent weights are dense FP).
  The Newton-Schulz operates on gradient momentum, not the weights themselves.
  STE passes gradients to latent weights. **Worth trying.**
- **Potential issue:** Muon's spectral updates may fight the ternarization (preserving
  singular structure that ternarization destroys). This interaction is unstudied.
- **Alternative:** Use AdamW for ternary (proven), Muon for FP16 baseline.
- **Full-Stack FP4** shows Newton-Schulz can run in NVFP4, suggesting Muon is compatible
  with aggressive quantization in principle.

---

## Key Papers Reference Table

| Paper | Year | Venue | Precision | Scale | Key Result |
|---|---|---|---|---|---|
| BitNet b1.58 | 2024 | arXiv→JMLR | Ternary W, INT8 A | 125M–100B | Matches FP16 at ≥3B |
| BitNet 2B4T | 2025 | arXiv | Ternary W, INT8 A | 2B | Open native ternary, 4T tokens |
| TriLM/Spectra | 2025 | ICLR | Ternary W, INT8 A | 99M–3.9B | Scaling laws, 300B tokens |
| TriTera | 2025 | ACL | Ternary W | Up to 1.2T tokens | TriRun GPU kernel, 5× inference |
| FBI-LLM | 2024 | arXiv | Binary W | 130M–7B | Binary from scratch via distillation |
| QuEST | 2025 | arXiv | 1–8 bit W+A | Up to 1.6B | Pareto-optimal at 4-bit, stable at 1-bit |
| CAT-Q | 2026 | ICML (Oral) | Ternary PTQ | 1.7B–235B | 512 samples ≈ 100B token QAT |
| FP4 All the Way | 2025 | arXiv | NVFP4 W+A+G | 7B | First full FP4 training |
| Quartet | 2025 | arXiv | MXFP4 W+A+G | Up to 7B | ~2× vs FP8 on RTX 5090 |
| Metis | 2025 | arXiv | NVFP4 W+A+G | 8B | 0.4% loss gap vs BF16 |
| TetraJet-v2 | 2026 | arXiv | NVFP4 W+A+G | 70M–370M | 51.3% gap reduction |
| Full-Stack FP4 | 2026 | arXiv | NVFP4 full stack | 3B | NVFP4 Muon (!), 1.47% gap |
| LOTION | 2025 | arXiv | INT4/8/FP4 | 150M–300M | Loss smoothing vs STE |
| LNS-Madam | 2022 | IEEE TCAD | 8-bit LNS | CV+NLP | LNS + multiplicative update |
| Neutrino-1 8B | 2026 | (Fermion) | Ternary (proprietary) | 8B | Native ternary, MMLU 72.1 |
| 8-bit Muon | 2025 | arXiv | 8-bit optimizer | Up to 2.7B | Muon robust to quantization |
| MuonQ | 2026 | arXiv | 4-bit optimizer | GPT-2, LLaMA | 4-bit Muon, 7.3× state reduction |

---

## Appendix: Key Links

**Code/Models:**
- BitNet 2B4T: [huggingface.co/microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- Spectra Suite: [github.com/NolanoOrg/SpectraSuite](https://github.com/NolanoOrg/SpectraSuite)
- bitnet.cpp: [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet)
- T-MAC: [github.com/microsoft/T-MAC](https://github.com/microsoft/T-MAC)
- CAT-Q: [github.com/IntelChina-AI/BitTern](https://github.com/IntelChina-AI/BitTern)
- QuEST: [github.com/IST-DASLab/QuEST](https://github.com/IST-DASLab/QuEST)
- NVFP4 on sm_120: [github.com/Infatoshi/nvfp4-sm120](https://github.com/Infatoshi/nvfp4-sm120)
- FBI-LLM: [github.com/LiqunMa/FBI-LLM](https://github.com/LiqunMa/FBI-LLM)
- Neutrino-1: [huggingface.co/FermionResearch/Neutrino-8B](https://huggingface.co/FermionResearch/Neutrino-8B)
- Binary 1-bit LM (practical): [huggingface.co/kobimusic/bitnet-1bitllm](https://huggingface.co/kobimusic/bitnet-1bitllm)
- MuonQ: [github.com/YupengSu/MuonQ](https://github.com/YupengSu/MuonQ)
- 4-bit Muon-GRASP: [github.com/wuhuaijin/lowbit-Muon](https://github.com/wuhuaijin/lowbit-Muon)
