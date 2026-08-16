# Frontier Training Speedups — Deep Research Synthesis (August 2026)

A comprehensive survey of state-of-the-art techniques that could make Flower's
training faster, larger, and better. Covers optimizers, CUDA kernels, low-precision
training (FP4/ternary), attention patterns, architecture ideas, and data strategy.

**Hardware target:** RTX 5090 (Blackwell sm_120, 32GB), single GPU
**Model scale:** 100M–600M (goal: 1B+)
**Current stack:** Muon(quintic5, batched NS) + AdamW, FlexAttention, FP8 tensorwise,
torch.compile(max-autotune-no-cudagraphs), FFN-only checkpointing, FineWeb-Edu

**Companion reports (detailed per-area research, all in this repo):**
- `docs/OPTIMIZER_RESEARCH_2026.md` — optimizer deep dive
- `research/low_precision_kernel_research_2026-08.md` — CUDA/FP4/FP8 kernels
- `research/ternary-binary-training-2026.md` — ternary/binary weight training
- `reports/research_attention_mechanisms_2026-08.md` — attention patterns
- `reports/research_novel_architectures_2026-08.md` — architecture ideas
- `reports/research_training_data_strategy_2026-08.md` — data strategy

---

## EXECUTIVE SUMMARY: The 10 Highest-ROI Actions

Ranked by (expected impact × ease of implementation × relevance to Flower's research goals):

| # | Action | Category | Effort | Expected Gain |
|---|--------|----------|--------|---------------|
| 1 | **Contra-Muon** (1-line add to Muon.step) | Optimizer | Trivial | +0.5–1% loss, free |
| 2 | **Per-head Muon** for attention weights | Optimizer | Small | +0.5–1% loss, Kimi K3 validated |
| 3 | **Hybrid attention schedule** (6:1 window:full, full in middle) | Architecture | Small | Better long-context, every SOTA model does this |
| 4 | **Add attention sinks** (4 tokens, full attention) | Architecture | Trivial | Provably necessary, free win |
| 5 | **Add DCLM-Baseline** to data mix (50/50 with FineWeb-Edu) | Data | Low | Complementary quality signal (SmolLM2 validated) |
| 6 | **Mix in ~10–15% code** (Stack-Edu or StarCoderData) | Data | Low | Better reasoning (SmolLM2, OLMo 2) |
| 7 | **Pre-tokenize and cache to disk** (stop streaming+tokenizing) | Infrastructure | Medium | Eliminates the 17% throughput gap |
| 8 | **Muon²-F** (factored second moment before NS) | Optimizer | Small | -40% NS iterations, near-zero memory cost |
| 9 | **LoRA-Pre** momentum compression | Optimizer | Medium | Fit 1B+ models on 32GB (ICLR 2026 Oral) |
| 10 | **Shorter attention window** (w=512 or stochastic 128/2048) | Architecture | Trivial | Forces memory mechanisms to engage (SWAX finding) |

**The meta-finding:** At Flower's scale (100M–600M), optimizer and kernel work
are near their ceilings (67% MFU, Muon already adopted). The biggest remaining
wins are (a) fitting larger models via memory-efficient optimizers, (b) better
data and training schedules, and (c) architectural changes that make the memory
mechanism thesis testable (shorter windows, hybrid attention, additive routing).

---

## 1. OPTIMIZERS

### Current state in Flower
- Muon (quintic5, batched NS) for 2D weights, AdamW/CautiousAdamW for 1D/embedding
- Aurora implemented but optional
- NorMuon, Cautious WD, cubic5/hybrid_v4 schedules wired
- NS already batched (4 shape groups, zero singletons)

### The reality check (important context)
Against well-tuned AdamW, no optimizer exceeds ~1.4x speedup at 100M–1B scale,
and that advantage shrinks toward 1.1x at 1B+. Muon's gains are real but near
ceiling. LR tuning matters more than optimizer choice. Source: Wen et al.
arXiv:2509.02046, NVIDIA arXiv:2607.20548.

### Muon family: what's new

**Contra-Muon** (github.com/nilin/contra-muon) — TRIVIAL, DO NOW
- After NS update, subtract a fraction of operator-normalized momentum gradient
- One line: `ortho = ortho - 0.2 * (ortho / ortho.norm()) * momentum.norm()`
- Multiple Track-3 nanoGPT speedrun records (PR #275: 3225 steps, PR #301: 2995)
- Composable with NorMuon, Aurora, everything

**Per-head / Group Muon** (arXiv:2605.08933, Kimi K3 §2.5) — EASY
- Orthogonalize attention heads SEPARATELY instead of full QKV matrix
- Reshape momentum along head dim, run NS per head or per group (g=6 random)
- Equalizes update scale across heads (large heads no longer dominate)
- Kimi K3 uses this at 2.8T params; nanoGPT PR #253 saves ~10 steps

**Muon²-F** (arXiv:2604.09967) — EASY, CUTS NS COST
- Apply Adafactor-style factored second moment BEFORE Newton-Schulz
- Improves spectral conditioning of NS input (singular values shift 10x larger)
- **Reduces required NS iterations by 40%** (3-step Muon² beats 5-step Muon)
- Muon²-F variant: factored second moment = just 2 vectors, near-zero memory
- Directly relevant: cubic5 already cuts matmuls 15→10; Muon²-F could cut further

**OKLS / Online KL Shampoo** (Tilde Research, Jul 2026) — BIGGEST ALGORITHMIC GAIN
- **1.45x Muon's parameter efficiency** at 98% throughput
- Zero-staleness Kronecker-factored optimizer using Scaled CANS (Chebyshev NS)
- A 200M–1B OKLS model matches a Muon model ~1.5x larger
- Memory cost: momentum + two (m,m) + two (n,n) covariance factor copies in FP32
- For 600M this is significant but fits 32GB
- Code: github.com/tilde-research/online-kl-shampoo-release

**Compositional Muon** (Tilde Research, Jun 2026) — FOR ATTENTION
- Partner-whitened updates for QK^T and OV circuits (not individual matrices)
- The loss sees compositions (QK^T), not individual matrices
- Cheap "isotropic rule" approximation = partner-rescaled Muon, near-zero overhead
- nanoGPT Track-3 record: 2875 steps
- Code: github.com/tilde-research/comp-muon-release

**Newton-Muon** (arXiv:2604.01472) — PRINCIPLED
- Right-precondition gradient with inverse activation second moment before NS
- Muon implicitly assumes isotropic activations (they aren't)
- 6% fewer iterations, 4% wall-clock reduction, 1.8% per-step overhead
- Code: github.com/zhehangdu/Newton-Muon

**SODA wrapper** (arXiv:2605.11172) — ELIMINATES WD TUNING
- Wraps ANY base optimizer (including Muon) with zero new hyperparameters
- Theoretically-grounded 1/k weight decay schedule
- Consistently improves baselines even WITH tuned weight decay
- Code: github.com/tmpethick/soda_code

### Newton-Schulz iteration improvements

**Scaled CANS** (arXiv:2506.10935) — Chebyshev-optimal coefficients
- Remez-algorithm-computed coefficients replace fixed quintic5
- Same matmul count, better convergence per iteration
- The OKLS variant: 10 iters, 27 FP16 GEMMs, computes matrix INVERSE square root

**Polar Express** — Concurrent optimal-polynomial method (independently derived)
- Already in the current nanoGPT speedrun record stack

### Different paradigms

**LoRA-Pre** (arXiv:2602.24283, ICLR 2026 Oral) — KEY FOR FITTING BIGGER MODELS
- EMA momentum ≡ online linear regression → decompose m = m_B · m_A (rank r)
- Memory: p×q → (p+q)×r. **1/8 the rank of GaLore for same quality**
- Works for BOTH Adam AND Muon
- Directly addresses the "fit 1B on 32GB" goal
- Code: github.com/mrflogs/LoRA-Pre

**Scion** (EPFL LIONS, arXiv:2502.07529) — MOST MEMORY-EFFICIENT
- Only 1 set of weights + 1 gradient (half-precision). Zero-shot HP transfer.
- Code: github.com/LIONS-EPFL/scion

**SF-NorMuon** (arXiv:2605.23061) — SCHEDULE-FREE
- Schedule-free + spectral (NorMuon): no LR schedule needed
- Matches tuned AdamW across 1–8x Chinchilla WITHOUT a schedule
- 35–52% speedup over SF-AdamW. Enables anytime checkpointing.

### Bottom line on optimizers
Flower's Muon is near ceiling. The highest-ROI moves are:
1. Contra-Muon + Per-head Muon (near-free, do immediately)
2. Muon²-F to cut NS cost on Blackwell
3. LoRA-Pre to unlock 1B-scale models
4. OKLS is the only method offering a step-change (1.45x param efficiency), but
   heavy memory cost means it needs LoRA-Pre first for 600M+ on 32GB

---

## 2. CUDA / KERNEL / LOW-PRECISION

### SM120 reality: it's a chimera, not datacenter Blackwell

| Feature | Source | SM120 (5090) |
|---------|--------|--------------|
| MMA instructions | SM80 (Ampere) | warp-level mma.sync, NOT tcgen05/UMMA |
| TMA | SM90 (Hopper) | ✓ async bulk loads |
| FP4/FP8 block scaling | SM100 (B200) | ✓ mxf4nvf4.block_scale MMA |
| Tensor Memory (TMEM) | SM100 only | **DOES NOT EXIST** |
| Shared memory | Unique | **99 KB/SM** (vs 228 KB on B200) |
| Cluster multicast | SM100 | **Not supported** (1×1×1 only) |

Any kernel written for SM100 (DeepGEMM, tcgen05 paths, WGMMA Flash Attention)
will fail on SM120. FP4 IS hardware-supported — but via a completely different
instruction path than datacenter Blackwell.

### What Flower already measured (and what's still true)

| Technique | Result on 5090 | Status |
|-----------|----------------|--------|
| FP8 tensorwise (torchao) | 1.39x/block, 1.31x full-stack | ✓ Production |
| FP8 rowwise | 1.01x (pointless) | Confirmed dead |
| NVFP4/MXFP4 GEMM | 1.02x bf16 (no fast kernel) | Still dead for training |
| Hand-written FP8 attention | 18.4x error, rejected | Confirmed dead |
| Hand-written bf16 attention | 0.93x flex (flex wins) | Confirmed |
| FlexAttention block tuning | 1.028x (inside noise) | Defaults optimal |
| Liger RMSNorm/SwiGLU | 2x SLOWER under compile | Confirmed |
| CUDA Graphs at 450M | OOM | Shape-dependent |
| Batch 3 (vs batch 2) | 0.976x, +6.8 GB | Confirmed slower |

### What's NEW that could change the picture

**A4Q Kernel** (Jetha Chan, Jul 2026) — NVFP4 attention for sm_120
- Native `mma.sync.kind::mxf4nvf4.block_scale` QK^T — zero dequantization
- **Inference only** (KV cache). Not a training kernel.
- Measured: -39% TTFT at 256K context, +67% KV capacity vs FP8
- Relevant if Flower ever does long-context inference/eval
- Being upstreamed to FlashInfer/vLLM

**CODA** (arXiv:2605.19269, Han Guo + Tri Dao) — GEMM-epilogue fusion ⭐
- Reparameterizes Transformer ops as GEMM + epilogue programs
- Fuses RMSNorm, SwiGLU, residuals, RoPE, CE INTO the GEMM output tile
- **Directly solves Flower's "Liger RMSNorm/SwiGLU 2x slower under compile" problem**
- `linear_swiglu`, GEMM+RMSNorm+RoPE, GEMM+CE all implemented
- Built on CuTeDSL. Targets H100 but adaptable to SM120 (both have TMA)
- Code: github.com/open-lm-engine/coda-kernels (235 stars)

**FlashAttention-4 SM120 backend** (PR #2634)
- CuTeDSL-based, SM80-base kernels (cp.async + TMA + warp-specialized)
- Forward: head dims 64–256. Backward: 64–256 (d=192 limited by 99KB smem)
- fp8 KV-cache decode: ~1.6–1.9x faster than bf16
- Not merged yet — watch for it

**FP4 training on SM120 — the frontier**
- CUTLASS examples 79a-d: NVFP4 forward GEMM works on SM120 (2x MXFP8 throughput)
- Harry-Chen/fp4_sm120: polyfills stochastic rounding + RHT for SM120
  - `cvt.rs.satfinite.e2m1x4.f32` missing on SM120 → polyfilled, bit-exact vs SM100
  - RHT GEMM polyfilled with WMMA, 0% mismatch vs SM100 reference
- Transformer Engine NVFP4 training: **crashes on SM120** (99KB smem cap)
- **No FP4 backward GEMM exists for SM120 yet** → no training path
- Estimate: late 2026/early 2027 for usable community path

**UFP4 / Shrinkage Bias** (arXiv:2606.20381) — important for when FP4 training works
- E2M1 format (used by NVFP4/MXFP4) has inherent **shrinkage bias**
- Systematic negative rounding error from geometric asymmetry of representable bins
- Accumulates multiplicatively across layers, amplified by RHT
- Solution: UFP4 — use uniform E1M2/INT4 grids instead of E2M1
- Future accelerators should support E1M2 as first-class training primitive

**Token Superposition Training (TST)** (arXiv:2605.06546, Nous Research) ⭐⭐⭐
- **2.5x wall-clock pretraining speedup**, zero architecture change
- Phase 1: compress s consecutive tokens into bags, train with multi-hot CE
- Phase 2: revert to standard NTP (model is architecturally identical)
- Validated at 270M, 600M, 3B, 10B MoE
- Already spec'd in `docs/training-speedups.md` Section 9 — implement it
- Combinable with MTP (additive gains confirmed)

**Lighthouse Attention** (arXiv:2605.06554, Nous Research)
- Training-only hierarchical attention: 1.4–1.7x e2e pretraining at 98K context
- 17x faster forward at 512K context
- Symmetric Q,K,V pooling + parameter-free scoring + stock FlashAttention
- Two-stage: train majority with Lighthouse, brief SDPA resume at end
- After resumption, matches dense-from-scratch baseline

### Bottom line on kernels
The FP8 tensorwise stack is the whole of the low-precision win on SM120. The next
big lever is **TST** (2.5x throughput, zero kernel work) and **CODA** (fixes the
Liger-under-compile problem). FP4 training remains blocked on SM120 software
support — the hardware has the instructions, the backward kernels don't exist yet.

---

## 3. TERNARY / BINARY WEIGHTS

### The state of the art

**BitNet b1.58 / TriLM scaling study:**
- Ternary {-1,0,1} weights match FP16 perplexity at ≥3B params
- At 100M–600M: measurable perplexity gap, but downstream tasks surprisingly competitive
- Same scaling rate (α=0.26), higher offset (ε=1.76 vs 1.67 for FP16)
- Gap closes with scale

**Training mechanism (simple):**
- Maintain FP16 "latent/master" weights
- Ternarize on-the-fly in forward: absmean scaling → round to {-1,0,1}
- Backpropagate via straight-through estimator (STE)
- Standard AdamW works. No special optimizer needed.

**Critical training schedule (TriLM/Spectra):**
1. Halve LR at ~50% of training → sharp loss drop
2. Remove weight decay at ~66% of training (ternarization provides regularization)
- Without both, smaller models (<1B) diverge after halfway mark

**The Muon question (open research):**
- Muon operates on FP16 latent weights → mathematically should work
- But spectral philosophy of Muon interacts oddly with extreme quantization
- Nobody has published Muon + ternary from scratch → **potentially novel**
- Full-Stack FP4 (arXiv:2607.04422) shows NS can run in NVFP4

**Why no training speedup:**
- During training, ternary weights are dequantized to FP16 for matmul each forward
- No fast ternary GEMM training kernels exist (all target inference)
- Memory savings come only at checkpoint time, not during training
- Training memory = same as FP16 (latent weights + Adam states)

**Selective ternarization (consensus architecture):**
- Transformer linears: ternary
- Embeddings: int8 (not ternary)
- Norms: FP32
- LM head: FP16/FP32

**Fermion Neutrino-1 (8B):** QAT from Qwen3-8B into ternary. MMLU 72.1 (beats
FP16 Llama-3.1-8B at 68.3 and Gemma-4-E4B at 70.57). 3.88 GB on disk. But training
methodology is proprietary — "native format" language may be marketing.

**CAT-Q (Intel, ICML 2026 Oral):** Post-training ternary quantization that narrows
the rounding cliff. 512 calibration samples match QAT results. Scales to 235B.
Code: github.com/IntelChina-AI/BitTern

### Bottom line on ternary
Validated research direction for Flower (especially Muon compatibility is novel),
but **not a training speedup**. The win is checkpoint size and a research question
about Muon+ternary interaction. For practical training speed, FP4 via Quartet-style
MXFP4 kernels is the better bet — but those don't exist on SM120 yet either.

---

## 4. ATTENTION PATTERNS

### The universal consensus: hybrid > homogeneous

Every frontier model in 2025–2026 uses hybrid attention. The question is what
ratio and which layers.

**Optimal ratio (Meta systematic study, arXiv:2510.04800):**
- 1:5 (full:window) is Pareto-optimal for quality vs efficiency
- MiMo-V2.5: 6:1 window:full at window=128
- Laguna-XS: 3:1 at window=512

**Critical: full-attention layers go in the MIDDLE, not front.**
- Front placement consistently hurts quality
- Evenly scatter full-attention blocks across depths
- For 20 layers at 6:1: positions ~3, 7, 11, 15 (NOT position 0)

**The SWAX finding (directly relevant to Flower):**
- Shorter sliding windows IMPROVE long-context performance
- Because they force the model to rely on recurrent/external memory
- Best: stochastic windows (alternating 128 and 2048)
- This validates Flower's seq >> window thesis — but suggests window=2048 at
  seq=8192 may be too large (4:1 ratio) for memory to matter

### Linear attention: now beats full attention

**KDA (Kimi Delta Attention)** — current SOTA linear attention
- Channel-wise gating on the delta rule
- Kimi Linear (KDA + periodic MLA) **outperforms full MLA** in fair comparison
- FlashKDA kernel: chunk=16, bf16 state, O(L·d²) training, constant inference memory
- Code: github.com/MoonshotAI/FlashKDA

**Gated DeltaNet-2** (NVIDIA, May 2026)
- Decouples erase gate from write gate (Flower's memory variants likely use single gate)
- Strongest overall among Mamba-2, GDN, KDA, Mamba-3 at 1.3B
- Code: github.com/NVlabs/GatedDeltaNet-2

### Sparse attention (natively trainable)

**NSA (Native Sparse Attention)** — best for training
- Coarse-grained compression + fine-grained token selection
- **Maintains or exceeds full attention** in pretraining
- 11.6x decode, 9.0x forward, 6.0x backward
- FlexAttention-compatible (hierarchical block-sparsity → block masks)

**LongCat LSA / IndexShare** — cross-layer index sharing
- One layer runs indexer, 3 neighbors reuse selected tokens (70–100% overlap)
- Near-free efficiency win if adding expensive memory retrieval

### Attention residuals (Flower has AttnRes)

**CRITICAL: Use additive routing, NOT replacement routing.**
- Kimi's replacement routing AttnRes DEGRADES at scale (+6.9% worse at 1B)
- Additive routing (Delta Block) improves at all scales (-8.2% PPL at 7.6B)
- Flower should verify its AttnRes uses additive routing

### Position encoding

**RoPE provably fails at long context** (arXiv:2605.15514):
- Loses locality bias AND token consistency as context grows
- Cannot preserve both position and token discrimination

**NoPE gives better length generalization.**
- Cohere's hybrid RoPE+NoPE+SWA is SOTA: RoPE in windowed layers, NoPE in full/memory layers
- QK-Norm HURTS long-context (higher entropy, worse retrieval)

### Attention sinks

**Provably necessary under softmax** (arXiv:2603.11487).
- Keep 4 sink tokens at sequence start with full attention
- All SOTA streaming/hybrid models do this
- Trivial to add, free win

---

## 5. ARCHITECTURE IDEAS FOR MEMORY MECHANISMS

### Why prior memory variants showed null results (cross-cutting diagnosis)

1. **NTP val loss is blind to recall capacity** — compressed-memory variants can
   tie on perplexity while collapsing on exact associative recall (UniMatrix
   arXiv:2604.25930 reproduces Flower's exact null pattern)
2. **TTT/memorization mechanisms are secretly linear attention** (NVIDIA
   arXiv:2602.21204) — gradient ascent works, queries ≈ keys, more optimization
   ≠ better. The value is representational mixing, not faithful KV storage.
3. **Window too large** — at 4:1 window:seq ratio, local attention suffices
4. **Single write/erase gate** — GDN-2 shows decoupled gates are better

### Most promising new memory ideas

**FwPKM (Fast-weight Product Key Memory)** — reuses existing bloom/surprise code
- Sparse gradient updates only on activated slots
- Cleanest synthesis of tested mechanisms

**GDN-2's erase/write gate decoupling** — directly applicable to Flower
- Current memory variants use single gate for both forgetting and writing
- Separating them could improve memory utilization

**KVM (Key-Value Means)** — simplest memory variant to add
- Block-recurrence with mean-of-block compressed KV state
- No custom kernels, growable or fixed-size
- If Flower's summary_memory underperforms, KVM's simpler mean-of-block may be stronger

**Persistent memory (Titans)** — component Flower likely lacks
- Learnable input-independent parameters encoding task priors
- Distinct from long-term memory — gates the memory read

**Additive routing for memory gates** — same lesson as AttnRes
- Memory should AUGMENT the hidden state, not REPLACE it
- Preserve the residual stream highway

### Looped / recurrent architectures

**Hyperloop Transformers** (MIT, arXiv:2604.21254) — begin/middle/end with
hyper-connections. ~50% fewer params, outperforms depth-matched. Matrix-valued
residual streams at loop boundaries (~150-300K extra params).

**The "recurrent transformer" wave (2026):**
- The Recurrent Transformer (arXiv:2604.21215): position t at each layer attends
  to KV from earlier blocks, not just current block
- Context-Ready Transformer (arXiv:2606.27538): pre-computes context
- CART (arXiv:2606.01495): parameter-efficient with learned stability
- Block recurrence is the natural home for external memory

### Novel components (trivial to add, potentially impactful)

**Gated Norm** (A.X K2) — input-dependent gate post-RMSNorm
- Suppresses massive activations — stabilizes memory-gated branches at low precision
- Trivial: one gate scalar per feature

**SiTU-GLU** (Kimi K3) — bounded activation for routed/memory-gated experts
- Prevents coincident-large-value blowups
- Drop-in replacement for SwiGLU in memory-gated FFNs

**Engram conditional memory** (DeepSeek)
- N-gram → embedding O(1) lookup, orthogonal static memory axis
- Flower already has `engram_lite` — this validates the direction

---

## 6. DATA STRATEGY

### Datasets: what to use

**Tier 1 (use now):**
- FineWeb-Edu score-3 (current) — ✅ solid choice
- **DCLM-Baseline** — add as 50/50 mix. SmolLM2 controlled ablations show this
  mix beats either alone. Filters for instruction-following/Q&A style (complementary
  to FineWeb-Edu's educational focus)

**Tier 2 (add for specific gains):**
- **Stack-Edu / StarCoderData** (~10–15% of mix) — code improves reasoning
- **Cosmopedia-v2** (synthetic textbooks) — density/quality for small models
- **FineMath** — math reasoning

**Skip:** RedPajama, RefinedWeb, The Pile (all superseded by FineWeb-Edu + DCLM)

### Tokenizer

**16k vocab is well-calibrated for 100M–600M.** NeurIPS 2024 scaling laws with
vocabulary confirm this. Don't go to 4k (too coarse) or 50k (wastes params).
Flower's current 16k BPE is near-optimal.

### Training schedule

**Overtrain aggressively.** This is the #1 lever for small models.
- 50–200x Chinchilla ratio for small models (SmolLM2, de Vries)
- At 600M: train on 30–120B tokens (not the Chinchilla-optimal 12B)
- Overtraining reliably improves all downstream tasks

**Add a decay phase.** WSD (warmup-stable-decay):
- 10–20% of training with LR → 0
- Significant val-loss drop (the Delphi recipe)
- Flower already has `lr_schedule: wsd` wired but it's worth verifying the ratio

### Data selection (when data-constrained)

**Perplexity-based pruning** (ICLR 2025):
- Train 100M model on sample → score all docs → keep best 50–70%
- 125M model pruning improves 3B model by 2.04 avg downstream points
- Works in over-trained and data-constrained regimes

### Infrastructure

**Pre-tokenize and cache to disk.** Flower's own profiling shows 17% throughput
gap between synthetic-token profiling (52k tok/s) and real training (42–45k tok/s).
The FineWeb-Edu loader sustains 1.08M tok/s at 2 workers — 20x the model's
consumption rate — but the streaming + tokenization overhead is real. Pre-tokenize
to memmap .bin files for maximum throughput.

---

## 7. CROSS-CUTTING SYNTHESIS: WHAT MAKES MEMORY MECHANISMS WORK

The research converges on a clear diagnosis of why Flower's memory variants showed
null results at small scale, and what would change that:

### The four problems

1. **Scale**: Memory mechanisms need ~500M+ params (Flower's finding, confirmed
   by the emergence literature). Below this, capacity is spent on basics.

2. **Window:seq ratio**: At 4:1 (window=2048, seq=8192), local attention suffices.
   The SWAX paper shows you need SHORTER windows (or stochastic) to force memory
   engagement. Try window=512 at seq=8192 (16:1 ratio) or stochastic 128/2048.

3. **NTP blindness**: Val loss cannot distinguish a model that recalls from memory
   vs one that guesses. Need associative-recall probes (MQAR, needle-in-haystack)
   as primary metrics, not just perplexity.

4. **Single-gate memory**: Modern linear attention (GDN-2, KDA) decouples erase
   from write gates. Flower's memory variants likely conflate them.

### The testable thesis (combining all findings)

To make memory mechanisms show value:
- Train at ≥400M params (the scale-up target)
- Use **hybrid attention** (6:1 window:full, full in middle) so the model has
  global reach without full attention everywhere
- Use **shorter window** (512 or stochastic) to create memory pressure
- Add **attention sinks** (4 tokens)
- Decouple **erase and write gates** in the memory update
- Use **additive routing** (memory augments residual, doesn't replace)
- Evaluate on **associative recall probes**, not just perplexity
- Consider **KDA/GDN-2 as a linear-attention baseline** — if explicit external
  memory can't beat implicit recurrent memory (KDA), that's an important finding

---

## CITATION INDEX (key papers by arXiv ID)

### Optimizers
- 2302.06675 — Lion
- 2402.11858 — PSGD Lie-group theory
- 2502.07529 — Scion (ICML 2025)
- 2506.10935 — CANS (Chebyshev NS)
- 2509.02046 — Optimizer reality check (Wen et al.) ⚠️
- 2509.03378 — KL-Shampoo/SOAP (ICLR 2026)
- 2510.05491 — NorMuon
- 2602.24283 — LoRA-Pre (ICLR 2026 Oral) ⚠️
- 2604.01472 — Newton-Muon
- 2604.09967 — Muon²
- 2605.08933 — Group Muon
- 2605.11172 — SODA
- 2605.23061 — SF-NorMuon (schedule-free spectral)
- 2606.00371 — cubic5 NS schedule
- 2606.19348 — DeepSeek-V4 NS schedule
- 2607.20548 — SOAP at scale (NVIDIA) ⚠️
- 2607.24653 — Kimi K3 (Per-Head Muon)
- 2607.26001 — SAM-Muon

### Kernels / Low-precision
- 2509.25149 — NVFP4 Pretraining (NVIDIA, 12B/10T tokens)
- 2603.05451 — FlashAttention-4
- 2605.06546 — TST (Token Superposition Training) ⚠️
- 2605.06554 — Lighthouse Attention
- 2605.19269 — CODA (GEMM-epilogue fusion)
- 2606.19348 — DeepSeek-V4 (CSA/HCA, MXFP4 QAT)
- 2606.20381 — UFP4 (shrinkage bias in E2M1)
- 2607.20466 — JAXBench (TPU kernel benchmark)

### Attention
- 2405.21060 — Mamba-2 (SSD)
- 2502.11089 — NSA (Native Sparse Attention)
- 2509.24552 — SWAX (shorter windows improve long-context)
- 2510.04800 — Meta hybrid attention systematic study
- 2510.26692 — Kimi Linear (KDA beats full attention)
- 2603.11487 — Attention sinks provably necessary
- 2605.15514 — RoPE provably fails at long context
- 2605.18855 — Delta Attention Residuals (additive > replacement)
- 2605.22791 — Gated DeltaNet-2 (erase/write decoupling)
- 2607.09694 — Low-Rank AttnRes
- 2608.01662 — LongCat Sparse Attention

### Architecture
- 2402.17764 — BitNet b1.58
- 2501.00663 — Titans (Learning to Memorize at Test Time)
- 2504.12285 — BitNet b1.58-2B4T
- 2602.21204 — TTT is secretly linear attention ⚠️
- 2604.21254 — Hyperloop Transformers
- 2605.12466 — Attractor Models
- 2605.20613 — HRM-Text-1B
- 2606.26650 — CAT-Q (ICML 2026 Oral)

### Data
- 2406.11794 — DCLM (NeurIPS 2024)
- 2406.17557 — FineWeb-Edu (NeurIPS 2024)
- 2502.02737 — SmolLM2 (ICML 2025) ⚠️

⚠️ = highest-signal references for decision-making

---

*Research compiled August 9, 2026. All sources verified via web search/extraction.
Detailed per-area reports in the companion files listed at the top.*
