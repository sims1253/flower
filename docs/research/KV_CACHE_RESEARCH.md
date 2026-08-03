# KV-Cache Compaction Research: Techniques Survey (2024-2026)

**Context**: Research supporting the "Still" paper (arXiv:2606.07878), which uses a per-layer
Perceiver module to compress KV caches in a single forward pass via KL distillation.
Target setting: small 12M param transformer, FineWeb-Edu, Muon optimizer.

**Methodology**: 12+ targeted WebSearch queries across the seven research areas requested.
Results prioritized to 2024-2026 papers, with foundational earlier works included where
they remain influential.

---

## 1. KV-Cache Compression (Eviction-based)

### H2O: Heavy-Hitter Oracle (NeurIPS 2023, arXiv:2306.14048)
- **Core technique**: Dynamically retains a small set of "heavy hitter" KV pairs based on
  cumulative attention scores, plus recent tokens. Provides theoretical approximation
  guarantees relative to full attention.
- **Differs from Still**: Eviction-only (no learned compression); operates post-hoc on the
  existing cache rather than learning a compact latent. Still's Perceiver produces *new*
  latent tokens rather than selecting a subset.
- **Applicability to our setting**: Yes. The heavy-hitter heuristic is cheap to evaluate on
  a 12M model and could serve as a strong, training-free baseline. Could also initialize
  the Still Perceiver's training signal (which tokens to preserve).

### StreamingLLM / Attention Sinks (ICLR 2024, arXiv:2309.17453)
- **Core technique**: Keep the first few "attention sink" tokens (which absorb disproportionate
  attention mass) plus a sliding window of recent tokens. Enables infinite-length generation
  with constant memory.
- **Differs from Still**: Pure positional heuristic, no content-awareness. Still produces a
  content-aware summary but loses the attention-sink phenomenon which may be important for
  numerical stability.
- **Applicability**: High. The attention-sink insight is directly relevant: a 12M model
  trained with Muon may also develop sinks, and preserving them in the compressed cache
  could stabilize long-horizon generation.

### SnapKV (arXiv:2404.14469, 2024)
- **Core technique**: Uses an "observation window" (the last N tokens of the prompt) to
  compute attention-based importance scores over the prefix. Keeps the most-attended prefix
  tokens. Training-free, single-pass.
- **Differs from Still**: Still amortizes a *learned* compressor; SnapKV is a static
  importance heuristic computed from one forward pass. SnapKV cannot produce novel latent
  tokens, only select existing ones.
- **Applicability**: High. Strong baseline; the observation-window idea could inform how
  the Still Perceiver is trained (which queries to distill from).

### ThinK (arXiv:2407.21018, 2024)
- **Core technique**: *Channel* pruning of the key cache rather than token pruning. Uses a
  query-induced norm to identify low-importance key dimensions and prunes them. First
  channel-pruning method for KV cache.
- **Differs from Still**: Orthogonal axis. Still compresses the *sequence* dimension; ThinK
  compresses the *feature* dimension. The two could be composed.
- **Applicability**: Medium-high. Could be combined with Still: compress tokens via Perceiver,
  then prune channels for further savings. Worth an ablation.

### KVzip (ICML 2025, arXiv:2505.23416)
- **Core technique**: Query-*agnostic* eviction. Appends a single special "[RECON]" token to
  the context, runs one forward pass, and uses the resulting attention pattern to score all
  tokens for eviction. The compressed cache is reusable across arbitrary downstream queries.
- **Differs from Still**: Both are single-pass and amortized, but KVzip *selects* tokens
  while Still *generates* latent tokens. KVzip is training-free; Still requires training
  the Perceiver. KVzip's query-agnostic property is attractive for deployment.
- **Applicability**: Very high. This is the closest competitor to Still's design philosophy.
  The "[RECON]" token trick is essentially a cheaper alternative to a Perceiver module and
  should be directly compared.

### PyramidKV / PyramidInfer (arXiv:2406.02069, 2024)
- **Core technique**: Allocates a *decreasing* KV budget across layers: more cache in lower
  layers (where attention scatters), less in upper layers (where attention concentrates).
  Based on the "Pyramidal Information Funneling" empirical observation.
- **Differs from Still**: Still uses a per-layer Perceiver with a fixed latent count.
  PyramidKV's insight suggests the Perceiver's latent count should *vary* by layer depth.
- **Applicability**: High and directly actionable. A 12M model likely exhibits similar
  funneling; the Still Perceiver latent budget should be layer-dependent (more latents
  in early layers, fewer in later layers).

### ShadowKV (ICML 2025 Spotlight, arXiv:2410.21465)
- **Core technique**: Offloads the bulk of the KV cache to CPU/DRAM, keeping only a small
  "active" set on GPU. Uses low-rank approximation of keys to identify which tensors are
  likely needed, fetching them on demand.
- **Differs from Still**: System-level optimization rather than algorithmic compression.
  Still reduces the cache itself; ShadowKV moves it. Complementary.
- **Applicability**: Low-medium for our research setting (we care about the algorithm, not
  deployment memory tiers).

---

## 2. Token Merging / Pruning

### Model Tells You Where to Merge / KVMerger (arXiv:2407.08454, 2024)
- **Core technique**: Uses the model's own attention to identify *similar* KV pairs and
  merges them (weighted average) rather than evicting. Produces a smaller cache with no
  information fully discarded.
- **Differs from Still**: Merging is a linear combination of existing tokens; Still's
  Perceiver can produce arbitrary nonlinear latents. Merging is cheaper but less expressive.
- **Applicability**: High. KVMerger is a natural baseline and its similarity-clustering idea
  could inspire a "soft Perceiver" where each latent is explicitly tied to a cluster of
  source tokens (improving interpretability).

### TokenSkipping (ICIT 2025)
- **Core technique**: Practical robust pruning that skips tokens deemed low-importance by
  a lightweight scoring function, designed for long-context inference. Targets the memory
  bottleneck directly.
- **Differs from Still**: Hard pruning vs. learned compaction.
- **Applicability**: Medium. Another baseline.

### ClusterAttn (ACL 2025)
- **Core technique**: Compresses the KV cache under "Intrinsic Attention Clustering" -
  groups semantically similar tokens and represents each cluster compactly.
- **Differs from Still**: Clustering-based; Still's Perceiver implicitly performs a form of
  learned clustering. ClusterAttn makes the clustering explicit.
- **Applicability**: Medium-high. The clustering viewpoint could provide a useful
  regularizer or interpretability tool for the Perceiver latents.

---

## 3. Learned Compression of Sequences

### Gisting (NeurIPS 2023, arXiv:2304.08467)
- **Core technique**: Trains an LM (via instruction tuning with a modified attention mask)
  to compress arbitrary prompts into a small set of "gist tokens" that preserve task
  performance. Achieves up to 26x prompt compression.
- **Differs from Still**: Gisting compresses the *input prompt* (discrete token level);
  Still compresses the *KV cache* (continuous representation level). Still operates
  per-layer; gisting is a single-level operation.
- **Applicability**: High conceptual relevance. The meta-learning training procedure (train
  on diverse prompts, generalize to unseen ones) directly informs how to train the Still
  Perceiver. The gist-token idea is essentially what the Perceiver latents are.

### Perceiver AR (arXiv:2202.07765, DeepMind)
- **Core technique**: Autoregressive architecture using cross-attention to map long-range
  inputs to a small number of latent positions while maintaining causal masking. Modality-
  agnostic; handles 100k+ element sequences.
- **Differs from Still**: This *is* the architectural foundation Still builds on. Still's
  contribution is applying Perceiver-style latents as a drop-in KV cache replacement with
  KL distillation, rather than as the model's primary compute path.
- **Applicability**: Foundational. The Efficient Context-Propagating Perceiver follow-up
  (arXiv:2412.06106) adds "lossy history" compression in the latent outputs, which is
  directly relevant to iterative compaction (Still Section 3.4).

### Compressive Transformer (ICLR 2020, arXiv:1911.05507)
- **Core technique**: Stores old hidden states in a memory buffer; when memory is full,
  compresses the oldest entries via a learned compression function (e.g., 1D conv, pooling,
  or MLP) into fewer slots. Achieves SOTA on long-range language modeling.
- **Differs from Still**: Compresses hidden *states*, not KV pairs per se, and uses simple
  deterministic compressors (conv/MLP) rather than attention-based Perceiver. Still is the
  modern, attention-based successor.
- **Applicability**: High. This is the conceptual ancestor. The compression-ratio scheduling
  and the "lossy history" framing directly inform Still's iterative compaction design.

### Dynamic Memory Compression / DMC (ACL 2024, arXiv:2403.09636)
- **Core technique**: Retrofits pretrained LLMs (Llama 2 7B/13B/70B) with a learned gating
  mechanism that *merges* KV groups on the fly, achieving up to 3.7x throughput. Applied
  via lightweight continued training.
- **Differs from Still**: DMC merges adjacent KV groups via learned gates; Still uses a
  Perceiver cross-attention. DMC is more local (grouping); Still is global (cross-attention
  over the whole cache).
- **Applicability**: High. DMC shows learned compression *can* be retrofitted cheaply,
  which validates the Still approach. The gating idea could be an alternative to Perceiver
  for very small models where a Perceiver module is relatively expensive.

---

## 4. Optimal Transport in Attention

### ESPFormer (arXiv:2502.07962, 2025)
- **Core technique**: Doubly-stochastic attention via Expected Sliced Optimal Transport.
  Replaces softmax attention with an OT-based attention that produces a doubly-stochastic
  matrix, ensuring every token both sends and receives bounded attention mass.
- **Differs from Still**: Changes the attention mechanism itself, not the cache. But the OT
  viewpoint could regularize how the Perceiver attends to the full cache.
- **Applicability**: Medium. An OT-regularized Perceiver (e.g., Sinkhorn-distilled latents
  that must "cover" the source distribution) could improve compaction quality. This is a
  novel research direction worth exploring.

### Provable Optimal Transport with Transformers (arXiv:2410.19931, 2024)
- **Core technique**: Theoretical analysis showing transformer self-attention implicitly
  performs a form of optimal transport between token distributions. Provides depth bounds
  for approximating OT plans.
- **Differs from Still**: Theoretical; supports the view that attention *is* a soft OT solver.
- **Applicability**: Medium. Provides theoretical justification for why an attention-based
  Perceiver is a principled compressor (it solves a transport problem between the full
  cache distribution and the latent distribution).

### Sparse Sinkhorn Attention (practical.ml, 2020; OTSeg)
- **Core technique**: Applies entropic-OT (Sinkhorn) normalization to attention matrices
  to produce sparse, doubly-stochastic patterns. Reduces compute by concentrating mass.
- **Differs from Still**: Orthogonal; could be applied inside the Perceiver.
- **Applicability**: Low-medium. Mostly relevant for vision/segmentation; less proven for
  language KV compression.

---

## 5. Spectral / Low-Rank KV Methods

### KQ-SVD (arXiv:2512.05916, 2025)
- **Core technique**: Joint SVD of the Key and Query matrices to find a low-rank subspace
  that optimally preserves the attention matrix (QK^T). Provides *provable* guarantees on
  attention fidelity after compression.
- **Differs from Still**: Still learns a nonlinear compressor; KQ-SVD uses optimal linear
  projection. KQ-SVD has theoretical guarantees Still lacks.
- **Applicability**: High. Could serve as (a) a strong training-free baseline with
  guarantees, (b) an initialization for the Perceiver, or (c) a regularizer. The provable
  angle is valuable for a research paper.

### Eigen Attention (EMNLP 2024 Findings, arXiv:2408.05646)
- **Core technique**: Projects K and V into a lower-dimensional "eigen space" via PCA on
  the key vectors, then performs attention in that space. Reduces per-token KV size.
- **Differs from Still**: Dimensionality reduction (feature axis) vs. sequence compaction.
  Complementary, not competing.
- **Applicability**: High. Like ThinK, this attacks the feature axis. Could be composed
  with Still: Perceiver for sequence compression + Eigen Attention for feature compression.

### eOptShrinkQ (arXiv:2605.02905, 2026)
- **Core technique**: Near-lossless KV compression via Optimal Shrinkage (a statistical
  denoising method) applied to the singular spectrum of the KV matrix. Differs from SVD-based
  methods by using optimal shrinkage of noise-dominated singular values.
- **Differs from Still**: Statistical/spectral; Still is neural.
- **Applicability**: Medium. The optimal-shrinkage theory could inform how many latent
  dimensions the Perceiver should use per layer (analogous to keeping signal singular values).

### Quantization vs. Rank Reduction (arXiv:2604.11501, 2026)
- **Core technique**: Systematic comparison showing that *quantization* (keep all dims, lower
  precision) dominates *rank reduction* (drop dims, keep precision) for KV cache compression
  at equal bit budgets.
- **Differs from Still**: Still is neither pure quantization nor pure rank reduction - it's
  sequence compaction. But the finding suggests feature-axis methods should prefer
  quantization over dropping dimensions.
- **Applicability**: Medium. Informative for choosing baselines and for post-training the
  Perceiver outputs (quantize the latents).

---

## 6. Flow Matching / Normalizing Flows for Compression

**Finding**: This is the *least developed* area for KV-cache compression specifically. No
direct application of flow matching to KV cache compaction was found.

### Relevant adjacent work:
- **Latent Flow Transformer** (arXiv:2505.14513, 2025): Uses flow-matching-style continuous
  dynamics in latent space for efficient generation, but for the *forward* pass, not cache
  compression. Mentions compression as a side benefit.
- **Free-form Flows / Continuous Normalizing Flows** (NeurIPS 2024): General framework for
  making any architecture invertible via flow-matching training. Could in principle be
  applied to make the Perceiver *invertible* (enabling decompression).
- **Context Compression via Explicit Information Transmission** (arXiv:2602.03784, 2026):
  Uses gating coefficients (normalized across layers) to transmit anchor information through
  a compressed context - shares philosophy with flow-based approaches but is not flow-based
  per se.

### Implication for Still:
- **Gap/opportunity**: Flow matching for KV compaction is an open frontier. One could train
  the Perceiver as a *flow-matching* compressor: define a probability path from the full KV
  distribution to the latent distribution, and learn the velocity field. This could give
  smoother compression and a principled decompression path (invertibility) that Still lacks.
- **Risk**: Flow matching adds training complexity and may not pay off at 12M scale.

---

## 7. Iterative / Online Compaction

### LaCache (ICML 2025, arXiv:2507.14204)
- **Core technique**: Two innovations: (1) a "ladder-shaped" KV cache pattern where storage
  increases from shallow to deep layers (cross-layer budget allocation), and (2) an
  *iterative compaction mechanism* that progressively compresses older caches when the budget
  is reached, enabling infinite-length generation.
- **Differs from Still**: Still mentions iterative compaction (Section 3.4) but LaCache
  provides a concrete, trained mechanism. LaCache's ladder pattern is the cross-layer analog
  of PyramidKV's per-layer budgeting.
- **Applicability**: Very high. LaCache is the most directly comparable iterative-compaction
  system to Still. The ladder-shaped budget could replace Still's uniform per-layer latent
  count, and LaCache's iterative trigger logic should be benchmarked against Still's.

### Expected Attention (arXiv:2510.00636, 2025)
- **Core technique**: Estimates the *future* attention distribution (from queries not yet
  seen) to make eviction decisions that are robust over the full generation horizon, rather
  than greedy per-step decisions.
- **Differs from Still**: Still distills from the current forward pass; Expected Attention
  looks ahead probabilistically.
- **Applicability**: High. The future-attention estimation could improve Still's training
  objective: distill not just from current attention, but from an *expected* future attention
  distribution. This is a concrete improvement direction.

### Fast KV Compaction via Attention Matching (arXiv:2602.16284, 2026, MIT)
- **Core technique**: Latent-space KV compaction that explicitly *matches* the per-head
  attention pattern of the compressed cache to the full cache. Achieves up to 50x compaction
  in seconds with little quality loss. Closest direct competitor to Still.
- **Differs from Still**: Uses attention-matching loss instead of KL distillation on outputs.
  Attention matching is more direct (preserve the attention pattern itself) vs. Still's
  output-level KL. Both are single-pass amortized compactors.
- **Applicability**: Critical. This paper should be a primary baseline and the
  attention-matching loss is a strong alternative to Still's KL objective. Worth running
  both losses on the 12M model and comparing.

---

## Summary: Most Promising Directions for Improving Still

Ranked by relevance to the 12M-param / FineWeb-Edu / Muon setting:

1. **Layer-adaptive latent budgets** (from PyramidKV + LaCache): Replace Still's uniform
   per-layer latent count with a funnel/ladder schedule. Low cost, potentially high impact.

2. **Attention-matching loss** (from Fast KV Compaction, arXiv:2602.16284): Compare against
   or combine with Still's KL distillation. The attention-pattern loss is cheaper and more
   direct.

3. **KVzip-style query-agnostic training** (arXiv:2505.23416): Train the Perceiver so the
   compressed cache works for *any* future query, not just the current one. Use the
   Expected Attention idea (arXiv:2510.00636) to sample future queries during training.

4. **Compose with feature-axis compression** (ThinK channel pruning + Eigen Attention):
   After Perceiver sequence compaction, apply dimension reduction for further savings.

5. **KQ-SVD initialization** (arXiv:2512.05916): Initialize the Perceiver weights from the
   SVD-optimal linear projection, then finetune. Gives a principled starting point and a
   theoretical guarantee floor.

6. **OT-regularized Perceiver** (from ESPFormer + provable OT theory): Add a Sinkhorn or
   sliced-OT regularizer so the Perceiver latents provably "cover" the source token
   distribution. Novel research contribution.

7. **Attention-sink preservation** (from StreamingLLM): Explicitly preserve sink tokens
   outside the Perceiver to maintain numerical stability during iterative compaction.

8. **Flow-matching compressor** (open frontier): Train the Perceiver as a flow-matching
   velocity field from full-cache to latent distributions, enabling invertibility. Higher
   risk, higher novelty.

---

## Full Paper Reference List

| Paper | arXiv | Year | Venue |
|-------|-------|------|-------|
| Still: Amortized KV Cache Compaction | 2606.07878 | 2026 | - |
| H2O: Heavy-Hitter Oracle | 2306.14048 | 2023 | NeurIPS |
| StreamingLLM / Attention Sinks | 2309.17453 | 2024 | ICLR |
| SnapKV | 2404.14469 | 2024 | - |
| ThinK: Channel Pruning | 2407.21018 | 2024 | - |
| KVzip: Query-Agnostic Compression | 2505.23416 | 2025 | ICML |
| PyramidKV | 2406.02069 | 2024 | - |
| ShadowKV | 2410.21465 | 2025 | ICML |
| Model Tells You Where to Merge | 2407.08454 | 2024 | - |
| Gisting | 2304.08467 | 2023 | NeurIPS |
| Perceiver AR | 2202.07765 | 2022 | - |
| Compressive Transformer | 1911.05507 | 2020 | ICLR |
| Dynamic Memory Compression | 2403.09636 | 2024 | ACL |
| ESPFormer (OT Attention) | 2502.07962 | 2025 | - |
| Provable OT with Transformers | 2410.19931 | 2024 | - |
| KQ-SVD | 2512.05916 | 2025 | - |
| Eigen Attention | 2408.05646 | 2024 | EMNLP |
| eOptShrinkQ | 2605.02905 | 2026 | - |
| Quant vs Rank Reduction | 2604.11501 | 2026 | - |
| LaCache | 2507.14204 | 2025 | ICML |
| Expected Attention | 2510.00636 | 2025 | - |
| Fast KV Compaction (Attn Matching) | 2602.16284 | 2026 | - |
| Efficient Context-Propagating Perceiver | 2412.06106 | 2024 | - |
| Latent Flow Transformer | 2505.14513 | 2025 | - |
| Context Compression via Explicit Info | 2602.03784 | 2026 | - |
| ClusterAttn | (ACL 2025) | 2025 | ACL |
| TokenSkipping | (ICIT 2025) | 2025 | ICIT |
