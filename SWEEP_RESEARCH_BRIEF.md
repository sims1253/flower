# Flower Sweep Research Brief
## Memory-Augmented Attention for Small Transformers -- Investigation Menu
### Generated: 2026-05-09

Target: 2x RTX 5090 (Blackwell), 16-20M params, FineWeb-Edu, GPT-2 tokenizer.
Sweep 1 finding: hierarchical max-pooled memory = 2.6x PPL improvement over vanilla local attention.

This document is for the sweep design agent. Items are ordered by expected impact + novelty.
Each item has: what it is, why it might work, implementation complexity, and risk level.

---

## A. ARCHITECTURE VARIANTS TO SWEEP

### A1. Orthogonal State Recurrence (LATTICE) -- ICML 2025
**What**: Instead of max-pool to write memory, update each memory slot only with information *orthogonal* to its current state. Minimizes interference between stored memories. Input- and state-dependent "writing intensity" gating.
**Why**: Directly upgrades Flower's hierarchical max memory. Max-pool discards information; orthogonal updates preserve existing memories while adding only novel information. Principled alternative.
**Complexity**: Medium -- need the OSR update rule (gradient-projection style)
**Risk**: Low -- clean swap for existing memory write mechanism
**Source**: arxiv 2504.05646

### A2. Looped Transformer (Feedback Depth)
**What**: Loop the 4-layer model K times (K=2,3,4). Each loop iteration reads/writes to the memory bank. The memory bank acts as an anchor point across loops.
**Why**: 4 layers is shallow. Looping creates effective depth of 4*K with zero additional parameters. The memory bank becomes a scratchpad that persists across loops, giving the model "thinking time." Each loop refines the representation.
**Complexity**: Low -- modify the forward pass to loop, share memory bank across iterations
**Risk**: Low -- well-studied technique, easy to implement
**Source**: arxiv 2502.17416

### A3. TransformerFAM Feedback Connections
**What**: Add feedback connections from later layers back into earlier layers' attention. "Feedback Attention Memory" acts as working memory. No additional weights.
**Why**: For a 4-layer model, feedback from layer 4 to layer 1 creates an effective recurrence. The memory bank interfaces with both the forward and feedback streams. Inspired by cortico-thalamic loops.
**Complexity**: Low -- just add residual connections from later to earlier layers
**Risk**: Low -- zero extra parameters, just wiring change
**Source**: arxiv 2404.09173 (Google)

### A4. Surprise-Gated Memory Writes (Titans-style)
**What**: Gate memory writes by "surprise" -- how much the input violates the model's expectations (gradient of loss w.r.t. input). Surprising inputs get stored; expected inputs get skipped.
**Why**: Currently Flower writes to memory on every token. Surprise gating means the model learns WHAT is worth remembering. Also provides a principled forgetting mechanism (weight decay as memory decay).
**Complexity**: Medium -- need to compute surprise signal and use it as write gate
**Risk**: Medium -- surprise signal could be noisy at small scale
**Source**: arxiv 2501.00663 (Google Research)

### A5. Phase-Associative Memory (Complex-Valued)
**What**: Memory state is a d×d complex matrix (not vector). Uses outer products for binding, inner products (conjugate) for retrieval. O(d²) capacity per head -- for 256 dims, that's 65,536 effective memory slots per head with NO additional parameters.
**Why**: Completely different paradigm from attention-based memory retrieval. The phase-based binding/unbinding is mathematically clean. O(d²) capacity for free is remarkable. Complex-valued computation is supported natively in PyTorch.
**Complexity**: High -- rewrite memory module to use complex arithmetic
**Risk**: High -- unproven, but high reward
**Source**: arxiv 2604.05030

### A6. Product Key Memory (DeepMind PEER)
**What**: Decompose memory lookup into two independent hash functions, enabling O(1) retrieval from massive memory banks. For d=16 sub-dimensions each, get 16×16=256 effective hash buckets with only 32 hash computations.
**Why**: Currently Flower does attention over memory slots (quadratic in memory size). Product keys give O(1) retrieval from much larger memory banks. Decouples memory capacity from compute.
**Complexity**: Medium -- need product key decomposition layer
**Risk**: Medium -- proven technique but not widely used in transformers
**Source**: Google DeepMind PEER architecture, 2024

### A7. Hybrid SSM-Attention Heads (Hymba-style)
**What**: Run a lightweight SSM (state space model) head alongside the attention head in the SAME layer in parallel. SSM handles streaming/summarization; attention handles precise retrieval.
**Why**: Flower already has a local attention window + memory. The SSM head could replace the sliding window as a more efficient streaming mechanism, freeing attention capacity for memory retrieval. Cross-layer KV sharing reduces parameters.
**Complexity**: Medium -- add SSM head parallel to attention
**Risk**: Low-Medium -- NVIDIA validated at 1.5B scale
**Source**: arxiv 2411.13676 (NVIDIA)

### A8. Low-Dimensional Projected Attention (LPA)
**What**: Project queries/keys through a lower-dimensional space (e.g., 256 → 64) ONLY for attention/memory lookup. FFN stays at full dimension. Saves 12.4% time with 5% perplexity improvement at 130M-3B scale.
**Why**: Free perplexity improvement + faster memory retrieval. The bottleneck acts as regularization. The finding that FFN should NOT be compressed is useful.
**Complexity**: Low -- just add a projection layer before attention
**Risk**: Very Low -- well-validated
**Source**: arxiv 2411.02063

### A9. Holographic Reduced Representations (HRRs)
**What**: Compress structured information into memory slots using circular convolution for binding and circular correlation for unbinding. Encode role-filler pairs (subject-verb, entity-attribute) in fixed-size vectors.
**Why**: Instead of storing raw token representations, store holographically bound structures. Much denser information packing per memory slot. The math is clean (circular conv = FFT multiply).
**Complexity**: Medium -- need HRR encode/decode in the memory module
**Risk**: Medium -- HRR capacity is O(1/√n), may need large vectors
**Source**: Plate 1995/2003

### A10. Neural Paging (OS-Inspired Memory Management)
**What**: Learned page controller with KEEP/EVICT/PREFETCH operations on memory slots. Treats memory bank like virtual memory. O(N²) → O(N·K²) complexity.
**Why**: Currently Flower has no principled eviction policy. When the memory bank fills up, what gets overwritten? Neural paging learns this. Belady's optimality criterion (evict farthest-future-used) can be approximated by a tiny learned predictor.
**Complexity**: Medium-High -- need paging controller module
**Risk**: Medium -- conceptually clean but untested in small transformers
**Source**: Neural Paging paper, 2024

### A11. E8 Lattice Memory
**What**: Store memory entries on points of the E8 lattice (8-dimensional, optimal packing density). Lookup is O(1) regardless of memory size by exploiting lattice symmetries. Project queries to 8-dim lattice space, constant-time lookup, Gaussian kernel interpolation.
**Why**: O(1) memory access with no compute scaling. For a 256-dim model, project to 8-dim lattice space, look up, read back. Essentially unbounded memory.
**Complexity**: High -- need lattice operations and projection
**Risk**: High -- mathematically beautiful but the projection might lose too much info
**Source**: arxiv 2107.03474 (NeurIPS 2021)

### A12. Reservoir-Tier Hybrid
**What**: Replace one attention layer with a fixed random recurrent reservoir (echo state network). Only the readout is trained. The reservoir provides free long-term memory via recurrence.
**Why**: For a parameter-constrained model, replacing one attention layer with a parameter-free reservoir frees ~25% of parameters for the memory module. The reservoir provides a different kind of memory (attractor-based) that complements attention-based retrieval.
**Complexity**: Medium -- implement reservoir layer
**Risk**: Medium -- reservoir computing is well-established but not in transformers
**Source**: EMNLP 2025 (ResFormer)

---

## B. MEMORY UPDATE MECHANISMS (plug into existing hierarchical memory)

These are alternative ways to write to memory, directly comparable to current max-pool.

| Mechanism | Complexity | What changes |
|-----------|-----------|--------------|
| Max-pool (current best) | Baseline | Aggregate via max |
| Mean-pool | Low | Aggregate via mean |
| Orthogonal update (A1) | Medium | Project gradient orthogonal to current state |
| Surprise-gated (A4) | Medium | Only write when input is "surprising" |
| Flow-based evolution | High | Memory entries evolve via learned vector field |
| Bayesian update (Kanerva) | High | Uncertainty-aware write with posterior |
| Fast weights (Schmidhuber) | Medium | Memory weights updated per-sequence, not per-epoch |

---

## C. OPTIMIZER EXPERIMENTS

### C1. Muon (RECOMMENDED)
**What**: SGD-momentum + Newton-Schulz orthogonalization of update matrices. ~2x compute efficiency vs AdamW at 30M-200M scale. Best-evidenced at exactly Flower's scale.
**Why**: The NanoGPT speedrunning community has validated it across 12 consecutive records by 7 researchers. At 30M-200M, reaches target loss with 48-52% of AdamW's compute. Also: 8-bit quantized Muon gives 62% memory reduction.
**Key config**: LR ~0.02 for hidden weights, 3e-4 for aux (embeddings/heads). momentum=0.95, Nesterov=True, ns_steps=5.
**Important**: Only for 2D weight params. Embeddings/heads/scalars still use AdamW.
**Risk**: Very low -- extensively validated at this scale
**Source**: Keller Jordan et al., github.com/KellerJordan/Muon

### C2. Schedule-Free (Orthogonal to Muon)
**What**: Eliminates LR decay schedules entirely. No cosine/linear schedule needed.
**Why**: Removes one hyperparameter from sweeps. Can be combined with Muon (use Schedule-Free for the AdamW aux params). LR 1x-10x larger than with schedules.
**Risk**: Very low

### C3. SOAP
**What**: Shampoo + Adam in eigenbasis. >40% fewer iterations at large batch.
**Why**: Less compelling at small batch sizes. Only worth testing if Muon doesn't pan out.
**Risk**: Moderate -- mixed reproduction results

**Recommendation**: Use Muon for all sweeps. It's proven at this scale and effectively doubles sweep throughput. Add Schedule-Free to eliminate the schedule tuning dimension.

---

## D. TRAINING EFFICIENCY (Blackwell / 2x RTX 5090)

### Hardware Reality Check
- **RTX 5090**: Blackwell GB202, 32GB GDDR7, 5th-gen Tensor Cores, SM 12.0
- **NO NVLink** on consumer GeForce. 2-GPU via PCIe 5.0 x16 only.
- **PyTorch 2.8+ nightly** (cu128) required for SM 12.0 support
- **FlashAttention-2**: Works via SM80 backward compatibility
- **FlashAttention-3/4**: NOT available for consumer Blackwell yet
- **FP8 training**: Limited on consumer Blackwell (no second-gen Transformer Engine). Focus on bf16.
- **FP4 training**: Not ready for training, inference only.
- **torch.compile**: Works on Blackwell, recommended for kernel fusion speedups

### Recommended Training Setup
1. Linux, PyTorch 2.8+ nightly (cu128), CUDA 12.8, cuDNN 9, NCCL 2.25+
2. bf16 autocast as default mixed precision
3. FlashAttention-2 for attention
4. torch.compile for kernel fusion
5. DDP (not FSDP) for 2-GPU -- PCIe only
6. Set `NCCL_P2P_DISABLE=1` if hitting P2P errors
7. For sweeps: run 1 GPU per trial (2 concurrent trials) rather than 2 GPUs per trial

### Sweep Strategy
1. **Short trials first**: Run all variants for 10-25% of full steps (3-8k of 30k). Rank by trend, not final loss.
2. **Top-K full training**: Only fully train the 3-5 most promising variants.
3. **Early-stop aggressively**: Use patience-based stopping or GradES (gradient-based early stopping per component, 1.57-7.22x speedup).
4. **Compare at fixed token counts**: Track loss at 5k, 10k, 20k, 30k steps for fair comparison.
5. **Muon for all**: Don't sweep optimizers -- just use Muon. Focus the sweep budget on architecture.

---

## E. FLOW-MATCHING DIRECTIONS (don't give up yet)

The flow-based variants underperformed in Sweep 1, but the architecture is promising. These directions address likely failure modes:

### E1. Flow-Matched Memory State Evolution
**What**: Instead of static memory embeddings, maintain memory entries as points that evolve via a learned flow field as new information arrives.
**Why**: The flow-based models in Sweep 1 (flow_attention, flow_memory, fa_fm, fa_sm) applied flow matching to the attention mechanism itself. Instead, apply it to the MEMORY STATE. Memory doesn't store fixed embeddings -- it stores a state that evolves continuously.
**Failure mode addressed**: Flow matching on attention was too much architectural change at small scale. Flow matching on memory only is a smaller, more targeted change.
**Complexity**: High
**Risk**: High

### E2. More Epochs / Scale for Flow Variants
**What**: The flow variants may simply need more training. Run the best flow variant from Sweep 1 for 100k+ steps at larger scale (8 layers, 512 dims).
**Why**: Flow matching is mathematically richer but may need more data/epochs to converge. The Sweep 1 results showed flow variants close to baseline, not catastrophically worse.
**Complexity**: None (just longer training)
**Risk**: Low

### E3. Hybrid: Tonic Attention + Flow Memory
**What**: Use standard attention for the forward pass, but use a normalizing flow to learn the memory write/read transformation. The flow maps the raw token representation to a "memory-ready" representation and back.
**Why**: Decouples the flow from the attention mechanism. The flow only operates on the memory pathway, not the main computation graph. Much smaller surface area for the flow to learn.
**Complexity**: Medium
**Risk**: Medium

---

## F. CROSS-CUTTING IDEAS FROM COGNITIVE SCIENCE

### F1. Complementary Learning Systems (Dual Memory)
**What**: Two memory systems: hippocampal (fast, pattern-separated, orthogonal) and neocortical (slow, overlapping, distributed). Replay consolidates fast → slow.
**Why**: Directly maps to Flower's hierarchical memory. The "short-term" bank should use orthogonal representations (prevent interference). The "long-term" bank should use overlapping representations (compression). Consolidation = periodic transfer.
**Complexity**: Medium
**Source**: Cognitive science, McClelland et al. 1995

### F2. Dynamic Cheatsheet Finding (Quality Gating)
**What**: "Smaller models benefit from memory in limited amounts... they generate too few correct solutions... memory gets populated with flawed strategies."
**Why**: Critical warning for Flower at 16-20M params. Don't just increase memory capacity -- gate memory writes by confidence. Only store what the model is confident about. This argues for surprise-gating (A4) or orthogonal updates (A1) over simple capacity increases.

---

## G. WIKI PRIOR ART (from existing research)

These are from the wiki analysis, already available as context:

| Technique | Wiki Page | Relevance |
|-----------|-----------|-----------|
| DeepSeek Engram | [[deepseek-engram]] | Closest prior art. Lookup table parallel to attention, O(1) retrieval, vocab compression. |
| Nested Learning | [[nested-learning]] | Multi-frequency memory architecture. Optimizer design IS memory design. |
| Subquadratic Attention | [[subquadratic-attention]], [[subq]] | Memory augmentation IS subquadratic attention. Unexplored connection. |
| GOAT/OT Attention | [[optimal-transport-attention]], [[goat]] | OT enforces conservation laws on attention. Prevents degenerate patterns with memory. |
| BitNet Ternary | [[bitnet-ternary-training]] | 1.58-bit weights, 7x memory reduction. Unexplored with memory augmentation. |
| Flow Maps | [[flow-map-language-models]], [[discrete-flow-maps]], [[langflow]] | Three competing flow paradigms (Feb-Apr 2026). None consider memory. |
| ZAYA1/CCA | [[zaya1-8b]] | Learned Residual Scaling for stable memory injection. |
| SimpleSD | [[simple-sd]] | Self-distillation at optimal temperature. Post-training boost. |

---

## SUGGESTED SWEEP 2 PLAN

### Phase 1: Core Architecture (run all, 10k steps each, 1 GPU per trial)

| ID | Variant | What tests |
|----|---------|------------|
| S2.0 | vanilla_local | Baseline (re-run with Muon for fair comparison) |
| S2.1 | summary_hierarchical_max | Sweep 1 winner (re-run with Muon) |
| S2.2 | orthogonal_memory | A1: LATTICE orthogonal updates replacing max-pool |
| S2.3 | looped_4layer_x2 | A2: Loop 4-layer model 2x with memory persistence |
| S2.4 | looped_4layer_x3 | A2: Loop 3x |
| S2.5 | feedback_fam | A3: Feedback connections layer 4→1 |
| S2.6 | surprise_gated | A4: Surprise-gated memory writes |
| S2.7 | lpa_projected | A8: Low-dim projected attention (256→64) |
| S2.8 | flow_memory_v2 | E3: Standard attention + flow-transformed memory read/write |

### Phase 2: Top 5 from Phase 1, full 30k steps, both GPUs

### Phase 3: Scale winners to 8 layers, 512 dims, 100k+ steps (Sweep 3)

### Optimizer for ALL variants: Muon (hidden weights) + AdamW (aux)
### Schedule: Schedule-Free (no cosine/linear decay)
### Precision: bf16 autocast + FlashAttention-2 + torch.compile

---

## H. TOKENIZER INVESTIGATION: GPT-2 BPE IS LIMITING

### The Problem
Flower uses GPT-2 BPE tokenizer (50,257 vocab) with d_model=256.

**Parameter budget breakdown:**
- Embedding matrix (tied): 50,257 × 256 = **12.9M params**
- Transformer layers (4 layers): **3.1M params**
- Embedding fraction: **80.3%** of total parameters

This means 4 out of 5 parameters in the model are just lookup table entries.
The actual "thinking" part (transformer + memory) gets only 20% of the parameter budget.

### What the Research Says

**Scaling Laws with Vocabulary (NeurIPS 2024)** trained 33M-3B models and found:
- Optimal vocabulary size depends on compute budget
- Most LLMs use too-small vocabularies (e.g., Llama2-70B should use 216K, not 32K)
- BUT this is for LARGE models. The inverse is true for small models.
- Optimal embedding fraction is ~25-30% of total params

**Over-Tokenized Transformer (Jan 2025)** found:
- Scaling input vocabulary gives log-linear loss improvement
- But scaling OUTPUT vocabulary hurts small models
- A 400M model with 12.8M input vocab matches 1B baseline
- Key: decouple input vocab (can be huge) from output vocab (should be small)

**PanGu-π (Huawei, Tiny LM study, 2024)** found:
- Compact tokenizer crucial for tiny models
- Depth > width for small models
- Parameter inheritance from larger models helps

### Optimal Vocab for Flower

With 3.2M non-embedding params, targeting 25-30% embedding fraction:
- **Optimal vocabulary: ~3,000-5,000 tokens** (not 50,257!)

Alternatives:
| Vocab | Embed % | Total Params | Notes |
|-------|---------|-------------|-------|
| 256 (byte-level) | 2.0% | 3.2M | Pure bytes, 4x longer sequences |
| 1,024 | 7.7% | 3.4M | Compact BPE, 2-3x longer |
| 4,096 | 25.0% | 4.2M | Near-optimal per scaling law |
| 8,192 | 40.0% | 5.2M | Still much better than GPT-2 |
| 50,257 (GPT-2) | 80.3% | 16.0M | Current -- embedding dominated |

### Practical Options

**Option A: Train a custom BPE with 4K-8K vocab on FineWeb-Edu**
- Use tiktoken/sentencepiece to train a smaller BPE on FineWeb-Edu
- Fewer tokens per word means longer sequences but MUCH better parameter utilization
- Sequences would be ~2-3x longer (more compute per step, but better per-parameter efficiency)
- Would need to retokenize the dataset (one-time cost)

**Option B: Byte-level with local attention**
- 256-entry vocabulary, pure bytes
- Sequences ~4x longer, but embedding is negligible
- Works well with local attention windows (memory handles the length)
- BLT (Meta, 2025) showed byte-level can match tokenized LLMs at scale
- Natural fit for Flower's memory architecture: local byte processing + memory for long-range

**Option C: Over-Tokenized input, small output**
- Use large input vocabulary (multi-gram embeddings) but small output vocabulary
- The Over-Tokenized Transformer showed 5.7x convergence speedup
- Input embedding can be offloaded to CPU during inference (negligible latency)
- More complex to implement

**Option D: Character-level with longer context**
- ~128 vocabulary (printable ASCII + specials)
- Each "token" is one character
- Sequence is very long, but attention window + memory handles it
- Simplest tokenizer, no training needed

### Recommendation

The GPT-2 tokenizer is **severely limiting** Flower. The 80% embedding fraction is an enormous waste. Switching to a 4K-8K vocab trained on FineWeb-Edu would be the simplest fix. A byte-level approach with Flower's memory architecture would be the most architecturally clean (Keller's "data locality" principle -- simple tokens, complex memory).

This is probably the single highest-impact change to make before Sweep 2.

---

## I. CARMACK / KELLER SEEDS -- ARCHITECTURAL IDEAS

These are ideas inspired by John Carmack and Jim Keller's engineering philosophies, translated to Flower's context.

### I1. PVS-Inspired Memory (Precalculate Visibility)
**What**: Carmack's Quake breakthrough -- precompute which memory slots are "visible" (relevant) from which regions of input space. Store as a sparse bit vector. At runtime, just look up which slots to attend to.
**Why**: Eliminates the need for full attention over memory. O(1) routing, O(k) attention where k = precomputed visible set.
**Complexity**: Medium -- need a precomputation step (could be a learned router frozen after training)
**Risk**: Low

### I2. Virtual Texturing for Memory (Page in What You Need)
**What**: Carmack's megatexture system: divide memory into pages, keep hot pages in GPU tensor, cold pages in compressed form. Coarser mip = compressed summary as fallback when fine page not loaded.
**Why**: Don't keep all memories at full resolution. Hot memories = full detail. Cold memories = compressed summaries. Natural hierarchy.
**Complexity**: Medium
**Risk**: Low

### I3. BSP Front-to-Back Rendering (Cheap Elimination of 95% of Candidates)
**What**: Carmack's insight: don't solve the general problem, solve YOUR problem with the cheapest structure that works. Doom's implicit z-buffer was 5% of the cost for 90% of the benefit because the constraints simplified the problem.
**Why**: For memory retrieval, you don't need full attention. Use a cheap test (hash, LSH, dot product threshold) to eliminate 95% of memory slots, then only do expensive attention on the remaining 5%.
**Complexity**: Low-Medium
**Risk**: Low

### I4. Colocated Compute + Data (Keller's Partitioned Memory Banks)
**What**: Keller: break data into small pieces, put processing next to each piece. Don't have one big memory bank with attention over all of it.
**Why**: Partition memory into small local banks (16-32 slots each). Each bank has its own tiny readout. A lightweight router selects which bank(s) to query. O(1) per bank + O(num_banks) routing vs O(slots²) full attention.
**Complexity**: Medium
**Risk**: Low-Medium

### I5. Zero-Copy Memory Pipeline (Keller's Data Flow)
**What**: Keller: output of one computation should flow directly to the next without intermediate writes.
**Why**: Design so memory read flows directly into the next layer's computation, and the layer's output flows directly into the memory write. No separate "read from memory, then process" step.
**Complexity**: Low-Medium (architectural wiring)
**Risk**: Low

### I6. Branch Predictor for Memory Access (Keller's Found Parallelism)
**What**: Keller: CPUs use neural-network-like predictors to guess branch outcomes. Train a tiny predictor network that guesses which memory slots will be relevant for the next few tokens based on current hidden state.
**Why**: If the predictor is right 95% of the time, expensive attention only happens 5% of the time. The predictor learns the access pattern like a CPU branch predictor.
**Complexity**: Medium
**Risk**: Medium

### I7. Memory as the Architecture, Not the Bolt-On (Keller's Rewrite Principle)
**What**: Keller rewrites architectures from scratch every 5 years. Don't add memory to a transformer -- design a memory system with minimal attention on top.
**Why**: Invert the default. What if the primary mechanism IS memory retrieval, with a thin reasoning layer? The "transformer + memory bolt-on" approach may be fundamentally suboptimal.
**Complexity**: High (conceptual redesign)
**Risk**: High but aligned with the research goal

### I8. "Learn Slow So You Can Learn Fast" (Carmack on Plasticity)
**What**: Train backbone slowly with strong regularization, let memory bank be fast-adapting. Two learning rates.
**Why**: The weights build solid representations (slow), the memory captures specifics (fast). Directly from Carmack's observation that biological systems sacrifice initial speed for long-term adaptability.
**Complexity**: Low -- just different LRs for memory params vs rest
**Risk**: Very Low

---

## J. SUMMARY PRIORITY TABLE (ALL IDEAS)

| ID | Idea | Impact | Novelty | Complexity | Risk | Source |
|----|------|--------|---------|------------|------|--------|
| **TOK** | Smaller tokenizer (4K-8K vocab) | VERY HIGH | LOW | Medium | LOW | Scaling laws |
| A2 | Looped transformer (2-3x) | HIGH | MED | LOW | LOW | arxiv 2502.17416 |
| A1 | Orthogonal memory updates (LATTICE) | HIGH | HIGH | MED | LOW | ICML 2025 |
| A3 | Feedback connections (FAM) | MED-HIGH | MED | LOW | LOW | Google 2024 |
| A8 | Low-dim projected attention | MED | LOW | LOW | V.LOW | arxiv 2411 |
| I3 | Cheap pre-filter for memory retrieval | MED-HIGH | MED | LOW | LOW | Carmack/BSP |
| I8 | Dual LR (slow backbone, fast memory) | MED | LOW | V.LOW | V.LOW | Carmack/plasticity |
| C1 | Muon optimizer | MED | MED | LOW | V.LOW | Keller Jordan |
| A4 | Surprise-gated memory writes | MED | HIGH | MED | MED | Google Titans |
| I4 | Partitioned memory banks | MED | MED | MED | LOW | Keller/Tenstorrent |
| I1 | PVS precomputed visibility | MED | MED | MED | LOW | Carmack/Quake |
| A6 | Product key memory | MED | MED | MED | MED | DeepMind PEER |
| A7 | Hybrid SSM-attention heads | MED | MED | MED | MED | NVIDIA Hymba |
| A5 | Phase-associative memory (complex) | MED-HIGH | VERY HIGH | HIGH | HIGH | arxiv 2604 |
| I6 | Branch predictor for memory | LOW-MED | HIGH | MED | MED | Keller |
| E3 | Flow-transformed memory read/write | MED | VERY HIGH | MED | MED | Novel combo |
| A11 | E8 lattice memory | MED | VERY HIGH | HIGH | HIGH | NeurIPS 2021 |
| I7 | Memory-first architecture | HIGH | VERY HIGH | HIGH | HIGH | Keller philosophy |
