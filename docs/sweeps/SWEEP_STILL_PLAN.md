# Still KV-Cache Compaction: Reproduction + Novel Variants

## Paper

**Still: Amortized KV Cache Compaction in a Single Forward Pass**
O'Neill et al. (arXiv:2606.07878, Jun 2026, Baseten)

## Core Idea

Still is a small per-layer Perceiver trained once against a frozen base model.
At inference, it takes the full per-layer KV cache and produces compact keys and
values in a single forward pass. The compact cache replaces the original prefix
cache for subsequent tokens.

Key design choices:
- Per-layer Perceiver with cross-attention (into [K;V] cache) + self-attention
- Position-free compaction: keys un-rotated, compactor uses own RoPE, re-rotated
- Identity-style initialization for stable early training
- KL distillation loss: forward KL from teacher (full cache) to student (compact)
- Top-200 teacher-vocabulary support with gold token forced in
- Canonical config: d_latent=256, B=2 blocks, nc=ns=1 heads, no FFN, beta=0

## What We Implement

### Core Architecture (`flower/models/still.py`)
- `StillCompactor`: Per-layer Perceiver KV-cache compactor
  - CompactorCrossAttention with QK-norm + d_latent logits scale
  - CompactorSelfAttention (latent refinement)
  - Position-free compaction (inverse-RoPE + re-RoPE)
  - Identity-style initialization (Appendix C)
  - Configurable compact_len, num_blocks, d_latent

### Training Wrapper (`flower/models/still_lm.py`)
- `StillLM`: frozen base model + per-layer compactors
  - Teacher pass: base model with full KV cache (no grad)
  - Student pass: base model with compacted KV cache (grad flows to compactor)
  - Top-k KL distillation loss
  - Optional CE auxiliary loss
  - step tracking for `compact_from_step` warmup

### Novel Variants (from wiki ideas)

1. **OT-Compactor** (`still_ot`): Sinkhorn OT coupling replaces softmax in
   cross-attention. Inspired by [[optimal-transport-attention]] and our
   flow_ot_memory. The transport plan enforces marginal conservation, giving
   a structured compression that doesn't allow attention to collapse.

2. **Energy-Read Compactor** (`still_energy`): Log-sum-exp sharpening of the
   compact latent representation before key/value projection. From
   [[agent-memory-architecture]] energy-based read and Sweep 7 B2. Beta controls
   the sharpness: low beta = mean-like, high beta = max-like.

3. **Frequency-Decay Compactor** (`still_freq`): Per-slot spaced-repetition decay
   applied to compact values between compaction passes. From
   frequency_decay_memory. Rarely-written slots are preserved more; heavily-
   written slots get decayed, preventing overwriting of distinctive content.

4. **Full Stack** (`still_all`): OT + energy + frequency decay combined.

5. **CE Auxiliary** (`still_ce`): KL + direct CE supervision (weight 0.5).

### Evaluation Probes (`flower/probes/still_eval.py`)
- `compaction_kl_curve`: teacher-student KL at multiple compression ratios
- `needle_through_compaction`: exact-retrieval after compaction
- `iterative_compaction_sweep`: quality degradation under repeated compaction
- `compression_utility`: normalized utility frontier

## What We Cannot Reproduce at This Scale

- **Model size**: Paper uses Qwen3-4B (4B params). We use ~12M param base
  models. The compactor ratio is ~16% vs paper's ~1%.
- **Context length**: Paper uses 8k-128k contexts. We use 512-2048.
- **Training data**: Paper uses 4-domain MCQ dataset with ~120k items. We use
  synthetic tokens or FineWeb-Edu.
- **8x H200 compute**: We have a single RTX 5090 (32GB).

The goal is not to match the paper's absolute numbers, but to:
1. Verify the architecture works (KL distillation, compaction, position-free)
2. Test whether our novel ideas (OT, energy, freq decay) improve on baseline
3. Develop evaluation methodology for compaction quality at small scale

## 5090 Feasibility

Qwen3-4B inference:
- 4B params in bf16 = ~8GB weights
- 8k context KV cache: 36 layers x 8 KV heads x 128 dim x 8192 tokens x 2 bytes x 2 (K+V)
  = ~150MB per layer x 36 = ~5.4GB
- Compactor forward: +50M params + activations
- Total: ~14GB for 8k context, well within 32GB

Qwen3-4B training:
- Forward + backward activations would be much larger
- Full-scale training (8x H200 in paper) is not feasible on a single 5090
- But inference + compaction at 8k should work, enabling evaluation

**Decision**: Focus on training compactors on our small Flower models (where we
control the base model), then potentially evaluate on larger open-weight models
(Qwen3-4B, Gemma-3-4B) if the compactor transfers.

## Sweep Config

`configs/sweep_still_compaction.yaml`: 10 variants, 3 seeds, synthetic data
for fast iteration. Variants:
- still_baseline (canonical Still)
- still_ot, still_energy, still_freq, still_all (novel variants)
- still_ce (CE auxiliary)
- still_c8, still_c16, still_c32 (compression ratio sweep)
- vanilla_local_control (no compactor)

## Implementation Status

- [x] StillCompactor module with identity init
- [x] StillCompactorOT (OT-coupled variant)
- [x] StillLM wrapper with KL distillation
- [x] Compaction eval probes
- [x] Config fields and variant registration
- [x] Sweep config
- [x] Tests (16/16 passing)
- [ ] Run on GPU with real data
- [ ] Evaluate on larger open-weight models
- [ ] Compare novel variants vs baseline

## Key Connections to Wiki

- [[per-layer-kv-cache-conditioning]]: Same pattern of per-layer KV access
- [[subquadratic-attention]]: Compaction is complementary (reduces KV cache size)
- [[rope-long-context-failure]]: Position-free compaction sidesteps RoPE issues
- [[optimal-transport-attention]]: OT-coupled compactor variant
- [[agent-memory-architecture]]: Energy-read variant, memory hierarchy connection
- [[efficiency-frontier]]: Compaction moves the Pareto frontier for context cost
- [[evaluation]]: Our composite probes add compaction-specific metrics
- [[reward-hacking]]: KL distillation avoids proxy compression issues
