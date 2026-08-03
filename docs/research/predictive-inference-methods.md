# Predictive/Speculative Inference Methods — Training vs Inference Classification

Research note comparing EAGLE, MTP, DSpark/DS4, Medusa, TST, and emerging methods.
Purpose: determine which methods offer **training-time** speedups applicable to
Flower (600M dense transformer, trained from scratch), vs which are inference-only.

**Key finding**: Most "speculative decoding" methods (EAGLE, Medusa, DSpark) are
inference-only. Only a small family of **auxiliary-loss / objective-modification**
methods (MTP, TST, FSP, MuToR) are training techniques. TST is **distinct from
and complementary to** MTP — the existing `training-speedups.md` Section 8 cites
TST as validation for MTP, but they are different methods targeting different
bottlenecks.

---

## 1. Classification Table

| Method | Category | Training or Inference | Changes loss surface? | Speedup type |
|---|---|---|---|---|
| **EAGLE 1/2/3** | Speculative decoding (draft model) | **Inference only** | No (draft trained post-hoc) | Inference latency |
| **DSpark (DS4)** | Speculative decoding (semi-AR draft) | **Inference only** | No | Inference latency + serving throughput |
| **DFlash** | Parallel draft speculation | **Inference only** | No | Inference latency |
| **Medusa 1/2** | Multi-head speculative decoding | **Inference only** (needs head training) | No (Medusa-1) / Minor (Medusa-2) | Inference latency |
| **MTP (Gloeckle / DeepSeek-V3)** | Auxiliary prediction objective | **TRAINING** | **Yes** | Sample efficiency (fewer tokens to target loss) |
| **TST (Token Superposition)** | Two-phase objective + input compression | **TRAINING** | **Yes** (reverts in phase 2) | Wall-clock (more tokens/FLOP) |
| **FSP (Future Summary Prediction)** | Auxiliary head (compressed future) | **TRAINING** | **Yes** | Sample efficiency (long-horizon) |
| **MuToR (Register MTP)** | Interleaved register tokens | **TRAINING** | **Yes** | Sample efficiency |
| **JTP (Joint MTP)** | Bottleneck teacher-forced MTP | **TRAINING** | **Yes** | Representation quality (early-stage) |
| **Token Order Prediction** | Auxiliary order head | **TRAINING** | **Yes** | Sample efficiency |

---

## 2. Inference-Only Methods (NOT useful for Flower training)

### 2.1 EAGLE (v1/v2/v3) — arXiv:2401.15077, 2406.16858, 2503.01840

- **How**: Trains a lightweight draft model that autoregressively predicts the
  target model's *second-to-top-layer features* (v1/v2) or directly predicts
  tokens via "training-time test" (v3). Draft proposes a tree of tokens; target
  verifies in one forward pass.
- **Category**: Pure inference acceleration. The draft model is trained *after*
  the main model is frozen. Does not touch the main training loss.
- **Speedup**: 2.7-3.5x (v1, LLaMA2-70B), 4x (v2), up to **6.5x** (v3, 13B).
- **Loss surface impact**: None on the target model.
- **Applicability to Flower**: **None for training.** Only relevant if Flower
  is ever deployed for inference — irrelevant to the training-speedups doc.

### 2.2 DSpark ("DS4") — arXiv:2607.05147 (DeepSeek, 2026)

- **How**: Semi-autoregressive draft = parallel backbone + lightweight serial
  head (kills "suffix decay") + confidence head + hardware-aware scheduler that
  skips verifying low-confidence tokens under load. Deployed in DeepSeek-V4.
- **Category**: Inference-only serving optimization. This is the "DS4" method.
- **Speedup**: 57-85% faster per-user generation at matched throughput vs
  MTP-1 baseline, in DeepSeek-V4 production.
- **Loss surface impact**: None.
- **Applicability to Flower**: **None for training.** Production serving tech.

### 2.3 DFlash — arXiv:2602.06036 (DeepSeek)

- **How**: Fully parallel draft backbone (no inter-token dependency within a
  block). Faster but suffers suffix decay — DSpark fixes this.
- **Category**: Inference-only.
- **Applicability to Flower**: None for training.

### 2.4 Medusa — arXiv:2401.10774

- **How**: Adds extra decoding heads to predict future tokens in parallel; tree
  attention verifies candidates. Medusa-1 freezes backbone, trains heads only.
  Medusa-2 jointly trains backbone + heads with a careful recipe.
- **Category**: Inference-only. Medusa-2's joint training is a fine-tuning step,
  not a pretraining objective.
- **Speedup**: ~2.3-2.8x latency at batch=1.
- **Applicability to Flower**: **None for training.** The head-training recipe
  could be repurposed but it targets inference, not sample efficiency.

---

## 3. Training-Time Methods (Candidates for the training-speedups doc)

### 3.1 MTP (Multi-Token Prediction) — ALREADY IN DOC (Section 8)

- **How**: k independent heads (Gloeckle) or D sequential causal-chain modules
  (DeepSeek-V3) predict future tokens. Auxiliary loss added to next-token loss.
- **Training/Inference**: Training objective. Heads can be discarded for
  inference (DeepSeek discards them; main model works standalone).
- **Speedup claim**: ~1.5-2x sample efficiency (fewer tokens to reach target
  loss). Used in DeepSeek-V3 (D=1), Qwen3-Next, Step 3.5, Nemotron 3.
- **Loss surface**: **Yes, changes it.** This is why Section 8 gates it behind
  "final runs only."
- **Scale dependence**: "Increasingly useful for larger model sizes" (Gloeckle);
  known to **degrade performance for smaller models** (<1B). At Flower's 600M,
  borderline — DeepSeek's own D=1 choice suggests conservative depth.
- **Newer developments (2025-2026)**:
  - **Pre-Training Curriculum for MTP** (Aynetdinov & Akbik, ACL 2025): forward
    curriculum (NTP→MTP) helps small models leverage MTP. Relevant if Flower
    adds MTP — start with NTP, ramp to MTP.
  - **Megatron-Bridge** has production MTP support (mtp_num_layers,
    mtp_loss_scaling_factor default 0.1).
- **Status in doc**: Correctly covered. Recommendation: keep as-is, optionally
  note the curriculum variant and the D=1 default from DeepSeek-V3.

### 3.2 TST (Token Superposition Training) — arXiv:2605.06546 (Nous Research)

**THIS IS THE KEY FINDING. The existing doc conflates TST with MTP. They are
different methods.**

- **How**: Two-phase training.
  - **Phase 1 (superposition)**: Average embeddings of contiguous s-grams into
    "s-tokens" (input compression → s-fold more data per FLOP). Predict the
    *next bag of s tokens* with **multi-hot cross-entropy (MCE)** — a single
    head, order-independent.
  - **Phase 2 (recovery)**: Revert to standard next-token prediction. Model
    quickly recovers and surpasses equal-FLOP baseline.
- **Training/Inference**: Pure training method. **Inference architecture is
  identical to baseline** — no extra heads, no draft model.
- **Speedup claim**: Up to **2.5x wall-clock** reduction at 10B-A1B MoE scale.
  Validated at **270M and 600M** (directly relevant to Flower). Same loss at
  ~half the compute.
- **Loss surface**: **Yes**, but only temporarily (phase 1). Phase 2 reverts to
  standard NTP, so the *final* model is directly comparable to baselines. This
  is a critical advantage over MTP for Flower's architecture-comparison
  research: the final model has the standard loss surface.
- **Why it's NOT MTP**: MTP adds k heads and an auxiliary loss but processes
  the same tokens/FLOP. TST changes tokens/FLOP via input compression and
  replaces the target (single head, MCE loss). The TST paper explicitly states:
  > "MTP and its variants do not increase training-time throughput... TST
  > occupies a different point in the design space... we view TST as orthogonal
  > to, rather than competing with, auxiliary-loss methods, and combining the
  > two is a natural direction for future work."
- **Hyperparameters**: bag size s ∈ [4,8], step ratio r ∈ [0.2,0.4]. Robust.
- **Limitation**: Trades compute for data — consumes s× more tokens. Only
  favorable when compute-bound (not data-bound). Flower is compute-bound.
- **Direct evidence of TST+MTP complementarity**: The AOMTS experiment series
  (hudsongouge/AOMTS-TST-s6-100M-3k, HuggingFace) tested all combinations at
  100M/3k steps:
  - Base (no TST, no MTP): 2.287 nats
  - MTP=1 alone: 2.276 (-0.011)
  - TST alone: 2.214 (-0.073)
  - **TST + MTP=1: 2.205 (-0.083, best)** — gains are additive.
  - TST + MTP=2: 2.215 (diminishing returns past 1 MTP head at this scale).

### 3.3 FSP (Future Summary Prediction) — arXiv:2510.14751 (Mahajan et al.)

- **How**: Single auxiliary head predicts a *compressed representation* of the
  future (τ=12-100 tokens ahead), not individual tokens. Two variants:
  handcrafted (bag-of-words) or learned (reverse-LM embedding, RevLM).
- **Training/Inference**: Training objective.
- **Speedup claim**: Up to +5% on math/coding benchmarks at 8B over NTP and MTP.
  Addresses MTP's weakness: MTP captures only short-range dependencies.
- **Loss surface**: Yes (auxiliary head + loss).
- **Applicability to Flower**: **Promising but heavier.** Requires training a
  reverse LM (RevLM) offline to generate targets. More complex than MTP/TST.
  Defer unless long-horizon reasoning is a Flower goal.

### 3.4 MuToR (Register-based MTP) — NeurIPS 2025

- **How**: Inserts interleaved learnable "register" tokens into sequences; each
  register predicts a future token at offset d. Single shared embedding +
  positional encoding of offset. No extra output heads.
- **Training/Inference**: Training objective. Compatible with SFT and PEFT.
- **Speedup claim**: Outperforms both parallel MTP and DeepSeek sequential MTP
  on math/summarization. Works for vision + language.
- **Loss surface**: Yes (auxiliary register loss).
- **Applicability to Flower**: Interesting alternative to MTP — fewer
  parameters (single embedding vs k heads), scales prediction horizon cheaply.
  But increases sequence length (more compute/step). Newer, less battle-tested.

### 3.5 JTP (Joint Multi-Token Prediction) — arXiv:2503.21801

- **How**: Teacher-forces future tokens through a lightweight bottleneck
  ("Fetch" module, self-attention only, no MLP) to enrich the hidden state with
  multi-step "planning" info. Models joint distribution, not independent heads.
- **Training/Inference**: Training objective. Minimal overhead (lightweight
  module).
- **Status**: Preliminary/work-in-progress. Validated mainly on synthetic star-
  graph tasks. Not yet proven at Flower's scale.
- **Applicability to Flower**: Watch but don't adopt yet.

### 3.6 Token Order Prediction — Zuhri et al.

- **How**: Single auxiliary head predicts the *relative order* of future tokens
  instead of the tokens themselves. Simpler than MTP, single head.
- **Applicability to Flower**: Niche; less evidence than MTP/TST.

---

## 4. Recommendations for `training-speedups.md`

### 4.1 CRITICAL FIX: Separate TST from MTP in Section 8

The current Section 8 cites TST (arXiv:2605.06546) as "validation" for MTP
("Also validated at scale: TST... shows 1.8-2.5x wall-clock speedup"). This is
**misleading** — TST is a distinct method, not evidence that MTP works. The TST
paper explicitly positions itself as *orthogonal* to MTP.

**Recommended action**: Split Section 8 into:
- **Section 8a: MTP** (auxiliary prediction heads — sample efficiency)
- **Section 8b: TST** (token superposition — wall-clock throughput)

### 4.2 ADD: TST as a new section (high priority for Flower)

TST is arguably **better suited to Flower than MTP** because:
1. **Directly validated at 600M** (Flower's target scale) by the original paper.
2. **Final model is architecturally identical to baseline** — phase 2 reverts
   to standard NTP. This preserves comparability for Flower's architecture
   research (the doc's core concern). MTP permanently changes the loss surface.
3. **2.5x wall-clock** is a bigger, more reliable speedup than MTP's ~1.5x
   sample efficiency.
4. **Drop-in**: no architecture, optimizer, tokenizer, or data changes.
5. **Orthogonal to MTP**: AOMTS experiments show TST+MTP=1 is additive.

Suggested placement: **above MTP in priority** (currently Section 8 is last).
TST should be a "final runs" technique like MTP, but it's the stronger option.

### 4.3 ADD: MTP Curriculum note (minor)

If MTP is used, cite Aynetdinov & Akbik (ACL 2025): forward curriculum (NTP→MTP)
helps small models. Relevant since Flower is at the small end of MTP's useful
range.

### 4.4 WATCH (do not add yet): FSP, MuToR, JTP

- **FSP**: Strongest evidence beyond MTP for reasoning tasks, but requires
  offline reverse-LM training. Add if Flower targets long-horizon reasoning.
- **MuToR**: Elegant, parameter-efficient MTP alternative. Needs more scale
  evidence.
- **JTP**: Too preliminary.

### 4.5 DO NOT ADD (inference-only, confirmed)

EAGLE, DSpark/DS4, DFlash, Medusa — all inference-only. None help training.
They belong in an inference-serving doc, not training-speedups.

---

## 5. Summary Decision Matrix for Flower (600M, from scratch)

| Method | Add to doc? | Priority | Why |
|---|---|---|---|
| **TST** | **YES (new section)** | **HIGH** | 2.5x wall-clock, validated at 600M, final model = baseline arch |
| **MTP** | Keep (fix TST citation) | MEDIUM | Already covered; D=1, curriculum variant |
| **FSP** | Watch | LOW | Needs reverse-LM; defer |
| **MuToR** | Watch | LOW | Promising but new |
| **JTP** | No | — | Too preliminary |
| EAGLE/DSpark/DFlash/Medusa | No | — | Inference-only |
