# Parameter-Efficient & Small-Model Architecture Research Notes

Sources surveyed: OpenAI Parameter Golf (github.com/openai/parameter-golf), arXiv:2606.19348 (DeepSeek-V4), Qwen3 (arXiv:2505.09388, incl. Qwen3-8B).

**Flower safety framing**: Changes that alter information pathways = DANGEROUS (confound memory-mechanism comparisons). Throughput/optimizer/init/eval tricks = SAFE.

---

## Source 1: OpenAI Parameter Golf

**What it is**: Competition to train the best LM fitting in a 16MB artifact, <10 min on 8×H100, scored on FineWeb val bits-per-byte. 2000+ submissions. Winning score: 1.1147 BPB (≈22M params, 11 layers, 512-dim, int6 quantized). Baseline: 1.2244 BPB.

**Scale**: 9-11 layer, 512-dim, ~22M params, vocab 1024-8192 BPE. This is BELOW Flower's 25M-600M range, but the *ideas* are architecture-level and many transfer.

### Ideas extracted

| Idea | What it does | New params? | Pathway change? | Flower verdict |
|------|--------------|-------------|-----------------|----------------|
| **XSA (Exclusive Self Attention)** [arXiv:2603.09078, Apple] | After standard attention output `y_i`, subtract projection onto self-value `v_i`: `z = y - (y·v̂)v̂`. Forces attention to capture ONLY contextual (orthogonal-to-self) info. Validated up to 2.7B; gains GROW with sequence length. | 0 | **YES — removes self-information from attention output** | ⚠️ DANGEROUS but theoretically aligned with Flower's thesis (see below) |
| **SmearGate** [#65] | Learned per-dim gate blends token emb with prev-token emb: `out = σ(g)·x + (1-σ(g))·x_prev`. ~512 params. | ~d_model | YES (adds bigram pathway pre-attention) | ❌ DANGEROUS — competes with memory's role of carrying prev-context |
| **BigramHash** [#65] | Hash table `(prev·92821+cur)%B` → learned pair embeddings. 2048-4096 buckets. | ~B×dim | YES (shortcut token-pair features) | ❌ DANGEROUS — micro-scale only, confounds |
| **Partial RoPE (16/64 dims)** [#315] | Apply rotary PE to only first 16 of 64 head dims; rest attend position-invariant. | 0 | Mild (changes positional encoding) | ⚠️ Borderline — zero-param, test as ablation |
| **LN Scale = 1/√(layer+1)** [#315] | Scale RMSNorm output by decay factor per layer. Damps deeper layers. | 0 | Minimal (scalar rescale, not routing) | ✅ SAFE-ish — stability trick, test it |
| **U-Net skip connections** [#289] | Encoder-decoder learned skip weights across layers. | moderate | YES (alternative information pathway) | ❌ Already in "What NOT to implement" |
| **Mini depth recurrence** [#1204, @msisovic] | Repeat layers 4&5, delay recurrence until mid-training, partially untie repeated MLPs. First leaderboard row to make recurrence work. | -50% params for repeated layers | YES (weight-shared depth) | ⚠️ Related to HRM-Text/Hyperloop in existing skill; partially novel (partial untying + delayed start) |
| **LeakyReLU(0.5)²** [#493] | Activation: `(LeakyReLU(0.5)(x))²`. Replaces ReLU². | 0 | Mild (changes nonlinearity) | ❌ DANGEROUS — SwiGLU preferred at Flower scale |
| **Orthogonal init** [#164] | `nn.init.orthogonal_()` on all weight matrices. Synergizes with Muon (NS step preserves orthogonality). | 0 | No (init only) | ✅ SAFE — free convergence boost, test |
| **Parallel Muon + Parameter Banking** [#399] | Run Muon NS iterations in parallel across params; "bank" params to reduce sync. | 0 | No | ✅ SAFE — throughput |
| **EMA weight averaging** [#401] | Exponential moving average of weights (decay 0.997), evaluated at inference. | 0 (inference) | No | ✅ SAFE — free quality boost |
| **Sliding window eval (stride 64)** | Overlapping eval windows so every token gets ~full context. Eval-only. | 0 | No (eval only) | ✅ SAFE — improves measured BPB without changing model |
| **GPTQ int6/int5 quantization** [#414,#535,#609] | Post-training weight quantization with Hessian-based calibration. AR self-gen calibration [#1019]. | 0 (export) | No | ✅ SAFE — inference/export only |
| **Logit softcap (30.0)** | Clamp logits to [-30,30]. | 0 | Minimal | Already in "What NOT" (overfit to target) |

### Parameter Golf — verdict for Flower
Most wins are **quantization + compression tricks** (irrelevant to Flower's training-time research) or **micro-optimizations** that confound memory comparisons. The two genuinely interesting architectural finds:

1. **XSA** — see dedicated analysis below.
2. **Mini depth recurrence with partial untying** — adds a nuance to the existing HRM-Text/Hyperloop coverage: *partially untying* shared layers and *delaying recurrence onset* until mid-training both helped. Worth noting if Flower ever tests recurrent depth.

---

## Source 2: arXiv:2606.19348 — DeepSeek-V4

**What it is**: DeepSeek-V4 preview. **NOT a small-model paper** — 1.6T (49B active) MoE and 284B (13B active) MoE. 1M-token context. The relevance to Flower is *architectural ideas*, not scale.

### Ideas extracted

| Idea | What it does | Scale tested | Pathway change? | Flower verdict |
|------|--------------|--------------|-----------------|----------------|
| **mHC (Manifold-Constrained Hyper-Connections)** [Xie et al. 2026] | Expands residual stream width by factor `hc`. Residual mapping constrained to **Birkhoff polytope** (doubly stochastic matrices) via Sinkhorn-Knopp (20 iters). Guarantees spectral norm ≤1 → non-expansive → stable deep stacking. Input/output maps Sigmoid-bounded. Dynamic (input-dependent) + static parameter decomposition. | 284B-1.6T | **YES — widens residual stream, adds parallel residual pathways** | ⚠️ DANGEROUS but novel stability insight for Hyper-Connections (see below) |
| **CSA (Compressed Sparse Attention)** | Compress KV cache τ:1 via learned weighted pooling, then DeepSeek Sparse Attention (top-k via "lightning indexer" with low-rank multi-head query). + sliding window branch + attention sink. | 1M ctx | YES (compression discards KV entries) | ❌ DANGEROUS — directly changes what info survives across context |
| **HCA (Heavily Compressed Attention)** | Same compression but τ′≫τ, no sparse selection. Cheaper, less granular. | 1M ctx | YES | ❌ DANGEROUS |
| **Hybrid CSA/HCA interleaved** | Alternate CSA and HCA layers. | 1M ctx | YES | ❌ DANGEROUS |
| **Hash routing for early layers** | Replace dense FFN in first N blocks with MoE using **hash-based routing** (deterministic, token-ID-based). Removes learned-router overhead at shallow depth. | 284B+ | YES (changes FFN routing) | ⚠️ DANGEROUS — but interesting for efficiency |
| **Sqrt(Softplus(·)) MoE affinity** | Replace `Sigmoid(score)` with `Sqrt(Softplus(score))` for expert affinity. | 284B+ | Mild (routing computation) | ⚠️ DANGEROUS if Flower uses MoE |
| **Hybrid Newton-Schulz (Muon)** | 10 NS iterations in 2 stages: first 8 with (3.4445, -4.7750, 2.0315) for rapid convergence, last 2 with (2, -1.5, 0.5) to pin singular values at 1. BF16-stable. | 284B+ | No | ✅ SAFE — optimizer improvement |
| **Attention sink (learnable sink logits)** | Learnable logits `s'_h` added to softmax denominator, allowing attention sum ≠ 1. | 284B+ | Mild (changes attention distribution) | ⚠️ Borderline — could interact with memory read |
| **Partial RoPE (last 64 dims)** | Apply RoPE to only last 64 dims of Q/KV; also apply to attention *output* with position offset for relative encoding. | 284B+ | Mild | ⚠️ Same as Parameter Golf finding |
| **RMSNorm on Q/K heads** | Normalize each Q head and the KV head before softmax. Prevents exploding logits. | 284B+ | Minimal | ✅ SAFE — this IS QK-Norm (see Qwen3) |
| **FP4 routed expert params** | Store/compute routed experts in FP4. | 284B+ | No | ✅ SAFE — precision (already in speedups doc §10) |

### DeepSeek-V4 — verdict for Flower
The headline ideas (CSA, HCA, mHC) are **large-scale long-context efficiency** mechanisms. They are DANGEROUS for Flower because they *are* alternative memory/compression pathways — testing them alongside Flower's bloom/summary memory would conflate two compression mechanisms.

**The one transferable insight**: mHC's core contribution is a *stability fix* for Hyper-Connections (the doubly-stochastic constraint via Sinkhorn-Knopp). If Flower ever experiments with expanded residual streams or Hyper-Connections (related to "Hyperloop" in the existing skill), the Birkhoff-polytope constraint is the key to making it trainable at depth. The hybrid Newton-Schulz schedule is a safe Muon upgrade.

---

## Source 3: Qwen3 (arXiv:2505.09388) — incl. Qwen3-8B

**What it is**: Qwen3 technical report. Dense models 0.6B-32B, MoE 30B-A3B / 235B-A22B. Note: `qwen.ai/blog?id=qwen3.8` resolved to **Qwen3.8-Max** (2.4T params, a different/later model built on Qwen3.5 architecture) — not the small-model source. The *small-model* architecture is Qwen3 itself.

**Qwen3-8B specs**: 36 layers, 32 Q heads / 8 KV heads (GQA 4:1), d_model≈4096, 8.2B params, 32K native context (131K with YaRN).

### Ideas extracted

| Idea | What it does | Scale | Pathway change? | Flower verdict |
|------|--------------|-------|-----------------|----------------|
| **QK-Norm** [Dehghani et al. 2023] | RMSNorm applied to Q and K **per head** (on head_dim) after projection, before RoPE. Stabilizes training — prevents attention-logit explosion. Removed need for QK-Clip. | 0.6B-235B | Minimal (normalization, not routing) | ✅ **SAFE — high-value, low-risk. Strongly recommend.** |
| **Removed QKV-bias** | Qwen2 had bias terms on QKV projections; Qwen3 removes them. | all | No | ✅ SAFE — Flower likely already bias-free |
| **Sliding window attention layers** | HF impl shows `layer_types` with `"sliding_attention"` — interleaved sliding-window and full-attention layers. | 8B+ | YES (limits attention range per layer) | ⚠️ Flower already has local_window; interleaving is a research variable |
| **GQA 4:1** (Qwen3-8B: 32Q/8KV) | Standard grouped-query attention. | all | No | ✅ SAFE — Flower should use GQA at 600M |
| **Strong-to-weak distillation** (logit distillation from large teacher) | Train small models by distilling from Qwen3-235B logits. 1/10 the GPU hours vs 4-stage RL. | 0.6B-32B | No (training method) | ℹ️ Not applicable — Flower trains from scratch |

### Qwen3 — verdict for Flower
Qwen3's architecture is **deliberately conservative** — the wins come from data (36T tokens), QK-Norm for stability, and distillation. The single most transferable finding is **QK-Norm**: it's a zero-parameter, zero-pathway-change stability fix that DeepSeek-V4 independently arrived at (RMSNorm on Q/K heads). Two independent frontier labs converging on it is a strong signal.

---

## Cross-source synthesis: What's NEW vs the existing skill

The existing skill covers: HRM-Text, Hyperloop, Ouro, Attractor Models, FLT, GatedDeltaNet, MoE for small models, linear/recurrent hybrids (3:1 ratio).

**Genuinely new ideas from these three sources:**

1. **XSA (Exclusive Self Attention)** — NOT in existing skill. Novel attention modification.
2. **mHC stability constraint (Birkhoff polytope / Sinkhorn-Knopp for Hyper-Connections)** — New nuance for the Hyperloop-adjacent area. The existing skill covers Hyperloop; mHC adds the *stability mechanism* that makes deep HC trainable.
3. **QK-Norm** — NOT in existing skill (it's a stability trick, not an architecture innovation per se, but frontier-standard now).
4. **Mini depth recurrence with partial untying + delayed onset** [#1204] — Adds "partial untying" and "delayed recurrence start" to the depth-recurrence coverage.
5. **Hash routing for shallow layers** (DeepSeek-V4) — deterministic routing for early FFN layers.
6. **Attention sink (learnable sink logits)** — implicit attention-sink mechanism.

Everything else (SmearGate, BigramHash, U-Net, quantization, compression, eval tricks) is either already covered, scale-inappropriate, or explicitly excluded.

---

## Recommendations for Flower

### Add to training-speedups.md (SAFE improvements)

1. **QK-Norm** — NEW SECTION. Apply RMSNorm to Q and K per-head after projection, before RoPE. Zero params (well, negligible: per-head scale). Zero pathway change. Two frontier labs (Qwen3, DeepSeek-V4) independently adopted it for stability. This is the highest-value safe addition.

2. **Orthogonal weight initialization** — Pair with Muon (Newton-Schulz preserves orthogonality). Zero pathway change, init-only. Parameter Golf found it accelerates early convergence.

3. **Hybrid Newton-Schulz schedule for Muon** — 10 iterations, 2-stage coefficients (rapid-converge then pin-to-1). DeepSeek-V4's refinement. Safe optimizer upgrade, complements existing §5 (NorMuon) and §6 (CWD).

4. **EMA weight averaging** — Evaluate with EMA(decay=0.997) of weights. Parameter Golf found consistent gains. Inference-time, zero training change.

5. **Sliding-window evaluation** — Overlapping eval windows (stride=64) so all tokens get near-full context. Parameter Golf showed large measured-BPB gains. Pure eval improvement, doesn't touch training.

### Do NOT add (DANGEROUS — confounds memory research)

- **XSA** — *theoretically interesting* (it forces attention to be purely contextual, which is exactly the role Flower's memory mechanisms play — so XSA + memory could be synergistic OR redundant). But it changes attention's information content, so it must be tested as a *research variable*, not a default. If tested, run it as a separate ablation axis: {vanilla, XSA} × {no-memory, bloom-memory, summary-memory}.
- **mHC** — widens residual stream, adds parallel pathways. Directly competes with memory mechanisms as an alternative info-routing channel.
- **CSA/HCA** — these *are* compression/memory mechanisms. Testing them alongside Flower's memory would conflate two compression designs.
- **SmearGate / BigramHash** — add shortcut pathways that could make external memory unnecessary.
- **U-Net skips** — already excluded.
- **Hash routing** — changes FFN routing.

### Research-variable candidates (test in dedicated ablation, not default)

- **XSA** — The most theoretically aligned idea for Flower's thesis. XSA removes self-information from attention, forcing attention to be purely about *gathering context*. Flower's memory mechanisms exist to carry context across the sliding-window boundary. Hypothesis: XSA makes the model *more dependent* on cross-window context → memory mechanisms become *more* important → larger memory signal. This is a testable, publishable hypothesis. But it must be an ablation axis, not a default.
- **Partial RoPE (16/64 or 64 dims)** — Zero-param. Could be tested as a minor ablation.

### Bottom line
Of ~25 ideas across the three sources, **5 are SAFE** (QK-Norm, orthogonal init, hybrid NS, EMA, sliding-window eval) and worth adding to training-speedups.md. **XSA is the one DANGER-zone idea worth a dedicated research ablation** because its mechanism (force attention to be purely contextual) directly intersects Flower's research question (does external memory help when local attention can't cover context?). Everything else is either already covered, scale-inappropriate, or explicitly excluded.
