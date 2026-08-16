# LLM Optimizer State-of-the-Art — August 2026

Research survey for the Flower project (Muon + AdamW, RTX 5090/Blackwell sm_120,
100M–600M models, seq 8192–32768). Focus: what could train faster/better than
the current Muon(quintic5) + CautiousAdamW setup.

Key takeaway up front: **The field has converged on a clear hierarchy.** Against
well-tuned AdamW, no optimizer exceeds ~1.4× token-efficiency speedup at the
100M–1B scale, and that advantage shrinks toward 1.1× as you approach 1B.
Within that envelope, the Muon family still leads on simplicity/throughput, but
OKLS (Online KL Shampoo) and KL-SOAP now match-or-beat it on parameter
efficiency. The biggest practical wins for a single-GPU project are likely
(a) memory-compression (LoRA-Pre) to fit larger models, (b) cheaper NS
iterations (Muon2, CANS, cubic5), and (c) the zero-staleness insight (even one
step of delayed preconditioning destabilizes).

---

## 1. MUON FAMILY EVOLUTION

### 1.1 OKLS — Online KL Shampoo (Tilde Research) ⭐ TOP CONTENDER
- **Paper/source**: Zhang, Keigwin, Pai, Dewulf — "Online KL Shampoo",
  blog.tilderesearch.com/blog/online-kl-shampoo, July 2026. Builds on
  arXiv:2509.03378 "Understanding & Improving Shampoo/SOAP via KL Minimization"
  (Lin et al., Microsoft, ICLR 2026).
- **Core mechanism**: Zero-staleness Kronecker-factored optimizer that
  approximates full-matrix AdaGrad. Updates KL-optimal left+right covariance
  factors AND computes fresh inverse-square-root preconditioners within the
  same step (no delay). Uses **Scaled CANS Coupled Newton–Schulz** (10 iters,
  27 FP16 GEMMs w/ FP32 accumulation) to make zero-staleness affordable.
- **Measured**: **1.45× Muon's parameter efficiency** while retaining **~98% of
  Muon's training throughput**. A 200M–1B OKLS model matches a Muon model ~1.5×
  larger. muP scaling rules derived from first principles and validated.
- **Practicality for Flower**: Memory cost = momentum (m,n) + two (m,m) + two
  (n,n) factor copies in FP32. For 600M this is significant but fits 32GB at
  the 100M–600M range. The covariance factors add ~5× the weight matrix size in
  state — much heavier than Muon (1× momentum). Throughput at 98% of Muon is
  the key enabler.
- **Code**: github.com/tilde-research/online-kl-shampoo-release (drop-in
  torch.optim.Optimizer; `OnlineKLShampoo`).
- **Key insight**: Muon = spectral-norm steepest descent (ignores gradient
  history); OKLS = AdaGrad family (captures correlations + history). The
  zero-staleness principle is the differentiator vs. SOAP/Shampoo.

### 1.2 Compositional Muon (Tilde Research) ⭐ BEST FOR ATTENTION
- **Paper/source**: Dewulf et al. — "Towards Compositional Steepest Descent",
  blog.tilderesearch.com/blog/compositional-muon, June 2026.
- **Core mechanism**: Extends Muon from individual matrices to *composed*
  transformer circuits (QK^T, OV). Derives partner-whitened update rules where
  each factor's gradient is rescaled by the spectral geometry of its partner.
  For QK: head-local half-split rule. For OV: hybrid (V per-head, W_O as
  single aggregating matrix).
- **Measured**: Consistent gains over Muon at 340M and 1B. Set nanoGPT speedrun
  Track-3 record (PR #311): reached 3.28 target at step 2875 vs baseline 2890
  (~15 steps faster from a single coupled WV↔WO update).
- **Practicality**: Cheap "isotropic rule" approximation = partner-rescaled
  Muon, near-zero overhead. Full version needs coupled NS on paired matrices.
- **Code**: github.com/tilde-research/comp-muon-release
- **STATUS: SCREENED AND REJECTED (2026-08-12).** CM beats the baseline
  (-0.0115 val_bpb) but **loses to plain per-head Muon by +0.016**, which is
  free where `cm_full` costs 10% throughput. The step-size control
  (`cm_full_mp_0p5`) went the wrong way, ruling out mis-scaling. Mechanism: the
  partner scalar `s_K[h]` varies per head, so CM orthogonalises per head and
  then re-introduces the cross-head scale variation that per-head
  orthogonalisation removes. Isotropic and full whitening measured within 0.0005
  of each other, so the granularity is irrelevant — the whitening itself is the
  defect. Full write-up and the one open follow-up:
  `docs/profiling/comp_muon_screen_results.md`. Keep `muon_per_head: true`.
- **IMPLEMENTED** (`training.comp_muon`, `flower/optim.py`). The gate
  `configs/muon_screen_450m.yaml` set for the heavier methods — "worth doing only
  if the cheap arms show the optimizer is a live axis at this scale" — was opened
  by that screen: `muon_per_head` measured **-0.027 val_bpb** at 1500 steps,
  ~2x the 600-step seed band. CM is the principled version of exactly that arm
  (its QK rule is head-local by construction, plus the partner whitening plain
  splitting lacks), so it is the natural follow-up rather than a new direction.
  Both whitening regimes are vendored: `comp_muon_isotropic: true` is the scalar
  Gram-root approximation (degenerates to partner-rescaled per-head Muon),
  `false` is the full coupled-Newton-Schulz inverse root. The reference's other
  knobs (joint budget, gauge/connection Sylvester solve, momentum reprojection,
  leg-norm-restore) are ablations and are deliberately not vendored. Screen:
  `configs/comp_muon_screen_450m.yaml`; tests: `tests/test_comp_muon.py`.
  Deviation from the reference: uses Flower's Newton-Schulz (`muon_ns_schedule`)
  rather than 8-step polar_express, so a CM arm differs from `muon_baseline` in
  the CM rule only and not also in the orthogonaliser.
- **Key insight**: Optimizer-architecture co-design — the loss sees *compositions*
  (QK^T, OV), not individual matrices. Updating each factor independently
  controls the wrong object.

### 1.3 Aurora (Tilde Research) — ALREADY IN PROJECT
- **Paper/source**: Dewulf, Pai, Yang, Zhang, Keigwin —
  blog.tilderesearch.com/blog/aurora, May 2026.
- **Core mechanism**: Solves steepest descent under *joint* constraints of
  row-norm uniformity + orthogonality. Fixes Muon's neuron-death pathology in
  MLP layers (tall matrices inherit row-norm anisotropy → >25% of MLP neurons
  die permanently). 2 damping iterations, β=0.5.
- **Measured**: +25 steps over NorMuon on nanoGPT (pure); NEW SOTA 3175 steps
  with Aurora + Contra-Muon + update/weight flooring. Gains scale with MLP
  width (large expansion factors benefit most).
- **Practicality**: Untuned Aurora = only ~6% overhead over Muon, drop-in
  replacement. Implemented in flower already (class `Aurora`, `flower/optim.py`).
- **Code**: github.com/tilde-research/aurora-release
- **Update check**: Project implementation should verify it uses pp_iterations=2,
  pp_beta=0.5 defaults and applies to MLP up/gate projections specifically.
- **STATUS: SCREENED (2026-08-12).** Now run for the first time. Works, but not
  enough to pay for itself: **-0.010 val_bpb at -7.9% throughput** (57,846 vs
  62,809 tok/s). It beats the baseline and loses to per-head Muon, which is
  free. The throughput cost matched the isolated optimizer bench's -9%
  prediction; note the earlier CPU measurement (2.4x optimizer step)
  **understated** the GPU figure (4.65x) — CPU optimizer timings do not
  transfer. Do not deploy. See `docs/profiling/comp_muon_screen_results.md`.
- **Previously: IMPLEMENTED BUT NEVER RUN.** Vendored, config-gated
  (`optimizer: aurora`, `aurora_pp_iterations`, `aurora_pp_beta`,
  `aurora_weight_decay`) and exposed on the CLI, but no config had ever set it
  and no run directory exists — it was dead code carrying an untested
  hypothesis. Now an arm in `configs/comp_muon_screen_450m.yaml`. Worth noting
  its neuron-death claim targets the same pathology as NorMuon, whose arm in the
  first screen was catastrophic (+0.517 val_bpb) for an unrelated reason since
  fixed. `aurora_weight_decay` defaults to 0.0 here but the release ships 0.025,
  so the screen runs both.

### 1.4 NorMuon / U-NorMuon / Contra-Muon / Soft-Muon (nanoGPT community)
- **NorMuon** (arXiv:2510.05491, Li Zichong): Scales each row by inverse RMS
  norm after orthogonalization (Adam-like per-row normalization). Was Track-1
  SoTA (record #41, 2.345 min). **U-NorMuon** variant normalizes tall matrices
  to uniform row norms — prevents neuron death (same insight as Aurora).
- **Contra-Muon** (github.com/nilin/contra-muon): After Muon's NS update,
  subtract a fraction of the operator-normalized momentum gradient to boost
  small singular directions (CONTRA_MUON=0.4 → subtract 0.2×normalized_grad).
  Exaggerates Muon, increases singular-value diversity. Multiple Track-3
  records (PR #275: 3225 steps, PR #301: 2995 steps w/ NorMuon+Contra).
- **Soft-Muon**: Convex combination of NS iterates → underweights small SVs vs
  Muon (middle ground between SGD and Muon).
- **Practicality**: All near-zero overhead, composable. **Contra-Muon is a
  trivial 1-line add to the existing Muon.step()** — worth trying immediately.

### 1.5 SOAP / KL-SOAP (NVIDIA NeMo) ⭐ STRONGEST AT SCALE
- **Paper/source**: Khona, Vavre, Wang et al. (NVIDIA) — "SOAP, Muon, and
  Beyond: Pushing LLM Pretraining Scales", arXiv:2607.20548, July 2026.
- **Core mechanism**: Diagnosed SOAP's "slingshot" instability at large batch —
  root cause = preconditioner staleness (QR eigenbasis recomputed every 10
  steps EXCLUDING current gradient). Fix: (1) per-step QR with current gradient
  included, (2) KL-divergence covariance accumulation (from KL-Shampoo).
- **Measured**: On Qwen-3-30B-A3B and 72B Hybrid Mamba-Transformer: SOAP & Muon
  consistently beat AdamW; **KL-SOAP has slight edge over Muon**. Both stable
  at batch sizes up to 100M tokens where AdamW degrades. NVIDIA explicitly
  recommends KL-SOAP over Muon when memory isn't limiting.
- **Practicality**: Heavy memory (Kronecker factors + Adam in eigenbasis). For
  single 32GB GPU at 100M–600M, OKLS is the more practical KL-flavored option.
- **Code**: github.com/NVIDIA-NeMo/Emerging-Optimizers (Megatron-LM integrated).
- **Key insight**: **Zero-staleness is non-negotiable.** Even one step of
  delayed preconditioning destabilizes. This validates OKLS's design choice.

### 1.6 SODA — Optimistic Dual Averaging (EPFL LIONS) ⭐ UNIFYING FRAMEWORK
- **Paper/source**: Pethick, Xie, Machacek, Cevher — arXiv:2605.11172, May 2026.
- **Core mechanism**: Generalizes Optimistic Dual Averaging → unifies Muon,
  Lion-K, NAdam, AdEMAMix as special cases. Provides a **SODA wrapper** (zero
  new hyperparameters) that eliminates weight-decay tuning via theoretically-
  grounded **1/k decay schedule** anchored at initialization.
- **Measured**: Wrapper consistently improves Adam, Muon, Scion across scales
  and horizons — even beats baselines *with tuned weight decay*. SODA†
  (optimistic variant) best under 1× and 4× Chinchilla.
- **Practicality**: Wrapper is lightweight, wraps ANY base optimizer including
  the project's Muon. Directly addresses the weight-decay tuning pain point.
- **Code**: github.com/tmpethick/soda_code (soda_wrapper.py)
- **Key insight**: Weight decay isn't just regularization — it's primal iterate
  averaging. The 1/k schedule falls out of the theory and transfers across
  horizons without tuning.

### 1.7 Newton-Muon (arXiv:2604.01472) ⭐ PRINCIPLED, EASY ADD
- **Paper/source**: Zhehang Du — arXiv:2604.01472, April 2026.
- **Core mechanism**: Right-preconditions gradient by inverse activation second
  moment BEFORE Newton-Schulz: `W ← W − η·msgn(G(ZZ^T)^−1)`. Derives Muon as a
  special case (when ZZ^T ∝ I, which it ISN'T in practice — highly anisotropic).
- **Measured**: 6% fewer iterations, ~4% wall-clock reduction over Muon on
  nanoGPT Record #4 reproduction. 1.8% per-step overhead from preconditioning.
- **Practicality**: Maintains running ZZ^T estimate per layer, recomputes
  inverse periodically via Cholesky. Cost ≤ 10× a single matmul. **The only
  algorithmic change is right-preconditioning — standard Muon runs after.**
  Very implementable. MLP contraction matrix uses 4-block diagonal approx.
- **Code**: github.com/zhehangdu/Newton-Muon
- **Key insight**: Muon implicitly assumes isotropic input activations.
  Incorporating the actual (anisotropic) input geometry is a free win.

### 1.8 Muon² (arXiv:2604.09967) ⭐ CHEAPEST NS COST REDUCTION
- **Paper/source**: arXiv:2604.09967, April 2026.
- **Core mechanism**: Apply Adam-style per-parameter second-moment scaling to
  the momentum matrix BEFORE orthogonalization. Improves spectral conditioning
  of the NS input (shifts singular values 10× larger, fewer in "dead zone").
- **Measured**: **Reduces required NS iterations by 40%** (3-step Muon² beats
  5-step Muon). Saves up to 25% GPU-hours over Muon at same loss. Scaled to 13B
  MoE. **Muon²-F** (Adafactor-style factored second moment) ≈ same gains, ~zero
  extra memory.
- **Practicality**: Muon²-F is nearly free — factored second moment is just 2
  vectors. Directly relevant to Flower's NS-cost bottleneck (cubic5 already
  cuts NS matmuls 15→10; Muon²-F could cut further to ~9).
- **Code**: Not yet found in search (paper from major lab; check HF papers).
- **Key insight**: The NS iteration's enemy is ill-conditioned input. Pre-whiten
  with a cheap diagonal/factored preconditioner and NS converges in fewer steps.

### 1.9 Scion (EPFL LIONS, arXiv:2502.07529) ⭐ MOST MEMORY-EFFICIENT
- **Paper/source**: Pethick, Xie, Antonakopoulos, Zhu, Silveti-Falls, Cevher —
  arXiv:2502.07529, Feb 2025 (ICML 2025).
- **Core mechanism**: Norm-constrained LMO (linear minimization oracle)
  optimizer. Unifies SGD, SignSGD, Lion, Muon under one framework. SCION uses
  operator (spectral) norms. **Zero-shot hyperparameter transferability** across
  model sizes.
- **Measured**: Significant speedups on nanoGPT **without any Adam**. Requires
  only 1 set of weights + 1 gradient (half-precision) — most memory-efficient
  of the matrix methods. Scales to 3B.
- **Practicality**: Extremely memory-lean. Config: Sign→Spectral→Sign across
  layers. Good candidate if memory is the binding constraint.
- **Code**: github.com/LIONS-EPFL/scion (ScionLight reuses p.grad).
- **Key insight**: Hyperparameter transferability is a first-class feature —
  tune once on a proxy, transfer to target. No Adam dependency at all.

### 1.10 SAM-Muon / SpecSAM-Muon (arXiv:2607.26001)
- **Paper/source**: "Sharpness-Aware Minimization and Muon: Robustness under the
  Spectral Norm", arXiv:2607.26001, July 2026.
- **Core mechanism**: Spectral inner perturbation (layerwise spectral-norm SAM)
  + Muon outer step. The spectral inner geometry selectively amplifies Muon's
  advantage (doesn't help AdamW much).
- **Measured**: Best validation accuracy on ViT-S/16 (80.23) and ResNet-50
  (78.55) ImageNet. SpecSAM-Muon > SAM-Muon > SAM-AdamW.
- **Practicality**: 2× forward/backward cost of SAM. Vision-focused; LLM
  applicability unproven. Lower priority for Flower.
- **Key insight**: Inner/outer geometry should match — spectral perturbation
  pairs with spectral (Muon) outer update.

### 1.11 Per-Head Muon / Group Muon (Kimi K3, nanoGPT) ⭐ EASY WIN
- **Sources**: Kimi K3 (arXiv:2607.24653, §2.5); Group Muon (arXiv:2605.08933);
  nanoGPT PR #253 (samacqua, paired-head Muon).
- **Core mechanism**: Orthogonalize attention head blocks SEPARATELY instead of
  the full QKV matrix. Per-head equalizes update scale across heads (large-scale
  heads no longer dominate). Group Muon treats group size + grouping rule as
  hyperparameters.
- **Measured**: Kimi K3 uses it at 2.8T params for stability. nanoGPT PR #253:
  -10 steps (1480 vs 1490). Group Muon: intermediate grouping (g=6, random)
  beats both full-QKV and head-wise MuonSplit.
- **Practicality**: **Trivial to implement** — reshape momentum along head dim,
  run NS per head/group. Slightly reduces overhead (taller blocks cheaper).
  Stage-dependent: fine-splitting helps early, coarser grouping wins later.
- **Key insight**: Muon's matrix scope is a design choice, not a fixed property.
  Attention heads are the natural unit.

---

## 2. ENTIRELY DIFFERENT PARADIGMS

### 2.1 Lion / Lion-K family
- **Lion** (Chen et al., arXiv:2302.06675): sign(momentum) update, no second
  moment. Memory = 1 momentum buffer (vs Adam's 2). Needs larger weight decay
  (~0.6 vs AdamW ~0.1), smaller LR. Advantage grows with batch size.
- **Lion-K** (arXiv via UT Austin slides): generalizes sign→∇K for any convex K.
  Lion is K(x)=‖x‖₁. Lyapunov analysis shows it solves constrained min of
  f(x) + (γ/λ)K*(x).
- **2026 status**: Subsumed under SODA framework. On well-tuned LLM benchmarks,
  Lion underperforms Muon/SOAP matrix methods. Still memory-efficient baseline.

### 2.2 PSGD — Preconditioned SGD (Xi-Lin Li)
- **Source**: github.com/lixilinx/psgd_torch (active, v2.0 Dec 2025); JAX port
  github.com/evanatyourservice/psgd_jax. Theory: arXiv:2402.11858.
- **Core mechanism**: General 2nd-order optimizer. Fits preconditioner Q on Lie
  groups (Kronecker, affine, low-rank). Kron variant = matmul-only, inverse-free,
  recovers Newton-Schulz for inverse 4th root of E[gg^T]. "Whitening" mode uses
  gradient outer products (no Hessian-vector products needed).
- **Practicality**: Kron variant is a drop-in Adam replacement, fewer
  hyperparameters. Mature, well-maintained codebase. Less LLM-benchmarked than
  Muon but strong on diverse tasks. Worth a bakeoff run.
- **Key insight**: Preconditioner fitting is strongly convex on Lie groups →
  multiplicative updates avoid explicit matrix inverse.

### 2.3 Sophia (arXiv:2305.14342)
- **Core mechanism**: Diagonal Hessian estimate (Hutchinson or Gauss-Newton-
  Bartlett) every k steps as preconditioner + element-wise clipping.
- **2026 status**: Original claim 2× over Adam didn't hold under fair tuning
  (Wen et al. reality check). Still competitive on validation loss in multi-epoch
  small-model regimes. Bias-correction framework (arXiv:2605.20756) gives
  Sophia -0.07 nats on Qwen2.5-0.5B. Largely superseded by matrix methods.

### 2.4 Schedule-Free (Defazio) + SF-NorMuon / AMUSE
- **Schedule-Free** (Defazio et al., NeurIPS 2024): removes LR schedule via
  aggressive iterate averaging. Won AlgoPerf self-tuning track.
- **ScheduleFree+** (arXiv:2605.19095): fixes for large batch + model scale.
  Polyak step → fully LR-free. 31% training-time reduction at 1000 TPP.
- **SF-NorMuon** (arXiv:2605.23061): Schedule-free + spectral (NorMuon).
  Matches tuned AdamW across 1–8× Chinchilla at 125M & 772M WITHOUT a schedule.
  35–52% speedup over SF-AdamW. Proves Ō(T^-1/4) stationarity.
- **AMUSE** (OpenReview, ICLR 2026 sub): Anytime Muon + Stable Evaluation.
  Shifts gradient evaluation from fast Muon sequence → stable averaged sequence.
- **Practicality**: SF-NorMuon is highly relevant for Flower — eliminates
  schedule tuning, supports anytime checkpointing. Composes with existing Muon.

### 2.5 Adafactor / Factored Second-Moment
- Still relevant as the *factored* building block inside Muon²-F and others.
  Not competitive standalone vs Muon at these scales.

---

## 3. PRACTICAL FINDINGS (THE REALITY CHECKS)

### 3.1 The NVIDIA / Wen et al. reality check ⚠️ CRITICAL CONTEXT
- **"Fantastic Pretraining Optimizers and Where to Find Them"** (Wen et al.,
  arXiv:2509.02046): Rigorous study of 10 optimizers, 0.1B–1.2B, 1–8× Chinchilla.
  **Against well-tuned AdamW, no optimizer exceeds 1.4× speedup.** Matrix-based
  optimizers (Muon, SOAP, Kron, Scion) consistently beat scalar-based, but the
  speedup DECREASES with model size: 1.4× at 100M → 1.1× at 1.2B.
- **Caveat (arXiv:2607.xxxxx, NYU/Andrew Wilson)**: "Hyperparameter Transfer
  Enables Consistent Gains" — with correct µP + 1/width weight decay scaling,
  Muon/SOAP/Shampoo hold ~1.4× consistently to 1.4B. The shrinkage in Wen et al.
  may reflect poor HP transfer, not fundamental convergence.
- **Implication for Flower**: At 100M–600M, Muon-family gains are real and near
  the ceiling. Don't expect >1.5× from switching optimizers alone. The win is
  fitting a LARGER model in the same memory (LoRA-Pre) or cheaper NS (Muon²-F).

### 3.2 LR matters more than optimizer at scale
- Wen et al.: tuning a single HP (LR) in the GPT-3 recipe yields up to 2×
  speedup at 100M — bigger than any optimizer switch. Optimizer HP transfer is
  non-trivial (Lion WD ≈ 0.6 vs AdamW ≈ 0.1).

### 3.3 LoRA-Pre (ICLR 2026 Oral) ⚠️ KEY FOR FITTING BIGGER MODELS
- **Paper**: "Taming Momentum: Rethinking Optimizer States Through Low-Rank
  Approximation", arXiv:2602.24283. Code: github.com/mrflogs/LoRA-Pre.
- **Core**: EMA momentum ≡ online linear regression. Decompose momentum m =
  m_B·m_A (rank r ≪ min(p,q)). Memory: p×q → (p+q)×r. Closed-form updates, no
  backprop. Works for BOTH Adam and Muon.
- **Measured**: SOTA 60M→1B pretraining. **1/8 the rank of GaLore** for same
  quality. +3.14 pts on Llama-3.1-8B fine-tuning over LoRA.
- **Practicality for Flower**: Directly addresses the "fit 1B on 32GB" goal.
  Muon momentum is the largest optimizer state — compressing it 8× is enormous.
  **High-priority integration candidate.**

### 3.4 Zero-staleness preconditioning
- OKLS and NVIDIA's SOAP fix both independently discovered: **even one step of
  delayed preconditioner application destabilizes training** (loss spikes,
  "slingshot" oscillations). Compute fresh inverse-roots every step.
- Implication: any SOAP/Shampoo-style method must be zero-stale. Muon is
  inherently zero-stale (no persistent preconditioner) — part of its robustness.

---

## 4. NEWTON-SCHULZ ITERATION IMPROVEMENTS

### 4.1 Coefficient schedules (already partially in project)
- **quintic5** (3.4445, -4.7750, 2.0315): speedrun standard, 15 matmuls. ✓ in project.
- **cubic5** (1.5, -0.5, 0.0): 10 matmuls, -33% orth compute, ~1e-3 val loss
  (arXiv:2606.00371 "How Much Orthogonalization Does Muon Need?"). ✓ in project.
- **hybrid_v4** (8× quintic + 2× stabilize): DeepSeek-V4 final-step schedule
  (arXiv:2606.19348), pins SVs at 1. ✓ in project.
- **stabilize** (2.0, -1.5, 0.5): the DeepSeek-V4 terminal schedule.

### 4.2 Scaled CANS — Chebyshev-Accelerated Newton-Schulz ⭐ STATE OF THE ART
- **Paper**: Grishina et al. — "Accelerating Newton-Schulz Iteration via
  Chebyshev-type Polynomials", arXiv:2506.10935 (CANS). ICLR 2026 blogpost on
  CANS-SVD (iclr-blogposts.github.io/2026/blog/2026/polar-svd/).
- **Core**: Chebyshev-optimized coefficients (Remez algorithm) → optimal 3rd-
  order polynomials + higher-degree via Remez. δ-orthogonalization for
  controlled approximate schemes.
- **Measured**: CANS polynomial (12 matmuls) outperforms Muon polynomial with
  same matmul count. Up to 2× SVD speedup on B200 with TF-32. Can run in
  BF16/TF-32 (cuSOLVER can't).
- **Scaled CANS Coupled** (OKLS's variant): 10 iterations, 27 FP16 GEMMs, FP32
  accumulation, deterministic per-step scales. Computes matrix INVERSE square
  root (not just polar factor) — enables OKLS's zero-staleness.
- **Practicality**: Drop-in replacement for the NS coefficient schedule. The
  Remez-computed coefficients can replace quintic5/cubic5 directly. For
  Blackwell sm_120, TF-32/BF16 matmuls are very fast.
- **Key insight**: Fixed NS coefficients are suboptimal. Chebyshev-optimal
  coefficients converge faster for the same matmul budget.

### 4.3 Polar Express (Amsel et al., 2025)
- Concurrent optimal-polynomial NS method. Same optimal 3rd-order polynomial as
  CANS (independently derived). Proves optimality of the composition scheme.
- Already used in the current nanoGPT speedrun record stack.

### 4.4 Batched / Grouped NS
- Project already has batched NS (test_newton_schulz.py validates batched equiv
  of legacy). Group Muon / Per-head Muon (§1.11) extend this to grouped blocks.
- Dion / Dion2 (Ahn et al., 2025): distributed low-rank / sampled-row
  orthonormalization for sharded weights. Relevant for multi-GPU, not single.

---

## 5. PRIORITY RECOMMENDATIONS FOR FLOWER

Ranked by (expected gain × ease of integration) for a 100M–600M model on RTX 5090:

| # | Method | Effort | Expected Gain | Why |
|---|--------|--------|---------------|-----|
| 1 | **Contra-Muon** (1-line add) | Trivial | +0.5–1% loss | Free, composable, multiple Track-3 records |
| 2 | **Per-head / Group Muon** for attention | Small | +0.5–1% loss | Reshape + per-head NS; Kimi K3 validated |
| 3 | **Muon²-F** (factored 2nd moment pre-NS) | Small | -40% NS iters | Cheaper NS, near-zero memory; ~25% GPU-hr |
| 4 | **Newton-Muon** right-preconditioning | Medium | ~6% fewer steps | Principled; periodic ZZ^T inverse via Cholesky |
| 5 | **Scaled CANS coefficients** | Small | -1-2 matmuls | Replace quintic5 with Remez-optimal poly |
| 6 | **LoRA-Pre** momentum compression | Medium | Fit 1B+ models | 8× rank efficiency; ICLR Oral; Muon variant exists |
| 7 | **SODA wrapper** (1/k WD schedule) | Trivial | Eliminate WD tuning | Wraps existing Muon; zero new HPs |
| 8 | **Aurora on MLP** (verify current impl) | Small | Fix neuron death | Already in project; verify pp_iters=2, β=0.5 |
| 9 | **OKLS** (full switch) | Large | 1.45× param eff. | Biggest algorithmic gain; heavy memory |
| 10 | **SF-NorMuon** (schedule-free) | Medium | Anytime training | No schedule tuning; matches tuned AdamW |
| 11 | **Compositional Muon** (QK/OV coupling) | Medium | Attention gains | Partner-whitened; cheap isotropic approx |

**Bottom line**: The project's Muon+AdamW is already near the practical ceiling.
The highest-ROI moves are (1) Contra-Muon + Per-head Muon (near-free), (2)
Muon²-F to cut NS cost on Blackwell, and (3) LoRA-Pre to unlock 1B-scale models.
OKLS is the only method offering a step-change in parameter efficiency (1.45×),
but at a memory cost that may not fit 600M+ on 32GB without LoRA-Pre first.

---

## KEY REFERENCES (by arXiv ID)

- 2302.06675 — Lion (Chen et al.)
- 2305.14342 — Sophia (Liu et al.)
- 2402.11858 — PSGD Lie-group theory (Li)
- 2502.07529 — Scion (Pethick et al., ICML 2025)
- 2506.10935 — CANS (Grishina et al.)
- 2509.02046 — Optimizer reality check (Wen et al.) ⚠️
- 2509.03378 — KL-Shampoo/SOAP (Lin et al., ICLR 2026)
- 2510.05491 — NorMuon (Li Zichong)
- 2602.24283 — LoRA-Pre (ICLR 2026 Oral) ⚠️
- 2604.01472 — Newton-Muon (Du)
- 2604.09967 — Muon² (Boosting Muon)
- 2605.08933 — Group Muon (head grouping theory)
- 2605.11172 — SODA (Pethick et al.)
- 2605.23061 — SF-NorMuon (schedule-free spectral)
- 2606.00371 — "How Much Orthogonalization Does Muon Need?" (cubic5)
- 2606.19348 — DeepSeek-V4 NS schedule (hybrid_v4/stabilize)
- 2607.20548 — SOAP, Muon, and Beyond (NVIDIA) ⚠️
- 2607.24653 — Kimi K3 (Per-Head Muon §2.5)
- 2607.26001 — SAM-Muon (SpecSAM-Muon)

⚠️ = highest-signal references for decision-making.
