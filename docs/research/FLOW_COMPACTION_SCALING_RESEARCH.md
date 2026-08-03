# Flow-Matching KV-Cache Compactor: Scaling & Compaction Research

**Scope:** Ideas for improving the flow-matching KV-cache compactor (a CNF with a
FiLM-conditioned velocity field, Euler integration, 5 steps; currently beats the
Still paper's Perceiver approach by ~3.5%) before scaling from 12M to ~50M+
params and longer training.

**Method:** 13 targeted web searches (focus 2024-2026 papers). One framing
insight runs through every recommendation: **our task is not generation, it is a
deterministic, paired-data transfer** (full KV cache -> compressed KV cache).
This makes the "transfer / paired-data" branch of the flow-matching literature
directly applicable, and several generative-acceleration tricks unnecessary.

Each section below gives: source, technique, application to our compactor,
implementation difficulty, and likelihood of helping at scale.

---

## Priority shortlist (TL;DR)

Ranked by expected impact / effort for our specific deterministic-compaction use case:

1. **MeanFlow average-velocity parameterization** (Sec 10) - drop the 5-step
   Euler loop at inference entirely; ~5x cheaper compaction. Highest impact.
2. **Minibatch OT coupling (OT-CFM)** (Sec 8) - straight geodesic paths from
   natural paired data; fewer steps, more stable training. Best bang-for-buck.
3. **Rectified-flow reflow / straightening** (Sec 5) - distill to near-linear
   paths, 1-2 steps. Complements #2.
4. **Velocity-net architecture upgrade** (Sec 4) - small transformer velocity
   net (AdaLN-Zero / boundary-conditioned) replacing the FiLM MLP. Key at 50M+.
5. **Higher-order / optimal-step solver** (Sec 1) - Heun or DPM-Solver++ at the
   current 5 steps, or optimal non-uniform step placement. Cheap, moderate gain.
6. **Training-stability hardening** (Sec 3) - spectral norm / Lipschitz clamp /
   gradient clipping on the velocity net. Necessary insurance at scale.

Distillation (#6 of the requests, flow-matching distillation) and adaptive step
sizing (#9) are lower priority: distillation only helps inference cost (which
MeanFlow already solves more cleanly), and adaptive stepping is marginal vs a
good OT-straightened fixed schedule.

---

## 1. Flow-matching ODE solver steps (Euler vs higher-order vs adaptive)

**Source / concept:**
- *From Euler to Dormand-Prince: ODE Solvers for Flow Matching* (arXiv 2605.00836, 2026) - derives and benchmarks Euler, midpoint/Heun, RK, and Dormand-Prince solvers for FM; sampling cost is dominated by NN evaluations.
- *Optimal Flow Matching: Learning Straight Trajectories in Just One Step* (NeurIPS 2024).
- *Optimal Stepsize for Diffusion Sampling* (arXiv 2503.21774, 2025) - optimizing step discretization.
- *DPM-Solver / DPM-Solver++* (NeurIPS 2022) - fast dedicated ODE solvers for diffusion/FM, ~10 steps; *Fast ODE-based Sampling in Around 5 Steps* (CVPR 2024).
- *FlowMatch Euler Discrete Scheduler* - refines the Euler noise/timestep schedule.

**Technique:** The number of Euler steps trades off against per-step network
cost; path straightness (Sec 5) determines how few you need. Higher-order
solvers (Heun/midpoint = 2 evals/step, RK4 = 4) and *dedicated* solvers
(DPM-Solver++) extract more accuracy per network call. Crucially, **non-uniform
timestep placement** (concentrating steps where the velocity field is most
curved) often beats more uniform steps.

**Application to our compactor:** 5 Euler steps is reasonable for near-straight
paths but on the low side if the path is curved. Cheap wins: (a) swap Euler for
Heun/midpoint at the same 5 "macro-steps" (~2x NN calls, better accuracy); (b)
use a non-uniform schedule (more steps near t=0 and t=1, where endpoint behavior
dominates); (c) if we adopt OT-coupling / reflow (Sec 5, 8), the path becomes
near-linear and 5 steps is already overkill - 2-3 suffice.

**Implementation difficulty:** Low-Medium. Swapping the integrator is a small
change in the sampling loop; non-uniform schedules are a one-liner.

**Likely to help at scale:** Moderate. Real wins come from straightening the
path (Sec 5/8); the solver choice is a multiplier on top.

---

## 2. FiLM conditioning improvements

**Source / concept:**
- *FiLM: Visual Reasoning with a General Conditioning Layer* (Perez et al., arXiv 1709.07871) - feature-wise affine modulation (gamma, beta) per conditioning signal.
- *Time-FiLM Conditioning* (2025) - dynamic, time-dependent FiLM for sequence/control tasks.
- *Neural Field Conditioning Strategies* (2D semantic segmentation) - survey of conditioning options beyond FiLM (cross-attention, modulation, hypernetworks).
- DiT/AdaLN-Zero ( Peebles & Xie, used across modern FM) - adaptive layer-norm with zero-init, the de-facto SOTA conditioning for FM velocity nets.

**Technique:** FiLM applies only feature-wise affine modulation, which is
expressive but limited. Stronger schemes: **AdaLN-Zero** (modulate scale+shift
per token after each layer norm, zero-initialized residual gating - very stable
at scale), **cross-attention** from conditioning tokens (best when conditioning
is itself a token sequence), **Time-FiLM** (modulation that depends on the
integration time t), and **hypernetwork** conditioning (generate weights).

**Application to our compactor:** Two distinct conditioning signals to separate:
(i) the integration time `t`, and (ii) the source KV-cache context that the
velocity field must attend to. For (i) use time-embedding + AdaLN scale/shift
(Time-FiLM). For (ii), since the source is a token sequence, **cross-attention
into the cache** is strictly more expressive than FiLM broadcasting. At 50M+
params, the FiLM MLP becomes a bottleneck; AdaLN-Zero + cross-attention scales
far better and is the documented recipe for large FM velocity nets.

**Implementation difficulty:** Medium. AdaLN-Zero is a small module change;
cross-attention adds a sub-layer.

**Likely to help at scale:** Yes, high. Conditioning quality matters more as
the velocity net grows; FiLM is the weakest of the standard options.

---

## 3. Continuous normalizing flow training stability

**Source / concept:**
- *FORT: Forward-Only Regression Training of Normalizing Flows* (Rehman et al.) - avoids backprop through the ODE.
- *Stability and convergence of Neural ODEs* (ICLR 2025) - smoothing the vector field with stochasticity during training improves stability/convergence.
- *Flow Matching for Generative Modeling* (Lipman et al., ICLR 2023) - simulation-free CFM training removes the adjoint method entirely.
- Apple ML research: *Normalizing Flows are Capable Generative Models*.

**Technique:** Failure modes at scale: (a) exploding/vanishing gradients through
long ODE integration; (b) vector-field stiffness causing finite-time blow-up;
(c) Lipschitz blow-up as width grows. Mitigations: **simulation-free CFM**
(we very likely already do this - no adjoint pass), **spectral/Lipschitz
normalization** on the velocity net, **velocity clipping**, **gradient
clipping**, **stochastic smoothing** (small noise injection during training),
and **forward-only regression** (FORT) to skip the backward-through-ODE cost.

**Application to our compactor:** As we scale from 12M to 50M+, the dominant
risk is training instability (the +3.5% margin could collapse). Concrete
hardening: add spectral norm (or a Lipschitz clamp) to every linear/conv layer
in the velocity net; clip predicted velocities to a sane range; keep
gradient-norm clipping; keep CFM (simulation-free) training so there is no
adjoint. FORT-style forward-only regression is worth benchmarking for speed but
is a behavior change.

**Implementation difficulty:** Low-Medium (spectral norm, clipping = easy;
FORT = larger refactor).

**Likely to help at scale:** Yes, critical. Stability is the single biggest
risk at 50M+; this is insurance, not optional.

---

## 4. Velocity-field parameterization / architecture

**Source / concept:**
- *Improving Rectified Flow with Boundary Conditions* (arXiv 2506.15864, 2025) - parameterize velocity so the boundary endpoints (t=0, t=1) are exactly satisfied; avoids uncorrected endpoint error.
- *Terminal Velocity Matching* (ICLR 2026, arXiv 2511.19797) - generalizes FM to model transitions between diffusion times, enabling high-fidelity one/few-step.
- *Flow Straighter and Faster* (arXiv 2511.23342, 2025).
- DiT-family (transformer + AdaLN-Zero) as the dominant FM velocity-net architecture.

**Technique:** The velocity net's architecture governs both quality and
step-count sensitivity. Key ideas: (a) **token-structured transformer velocity
nets** (self-attention over the evolving state) match the data topology of
sequence data far better than MLPs; (b) **boundary-conditioned
parameterization** guarantees the t=0/t=1 endpoints, removing a constant error
floor; (c) **terminal/mean-velocity** parameterizations (Sec 10) reframe what
the net predicts.

**Application to our compactor:** The KV cache is a token-structured tensor, so
a **small transformer velocity net** (mirroring the host LM's per-layer
structure, with AdaLN-Zero time conditioning + cross-attention to source cache,
see Sec 2) is the natural and scalable choice - not the current FiLM MLP.
Adding a **boundary-condition** term so that x(0)=source and x(1)=target are
exactly respected (or softly penalized) removes endpoint error that 5 Euler
steps cannot fully correct.

**Implementation difficulty:** Medium-High (architectural rewrite of the
velocity net). Boundary conditioning is easy to bolt on.

**Likely to help at scale:** Yes, high. Architecture is the largest single
quality lever as we scale.

---

## 5. Rectified-flow straightening (fewer steps)

**Source / concept:**
- *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* (Liu et al., ICLR 2023 spotlight, arXiv 2209.03003) - the "reflow" procedure straightens ODE paths; explicitly framed as **transfer between two empirically observed distributions** (exactly our setting).
- *Straightness Is Not Your Need in Rectified Flow* / *Rectified Diffusion* (ICLR 2025, arXiv 2410.07303) - caveat: straightness alone is not sufficient; the regression target must also be well-chosen.

**Technique:** Reflow: (1) train a flow, (2) simulate its trajectories to get
near-straight (source->target) pairs, (3) retrain a new flow to match those
straightened paths. The result is a flow solvable in very few Euler steps (even
1). Rectified flow is explicitly a *transfer* method, not just generation.

**Application to our compactor:** This is one of the two best fits. We already
have natural source/target pairs (full cache / compressed cache), so reflow is
cheap: train the current 5-step model, sample trajectories, retrain a 1-2 step
model on the straightened pairs. Combined with OT coupling (Sec 8) the very
first training already yields near-straight paths, so reflow may be a one-shot
fine-tune rather than multiple rounds.

**Implementation difficulty:** Low-Medium. Reflow is a standard, well-documented
recipe with reference implementations.

**Likely to help at scale:** Yes, high. Directly enables cutting 5 steps to 1-2.

---

## 6. Flow matching + knowledge distillation

**Source / concept:**
- *Self-Corrected Flow Distillation* (arXiv 2412.16906, 2024) - self-correction for consistent one/few-step flow distillation.
- *Score Distillation of Flow Matching Models* (arXiv 2509.25127, 2025).
- *Distilled Decoding* (ICLR 2025) - distillation to few-step.
- Distribution Matching Distillation (DMD, Yin et al. 2024).
- *How to build a consistency model: learning flow maps* (arXiv 2505.18825, 2025).

**Technique:** Train a multi-step FM "teacher", then distill into a 1-2 step
"student" via consistency / flow-map / distribution-matching losses. Eliminates
iterative sampling at inference.

**Application to our compactor:** Useful only for **inference cost** (compaction
latency). But MeanFlow (Sec 10) achieves the same one-step inference more
elegantly without a separate teacher. Distillation is therefore a backup if
MeanFlow underperforms. If pursued: train current 5-step compactor as teacher,
distill to 1-step student with a consistency loss.

**Implementation difficulty:** Medium-High (teacher-student pipeline, extra
training phase).

**Likely to help at scale:** Moderate for inference speed; **low for training
quality**. Deprioritize relative to MeanFlow.

---

## 7. KV-cache compaction scaling laws

**Source / concept:**
- *Expected Attention: KV Cache Compression by Estimating Attention Importance* (arXiv 2510.00636, 2025) - trainable compressors need substantial compute; importance estimation is the key signal.
- *CompressKV: Semantic Retrieval Heads Know What Tokens Are Not Important* (arXiv 2508.02401, 2025) - some heads (retrieval heads) need the full cache; per-head selectivity matters.
- *Inference-Time Hyper-Scaling with KV Cache Compression* (OpenReview 8ZiElzQxf1, 2025) - compression enables longer context at inference.
- *KVzip* (query-agnostic compression, 2025).
- *KV Cache Compression: A Review* (arXiv 2508.06297, 2025).

**Technique:** What works at scale: (a) **query-aware** compression (compress
based on what the upcoming queries attend to, not just static importance);
(b) **per-head / per-layer selectivity** - retrieval heads and certain layers
must retain full resolution; (c) **budgeted compute** - trainable compressors
scale but their compute/memory tradeoff must be explicitly budgeted;
(d) importance signals from attention statistics (Expected Attention) or learned
heads (CompressKV).

**Application to our compactor:** Three concrete scale-up actions: (1) make the
compactor **query-conditioned** (feed upcoming queries into the FiLM/cross-attention
conditioning, Sec 2); (2) add **per-layer and per-head compression ratios**
(retrieval heads compress less); (3) report the compute/memory Pareto
explicitly at 50M+ so we don't trade away the +3.5% margin for marginal memory
gains. The literature is clear that uniform compression degrades sharply at
scale.

**Implementation difficulty:** Medium (query conditioning ties into the
conditioning refactor in Sec 2; per-head ratios are config-level).

**Likely to help at scale:** Yes, critical - this is what prevents quality from
collapsing as the host LM and context grow.

---

## 8. Optimal-transport initial coupling (OT-CFM)

**Source / concept:**
- *Improving and Generalizing Flow-Based Generative Models with Conditional Flow Matching* (Tong et al., ICLR 2023, arXiv 2302.00482) - introduces minibatch **OT-CFM**.
- **TorchCFM** reference library (atong01/conditional-flow-matching) - drop-in OT-CFM implementations.
- *Weighted Conditional Flow Matching* (arXiv 2507.22270, 2025) - extends minibatch OT with a cost/weight term.
- *Optimal Flow Matching* (NeurIPS 2024) - straight trajectories, near one-step.

**Technique:** Instead of independent (source, target) pairs, solve a small
**optimal-transport assignment** within each minibatch to couple sources to
their nearest targets. The resulting conditional paths are **geodesics
(straight lines)**, which (a) train faster/stabler, (b) need far fewer Euler
steps, (c) reduce variance in the velocity regression target. Weighted-CFM adds
a cost weight to bias the coupling.

**Application to our compactor:** The single most natural fit in this list. We
have natural paired data, but random pairing within a minibatch is wasteful;
minibatch OT coupling gives straight paths for free. With OT coupling the
velocity field is a near-constant (target - source)/1 along the path, so 1-2
Euler steps suffice and training is more stable. Use the TorchCFM reference as
the starting point. If exact OT is too slow, entropic-OT or kNN-assignment are
good approximations.

**Implementation difficulty:** Low. OT-CFM is a well-documented, library-backed
recipe; the change is in data coupling, not architecture.

**Likely to help at scale:** Yes, high. Best bang-for-buck of all options.

---

## 9. Adaptive / learned ODE step sizing

**Source / concept:**
- *S-SOLVER: Numerically Stable Adaptive Step Size Solver for Neural ODEs* (ICANN 2023) - reliable local-truncation-error estimation for stable adaptive stepping.
- Adaptive-step-size neural-ODE implementations (Allauzen/adaptive-step-size-neural-ode).
- *Optimal Stepsize for Diffusion Sampling* (arXiv 2503.21774, 2025) - optimizing the *fixed* step schedule.

**Technique:** Either (a) error-estimation-based adaptive stepping (accept/reject
steps based on local truncation error, like S-SOLVER), or (b) a learned/predicted
step schedule, or (c) a well-chosen **fixed** optimal schedule.

**Application to our compactor:** At our current 5 steps, a good **fixed
non-uniform schedule** (Sec 1) already captures most of the benefit of adaptive
stepping at lower complexity. Adaptive stepping becomes worthwhile only if we
push to very few steps with a curved path - but OT-coupling/re/flow (Sec 5, 8)
removes the curvature that adaptive stepping would chase. So: deprioritize.

**Implementation difficulty:** Medium (error estimators add overhead and
bookkeeping).

**Likely to help at scale:** Low-Moderate. Marginal vs simpler OT-straightened
fixed schedules.

---

## 10. Mean-flow parameterization (average velocity)

**Source / concept:**
- *Mean Flows for One-step Generative Modeling* (Geng, Pokle, Kolter; arXiv 2505.13447, 2025) - replaces instantaneous velocity v(x,t) with an **average/mean velocity** over the integration interval, enabling true one-step generation.
- *Re-Meanflow: Efficient One-Step Generative Modeling* (2025).
- *Riemannian MeanFlow* (arXiv 2603.10718, 2026).
- *Discrete MeanFlow* (arXiv 2605.12805, 2026).
- MeanFlow tutorial (CVPR 2025).

**Technique:** Instead of learning the instantaneous velocity v(x,t) and
integrating it with N Euler steps, learn the **interval-averaged velocity** that
maps start -> end directly. Inference becomes a single network forward pass -
no ODE loop. Training is still simulation-free via a self-consistency / fixed
point loss on the mean velocity.

**Application to our compactor:** The highest-impact change for our setting.
MeanFlow lets us **eliminate the 5-step Euler loop at compaction time**,
turning compaction into one forward pass (~5x cheaper, deterministic, no
integrator). Since our task is a clean deterministic transfer (not generation),
the mean-velocity assumption is *more* justified here than in image generation.
Training cost is similar (self-consistency loss replaces the per-step velocity
loss).

**Implementation difficulty:** Medium. Reformulate the training loss (mean
velocity + self-consistency); inference simplifies dramatically.

**Likely to help at scale:** Yes - highest impact. Cuts compaction cost ~5x and
removes the integrator entirely, which also removes a class of numerical
instability.

---

## Bonus: directly relevant adjacent work

- **Simulation-Free Training of Neural ODEs on Paired Data** (Kim et al.,
  NeurIPS 2024, arXiv 2410.22918) - simulation-free NODE training for
  *deterministic mappings between paired data*. This is essentially our exact
  problem statement; its loss formulation should be checked against our current
  training objective. Likely already informs our design - worth a re-read before
  the 50M scale-up.

---

## Suggested scale-up sequencing

1. **Before touching architecture**, add OT-CFM minibatch coupling (Sec 8) and
   stability hardening (Sec 3). These are low-effort and de-risk everything.
2. **Rewrite the velocity net** to a small transformer with AdaLN-Zero time
   conditioning + cross-attention to the source cache (Sec 2, 4); add boundary
   conditions (Sec 4).
3. **Make the compactor query-aware** with per-head compression ratios (Sec 7).
4. **Switch to MeanFlow** (Sec 10) to collapse 5 steps to 1 at inference; fall
   back to rectified-flow reflow (Sec 5) if MeanFlow underperforms.
5. Keep higher-order / optimal-step solvers (Sec 1) and adaptive stepping (Sec 9)
   as tuning knobs on top of a straightened path, not primary levers.

## Sources (consolidated)

- Liu et al., *Flow Straight and Fast* (Rectified Flow), ICLR 2023, arXiv 2209.03003
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023
- Tong et al., *Improving and Generalizing Flow-Based Models with CFM (OT-CFM)*, ICLR 2023, arXiv 2302.00482; TorchCFM lib (atong01/conditional-flow-matching)
- Kim et al., *Simulation-Free Training of Neural ODEs on Paired Data*, NeurIPS 2024, arXiv 2410.22918
- *Optimal Flow Matching*, NeurIPS 2024
- *Fast ODE-based Sampling in Around 5 Steps*, CVPR 2024
- *Self-Corrected Flow Distillation*, arXiv 2412.16906 (2024)
- *Straightness Is Not Your Need in Rectified Flow* (Rectified Diffusion), ICLR 2025, arXiv 2410.07303
- Geng, Pokle, Kolter, *Mean Flows for One-step Generative Modeling*, arXiv 2505.13447 (2025)
- *How to build a consistency model: learning flow maps*, arXiv 2505.18825 (2025)
- *Improving Rectified Flow with Boundary Conditions*, arXiv 2506.15864 (2025)
- *Weighted Conditional Flow Matching*, arXiv 2507.22270 (2025)
- *Score Distillation of Flow Matching Models*, arXiv 2509.25127 (2025)
- *Optimal Stepsize for Diffusion Sampling*, arXiv 2503.21774 (2025)
- *Expected Attention* (KV compression), arXiv 2510.00636 (2025)
- *CompressKV*, arXiv 2508.02401 (2025)
- *KV Cache Compression: A Review*, arXiv 2508.06297 (2025)
- *Inference-Time Hyper-Scaling with KV Cache Compression*, OpenReview 8ZiElzQxf1 (2025)
- *Terminal Velocity Matching*, ICLR 2026, arXiv 2511.19797
- *From Euler to Dormand-Prince: ODE Solvers for Flow Matching*, arXiv 2605.00836 (2026)
- *S-SOLVER*, ICANN 2023
- Perez et al., *FiLM*, arXiv 1709.07871; *Time-FiLM Conditioning* (2025)
