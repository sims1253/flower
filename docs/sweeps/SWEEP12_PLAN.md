# Sweep 12 — Plan: Flow variance reduction for `StillCompactorFlow`

## TL;DR

Sweep 10 found the flow-matching KV compactor (`StillCompactorFlow`) is ~2× more
seed-variable than the standard compactor (val_ppl sd ~0.030 vs ~0.017), even though
its MEAN perplexity trends slightly better at both param budgets. Some flow seeds find
a good velocity-field trajectory, others don't. This sweep tests whether **reducing that
variance** turns flow's mean win into a statistically significant one. The decisive metric
is the **seed standard deviation of `val_perplexity`** (not the mean).

---

## 1. Hypothesis & mechanism

`StillCompactorFlow.forward` (still.py:757-775) refines the latent by Euler-integrating
a learned velocity field over `flow_steps` discrete steps:

```
dt = 1.0 / self.flow_steps
for step in range(self.flow_steps):
    t_val = step * dt
    d_k = self.velocity_keys(z, t_val, cond_k)
    d_v = self.velocity_vals(z, t_val, cond_v)
    velocity = 0.5 * (d_k + d_v)
    z = z + velocity * dt
```

At init this is exactly the identity: `_init_zero` (still.py:636-639) zeroes the final
layer of each `FlowVelocityNet`, and `_init_flow` (still.py:698-705) zeroes both
`context_proj_k`/`context_proj_v`, so the velocity output is **zero for every seed** at
step 0. The seed-variance therefore enters during *early training*, through the
**default-init** (PyTorch `kaiming_uniform_`) hidden layers of the velocity net —
`time_embed`, `cond_proj`, and the first two `Linear` layers of `self.net`
(still.py:625-633). Different seeds initialize these hidden weights differently; once
gradients break symmetry the velocity field takes a seed-dependent shape, and the Euler
rollout then **compounds** that shape error over `flow_steps` steps. Coarser integration
(fewer steps) leaves more discretization error for a bad trajectory to accumulate. The
hypothesis: by (a) integrating finer, (b) shrinking the velocity net's hidden-layer init
scale, and (c) adding a trajectory-consistency term that straightens the path, we
collapse the seed-to-seed trajectory variance.

---

## 2. Variance-reduction interventions

Ranked by expected impact × ease. Three independently-testable knobs.

### Intervention E1 — More Euler steps (5 → 10, 20). *Highest ease, expected medium impact.*

- **What:** Pure config change. `still_flow_steps: 10` and `still_flow_steps: 20` arms.
- **Code:** none. Config-only.
- **Mechanism:** finer `dt` → less per-step discretization error → a seed-dependent bad
  velocity shape accumulates less error before the final Perceiver blocks re-project.
- **Expected variance effect:** moderate downward.
- **Expected mean effect:** neutral-to-slightly-better.
- **Param cost:** zero. **Compute cost:** linear in `flow_steps`.

### Intervention E2 — Smaller velocity-net hidden-layer init. *High ease, expected medium-high impact.*

- **What:** Add a `velocity_init_scale` kwarg (default 1.0 = current) that multiplies the
  init scale of the **non-final** velocity-net layers only: `time_embed[0]`,
  `time_embed[2]`, `cond_proj`, `self.net[0]`, `self.net[2]` (still.py:625-633). Leave
  `_init_zero` (final layer) untouched. Test scale = 0.1.
- **Code:** ~6 lines in `FlowVelocityNet.__init__`; add `velocity_init_scale` to
  `StillCompactorFlow.__init__`; add `still_velocity_init_scale` config field; thread via
  build_still_model (mirror the existing `still_velocity_hidden` plumbing).
- **Mechanism:** the final layer is already zeroed, so init-time velocity is zero
  regardless. The remaining seed-sensitivity lives in the hidden layers' default
  `kaiming_uniform_` draw. Shrinking their scale keeps the net's effective gain near zero
  longer as training starts, so all seeds begin from a near-identical (near-zero) velocity
  field and diverge more slowly.
- **Expected variance effect:** the largest of the three knobs (targets the actual source).
- **Param cost:** zero.

### Intervention E3 — Trajectory consistency (MeanFlow-style) loss. *Medium ease, expected high impact.*

- **What:** Add a self-consistency auxiliary loss that straightens the velocity-field
  trajectory, mirroring the already-implemented `MeanFlowNet._consistency_loss`
  (still_flow2.py:221-271). During training, roll out the Euler path under `no_grad` to
  get the endpoint `z_target`, sample one intermediate time `t_k`, and add
  `L_cons = || v(z_tk, t_k) − (z_target − z_tk)/(1 − t_k) ||²` scaled by
  `still_flow_consistency_weight`. This pulls every seed's trajectory toward the same
  straight-line transport.
- **Code:** ~25 lines in a new `_consistency_loss` method on `StillCompactorFlow`; emit
  as `"flow_consistency_loss"` from `forward` when `self.training`; pick up in
  `StillLM.forward` alongside the existing `kl`/`ce` terms. Add
  `still_flow_consistency_weight` config field. **No teacher net, no new params.**
- **Expected variance effect:** high (published recipe for stable flow-matching).
- **Param cost:** zero.

### Considered and deferred

- **Deterministic conditioning / fixed noise:** rejected — no randomness enters the Euler
  loop (`t_val` deterministic, conditioning is mean/max pool, `dropout: 0.0`).
- **EMA of velocity-net weights:** deferred to a Sweep-13 candidate if E1-E3 don't close
  the gap.

---

## 3. Param audit

All three interventions are **param-neutral** by construction. Verified counts:

| arm | velocity_hidden | trainable/layer | ×6 layers |
|-----|-----------------|-----------------|-----------|
| B flow (sweep10 ref) | 256 (default) | 1,185,024 | **7,110,144** |
| D flow (sweep10 ref) | 8 | 407,792 | **2,446,752** |
| A std (sweep10 ref) | — | 328,704 | **1,972,224** |

- E1 (more steps): identical weights → identical counts.
- E2 (smaller init scale): multiplies init values only → identical counts.
- E3 (consistency loss): reuses `self.velocity_keys`/`self.velocity_vals` under `no_grad`,
  no new parameters → identical counts.

Every Sweep-12 flow arm is **bit-for-bit param-matched to sweep10's B (7,110,144) or D
(2,446,752)**. Standard-compactor variance floors: sweep10's C (7.1M, sd ~0.026) and E
(2.4M, sd ~0.017).

---

## 4. Arms

Anchor to sweep10's flow arms. **5 seeds each** (up from 3) because the sd of val_ppl
*is* the measured quantity.

| arm | budget | config | new knob | seeds |
|-----|--------|--------|----------|-------|
| `B0_flow_ref`     | 7.1M | still_flow, steps=5, hidden=default | — (clean-batch re-run) | 5 |
| `B1_flow_s10`     | 7.1M | still_flow, steps=10 | E1 | 5 |
| `B2_flow_s20`     | 7.1M | still_flow, steps=20 | E1 | 5 |
| `B3_flow_init01`  | 7.1M | still_flow, steps=5, velocity_init_scale=0.1 | E2 | 5 |
| `B4_flow_cons`    | 7.1M | still_flow, steps=5, consistency_weight=0.1 | E3 | 5 |
| `D0_flow_ref`     | 2.4M | still_flow, steps=5, hidden=8 | — | 5 |
| `D1_flow_s10`     | 2.4M | still_flow, steps=10, hidden=8 | E1 | 5 |
| `D3_flow_init01`  | 2.4M | still_flow, steps=5, hidden=8, velocity_init_scale=0.1 | E2 | 5 |
| `D4_flow_cons`    | 2.4M | still_flow, steps=5, hidden=8, consistency_weight=0.1 | E3 | 5 |

D2 (steps=20) dropped to save compute — E1's effect is budget-independent, B2 covers it.
Total: 9 arms × 5 seeds = 45 runs.

---

## 5. Config draft (`configs/sweep_still_flow_variance.yaml`)

```yaml
sweep:
  name: sweep_still_flow_variance

  defaults:
    model:
      vocab_size: 4096
      d_model: 384
      num_heads: 6
      num_layers: 6
      ffn_dim: 1536
      max_seq_len: 1024
      local_window: 256
      rope_base: 10000.0
      dropout: 0.0
      still_compact_len: 64
      still_num_blocks: 2
      still_d_latent: 128
      still_kl_topk: 200
      still_kl_weight: 1.0
      still_ce_weight: 0.1
      still_compact_from_step: 1500
      still_kl_temperature: 1.0
      still_base_warmup_steps: 1500

    data:
      dataset: fineweb_edu
      tokenizer: "custom:tokenizers/fineweb_4k.json"
      sequence_length: 1024
      eval_seq_len: 1024

    training:
      batch_size: 16               # 16 x accum 4 = effective 64. SAFE on 32GB.
      gradient_accumulation_steps: 4
      steps: 6500
      lr: 0.003
      warmup_steps: 200
      lr_schedule: linear_warmup
      grad_clip: 1.0
      eval_interval: 500
      checkpoint_interval: 1000
      save_checkpoints: true
      device: auto
      seed: 0
      seeds: [0, 1, 2, 3, 4]       # 5 seeds: sd is the measured quantity
      log_backend: tensorboard
      composite_eval: false
      optimizer: muon
      muon_lr: 0.003
      muon_momentum: 0.95
      validation_steps: 20

  variants:
    - name: B0_flow_ref
      model: { variant: still_flow, still_flow_steps: 5 }
    - name: B1_flow_s10
      model: { variant: still_flow, still_flow_steps: 10 }
    - name: B2_flow_s20
      model: { variant: still_flow, still_flow_steps: 20 }
    - name: B3_flow_init01
      model: { variant: still_flow, still_flow_steps: 5, still_velocity_init_scale: 0.1 }
    - name: B4_flow_cons
      model: { variant: still_flow, still_flow_steps: 5, still_flow_consistency_weight: 0.1 }
    - name: D0_flow_ref
      model: { variant: still_flow, still_flow_steps: 5, still_velocity_hidden: 8 }
    - name: D1_flow_s10
      model: { variant: still_flow, still_flow_steps: 10, still_velocity_hidden: 8 }
    - name: D3_flow_init01
      model: { variant: still_flow, still_flow_steps: 5, still_velocity_hidden: 8, still_velocity_init_scale: 0.1 }
    - name: D4_flow_cons
      model: { variant: still_flow, still_flow_steps: 5, still_velocity_hidden: 8, still_flow_consistency_weight: 0.1 }
```

New config fields: `still_velocity_init_scale` (float, default 1.0),
`still_flow_consistency_weight` (float, default 0.0).

---

## 6. GPU safety

**Every arm uses `batch_size: 16, gradient_accumulation_steps: 4` (effective 64).**
Sweep10's flow arms at batch 32×2 spiked to **31.9GB** during compaction and crashed. At
batch 16×4, flow arms sit at ~8-10GB peak. The Euler-step-heavy arms (B1/B2) reuse the
same `z` activation in-place across the loop, so peak is bounded by the single-step
footprint, not a multiple of `flow_steps`. **No arm may use physical batch > 16.** Confirm
B2_flow_s20 peak stays < 12GB in the smoke test before launching.

---

## 7. Analysis plan

Decisive metric: **seed standard deviation of `val_perplexity`** per arm. Adapt
`scripts/sweep10_analyze.py` → `scripts/sweep12_analyze.py`:
- Add per-arm sample `stdev` on `val_perplexity`.
- **Variance-reduction verdict:** does any intervention drop sd below the standard floor
  (~0.017 at 2.4M, ~0.026 at 7.1M)?
- **Mean-significance verdict:** with tightened ranges, is flow's val_ppl range
  **non-overlapping** with the standard arm's range?

**Statistical call (n=5):** report `sd` directly + min/max range; call non-overlapping
ranges a real effect (matches the project convention). Optionally report a bootstrap 95%
CI on the sd (1000 resamples) as a sanity check.

---

## 8. Verification (before the full sweep)

1. **Param audit (no training).** Instantiate `StillCompactorFlow` per arm; confirm
   B-cluster == 7,110,144 and D-cluster == 2,446,752 after E2/E3 code lands.
2. **Init-scale unit check (E2).** Build `velocity_init_scale=0.1`, run forward on dummy
   `(B=2, H=6, T=64, d=64)`, assert shapes unchanged and initial velocity still zero
   (final layer zeroed → identity init must not break).
3. **Consistency-loss unit check (E3).** Run one training step; assert
   `flow_consistency_loss` is a finite positive scalar in diagnostics.
4. **Regression:** `pytest tests/test_shapes.py -q`.
5. **100-step smoke** on `B2_flow_s20` (heaviest): confirm `val_perplexity` in metrics,
   peak VRAM < 12GB.
6. **Full sweep:** 9 arms × 5 seeds × 6500 steps = 45 runs.

---

## 9. What "interesting" looks like

(a) **Variance drops AND flow's mean win becomes non-overlapping.** → "Flow works for KV
compaction; you just had to stabilize the trajectory." Strongest outcome.

(b) **Variance drops but mean advantage vanishes.** → "The sweep10 mean win was lucky
high-variance seeds." Honest negative result; kills the lead cleanly.

(c) **Variance unchanged across all three knobs.** → "The instability is fundamental to
Euler-integration-of-a-learned-field." Points toward discrete-flow / diffusion
alternatives. Most scientifically interesting outcome.

A per-knob split (e.g. E3 alone drops variance) isolates *trajectory straightness* as the
load-bearing property of stable flow compaction.

---

### Files referenced

- `flower/models/still.py` — `StillCompactorFlow` (649-792), `FlowVelocityNet` (612-646),
  `_init_zero` (636-639), `_init_flow` (698-705), Euler loop (766-775)
- `flower/models/still_flow2.py` — `MeanFlowNet._consistency_loss` (221-271) reference for E3
- `flower/models/still_lm.py` — compactor construction (155-183), config threading (169-171),
  loss assembly (536-555)
- `flower/config.py` — `still_*` fields (103-127)
- `configs/sweep_still_flow_matched.yaml` — sweep10 setup
- `scripts/sweep10_analyze.py` — base for sweep12_analyze.py
