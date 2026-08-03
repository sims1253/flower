# Sweep 10 — Is the flow win real? (param-matched flow ablation)

## Why this sweep exists

The `sweep_still_novel` results looked like a clean "flow-matching compactor
beats everything" story:

| variant           | n  | mean train ppl | trainable params |
|-------------------|----|----------------|------------------|
| `novel_flow`      | 4  | **2.678**      | 7.11M            |
| `novel_freq`      | 6  | 2.826          | 1.97M            |
| `novel_baseline`  | 5  | 2.836          | 1.97M            |
| `novel_spectral`  | 5  | 2.842          | ~2.0M            |
| `novel_pyramid`   | 5  | 2.912          | <1.97M           |
| `novel_attnmatch` | 6  | 3.143          | 1.97M            |

But this is a **textbook confound**. `count_parameters` only counts *trainable*
params (`requires_grad`), and the base model (12.2M) is frozen during compactor
training. So the only thing that varies between arms is the **compactor's own
trainable capacity**:

- baseline compactor: 328,704 params/layer × 6 = **1.97M**
- flow compactor: 1,185,024 params/layer × 6 = **7.11M**

The flow arm has **3.6× more trainable compactor capacity** than every other
arm. Its lower train perplexity could be entirely explained by "more params,"
with zero credit to the flow mechanism. (The flow's extra mass comes from two
oversized `FlowVelocityNet`s per layer — 395k params *each*, hidden_dim =
`2 * d_latent = 256`.)

There is also a **second confound**: every reported perplexity is
**train-only** (`loss == train_loss`; the novel config never set
`validation_steps`). Train perplexity rewards extra capacity even more strongly
than held-out perplexity does, so the confound points in the same direction.

The current "flow wins" result is therefore **uninterpretable**. This sweep
makes it interpretable.

## Headline question

> Holding *trainable compactor parameter budget* fixed and measuring on a
> held-out set, does the **flow-matching mechanism** beat a **bigger standard
> Perceiver compactor**?

Three outcomes, all interesting:

1. **Flow wins at matched params.** The mechanism genuinely helps → real,
   publishable-ish "huh, that's interesting" result, and the natural seed for
   further flow-LM-as-compactor work (ties into the wiki's flow-matching /
   discrete-flow-maps / coupling-models cluster).
2. **Standard-at-flow-budget wins.** The original result was pure capacity →
   flow-matching for KV compaction is a dead end at this scale. Equally useful:
   kills a seductive false lead.
3. **Tie.** Flow offers no perplexity edge but may offer other properties
   (smoother curves, invertibility for decompression). Worth logging diagnostic
   curves either way.

## Design — one axis, controlled

All arms share the frozen `sweep_still_novel` setup (vocab 4096, d_model 384,
6 heads, 6 layers, seq 1024, base compactor config). The base model is
identical across arms (same arch, same seed, frozen after warmup). **Only the
compactor changes**, and every arm is built to hit the **same trainable
compactor param count** within ±2%.

### Arms (final, from the param audit)

All budgets verified within ±2.5% by instantiating each compactor:

| arm              | config                         | trainable compactor |
|------------------|--------------------------------|---------------------|
| `A std_2.0M`     | `still`, d_latent=128, bl=2    | 1,972,224           |
| `B flow_7.1M`    | `still_flow`, steps=5 (orig)   | 7,110,144           |
| `C std_7.1M`     | `still`, d_latent=192, bl=4    | 7,096,320  (0.19%)  |
| `D flow_2.4M`    | `still_flow`, steps=5, **hidden=8** | 2,446,752      |
| `E std_2.4M`     | `still`, d_latent=144, bl=2    | 2,384,640  (2.5%)   |

The decisive comparisons are **B vs C** (flow vs same-budget standard at high
capacity) and **D vs E** (flow vs same-budget standard at low capacity). The flow
can't be shrunk below ~2.4M: the two fixed `context_proj` (Linear 2d→4d, 32k
params each) impose a ~20% structural floor over the baseline — itself worth
noting in the writeup.

### Control knobs

- `still_num_blocks`, `still_d_latent` — already exist, drive standard-compactor
  capacity.
- `still_velocity_hidden` — **NEW** (this sweep), flows to both
  `FlowVelocityNet.hidden_dim` in `StillCompactorFlow`. Needed for arm D.
- `still_flow_steps` — already exists (Euler steps); keep at 5 to match the
  original.
- `training.validation_steps` — set > 0 so every arm reports `val_loss` /
  `val_perplexity`. Fixes the train-only confound.

## Config (`configs/sweep_still_flow_matched.yaml`)

Same defaults as `sweep_still_novel` except:
- `training.validation_steps: 20` (held-out eval each checkpoint + at end)
- `training.steps: 6500` (unchanged — power comes from matched comparison, not
  longer training)
- `training.seeds: [0, 1, 2]` (3 seeds; seed-noise floor on this setup is
  ~±0.02–0.03 ppl, so 3 seeds + non-overlapping means is enough to call it)

Variants: `A_std_2M`, `B_flow_7M`, `C_std_7M`, `D_flow_2M`, `E_std_2M`.

## Code change required

`StillCompactorFlow.__init__` hardcodes velocity-net `hidden_dim = 2*d_latent`.
Add an optional `velocity_hidden` kwarg (default None → current behavior),
thread it to both `FlowVelocityNet`s, and a `still_velocity_hidden` config field
in `build_still_model`. ~10 lines, backward-compatible.

Also verify the standard compactor's param count at `d_latent=192, blocks=4` is
within 2% of the flow compactor before launching arm C (param audit step below).

## Verification (run in order)

1. **Param audit (no training).** Instantiate each arm's compactor, print
   trainable params/layer and total. Confirm B≈C and A≈D within ±2%. Hard gate:
   do not launch the sweep until the budgets match.
2. **Hidden-dim unit check.** Build `StillCompactorFlow(velocity_hidden=24)`,
   run one forward on a dummy `(B=2, H=6, T=64, d=64)` cache, assert output
   shapes unchanged and loss is finite. Regression: `pytest tests/test_shapes.py -q`.
3. **100-step smoke** on one arm, confirm `val_perplexity` appears in the
   metrics JSON and tokens/sec is sane.
4. **Full sweep**, 4 arms × 3 seeds × 6500 steps ≈ 12 runs.

## Metrics & gate

Primary: **val_perplexity** (held-out), reported per arm as mean ± range over
seeds. Secondary: train_perplexity (to *quantify* the train/val gap and confirm
the capacity story), and the per-arm compactor param count logged into the
metrics JSON.

**Calling the result.** Non-overlapping seed ranges on val_perplexity between
B and C (or A and D) = real effect. Overlapping ranges = no effect, mechanism
indistinguishable from capacity.

## Out of scope (deferred)

- New compactor mechanisms (this sweep *judges* an existing one, doesn't add
  more). 
- Scale (1B+) — this is a 25M-class mechanism question.
- Longer horizons — 6500 steps matches the original run for direct comparison.

## Cost

Param audit + smoke: minutes. Full sweep: 12 runs × ~the novel-sweep per-run
cost. The flow arms are ~2× slower (5 Euler steps), so budget ~1.5–2× the
wall-clock of the equivalent novel arms.
