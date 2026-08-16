# Attention window screen — measured results (450M, RTX 5090)

**Headline: window 512 is better on quality AND 1.20x faster. My prediction was
backwards.** At equal wall-clock it is **-0.062 val_bpb** — 2.2x larger than the
best optimizer result ever measured on this model.

Screen: `configs/window_screen_450m.yaml`, `runs/window_screen_450m/`,
2026-08-12/13. 4 arms, seed 0, FP8 stack fixed, `vanilla_local`.

| arm | val_bpb | vs ctrl | tok/s | steps | wall-clock |
|---|---:|---:|---:|---:|---:|
| **`w512_isotime`** | **1.06594** | **-0.06194** | 80,021 | 1902 | 1.73 h |
| `w512` | 1.10995 | -0.01792 | 76,826 | 1500 | 1.42 h |
| `w1024` | 1.11602 | -0.01185 | 70,713 | 1500 | 1.54 h |
| `w2048_control` | 1.12787 | — | 64,221 | 1500 | 1.70 h |

The iso-time sizing worked: `w512_isotime` used 1.73 h against the control's
1.70 h, a 2% miss.

## The control has now reproduced three times

| run | screen | val_bpb |
|---|---|---:|
| `muon_baseline` | `muon_screen_450m` | 1.12785 |
| `cm_baseline` | `comp_muon_screen_450m` | 1.12783 |
| `w2048_control` | `window_screen_450m` | 1.12787 |

Total spread **0.00004**, across three screens, several days, and a reboot. That
is what makes a -0.062 result believable rather than suspicious.

## What I got wrong

`configs/window_screen_450m.yaml` states plainly: *"val_bpb is expected to get
WORSE at matched steps. The question is whether it gets worse by less than
1.268x more steps buys back."*

That was wrong. Shorter windows are better **at matched steps too** (-0.018 at
512, -0.012 at 1024). The model does better with strictly less context. The
iso-time arm then compounds a quality win with a throughput win instead of
trading one against the other, which is why -0.062 is so much larger than
anything else measured here.

Note this is not the throughput bench's 1.268x reappearing: measured in-run,
w512 is 76,826 vs 64,221 = **1.196x**. The bench (accum 4, synthetic, 5 steps)
overstated the in-run ratio by ~6%. The iso-time arm was sized on the bench
number and still landed within 2% of the control's wall-clock, so nothing is
invalidated — but quote 1.196x, not 1.268x, for a real run.

## Why this might be happening — and why it is not yet a deployment decision

The monotone trend (2048 -> 1024 -> 512, still improving, no turnover) says the
optimum is at or below 512, i.e. **outside the range measured**.

Candidate explanations, none yet tested:

1. **Training-length artifact.** This is a 1500-step screen; production runs are
   10k. Early in training the model cannot yet exploit long range, so a short
   window is a better prior. The advantage could shrink or invert by
   convergence. **This is the explanation that would most change the decision,
   and it is the cheapest to test.**
2. **Capacity misallocation at this scale.** At 450M, attention spent on
   2048-token range may be mostly noise, and narrowing it is a better inductive
   bias rather than a loss of information.
3. **The SWAX effect** (`frontier-speedups-2026-08.md` §4): shorter windows force
   reliance on other mechanisms. But that argument is about models WITH a memory
   mechanism, and this is `vanilla_local`, which has none. It does not explain
   this result and should not be cited for it.

## Consequences for the memory research — read this before using the number

`docs/training-speedups.md` warns that architecture changes confound the
memory-mechanism comparison, and this is one. Two specific implications:

- **The baseline just got much stronger.** Every memory variant is compared
  against vanilla. If vanilla at window 512 is -0.062 better for free, memory
  mechanisms have a substantially harder bar to clear than the finished sweeps
  assumed. Prior null results were measured against a weaker vanilla.
- **The memory thesis should be re-tested at window 512.** The frontier doc
  already flagged window 2048 at seq 8192 (a 4:1 ratio) as possibly too generous
  for memory to matter. This says 512 (16:1) is better for vanilla *too*, so the
  comparison should be redone there — for the memory variants, the effect could
  go either way and must be measured, not assumed to transfer.

**Do not port window 512 to a memory variant on the strength of this table.**
The whole premise of those variants is that they use context differently.

## Follow-up RESULTS (2026-08-14) — both questions answered

`runs/window_followup_450m/`.

### Q1: the curve turns at 512. The optimum is bracketed.

| window | val_bpb @1500 |
|---:|---:|
| 2048 | 1.12787 |
| 1024 | 1.11602 |
| **512** | **1.10995** |
| 256 | 1.11320 |

256 is *worse* than 512, so 512 is the optimum and the monotone trend that
prompted this follow-up has ended. No need to test 128.

### Q2: the quality win decays with horizon — by 61% from 1500 to 4000 steps

| horizon | short-window advantage at equal wall-clock |
|---:|---:|
| 1500 steps | **-0.06193** |
| 4000 steps | **-0.02388** |

This substantially supports explanation 1 above (training-length artifact) and
weakens 2 and 3. The advantage is shrinking as the model learns to use context,
exactly as a "short window is a better early prior" story predicts.

It has NOT vanished: -0.024 at 4000 steps is still comparable to per-head Muon's
-0.027, the best optimizer result on this model. But two points cannot
distinguish "decaying toward zero" from "asymptoting at ~-0.02", and production
runs are 10k steps. **Do not quote -0.062. Quote -0.024, and label it as still
decaying.**

### THE THROUGHPUT WIN IS THE DURABLE PART

The quality bonus is decaying; the 1.196x is not. Even if the gap reaches zero
at 10k, window 512 still reaches the same loss ~20% faster. That is the part of
this result that does not depend on the horizon, and it is the reason the finding
survives its own decay.

### Throughput numbers in this follow-up are CONTAMINATED — do not use them

The box was running heavy CPU compilation concurrently (user-reported, and
visible in the data): `w2048_4k` measured 47,081 tok/s against 64,221 for the
identical config in the main screen, i.e. **0.733x**. GPU utilisation was
observed at 17% / 152 W. All `tok/s` in this follow-up are invalid.

**The val_bpb numbers are unaffected.** They are a function of step count, seed
and data order, none of which CPU starvation changes — a starved loader makes
the same steps take longer, not different steps.

One real consequence for Q2: the iso-time sizing assumed a clean 1.196x, but the
arms were contaminated *unequally* (observed ratio 1.341x). Actual wall-clock was
6.19 h for `w2048_4k` and 5.52 h for `w512_4k_isotime` — the short-window arm got
**10.8% LESS** time than the control, not equal time. The contamination therefore
worked *against* the arm that won, which makes **-0.024 a conservative lower
bound** on the true equal-wall-clock advantage at this horizon. The direction of
the conclusion is safe; the magnitude is understated.

Re-measure throughput with `scripts/bench_arms.py` on an idle box before quoting
any tok/s from 2026-08-13 onward.

## Follow-up queued (now complete — see above)

`configs/window_followup_450m.yaml` (via `scripts/queue_window_followup.sh`):

- `w256` at 1500 steps — where does the curve turn? The optimum is below 512 and
  currently unbracketed.
- `w2048_4k` (4000 steps) vs `w512_4k_isotime` (4784 steps) — **the decision
  arm.** Does the iso-time win survive ~3x the horizon? If -0.062 holds at 4k it
  is real; if it decays toward zero it is a warmup artifact and the production
  default should not change.

Each arm gets a complete WSD schedule for its own step count, so the pair is a
fair comparison rather than one arm being caught mid-decay.
