# Sweep 9 — Optimizer (single axis = optimizer, LR-controlled)

## Why this sweep exists

Optimizer is the lever closest to the architecture question and the most
config-light of the deferred axes. The research report
(`research_training_efficiency_2026-05.md` §2) flags **Aurora** as a near-drop-in
Muon replacement that fixes Muon's neuron-death on tall MLP matrices (claimed
2.26 vs 2.31 loss at 1.1B, MIT-licensed, ~6% overhead).

**But the report's own reality check (§2.7) is the design constraint:** *against a
well-tuned AdamW, no optimizer exceeds ~1.4× speedup; Muon's edge shrinks from
1.3× at 0.1B to ~1.1× at 1.2B; LR matters more than optimizer choice.* Sweeps 4–5
already proved this the hard way — the AdamW arm looked terrible only because it ran
at a badly-tuned LR (`2.83e-4`), a confound, not a real optimizer gap.

**Therefore Sweep 9 controls for LR.** We do not compare optimizers at a single
borrowed LR. Each optimizer is given a short LR sweep first; the bake-off compares
each optimizer *at its own best LR*. Otherwise the result is uninterpretable.

## Phase 9A — LR calibration (cheap, gates 9B)

For each optimizer, sweep LR at 1 seed × short horizon (3k steps) and pick the LR
with the best val bpb. This is the analogue of Sweep 4 Phase 0 (which found
muon@3e-3) but done per-optimizer so no arm is handicapped.

- **muon** (anchor): LR ∈ {1e-3, 3e-3 (known), 6e-3} — confirm 3e-3 still wins.
- **adamw** (fair baseline): LR ∈ {1e-3, 2e-3, 3e-3} — give AdamW the well-tuned LR
  it never got in Sweeps 4–5, so the comparison is honest.
- **aurora**: LR ∈ {3e-3, 6e-3, 1e-2} (Aurora tolerates higher LR per the report).

Gate: each optimizer has a clear LR optimum (not at the edge of the swept range; if
it is, extend). ~9 short runs ≈ 2–3 h.

## Phase 9B — 3-seed bake-off at best LR (gated on 9A)

Run muon / adamw / aurora, each at its 9A-best LR, 3 seeds × 10k steps, on the
frozen Sweep-7 setup (energy-read hier_max_16, seq 2048, eval 8192, window 64,
batch 16/accum 2). Same validated discriminators as E5.

Optional 4th arm if cheap: **SODA** (report §2.2) as a thin wrapper that removes
weight-decay tuning — only if it drops into the factory in <1 day.

## Code change required

`flower/optim.py` is clean and localized — adding an optimizer means: implement the
class (mirroring `Muon`), add a branch in `build_optimizer` dispatch, add config
fields (`aurora_lr`, etc.) to `TrainingConfig`. The Muon/AdamW param-routing
(`_classify_params`: embeddings+1D→AdamW, 2D→matrix-optimizer) is reusable as-is —
Aurora is also a 2D-matrix optimizer, so it slots into the Muon code path.

**Aurora sourcing (agent-reported — VERIFY FIRST):** report cites
`github.com/tilde-research/aurora-release`, MIT. Before implementing:
1. Confirm the repo exists and the license permits vendoring.
2. Prefer `uv add` if it's pip-installable; else vendor the single optimizer file
   into `flower/optim_aurora.py` with attribution.
3. If the repo/paper cannot be verified (the arXiv IDs 2604/2605.* are
   future-dated, agent-reported), **do not fabricate an implementation** — fall
   back to a muon-LR-vs-adamw-LR bake-off (still a real, useful result) and report
   Aurora as blocked-pending-verification.

## Metrics & gate

Judge on the E5-validated FineWeb discriminators:
- **val bpb** — primary (lower = better).
- **time-to-fixed-loss** (nanoGPT-speedrun style) — the report stresses wall-clock
  to a target loss is more discriminative than raw final loss. Log it.
- **memory_ablation delta** + **needle_in_text** — does the optimizer change how
  load-bearing memory is / recall robustness? (Probably not, but it's free.)
- **blimp_mini** guardrail.

**Headline question:** at *each optimizer's own best LR*, does Aurora (or any arm)
beat the Muon anchor by more than the E5 seed-noise floor (±0.003 bpb)? The §2.7
prior says the gap will be small at 25M (~1.1–1.3×). A small-but-consistent,
non-overlapping-mean win is the bar; anything inside seed noise = "Muon stays."

## Cost

9A: ~9 short runs ≈ 2–3 h. 9B: 3 optimizers × 3 seeds × 10k ≈ 9 runs ≈ 15 h
(+3 h if SODA 4th arm). Launch with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `FLOWER_DATA_CACHE`.

## Verification (run in order)

1. New optimizer unit test: one step on a 2-layer toy model decreases a quadratic
   loss; param routing identical to Muon (embeddings/1D→AdamW).
2. `uv run python -m pytest tests/ -q` green (esp. `test_*optim*` / shapes).
3. 100-step sanity per optimizer: tokens/sec sane, no VRAM/shm spill, loss falling.
4. 9A LR sweep → pick best LR/optimizer. 9B 3-seed bake-off → 3×(metrics) table.

## Out of scope (deferred)
Data → Sweep 8. Parameterization / IsoFLOP ladder → Sweep 10. FP8/NVFP4 → mini
pass. LoRA-Pre momentum compression (report §2.6) is a memory tool for the 1–7B run,
not a 25M ranking question — defer to scale-out.
