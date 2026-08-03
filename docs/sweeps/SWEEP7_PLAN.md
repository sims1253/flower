# Sweep 7 — Fix the lens, then two bold bets

## Why this sweep exists

Sweeps 1–6 produced a consistent null result: architecture and memory-mechanism
changes barely move val-loss at 25.6M params / 10k steps. The six research
reports (`reports/research_*_2026-05.md`) converge on a single diagnosis — **the
null result is mostly a measurement problem, not (only) a model-size problem.**

Three concrete causes, all verifiable in the current code/results:

1. **Memory is redundant at seq=2048.** Memory mechanisms only separate when the
   context exceeds what attention already covers. We train at seq=2048 with
   `local_window=128`, and we *evaluate* recall at tiny lengths
   (`associative_recall_probe` builds seq≈24; `induction_copy_probe` caps
   `pattern_len` at 32). The memory module never has to be load-bearing, so all
   variants tie. (See `flower/probes/composite.py:60-95`, `:26-56`.)
2. **The recall probe is too easy.** `associative_recall_probe` is single-key,
   single-query, 8 pairs. It saturates or floors — it cannot trace the
   *capacity cliff* where compressed memories actually break. flow_ot's failure
   (Sinkhorn plan collapsed to uniform = mean-pooling, `ot_plan_entropy → log 16`
   in the Sweep 5 report) is exactly the kind of thing a point-estimate probe
   misses.
3. **The mechanism bake-offs ran on the wrong recipe.** Sweep 4 Phase 0 found
   `muon@lr3e-3 / ns5 / warm1k → val 3.413`
   (`configs/sweep4_phase1_memory_bake_off.yaml:71`), but Sweep 5/6 ran on
   `adamw / lr 2.83e-4 → control val 3.627`
   (`configs/sweep5_novel_memory_bake_off.yaml:30,40`). The mechanisms were
   compared on a ~0.2-nat-worse-trained model. A poorly trained model flattens
   real differences.
4. **No seed variance.** Sweep 5/6 ran one seed. At 25M, seed noise is large
   enough to fake a ranking. The config already declares `seeds=(0,1,2)`
   (`flower/config.py:132`) — it's half-wired, not absent.

**The deliverable of Sweep 7 is a trustworthy ruler.** Only after the ruler is
proven to discriminate do we spend GPU on the two new mechanism bets. We change
*one class of thing at a time*; data, optimizer family, parameterization, and
precision are deliberately deferred (see Roadmap).

This sweep runs in two phases, sequentially. Phase B is gated on Phase A passing.

---

## Phase A — Sharpen the lens

Goal: make long-context recall the primary, discriminating axis, and prove the
suite separates *known* variants before testing new ones.

### A1. Decouple eval context from train context

Add an `eval_seq_len` knob so we can evaluate at lengths beyond the training
seq_len (the extrapolation + memory-pressure regime).

- `flower/config.py` — add to `DataConfig` (or `EvalConfig` if cleaner):
  ```python
  # Sweep 7: evaluate at a longer context than we train at, to put memory under
  # pressure (recall across more than local_window tokens) and test length
  # extrapolation. None = use sequence_length. May exceed model.max_seq_len.
  eval_seq_len: int | None = None
  ```
- `flower/eval.py:124-125` — when `eval_seq_len` is set, use it instead of
  `min(sequence_length, max_seq_len)`. **Caveat:** RoPE cos/sin caches are sized
  to `max_seq_len` (`base.py` `RotaryEmbedding`). To eval beyond `max_seq_len`,
  size the RoPE cache to `max(max_seq_len, eval_seq_len)` at construction, or
  rebuild it lazily. This is a real edit, not a config-only change.
- The probes (`composite.py`) must take an explicit eval length argument instead
  of reading `cfg.model.max_seq_len`, so they can run at `eval_seq_len`.

Sweep-7 setting: train `seq_len=2048`, `eval_seq_len ∈ {2048, 4096, 8192}`,
`local_window=128` (unchanged). At 8192/128 the model must route ~64× the window
through memory or attention-extrapolation — that is where mechanisms diverge.

### A2. Replace the easy recall probe with a capacity-curve MQAR

Add `mqar_probe` (multi-query associative recall) to `composite.py`, and make
it sweep difficulty rather than report a single point:

- Multi-query: plant N key→value pairs, then query *several* keys (not one) at
  the end. This is the MAD-suite discriminant that separates memory
  architectures at small scale (per `research_why_null_results` and
  `research_frontier_obscure_mechanisms`; multiple 2026 papers reproduce our
  exact "ties on ppl, collapses on multi-key recall" pattern).
- Difficulty sweep: `num_pairs ∈ {16, 32, 64, 128}` × `delay ∈ {short, long}`,
  run at `eval_seq_len`. Report the **capacity curve** (accuracy vs num_pairs)
  and a scalar summary: **breaking point** = largest `num_pairs` with acc ≥ 0.5.
- Keep `induction_copy_probe` but run it at `eval_seq_len` too (long-range copy).

The capacity curve is the headline mechanism metric for Phase B, alongside the
existing `memory_ablation_probe` delta (which is already a good discriminator —
"how much does the model actually need its memory?", `composite.py:288-304`).

### A3. Multi-seed + a discrimination gate

- Wire `seeds=(0,1,2)` through the sweep runner so each variant runs 3 seeds;
  report mean ± std on every metric.
- **Discrimination gate (the exit criterion for Phase A):** run the three
  *known* variants — `vanilla_local`, `hier_max_16` (Sweep 4 winner),
  `bloom_memory` (Sweep 5 winner) — through the new suite on the fixed recipe
  (A4). The gate passes iff they produce a **consistent ranking on the MQAR
  capacity-curve breaking point across all 3 seeds, with non-overlapping
  mean±std**. If they don't separate, the ruler is still blind — fix the probe
  (harder/longer MQAR, different delay) before any Phase B work. **No new
  architecture runs until this passes.**
- **adamw recipe-cost control arm (decided in):** also run `hier_max_16` under
  the old Sweep-5 recipe (`adamw / lr 2.83e-4`) × 3 seeds. This quantifies how
  much the Sweep-5 recipe mistake cost us (val and capacity-curve delta vs the
  muon recipe) — a clean number for the blog and a sanity check that the recipe
  change is the improvement we think it is. +3 runs.

### A4. Fix the recipe to the best known one

Sweep 7 baseline recipe = the Phase-0 winner, not Sweep 5's adamw:
`optimizer: muon`, `muon_lr: 0.003`, `muon_ns_steps: 5`, `muon_momentum: 0.95`,
`warmup_steps: 1000`, `lr_schedule: linear_warmup`, `steps: 10000`,
`seq_len/max_seq_len: 2048`, `local_window: 128`, `batch_size: 16`,
`gradient_accumulation_steps: 2` (effective batch 32). This re-baselines the
control — continuity to Sweep 5's exact val numbers is already broken by the
eval changes, so we may as well train well.

### A5. Adopt Class-1 (math-preserving) throughput wins

These don't change *what* is learned, only wall-clock — and the saved time pays
for the 3× seed budget Phase A needs. Adopt if not already on, after a one-run
loss-neutrality check:

- `torch.compile` on the model.
- Flash-Attention path for the softmax-attention branch (FA2; **not** FA3 — see
  `research_cheap_1to7b_path`).
- Sequence packing in the dataloader (no padding waste at seq=2048).

Explicitly **NOT** here: FP8/NVFP4 (precision is a ranking variable, not free —
see Roadmap), dataset swap, optimizer family change, CompleteP/muP.

**Phase A cost estimate:** discrimination gate (3 variants + 1 adamw control
arm) × 3 seeds × 10k steps ≈ 12 runs ≈ 4–5 h on the 5090 (less with A5).

---

## Phase B — Two bold, original bets (gated on A passing)

Not paper re-implementations. Each bet attacks a *specific observed failure* and
reuses existing Flower modules.

### B1 — Load-bearing memory

**Thesis:** memory mechanisms tie because at seq=2048 they're redundant with
attention. Make memory the *only* long-range path and they must differ.

- Regime, not a new module: strictly local attention (`local_window=128`) with
  both **train seq and eval seq ≫ window**, so any cross-window dependency *must*
  flow through the memory bank. Reuses `summary_memory` (hier_max_16) and
  `bloom_memory` unchanged.
- **Train-long (decided in):** extend the *training* seq_len, not just eval, so
  the memory is forced to be load-bearing during learning — the truer test of
  the hypothesis. Train at `seq_len ∈ {4096, 8192}`; eval at ≥ train. This costs
  memory and wall-clock: at seq=8192/batch=16 the 5090 will likely need
  `batch_size=4, gradient_accumulation_steps=8` (effective batch 32) and/or FFN
  gradient checkpointing. Confirm with the real-scale sanity run before
  launching; reduce batch before reducing seq.
- Optional forcing knob: shrink `local_window` (e.g. 64) to widen the
  window→context gap and increase memory pressure further.
- **The interesting result (blog-worthy):** a model whose `memory_ablation`
  delta is *large* — i.e. zeroing memory collapses it — proving the memory is
  genuinely used, not decorative. Then the MQAR capacity curve should *finally*
  separate summary vs bloom vs a no-memory control. This is the direct,
  demonstrable answer to "why didn't memory matter?"
- Arms: `vanilla_local` (no memory — should fail long-range), `hier_max_16`,
  `bloom_memory`, each at `local_window ∈ {128, 64}` × `eval_seq_len ∈ {2k,4k,8k}`.

### B2 — Energy-based memory read (sharp retrieval)

**Thesis:** our compressed-memory reads blur. flow_ot's coupling collapsed to
uniform = mean-pooling and bought nothing. The read is a convex weighted average
over slots — it *cannot* do sharp, selective retrieval.

- Replace the convex/softmax weighted-average read in `summary_memory`'s
  `mem_read` with a **log-sum-exp energy read with a learnable inverse
  temperature β**: as β grows the read sharpens from averaging toward
  hard-max retrieval. Inspired by the Free Energy Mixer
  (`research_frontier_obscure_mechanisms`, arXiv:2602.07160 — *agent-reported,
  verify before coding*), but applied to the memory read rather than to
  attention. This is an original application targeting our exact failure.
- New config fields (`ModelConfig`): `energy_read: bool = False`,
  `energy_beta_init: float = 1.0` (β learnable, init small so it starts near the
  current mean-pool behavior and can sharpen if it helps).
- Arms: `hier_max_16` (control, convex read) vs `hier_max_16_energy` (same
  module, energy read). Single clean axis: read operator.
- **The interesting result:** if the energy read lifts the MQAR capacity-curve
  breaking point, we've shown the bottleneck was *read sharpness*, not capacity —
  a crisp, publishable mechanistic finding.

### B2 × B1 interaction

Run B2's energy read *inside* B1's load-bearing regime. Sharp retrieval should
matter most precisely when memory is the sole long-range path. The headline
comparison is the MQAR capacity curve of `hier_max_16` vs `hier_max_16_energy`
at `eval_seq_len=8192, local_window=64`.

**Phase B cost estimate:** ~4 mechanism arms × relevant (window × seq) cells ×
3 seeds. With train-long arms at seq 4k/8k (slower per step, smaller batch),
budget Phase B at ≈ 12–18 runs ≈ 10–16 h on the 5090 — the long-seq training is
the dominant cost. Start with the seq=4096 cells; only run seq=8192 if 4096
already shows the capacity curves separating.

---

## What is deliberately out of scope (and why)

These are high-leverage but are *separate axes* — bundling them into Sweep 7
would destroy attribution. Each becomes its own sweep once the ruler is trusted.

- **Dataset quality** (DCLM-Edu / FineWeb-HQ / Ultra-FineWeb). Changes the
  ranking surface. → Sweep 8.
- **Optimizer family** (Aurora / SODA / Muon2). Aurora specifically fixes Muon's
  neuron death on MLP matrices (`research_training_efficiency`). → Sweep 9.
- **Parameterization** (CompleteP / muP). A confound that may mask differences,
  but orthogonal to "which mechanism." → folded into Sweep 10's ladder.
- **FP8 / NVFP4 precision.** Tempting on Blackwell (5090) but **not loss-neutral**
  — low precision can shift rankings, and B2's energy read (log-sum-exp + sharp
  β) is exactly the precision-sensitive case. → dedicated validation pass below.

---

## Roadmap — the sweeps after this one

A sketch, so Sweep 7's decisions are made with the sequence in mind. Order is
deliberate: trust the ruler, then vary one big lever at a time, then scale.

- **Precision validation pass (mini, between 7 and 8).** Take 1–2 Sweep-7
  variants, re-run in FP8/NVFP4, ask: *does low precision reproduce the bf16
  ranking and capacity curve?* If yes → adopt for all large runs (big throughput
  + memory win on Blackwell). If no → we know before it corrupts a decision.
  Pure infra gate, not an architecture sweep.
- **Sweep 8 — Data bake-off.** Single axis = dataset. FineWeb-Edu (anchor) vs
  DCLM-Edu vs FineWeb-HQ, best Sweep-7 mechanism fixed. The reports rate data as
  3–5× more impactful than architecture at 1B+, so this likely matters more than
  any mechanism for the end goal.
- **Sweep 9 — Optimizer.** Single axis = optimizer. Muon (anchor) vs Aurora vs
  SODA. Aurora is a near-drop-in Muon replacement claiming to fix MLP neuron
  death; if it transfers at 25M it carries to the big run.
- **Sweep 10 — IsoFLOP ladder + CompleteP.** Now that the ruler is trusted, run
  the winning mechanism at 25M / 70M / 160M under CompleteP and ask the question
  that started all this: **does the mechanism gap open with scale?** This is the
  go/no-go for committing to a large run.
- **Scale-out (toward the real goal).** Per `research_cheap_1to7b_path`: don't
  pre-train 1–7B from scratch. Continued-pretrain a strong multilingual base
  (Qwen3-4B-Base or Gemma-3-4B-pt) on DE+EN, then QLoRA+DoRA finetune on personal
  chat history, optionally on-policy distillation to repair chat ability. Sweep
  7–10 inform which mechanism (if any) is worth porting into that run, and — just
  as importantly — give the blog its spine: *how we learned to measure, and what
  actually moved the needle.*

---

## Verification (run in order, after Phase A edits)

1. **Tests pass** — `uv run python -m pytest tests/ -x -q` (CPU variants + the
   CUDA path). Pay attention to any RoPE-cache-length failure from the
   `eval_seq_len > max_seq_len` change; that's a real bug, fix the cache sizing.
2. **Probe smoke** — confirm `mqar_probe` runs at `eval_seq_len=8192` on a tiny
   synthetic model and returns a capacity curve (dict of num_pairs → accuracy).
3. **Discrimination gate** — run the 3 known variants × 3 seeds; confirm the
   MQAR breaking point separates them with non-overlapping mean±std. **This is
   the Phase A exit gate.** If it fails, harden the probe, do not proceed to B.
4. **Real-scale sanity** — one Sweep-7-recipe run, 100 steps, confirm tokens/sec
   and no OOM at seq=2048/batch=16 with `torch.compile` on. If OOM, drop batch to
   8 / accum to 4 before launching.

---

## Resolved decisions

- **B1 trains long, not just evals long.** Training seq extends to 4096/8192 so
  memory is load-bearing during learning. Accepted cost: smaller batch + grad
  accum + possible FFN checkpointing. (Folded into B1 and the Phase-B budget.)
- **adamw stays as a Phase-A control arm** to quantify the Sweep-5 recipe-cost
  delta. (Folded into A3, +3 runs.)
