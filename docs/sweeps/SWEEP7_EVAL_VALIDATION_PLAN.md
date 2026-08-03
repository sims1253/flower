# Sweep 7 — Eval-Framework Validation (make the ruler transfer to FineWeb)

## Why this plan exists

Sweep 7's premise is "fix the ruler, then make mechanism bets." Phase A produced a
ruler that *does* discriminate — but **only for models trained on a synthetic MQAR
curriculum**. The current Sweep-7 winner (`energy_read=true`, `energy_beta_init=0.25`)
is a real effect *inside that synthetic regime* and has **not been shown to transfer
to FineWeb-trained language models**, which is the actual target distribution.

Decision (2026-05-30): **the eval framework is not "trustworthy" until a mechanism
ranking established on the ruler reproduces on FineWeb.** No scaling / data /
optimizer sweeps until that transfer is demonstrated.

## The concrete failure the ruler still has

`mqar_probe` (`flower/probes/composite.py:124`) draws keys/values as random ids in
`0..synthetic_vocab_size` via `_probe_vocab` (`:17`). For a FineWeb-trained model
those ids are **off-manifold**, so every FineWeb variant floors at breaking-point 0
(Phase A report, "Original MQAR result" and "Repaired MQAR Eval"). The probe
therefore measures *"can this architecture learn MQAR when trained on MQAR,"* not
*"does a text-trained LM with this mechanism recall better."* Those are different
questions, and only the second one matters for the project's end goal.

What already works in-distribution:
- `memory_ablation_probe` (`:431`) runs on the real FineWeb validation stream and in
  Phase A gave a sensible, non-degenerate ranking (vanilla 0.00, hier 0.24,
  bloom 0.21, adamw 0.56). This is an existing FineWeb discriminator we under-use.

What is unverified:
- `tests/test_sweep7_probes.py` only checks tensor shapes / seen-lengths and the
  energy-read math. **No test proves a probe returns a high score for a
  known-perfect-recall model and a chance score for a memoryless one.** So the
  ruler's core claim — that breaking-point tracks real recall ability — is asserted,
  not tested.

## Exit criterion (the gate for "ruler is trustworthy")

A mechanism advantage observed on the synthetic MQAR ruler (energy-read beats
control at long-64/128) must reproduce as a **non-zero, seed-consistent signal on at
least one in-distribution FineWeb discriminator** (memory-ablation delta and/or the
new on-manifold recall probe), with non-overlapping mean±std across 3 seeds. If it
does not reproduce, the synthetic ruler is *not* a valid proxy for FineWeb and we
must say so explicitly before any further mechanism spend.

---

## Phase E — steps (in order)

### E1. On-manifold (real-text) recall probe
Add `text_recall_probe` to `composite.py`. Same multi-query associative-recall
structure as `mqar_probe`, but keys/values are drawn from **real, in-vocabulary
tokens** so a FineWeb-trained model is on-manifold:
- For text datasets: sample key/value tokens from the model's actual high-frequency
  token set (estimate from the FineWeb val stream once, cache the id list), not from
  `0..synthetic_vocab`. Optionally frame as a needle/passkey in real text context.
- Keep the proven **candidate-set scoring** (rank correct planted value against the
  other planted values from the same example) to avoid the full-vocab floor.
- Reuse `_eval_seq_len` / `_long_context_batch_size` so it honours `eval_seq_len`
  and the VRAM cap.
- Return the same `capacity_curve` / `breaking_points` / `breaking_point` shape so it
  plugs into `run_composite_eval` and the existing reporting.

### E2. Harden probe correctness with deterministic unit tests
In `tests/test_sweep7_probes.py`, add:
- **True-positive:** a hand-built `OracleRecallModel` whose logits copy the planted
  value for the queried key → `text_recall_probe`/`mqar_probe` must return a high
  breaking point (>= largest num_pairs).
- **True-negative:** a uniform/memoryless model → breaking point 0 and accuracy
  near `1/num_pairs` (candidate-set chance), not coincidentally passing.
This proves the ruler can register both a real recall capability and its absence —
the property Phase A assumed but never tested.

### E3. Checkpoint audit (do not silently trust `runs/`)
Add `scripts/audit_checkpoints.py`: enumerate `runs/*/variants/*/*.pt`, reconstruct
the model from each run's saved config, attempt `load_state_dict`, and report
{loads-clean, shape-mismatch, missing-config, stale}. Output a table so E4/E5 only
build on checkpoints that actually load. Treat any mismatch as stale, not as a bug
to force-fix.

### E4. Cross-probe correlation pass (no new training)
On the cleanly-loading FineWeb Phase A checkpoints (`vanilla_local`, `hier_max_16`,
`bloom_memory`, `hier_max_16_adamw` × seeds 0/1/2) run: `memory_ablation_probe`,
the new `text_recall_probe`, and the old `mqar_probe`. Report each discriminator's
ranking and the rank correlation between them.
- **Pass:** `text_recall_probe` separates variants and its ranking is consistent
  with `memory_ablation_probe` (both are in-distribution). The old synthetic
  `mqar_probe` is expected to stay near-zero — confirming the diagnosis rather than
  contradicting it.
- This is cheap (eval-only) and tells us whether E5 is even worth the GPU.

### E5. FineWeb transfer gate for the energy-read winner (new training)
Train `hier_max_16_energy_beta025` vs control `hier_max_16` on **FineWeb**, Sweep-7
recipe (Muon lr 3e-3 / ns5 / warmup1k, seq2048, eff. batch 32), 3 seeds. Measure
val loss, `memory_ablation_probe` delta, and `text_recall_probe`.
- **Gate:** the energy-read advantage seen on synthetic MQAR (long-64/128) shows up
  on at least one in-distribution FineWeb discriminator with non-overlapping
  mean±std. Pass → the ruler is a valid FineWeb proxy; proceed to the scaling
  roadmap. Fail → document that synthetic-MQAR rank does not transfer, and switch
  the project's headline discriminator to the in-distribution probes.

**Budget:** E1–E4 are code + eval-only (hours, no training). E5 is 2 variants × 3
seeds × 10k steps ≈ 6 runs ≈ 2–3 h on the 5090.

## Verification per step
1. `uv run python -m pytest tests/ -x -q` after E1/E2 (probe math + new oracle tests).
2. E3: audit table printed; note any stale checkpoints explicitly.
3. E4: correlation table; decision recorded in `reports/`.
4. E5: 3-seed table; gate decision recorded in `reports/`.

## Out of scope (unchanged from SWEEP7_PLAN.md)
Dataset (Sweep 8), optimizer (Sweep 9), parameterization/IsoFLOP ladder (Sweep 10),
FP8/NVFP4 precision. None start until the E5 gate is resolved one way or the other.
