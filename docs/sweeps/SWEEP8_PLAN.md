# Sweep 8 — Data Bake-Off (single axis = pretraining corpus)

## Why this sweep exists

The research reports rate **data quality as 3–5× more impactful than architecture**
at 1B+ scale. Sweep 7 proved the ruler is trustworthy on FineWeb-Edu (E5: the
needle + ablation discriminators catch real in-distribution effects). Now we vary
the one lever the reports say matters most, holding everything else fixed.

**Single axis: the training corpus.** Mechanism is frozen to the Sweep-7 winner
(`hier_max_16` summary_memory with `energy_read=true, energy_beta_init=0.25` — the
robustness win from E5), optimizer/recipe frozen to the Sweep-7 Muon recipe,
model size frozen at 25.6M. Only the dataset changes.

## The candidates

| Arm | HF dataset | Why |
|---|---|---|
| `fineweb_edu` (anchor) | `HuggingFaceFW/fineweb-edu` `sample-10BT` | Sweep 1–7 baseline; continuity. |
| `dclm` | `mlfoundations/dclm-baseline-1.0` (or `-parquet`) | Model-filtered web; the reports' top-rated general corpus. |
| `fineweb_hq` | `HuggingFaceFW/fineweb-2-hq` or `fineweb-edu` score-filtered top bucket | Higher-precision educational filter. |

(Exact HF repo IDs are **agent-reported — verify each loads via `datasets` streaming
before committing GPU**; some require a `name`/config and have different `text_field`.)

## The tokenizer confound (must resolve first)

`tokenizers/fineweb_4k.json` was **trained on FineWeb-Edu**. Using it to score DCLM
/ FineWeb-HQ would hand FineWeb-Edu a home-field advantage (lower bpb purely from
better token fit, not better data). Two clean options, pick one and apply to all arms:

- **(A) Shared neutral tokenizer (recommended):** train one 4k BPE on an *equal
  mix* of all three corpora (`scripts/train_custom_tokenizer.py`), use it for every
  arm. Removes the tokenizer as a confound; bpb stays comparable across arms.
- **(B) Per-corpus tokenizer + bits-per-byte only:** train a 4k tokenizer per
  corpus, compare on **bpb** (bits per UTF-8 byte, already the eval metric) which is
  tokenizer-invariant by construction. Cleaner in theory but doubles tokenizer work
  and the needle probe's planted tokens must stay shared.

Decision needed from user before launch. Default to (A).

## Code change required (Sweep 8 is not config-only)

`flower/data.py` hardcodes `HuggingFaceFW/fineweb-edu` / `sample-10BT` and the
`{cache}/sample/10BT/*.parquet` glob in two places (`_FineWebChunkStream.__iter__`,
`fineweb_validation_documents`). Generalize:

- Add `DataConfig` fields: `hf_dataset: str`, `hf_name: str | None`,
  `hf_split: str = "train"`, and keep `text_field`. Make `dataset="hf"` route through
  a generic loader that reads these (the existing `fineweb-edu` path becomes a thin
  preset that fills them in).
- The local-parquet cache glob must key off the dataset, e.g.
  `{FLOWER_DATA_CACHE}/{dataset_slug}/*.parquet`; add the slug to
  `scripts/prefetch_dataset.py` so each corpus caches to its own subdir.
- The hermetic VAL_DOCS train/val split logic is dataset-agnostic already — keep it.
- **Verify the needle probe** (`needle_in_text_probe`) still draws filler from the
  *arm's own* validation stream so "on-manifold" means on-manifold for that corpus.

## Recipe (frozen from Sweep 7)

`optimizer: muon`, `muon_lr 0.003 / ns5 / momentum 0.95`, `warmup 1000`,
`lr_schedule linear_warmup`, `steps 10000`, `seq_len 2048`, `eval_seq_len 8192`,
`local_window 64`, `batch_size 16 / grad_accum 2` (eff. 32), 3 seeds, `energy_read
true / beta_init 0.25`. Launch with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (WSL2 spill guard) and
`FLOWER_DATA_CACHE` pointed at the multi-corpus cache.

## Metrics & gate

Judge on the **validated FineWeb-era discriminators**, now computed per-corpus:
- **val bpb** (tokenizer-invariant) — primary.
- **memory_ablation delta** — is memory more load-bearing on richer data?
- **needle_in_text** (BP + late@8, mean±std over 3 seeds) — recall robustness.
- **blimp_mini** as the grammar guardrail (catch "ppl down, competence down").

**Headline question:** does a better corpus move val bpb by **more than the seed
noise floor measured in E5 (±0.003 bpb)** — and by more than any architecture
change did in Sweeps 1–7 (which were ≤0.03)? If data moves it 3–5× more than
mechanism did, that confirms the reports' prior and sets the priority for the big run.

## Cost

3 corpora × 3 seeds × 10k steps ≈ 9 runs. At the measured ~83 min/variant +
~14 min/variant data-stream startup ≈ **~15 h on the 5090**. Plus one-time:
tokenizer training (minutes) and prefetching DCLM/FineWeb-HQ shards (download-bound;
DCLM is large — cache a sample, not the full set). Prefetch + a 100-step streaming
sanity per new corpus **before** the full launch.

## Verification (run in order)

1. Each new corpus loads via `datasets` streaming and yields non-empty `text_field`
   (1-doc smoke), both from HF and from the local parquet cache.
2. Shared/neutral tokenizer trained and wired; `build_tokenizer` resolves it.
3. 100-step streaming sanity per corpus: tokens/sec sane, no shm/VRAM spill,
   needle probe returns a curve.
4. Full 9-run sweep; report the 3×3 table + per-metric mean±std.

## Out of scope (deferred, unchanged)
Optimizer family → Sweep 9. Parameterization / IsoFLOP ladder → Sweep 10.
FP8/NVFP4 precision → mini validation pass. Do not bundle these in; data is the
single axis here.
