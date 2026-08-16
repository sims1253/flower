# Flower

**Flow-attention memory experiments** — a research framework for training and
evaluating small language models with various memory-augmentation and
KV-cache-compaction strategies.

The codebase supports a sweep over many model variants
(vanilla attention, linear/summary/partitioned/phase memories, flow-matching
attention & memory, Titans MAC, Hamiltonian attention, surprise/bloom/frequency
memories, and the "Still" KV-cache compactor family) under a shared training,
evaluation, and optimizer stack, so results are directly comparable.

## Quick start

Requires Python ≥ 3.13. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # or: uv sync --extra dev  (adds pytest, ruff, ty, torchao)
```

Train a single variant:

```bash
uv run python -m flower.train --config configs/vanilla_local.yaml --variant vanilla_local
# or simply
uv run python main.py -- --config configs/vanilla_local.yaml
```

Run a parallel sweep (1 trial per GPU, round-robin across GPUs):

```bash
uv run python -m flower.sweep_parallel \
    --config configs/sweep4_phase0_remuon.yaml \
    --gpus 0,1 --steps 10000 --output-dir runs/sweep4
```

Common `flower.train` flags: `--variant`, `--steps`, `--batch-size`,
`--gradient-accumulation-steps`, `--optimizer {adamw,muon,aurora}`,
`--seed`, `--device`, `--smoke` (tiny config for a fast sanity check),
`--metrics-json`, `--output-dir`, `--log-backend {none,tensorboard}`.

## Repository layout

```
flower/        # training framework
  models/      # model variants behind a shared build_model() registry
  flows/       # flow-matching / CNF building blocks
  probes/      # evaluation harnesses (composite benchmark probes)
  config.py    # YAML config dataclasses + loader
  data.py      # FineWeb token streaming & validation batches
  optim.py     # adamw / muon / aurora optimizers
  train.py     # single-trial training loop (entrypoint: python main.py)
  sweep.py, sweep_parallel.py
scripts/       # tokenizers, result aggregation, plotting, Vast.ai automation
configs/       # experiment YAMLs (sweep definitions + per-variant configs)
tokenizers/    # BPE tokenizer models — NOT checked in; regenerate (see tokenizers/README.md)
tests/         # pytest suite (shapes, causal masks, smoke train, sweeps, scripts)
docs/          # research notes, sweep plans & results, training roadmap
```

## Documentation

Research notes and experiment records live under [`docs/`](docs):

- [`docs/training-speedups.md`](docs/training-speedups.md) — roadmap for the next
  iteration (the live work plan).
- [`docs/sweeps/`](docs/sweeps) — per-sweep plans and results write-ups.
- [`docs/research/`](docs/research) — background research notes.
- [`docs/OVERVIEW.html`](docs/OVERVIEW.html) — a standalone HTML project overview
  (open in a browser).
- [`NEXT_IDEAS.md`](NEXT_IDEAS.md) — prioritized queue of candidate experiments.

## Running on Vast.ai

The `scripts/vast_*.sh` helpers automate offer search, instance creation, sweep
launch, and teardown on Vast.ai. They read configuration defaults from
[`configs/vast_defaults.env`](configs/vast_defaults.env) and require credentials
**only** via environment variables (never committed to the repo):

```bash
export VAST_API_KEY=...   # required
export HF_TOKEN=...       # required when training pulls datasets
```

`scripts/vast_common.sh` documents the full set of variables and includes
safety guards (`--yes` is required for any spend or destroy action).

## Testing

```bash
uv run pytest            # full suite
uv run pytest tests/test_shapes.py   # one area
```

## Configuration

Experiments are defined declaratively in YAML (see `configs/`). A config
specifies the tokenizer, data, model variant(s), optimizer, training schedule,
and evaluation probes; sweeps expand a base config over a grid of variants and
seeds. See `configs/sweep4_phase0_remuon.yaml` and
`configs/sweep4_phase1_memory_bake_off.yaml` for representative examples.

## Status

Active research codebase. The framework and sweep infrastructure are stable; the
model variants are under active experimentation (see `NEXT_IDEAS.md`).
