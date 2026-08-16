#!/usr/bin/env python3
"""Measure training-stream token throughput vs DataLoader worker count.

WHY THIS EXISTS
  docs/profiling/baseline_profile.md measures 52,207 tok/s for the 450M
  `vanilla_matched` step on *synthetic* tokens, but the two finished seeds in
  runs/sweep13_450m_longctx logged 41,918 and 44,810 tok/s on real FineWeb-Edu.
  That ~17% gap is the input pipeline, not the model.

  The cause was structural: `token_batches` accepted `num_workers` /
  `prefetch_factor` but dropped them on the training path, so training always
  ran on `_fineweb_loader`'s 2-worker default regardless of configuration.
  Validation was the only path that honoured them. This script measures what
  the loader can actually sustain so `data.num_workers` gets set from evidence.

WHAT IT MEASURES
  Pure loader throughput — no model, no GPU compute. Each configuration builds
  the real training stream at the real shape (batch x seq from the config) and
  times how long it takes to pull `--batches` batches after a warmup. The
  number to compare against is the step's token consumption rate: if the loader
  sustains less than the model's synthetic-token rate, the loader is the
  bottleneck and the GPU is idling between steps.

  Workers each hold their own HF streaming iterator and sharding is
  `islice(worker_id, None, num_workers)` over documents, so every worker
  decodes the full parquet stream but keeps only its 1/N share. Decode work
  therefore scales with worker count while tokenization work does not — which
  is exactly why the useful worker count has to be measured rather than set to
  nproc.

USAGE
  FLOWER_DATA_CACHE=data_cache PYTHONPATH=. \
    uv run python scripts/bench_data_workers.py
  ... --workers 2,4,6,8,12 --batches 40
  ... --config configs/sweep13_400m_longctx32k_checkpoint.yaml
"""

from __future__ import annotations

import argparse
import tempfile
import time

import torch
import yaml

from flower.config import load_config
from flower.data import token_batches
from flower.sweep import load_sweep, select_variants


def load_variant_config(sweep_path: str, variant_name: str):
    """Merge a sweep variant and load it through the same path a real run uses.

    Unlike scripts/profile_step.py's namesake this keeps the *real* dataset —
    the whole point here is to measure FineWeb-Edu tokenization throughput, so
    substituting synthetic tokens would measure nothing.
    """
    _sweep_name, variants = load_sweep(sweep_path)
    selected = select_variants(variants, variant_name, limit=None)
    if not selected:
        raise ValueError(f"Variant {variant_name!r} not found in {sweep_path}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(selected[0]["config"], tf, sort_keys=False)
        return load_config(tf.name)


def bench_workers(cfg, batch_size: int, workers: int, batches: int, warmup: int) -> dict:
    """Time `batches` pulls from the training stream at `workers` DataLoader workers."""
    stream = token_batches(
        cfg.data,
        batch_size,
        torch.device("cpu"),
        seed=0,
        num_workers=workers,
        prefetch_factor=4,
    )

    # Warmup covers worker spawn, HF dataset open, and tokenizer build — all
    # one-time costs that would otherwise be charged to the first few batches.
    t_first = time.perf_counter()
    first = next(stream)
    startup = time.perf_counter() - t_first
    for _ in range(warmup):
        next(stream)

    seq_len = int(first.shape[-1])
    tokens_per_batch = batch_size * seq_len

    start = time.perf_counter()
    for _ in range(batches):
        next(stream)
    elapsed = time.perf_counter() - start

    return {
        "workers": workers,
        "startup_s": startup,
        "tok_s": tokens_per_batch * batches / elapsed,
        "batch_ms": elapsed / batches * 1e3,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep13_450m_longctx_memory.yaml")
    ap.add_argument("--variant", default="vanilla_matched")
    ap.add_argument("--workers", default="2,4,6,8,12")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    cfg = load_variant_config(args.config, args.variant)
    batch_size = cfg.training.batch_size

    print(f"config={args.config} variant={args.variant}")
    print(f"batch={batch_size} seq={cfg.data.sequence_length} tokenizer={cfg.data.tokenizer}")
    print(f"measuring {args.batches} batches after {args.warmup} warmup batches\n")
    print(f"{'workers':>7} {'tok/s':>10} {'ms/batch':>10} {'startup s':>10}")

    results = []
    for w in [int(x) for x in args.workers.split(",")]:
        r = bench_workers(cfg, batch_size, w, args.batches, args.warmup)
        results.append(r)
        print(f"{r['workers']:7d} {r['tok_s']:10,.0f} {r['batch_ms']:10.1f} {r['startup_s']:10.1f}")

    best = max(results, key=lambda r: r["tok_s"])
    base = next((r for r in results if r["workers"] == 2), None)
    print(f"\nbest: {best['workers']} workers at {best['tok_s']:,.0f} tok/s")
    if base and base is not best:
        print(f"vs 2-worker default ({base['tok_s']:,.0f} tok/s): {best['tok_s'] / base['tok_s']:.2f}x")
    print(
        "\nCompare against the model's synthetic-token rate for this config "
        "(450M vanilla_matched: 52,207 tok/s). A loader below that rate is the "
        "binding constraint on the training step."
    )


if __name__ == "__main__":
    main()
