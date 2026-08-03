#!/usr/bin/env python3
"""Benchmark precision/compile settings on the real model config.

Measures steady-state forward+backward+step throughput and peak VRAM for each
precision mode, using synthetic token batches so the number reflects compute
rather than data-loader behaviour.

  uv run python scripts/bench_precision.py --config configs/sweep13_100m_phase0.yaml

Compile timings exclude warmup (the first call pays 30-90s of Dynamo/Inductor
work). Reported tokens/sec is over `--steps` measured steps after `--warmup`.

WSL2 note: exceeding VRAM does NOT raise OutOfMemoryError there. The WDDM memory
manager silently spills into shared host RAM over PCIe and throughput collapses
by an order of magnitude, so a too-large batch looks like a very slow run rather
than a failure. `--mem-fraction` caps the caching allocator so it raises instead,
which is the only way to get an honest OOM boundary on this platform.
"""

from __future__ import annotations

import argparse
import time

import torch

from flower.config import load_config
from flower.models import build_model
from flower.optim import build_optimizer
from flower.train import autocast_ctx, configure_precision, initialize_lr_schedule

MODES = [
    ("fp32", False),
    ("tf32", False),
    ("bf16", False),
    ("bf16", True),
]


def bench(cfg, device: torch.device, precision: str, compile_model: bool, warmup: int, steps: int) -> dict:
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    amp_dtype = configure_precision(precision, device)
    model = build_model(cfg.model).to(device)
    optims = build_optimizer(model, cfg.training)
    optims = optims if isinstance(optims, list) else [optims]
    initialize_lr_schedule(optims)

    run = model
    if compile_model:
        model.collect_module_diagnostics = False
        run = torch.compile(model, dynamic=False)

    batch = cfg.training.batch_size
    accum = cfg.training.gradient_accumulation_steps
    seq = cfg.data.sequence_length
    vocab = cfg.model.vocab_size
    tokens_per_step = batch * accum * seq

    def one_step() -> float:
        for opt in optims:
            opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for _ in range(accum):
            ids = torch.randint(0, vocab, (batch, seq), device=device)
            with autocast_ctx(device, amp_dtype):
                out = run(ids, labels=ids)
                loss = out["loss"]
            (loss / accum).backward()
            total += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        for opt in optims:
            opt.step()
        return float(total) / accum

    losses = []
    for _ in range(warmup):
        losses.append(one_step())
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        losses.append(one_step())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated(device) / 1e9
    # Locals (model/optims/compiled wrapper) are released when this frame exits;
    # the caller calls empty_cache() before the next mode is built.
    return {
        "tokens_per_sec": steps * tokens_per_step / elapsed,
        "sec_per_step": elapsed / steps,
        "peak_gb": peak,
        "final_loss": losses[-1],
        "finite": all(t == t for t in losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.9,
        help="Cap the caching allocator at this fraction of VRAM so oversized "
        "batches raise OOM instead of silently spilling to host RAM (WSL2).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device)

    print(f"{torch.cuda.get_device_name(0)} | {cfg.model.variant} | ", end="")
    print(f"batch {cfg.training.batch_size} x accum {cfg.training.gradient_accumulation_steps} x seq {cfg.data.sequence_length}")
    print(f"{'mode':<16} {'tok/s':>10} {'s/step':>9} {'peak GB':>9} {'speedup':>9}  loss")
    print("-" * 68)

    baseline = None
    for precision, compile_model in MODES:
        label = f"{precision}{'+compile' if compile_model else ''}"
        try:
            r = bench(cfg, device, precision, compile_model, args.warmup, args.steps)
        except torch.OutOfMemoryError:
            print(f"{label:<16} {'OOM':>10}")
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        baseline = baseline or r["tokens_per_sec"]
        flag = "" if r["finite"] else "  NON-FINITE LOSS"
        print(
            f"{label:<16} {r['tokens_per_sec']:>10,.0f} {r['sec_per_step']:>9.3f} "
            f"{r['peak_gb']:>9.2f} {r['tokens_per_sec'] / baseline:>8.2f}x  {r['final_loss']:.4f}{flag}"
        )


if __name__ == "__main__":
    main()
