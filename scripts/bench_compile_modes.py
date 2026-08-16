#!/usr/bin/env python3
"""Task 1 benchmark: torch.compile `default` vs `reduce-overhead` (CUDA graphs).

Measures tokens/sec for both modes on the same config + seed, and reports any
`skipping cudagraphs` warnings or recompile-limit hits. Kernel time is
shape/dtype-dependent, not content-dependent, so synthetic tokens stand in for
fineweb_edu — this keeps the benchmark fast and data-independent while matching
the real training-step shape exactly.

USAGE
  uv run python scripts/bench_compile_modes.py --config configs/sweep13_100m_longctx_phase0.yaml
  uv run python scripts/bench_compile_modes.py --config ... --steps 30 --warmup 15
  uv run python scripts/bench_compile_modes.py --config ... --mode-only reduce-overhead
"""
from __future__ import annotations

import argparse
import io
import re
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout

import torch

from flower.config import DataConfig, load_config
from flower.data import token_batches
from flower.models import build_model
from flower.models.base import prebuild_attention_masks
from flower.optim import build_optimizer
from flower.train import autocast_ctx, configure_precision, configure_vram_limit, resolve_device


def _capture_cudagraph_warnings() -> tuple[list[str], io.StringIO, io.StringIO]:
    """Redirect torch's warning stream so we can grep for cudagraph skips."""
    err = io.StringIO()
    return [], err, err


def _run_mode(cfg, mode: str, warmup: int, steps: int, seed: int) -> dict:
    """Run `warmup` + `steps` synthetic training steps in one compile mode."""
    torch.manual_seed(seed)
    device = resolve_device(cfg.training.device)
    configure_vram_limit(device)
    amp_dtype = configure_precision(cfg.training.precision, device)
    model = build_model(cfg.model).to(device)
    opt = build_optimizer(model, cfg.training)
    optims = opt if isinstance(opt, list) else [opt]
    eager = model
    if getattr(eager, "collect_module_diagnostics", False):
        eager.collect_module_diagnostics = False
    if device.type == "cuda":
        prebuild_attention_masks(eager, cfg.data.sequence_length, device)
    model = torch.compile(model, mode=mode, dynamic=False)

    seq_len = cfg.data.sequence_length
    batch = cfg.training.batch_size
    # Synthetic tokens stand in for fineweb_edu — kernel time is shape/dtype-
    # dependent, not content-dependent, so this matches the real step exactly
    # without needing the dataset download.
    synth_cfg = DataConfig(
        dataset="synthetic",
        sequence_length=seq_len,
        synthetic_vocab_size=cfg.model.vocab_size,
    )
    stream = token_batches(synth_cfg, batch, device, seed=seed)

    # Warmup (includes compile + cudagraph capture for reduce-overhead).
    model.train()
    for o in optims:
        o.zero_grad(set_to_none=True)
    for _ in range(max(1, warmup)):
        ids = next(stream)
        labels = ids.clone()
        with autocast_ctx(device, amp_dtype):
            out = model(ids, labels=labels)
            loss = out["loss"]
        loss.backward()
        for o in optims:
            o.step()
        for o in optims:
            o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Timed steps.
    log_buf = io.StringIO()
    losses = []
    tokens = batch * seq_len * steps
    with redirect_stderr(log_buf), warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        start = time.perf_counter()
        for _ in range(steps):
            ids = next(stream)
            labels = ids.clone()
            with autocast_ctx(device, amp_dtype):
                out = model(ids, labels=labels)
                loss = out["loss"]
            loss.backward()
            for o in optims:
                o.step()
            for o in optims:
                o.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    log_text = log_buf.getvalue()
    skip_cg = bool(re.search(r"skipping cudagraphs|skipping cudagraph", log_text, re.I))
    recompile = bool(re.search(r"recompile_limit|Recompiling|recompilation", log_text, re.I))
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    return {
        "mode": mode,
        "tokens_per_sec": tokens / elapsed,
        "elapsed": elapsed,
        "mean_loss": sum(losses) / len(losses),
        "final_loss": losses[-1],
        "skip_cudagraphs_warning": skip_cg,
        "recompile_warning": recompile,
        "peak_mem_gb": peak_mem,
        "log_excerpt": log_text[:1500],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--mode-only", default=None, help="run only this mode")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    modes = ["default", "reduce-overhead"] if args.mode_only is None else [args.mode_only]
    print(f"config={args.config}  seq={cfg.data.sequence_length}  batch={cfg.training.batch_size}  "
          f"variant={cfg.model.variant}  d={cfg.model.d_model} L={cfg.model.num_layers}")
    results = []
    for mode in modes:
        torch._dynamo.reset()
        torch.cuda.reset_peak_memory_stats()
        r = _run_mode(cfg, mode, args.warmup, args.steps, args.seed)
        results.append(r)
        tag = " ⚠CUDAGRAPH-SKIPPED" if r["skip_cudagraphs_warning"] else ""
        tag += " ⚠RECOMPILE" if r["recompile_warning"] else ""
        print(f"[{mode}] {r['tokens_per_sec']:.0f} tok/s  "
              f"loss mean={r['mean_loss']:.4f} final={r['final_loss']:.4f}  "
              f"peak={r['peak_mem_gb']:.2f} GB  elapsed={r['elapsed']:.1f}s{tag}")

    if len(results) == 2:
        speedup = results[1]["tokens_per_sec"] / results[0]["tokens_per_sec"]
        print(f"\nspeedup reduce-overhead/default = {speedup:.3f}x")
        d_loss = abs(results[1]["final_loss"] - results[0]["final_loss"])
        print(f"final-loss delta = {d_loss:.5f} (should be ~0; same seed)")


if __name__ == "__main__":
    main()
