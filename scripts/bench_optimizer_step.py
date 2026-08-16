#!/usr/bin/env python3
"""Isolated optimizer-step cost for the Muon family (Muon / per-head / CM / Aurora).

WHY THIS EXISTS SEPARATELY FROM bench_arms.py
  bench_arms.py measures the whole training step, and at the production
  `gradient_accumulation_steps: 16` the optimizer is only ~2.8% of it
  (docs/profiling/speedup_results.md: 134 ms optimizer vs 1206 ms step at
  accum 4, and fwd/bwd is 4x larger at accum 16). Run-to-run drift on this box
  is ~2%. So bench_arms cannot resolve even a DOUBLING of optimizer cost — it
  would show up as ~2.8% against a ~2% noise floor.

  This script times `optimizer.step()` and nothing else, so a 2x regression
  reads as 2x. It then converts that back into percent-of-step using a
  `--step-ms` reference, which is the number that actually decides anything.

WHAT IT DOES NOT MEASURE
  Quality. Every arm here is a sample-efficiency play; this only prices them.
  A cheap arm that trains worse is not a win. Pair with
  configs/comp_muon_screen_450m.yaml.

WHY IT BUILDS THE REAL MODEL
  Optimizer cost is driven by the parameter SHAPE HISTOGRAM, not by parameter
  count: `Muon` groups same-shape matrices into one batched Newton-Schulz, so a
  model with 4 shape groups and no singletons behaves nothing like a synthetic
  list of random matrices. It also runs `maybe_convert_fp8` first, because FP8
  conversion rebinds weight Parameters and changes which tensors the optimizer
  sees. `torch.compile` is deliberately skipped: it costs minutes at 450M and
  does not lower `optimizer.step()`.

USAGE
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    uv run python scripts/bench_optimizer_step.py \
    --config configs/comp_muon_screen_450m.yaml \
    --arms cm_baseline,cm_per_head_ref,cm_isotropic,cm_full,aurora
  ... --step-ms 1206     # reference full-step time to express cost as % of step
  ... --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch

# Reuse profile_step's config merging so an arm here is the same arm a run would
# execute — including the FP8 conversion, which changes the parameter set.
from profile_step import load_variant_config

from flower.models import build_model
from flower.optim import build_optimizer
from flower.precision import maybe_convert_fp8
from flower.train import configure_precision, configure_vram_limit, resolve_device


def bench_arm(sweep_path: str, arm: str, *, warmup: int, steps: int, repeats: int) -> dict:
    torch.manual_seed(0)
    cfg = load_variant_config(sweep_path, arm, accum_override=None)
    device = resolve_device(cfg.training.device)
    configure_vram_limit(device, fraction=getattr(cfg.training, "vram_fraction", 0.85))
    configure_precision(cfg.training.precision, device)

    model = build_model(cfg.model).to(device)
    model, _ = maybe_convert_fp8(model, cfg.training, device)
    optim_or_list = build_optimizer(model, cfg.training)
    optims = optim_or_list if isinstance(optim_or_list, list) else [optim_or_list]

    # Static synthetic gradients. Optimizer cost depends on shape and dtype, not
    # on gradient CONTENT, and regenerating them per step would time the RNG
    # instead. They are assigned once and left in place; every arm sees bitwise
    # the same gradients because the seed is fixed before allocation.
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)

    def one_step():
        for o in optims:
            o.step()

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    ms_runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(steps):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms_runs.append((time.perf_counter() - t0) / steps * 1000)

    # Optimizer STATE, not model memory: Muon keeps one momentum buffer per 2D
    # param, Aurora the same, and CM's whitening allocates transiently. Reported
    # because a method that is fast but doubles state can still cost a run.
    state_gb = sum(
        v.numel() * v.element_size()
        for o in optims for s in o.state.values() for v in s.values()
        if torch.is_tensor(v)
    ) / 1e9

    return {
        "arm": arm,
        "ms": statistics.median(ms_runs),
        "ms_min": min(ms_runs),
        "ms_max": max(ms_runs),
        "state_gb": state_gb,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
        "optimizers": [type(o).__name__ for o in optims],
    }


def _run_arm_subprocess(args, arm: str) -> dict | None:
    """One arm per process, for the same reason bench_arms.py does it.

    Allocator state and autotune caches from an earlier arm survive
    `empty_cache()` and contaminate the next arm's peak-memory reading.
    """
    cmd = [
        sys.executable, __file__,
        "--config", args.config, "--arms", arm,
        "--warmup", str(args.warmup), "--steps", str(args.steps),
        "--repeats", str(args.repeats), "--_child",
    ]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
    print(f"  FAILED: {' | '.join(t.strip() for t in tail)}", flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/comp_muon_screen_450m.yaml")
    ap.add_argument("--arms", required=True, help="comma-separated; the FIRST is the control")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--step-ms", type=float, default=None,
        help="reference full-step wall-clock (ms) to express optimizer cost as %% of step",
    )
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args._child:
        r = bench_arm(args.config, arms[0], warmup=args.warmup, steps=args.steps, repeats=args.repeats)
        print("__RESULT__" + json.dumps(r), flush=True)
        return

    results = []
    for arm in arms:
        print(f"\n=== {arm} ===", flush=True)
        r = _run_arm_subprocess(args, arm)
        if r is None:
            continue
        results.append(r)
        print(f"  {r['ms']:.2f} ms/step   state {r['state_gb']:.2f} GB   {'+'.join(r['optimizers'])}", flush=True)

    if not results:
        print("\nno arm completed")
        return

    control = results[0]
    pct_hdr = f"{'% of step':>10}" if args.step_ms else ""
    print(f"\n{'arm':22} {'ms/step':>9} {'spread':>8} {'vs control':>11} {'state GB':>9}{pct_hdr}")
    for r in results:
        spread = (r["ms_max"] - r["ms_min"]) / r["ms"] * 100
        rel = r["ms"] / control["ms"]
        pct = f"{r['ms'] / args.step_ms * 100:9.2f}%" if args.step_ms else ""
        print(f"{r['arm']:22} {r['ms']:9.2f} {spread:7.1f}% {rel:10.3f}x {r['state_gb']:9.2f}{pct}")

    print(f"\ncontrol = {control['arm']}")
    if args.step_ms:
        print(
            f"Percentages are against --step-ms {args.step_ms:.0f}. The decision number is the\n"
            "DELTA in '% of step' between an arm and the control, not the ratio in 'vs control':\n"
            "a 2x optimizer that is 3% of the step costs 3% of throughput, not 100%."
        )
    else:
        print("Pass --step-ms <full step ms> to convert these into percent-of-step.")


if __name__ == "__main__":
    main()
