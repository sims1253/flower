#!/usr/bin/env python3
"""A/B training-step throughput across sweep variants ("arms").

WHY THIS EXISTS
  Every speedup candidate needs the same measurement or the comparison is
  meaningless. This runs each named variant of a sweep YAML through the exact
  train.py wiring (build_model -> maybe_convert_fp8 -> build_optimizer ->
  prebuild_attention_masks -> torch.compile) and reports steady-state tok/s and
  peak VRAM, so arms differ only by what the config says they differ by.

MEASUREMENT DISCIPLINE
  - Compare arms against the control arm in the SAME invocation. Do NOT compare
    against the tok/s in a run's metrics.json: that is a whole-run average
    including compile warmup, validation passes and checkpoint writes, and a
    multi-hour run also sits at a lower sustained clock than a 2-minute bench.
  - Synthetic tokens. Kernel time depends on shape/dtype, not token content,
    and the input pipeline is not the constraint here: the FineWeb loader
    sustains ~1.08M tok/s at the default 2 workers against a ~52k tok/s
    consumption rate (scripts/bench_data_workers.py).
  - `--repeats` runs the whole timed block N times and reports the spread. The
    finished 450M seeds differ by 6.9% in tok/s, so a single timing is not
    evidence for a single-digit claim. Trust a gap only if it clears the spread.

USAGE
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    uv run python scripts/bench_arms.py \
    --config configs/speedup_screen_450m.yaml \
    --arms bf16_baseline,fp8_tensorwise
  ... --accum 4          # shorter steps; relative gaps hold, wall-clock shrinks
  ... --repeats 3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time

import torch

# profile_step.py already mirrors train.py's wiring exactly; reuse it rather
# than maintaining a second copy that can drift.
from profile_step import build_everything, load_variant_config, make_synthetic_batch

from flower.models.base import count_parameters


def _autocast(amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=amp_dtype)


def bench_arm(sweep_path: str, arm: str, *, warmup: int, steps: int, accum_override: int | None, repeats: int) -> dict:
    torch.manual_seed(0)
    cfg = load_variant_config(sweep_path, arm, accum_override=accum_override)
    accum = max(1, int(cfg.training.gradient_accumulation_steps))
    batch = int(cfg.training.batch_size)
    seq = int(cfg.data.sequence_length)
    vocab = int(cfg.model.vocab_size)

    device, amp_dtype, eager_model, run_model, optims = build_everything(cfg)
    run_model.train()
    params = count_parameters(eager_model)

    def one_full_step():
        for o in optims:
            o.zero_grad(set_to_none=True)
        for _ in range(accum):
            ids, labels = make_synthetic_batch(batch, seq, vocab, device)
            with _autocast(amp_dtype):
                loss = run_model(ids, labels=labels)["loss"]
            loss.backward()
        torch.nn.utils.clip_grad_norm_(eager_model.parameters(), cfg.training.grad_clip)
        for o in optims:
            o.step()

    # Warmup absorbs torch.compile (minutes at 450M) and any autotuning.
    for _ in range(warmup):
        one_full_step()
    torch.cuda.synchronize()

    tok_s_runs = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(steps):
            one_full_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tok_s_runs.append(steps * accum * batch * seq / elapsed)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    # No teardown needed: each arm owns its process (see _run_arm_subprocess),
    # so the OS reclaims everything on exit. The previous in-process `del` +
    # empty_cache() could not free compile workspaces anyway, which is why the
    # subprocess split exists.

    return {
        "arm": arm,
        "params": params,
        "tok_s": statistics.median(tok_s_runs),
        "tok_s_min": min(tok_s_runs),
        "tok_s_max": max(tok_s_runs),
        "peak_gb": peak_gb,
        "accum": accum,
        "batch": batch,
        "seq": seq,
    }


def _run_arm_subprocess(args, arm: str) -> dict | None:
    """Measure one arm in a fresh process.

    Arms MUST NOT share a process. torch.compile workspaces, inductor autotune
    caches and cudagraph pools from an earlier arm survive `empty_cache()` and
    inflate the next arm's peak-memory reading — and can push a later arm into a
    spurious OOM. A subprocess per arm makes every peak_gb an independent
    measurement rather than a running total.
    """
    cmd = [
        sys.executable, __file__,
        "--config", args.config,
        "--arms", arm,
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--repeats", str(args.repeats),
        "--_child",
    ]
    if args.accum is not None:
        cmd += ["--accum", str(args.accum)]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    # Surface why the arm died — an OOM here is itself a finding about the arm.
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
    print(f"  FAILED: {' | '.join(t.strip() for t in tail)}", flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/speedup_screen_450m.yaml")
    ap.add_argument("--arms", required=True, help="comma-separated variant names; the FIRST is the control")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--accum", type=int, default=None, help="override gradient_accumulation_steps")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args._child:
        # One arm, this process, emit a machine-readable line for the parent.
        r = bench_arm(
            args.config, arms[0],
            warmup=args.warmup, steps=args.steps,
            accum_override=args.accum, repeats=args.repeats,
        )
        print("__RESULT__" + json.dumps(r), flush=True)
        return

    results = []
    for arm in arms:
        print(f"\n=== {arm} ===", flush=True)
        r = _run_arm_subprocess(args, arm)
        if r is None:
            continue
        results.append(r)
        print(
            f"  {r['tok_s']:,.0f} tok/s (min {r['tok_s_min']:,.0f} / max {r['tok_s_max']:,.0f})"
            f"   peak {r['peak_gb']:.2f} GB   params {r['params']:,}",
            flush=True,
        )
    if not results:
        print("\nno arm completed")
        return

    control = results[0]
    print(f"\n{'arm':22} {'tok/s':>10} {'spread':>8} {'vs control':>11} {'peak GB':>9}")
    for r in results:
        spread = (r["tok_s_max"] - r["tok_s_min"]) / r["tok_s"] * 100
        rel = r["tok_s"] / control["tok_s"]
        print(f"{r['arm']:22} {r['tok_s']:10,.0f} {spread:7.1f}% {rel:10.3f}x {r['peak_gb']:9.2f}")
    print(f"\ncontrol = {control['arm']}; batch {control['batch']} x accum {control['accum']} x seq {control['seq']}")
    print("A gap smaller than the spread column is not evidence. Re-run with more --repeats.")


if __name__ == "__main__":
    main()
