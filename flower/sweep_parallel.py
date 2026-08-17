"""Parallel sweep launcher — 1 trial per GPU, round-robin across `--gpus`.

Each variant runs in its own subprocess with `CUDA_VISIBLE_DEVICES` set to the GPU
it owns. The launcher maintains a fixed pool of size `len(gpus)` and spawns the
next pending variant whenever a slot frees up.

Usage:
  uv run python -m flower.sweep_parallel --config configs/sweep2_phase0_tokenizer.yaml \\
      --output-dir runs/vast/phase0 --gpus 0,1 --steps 10000

For a single-GPU instance just pass `--gpus 0`; the launcher then behaves like
the serial sweep but still gives you per-variant log files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from flower.sweep import expand_seed_variants, load_sweep, parse_seeds, select_variants, write_variant_config


def _spawn(
    variant: dict[str, Any],
    gpu_id: int,
    output_dir: Path,
    tmpdir: Path,
    steps: int | None,
    extra_args: list[str],
    device: str = "cuda",
) -> tuple[subprocess.Popen, str, Path]:
    name = variant["name"]
    config = deepcopy(variant["config"])
    metrics_path = output_dir / f"{name}.metrics.json"
    config.setdefault("training", {})["metrics_json"] = str(metrics_path)
    config["training"].setdefault("log_backend", "tensorboard")
    config["training"]["output_dir"] = str(output_dir / "variants" / name)
    config["training"]["device"] = device
    if steps is not None:
        config["training"]["steps"] = steps

    config_path = write_variant_config(config, tmpdir, name)
    log_path = output_dir / f"{name}.stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        "-m",
        "flower.train",
        "--config",
        str(config_path),
    ] + extra_args
    if steps is not None:
        cmd += ["--steps", str(steps)]
    # The child keeps its own dup of the fd, so closing the parent's handle
    # here is safe; previously it stayed open for the lifetime of the sweep
    # (one leaked fd per trial).
    with log_path.open("w") as log_fh:
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, name, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Sweep YAML")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids (e.g. '0,1')")
    parser.add_argument("--steps", type=int, default=None, help="Override training.steps")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds; overrides training.seeds")
    parser.add_argument("--output-dir", type=str, default="runs/sweep_parallel")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for trials (cuda|cpu). 'cpu' is for local smoke tests; in that mode --gpus is ignored for routing but still controls parallelism.",
    )
    args = parser.parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus must list at least one GPU id")

    sweep_name, variants = load_sweep(args.config)
    selected = select_variants(variants, args.variants, args.limit)
    selected = expand_seed_variants(selected, parse_seeds(args.seeds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip variants whose metrics_json already exists — survives interruption +
    # resume cycles, so re-launching the same sweep only re-runs the trials that
    # didn't finish writing a final metrics file.
    pre_existing = []
    fresh = []
    for v in selected:
        metrics_path = output_dir / f"{v['name']}.metrics.json"
        if metrics_path.exists():
            pre_existing.append(v["name"])
        else:
            fresh.append(v)
    if pre_existing:
        print(f"[sweep] skipping {len(pre_existing)} variant(s) with existing metrics: {pre_existing}", flush=True)
    selected = fresh

    extra_args: list[str] = ["--smoke"] if args.smoke else []

    summary: dict[str, Any] = {
        "sweep": sweep_name,
        "config": str(Path(args.config)),
        "output_dir": str(output_dir),
        "gpus": gpus,
        "started_at": time.time(),
        "variants": [],
    }

    pending = list(selected)
    running: dict[int, tuple[subprocess.Popen, str, Path]] = {}
    completed: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="flower-psweep-") as tmp:
        tmpdir = Path(tmp)
        while pending or running:
            # Fill free GPU slots from the pending queue.
            for gpu_id in gpus:
                if gpu_id in running:
                    continue
                if not pending:
                    break
                v = pending.pop(0)
                proc, name, metrics_path = _spawn(
                    v, gpu_id, output_dir, tmpdir, args.steps, extra_args, device=args.device
                )
                running[gpu_id] = (proc, name, metrics_path)
                print(f"[{args.device} {gpu_id}] starting {name} (pid {proc.pid})", flush=True)

            # Reap any finished trial.
            time.sleep(2)
            for gpu_id in list(running):
                proc, name, metrics_path = running[gpu_id]
                if proc.poll() is None:
                    continue
                rc = proc.returncode
                metrics: dict[str, Any] | None = None
                if metrics_path.exists():
                    try:
                        metrics = json.loads(metrics_path.read_text())
                    except json.JSONDecodeError:
                        metrics = None
                completed.append(
                    {
                        "name": name,
                        "gpu_id": gpu_id,
                        "returncode": rc,
                        "metrics": metrics,
                        "metrics_json": str(metrics_path),
                    }
                )
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[gpu {gpu_id}] finished {name}: {status}", flush=True)
                del running[gpu_id]

    summary["finished_at"] = time.time()
    summary["variants"] = completed
    summary["variant_count"] = len(completed)
    summary["failed"] = [c["name"] for c in completed if c["returncode"] != 0]
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "variants"}, indent=2))
    print(f"\nSummary written to {summary_path}")
    if summary["failed"]:
        print(f"WARNING: {len(summary['failed'])} variant(s) failed: {summary['failed']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
