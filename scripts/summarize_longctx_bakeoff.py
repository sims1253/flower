#!/usr/bin/env python3
"""Summarize the long-context memory bake-off from .metrics.json files.

Reads every <output_dir>/*.metrics.json, groups by base variant, and prints a
ranked table of mean val_bpb / val_loss across seeds. This is the quick read on
whether any memory arm beats the vanilla control at long context.

  PYTHONPATH=. uv run python scripts/summarize_longctx_bakeoff.py runs/sweep13_longctx_bakeoff
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    files = sorted(args.output_dir.glob("*.metrics.json"))
    if not files:
        print(f"No .metrics.json files in {args.output_dir}")
        return

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        m = json.loads(f.read_text())
        # name like "vanilla_local_seed0" -> base "vanilla_local"
        name = f.stem.replace(".metrics", "")
        base = name.rsplit("_seed", 1)[0]
        by_arm[base].append(m)

    rows = []
    for arm, runs in by_arm.items():
        val_losses = [r.get("val_loss") for r in runs if r.get("val_loss") is not None]
        val_bpbs = [r.get("val_bpb") for r in runs if r.get("val_bpb") is not None]
        rows.append({
            "arm": arm,
            "seeds": len(runs),
            "val_loss_mean": statistics.mean(val_losses) if val_losses else float("nan"),
            "val_loss_std": statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0,
            "val_bpb_mean": statistics.mean(val_bpbs) if val_bpbs else float("nan"),
        })

    rows.sort(key=lambda r: r["val_bpb_mean"])
    print(f"{'arm':<18} {'seeds':>5} {'val_loss':>10} {'±':>6} {'val_bpb':>9}")
    print("-" * 55)
    baseline = next((r for r in rows if r["arm"] == "vanilla_local"), None)
    for r in rows:
        delta = ""
        if baseline and r["arm"] != "vanilla_local":
            d = r["val_bpb_mean"] - baseline["val_bpb_mean"]
            delta = f"  ({'+'if d>=0 else ''}{d:.4f} bpb vs vanilla)"
        print(
            f"{r['arm']:<18} {r['seeds']:>5} {r['val_loss_mean']:>10.4f} "
            f"{r['val_loss_std']:>6.4f} {r['val_bpb_mean']:>9.4f}{delta}"
        )


if __name__ == "__main__":
    main()
