"""Analyze sweep_still_flow_euler: does more Euler steps reduce flow variance?

The decisive metric is the SEED STANDARD DEVIATION of val_perplexity (not just
the mean). Sweep 10 found flow sd ~0.030 vs standard ~0.017. If more Euler steps
drops sd toward the standard floor, the variance is integration-error-driven.
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict


def main() -> None:
    arm_map = {
        "B0_steps5": "B0 steps5 (reference)",
        "B1_steps10": "B1 steps10",
        "B2_steps20": "B2 steps20",
    }
    files = glob.glob("runs/sweep_still_flow_euler/*.metrics.json")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        name = f.split("/")[-1].replace(".metrics.json", "")
        base = re.sub(r"_seed\d+$", "", name)
        if base not in arm_map:
            continue
        d = json.load(open(f))
        if d.get("steps", 0) < 6500:
            continue
        by_arm[base].append(d)

    print(f"Found metrics for {sum(len(v) for v in by_arm.values())} completed runs.\n")

    # Standard compactor variance floor (from sweep 10, for reference).
    print("Reference: sweep10 standard compactor val_ppl sd ~0.017 (the floor to beat).")
    print("Reference: sweep10 flow B arm (steps=5) val_ppl sd ~0.028.\n")

    rows = {}
    order = ["B0_steps5", "B1_steps10", "B2_steps20"]
    for arm in order:
        runs = sorted(by_arm.get(arm, []), key=lambda d: d.get("seed", 0))
        if not runs:
            print(f"{arm_map[arm]:24s}  (no completed runs yet)")
            continue
        val = [r.get("val_perplexity") for r in runs if r.get("val_perplexity") is not None]
        if not val:
            continue
        m = sum(val) / len(val)
        sd = (sum((x - m) ** 2 for x in val) / len(val)) ** 0.5 if len(val) > 1 else 0.0
        params = runs[0].get("parameter_count")
        rows[arm] = dict(val=val, mean=m, sd=sd, min=min(val), max=max(val), n=len(val))
        print(
            f"{arm_map[arm]:24s} n={len(val)} val_ppl={[round(x,4) for x in val]}\n"
            f"{'':24s}      mean={m:.4f} sd={sd:.4f} range=[{min(val):.4f},{max(val):.4f}] params={params}\n"
        )

    print("=" * 70)
    print("VARIANCE VERDICT (lower sd = finer Euler integration helped)")
    print("=" * 70)
    ref = rows.get("B0_steps5")
    if ref:
        for arm in ["B1_steps10", "B2_steps20"]:
            r = rows.get(arm)
            if not r:
                print(f"\n{arm_map[arm]}: incomplete")
                continue
            sd_change = r["sd"] - ref["sd"]
            mean_change = r["mean"] - ref["mean"]
            print(
                f"\n{arm_map[arm]} vs B0 steps5:\n"
                f"  sd: {ref['sd']:.4f} -> {r['sd']:.4f} ({sd_change:+.4f}, {'LOWER' if sd_change < 0 else 'HIGHER'})\n"
                f"  mean: {ref['mean']:.4f} -> {r['mean']:.4f} ({mean_change:+.4f}, {'better' if mean_change < 0 else 'worse'})"
            )
            if r["sd"] < 0.017:
                print("  -> sd BELOW standard floor (0.017)! Variance tamed.")
            elif sd_change < -0.005:
                print("  -> variance meaningfully reduced.")
            else:
                print("  -> variance not reduced (integration error not the cause).")


if __name__ == "__main__":
    main()
