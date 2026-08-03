"""Analyze sweep_still_flow_matched: is the flow win real or a param-count confound?

Reads runs/sweep_still_flow_matched/*.metrics.json, groups by arm, and reports
train + val perplexity with seed ranges so the decisive comparisons are visible:

  B flow_7.1M  vs  C std_7.1M   (high-budget: mechanism vs capacity)
  D flow_2.4M  vs  E std_2.4M   (low-budget:  mechanism vs capacity)
  A std_2.0M                       anchor (low-budget, original baseline)

Non-overlapping seed ranges on val_perplexity = a real mechanism effect.
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict


def main() -> None:
    arm_map = {
        "A_std_2M": "A std_2.0M (anchor)",
        "B_flow_7M": "B flow_7.1M",
        "C_std_7M": "C std_7.1M (match to B)",
        "D_flow_2M": "D flow_2.4M",
        "E_std_2M": "E std_2.4M (match to D)",
    }
    files = glob.glob("runs/sweep_still_flow_matched/*.metrics.json")
    # Exclude any per-variant subdir metrics by keeping only top-level names.
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        name = f.split("/")[-1].replace(".metrics.json", "")
        # name like "A_std_2M_seed0" or "A_std_2M"
        base = re.sub(r"_seed\d+$", "", name)
        if base not in arm_map:
            continue
        d = json.load(open(f))
        if d.get("steps", 0) < 6500:  # skip partial/smoke
            continue
        by_arm[base].append(d)

    print(f"Found metrics for {sum(len(v) for v in by_arm.values())} completed runs.\n")

    rows = {}
    order = ["A_std_2M", "B_flow_7M", "C_std_7M", "D_flow_2M", "E_std_2M"]
    for arm in order:
        runs = sorted(by_arm.get(arm, []), key=lambda d: d.get("seed", 0))
        if not runs:
            print(f"{arm_map[arm]:28s}  (no completed runs yet)")
            continue
        val = [r.get("val_perplexity") for r in runs if r.get("val_perplexity") is not None]
        trn = [r.get("train_perplexity", r.get("perplexity")) for r in runs]
        params = runs[0].get("parameter_count")
        seeds = [r.get("seed") for r in runs]

        def stats(xs):
            if not xs:
                return None
            return (min(xs), sum(xs) / len(xs), max(xs), len(xs))

        vs = stats(val)
        ts = stats(trn)
        rows[arm] = dict(val=vs, trn=ts, params=params, seeds=seeds)
        label = arm_map[arm]
        if vs:
            print(
                f"{label:28s} params={params:>9} seeds={seeds}\n"
                f"  val_ppl  min={vs[0]:.4f} mean={vs[1]:.4f} max={vs[2]:.4f}  (n={vs[3]})\n"
                f"  trn_ppl  min={ts[0]:.4f} mean={ts[1]:.4f} max={ts[2]:.4f}  (n={ts[3]})\n"
            )

    # Decisive comparisons.
    print("=" * 70)
    print("DECISIVE COMPARISONS (non-overlapping val_ppl ranges = real effect)")
    print("=" * 70)

    def compare(name_a, label_a, name_b, label_b):
        a, b = rows.get(name_a), rows.get(name_b)
        if not a or not b or not a["val"] or not b["val"]:
            print(f"\n{label_a} vs {label_b}: incomplete (need both arms done)")
            return
        ar = (a["val"][0], a["val"][2])
        br = (b["val"][0], b["val"][2])
        overlap = ar[0] <= br[1] and br[0] <= ar[1]
        verdict = "OVERLAP (no clear effect)" if overlap else "NON-OVERLAP (real effect)"
        better = (
            label_a if a["val"][1] < b["val"][1] else label_b
        )
        print(
            f"\n{label_a} vs {label_b}\n"
            f"  {label_a}: mean val_ppl={a['val'][1]:.4f} range [{ar[0]:.4f}, {ar[1]:.4f}]\n"
            f"  {label_b}: mean val_ppl={b['val'][1]:.4f} range [{br[0]:.4f}, {br[1]:.4f}]\n"
            f"  lower-mean winner: {better}\n"
            f"  -> {verdict}"
        )

    compare("B_flow_7M", "B flow_7.1M", "C_std_7M", "C std_7.1M")
    compare("D_flow_2M", "D flow_2.4M", "E_std_2M", "E std_2.4M")


if __name__ == "__main__":
    main()
