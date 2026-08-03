"""Analyze sweep_still_taper: does the TLM "wider-early" result apply to KV compaction?

Reads runs/sweep_still_taper/*.metrics.json, groups by arm, reports val_perplexity
with seed ranges. Decisive comparisons:
  B taper_early vs A uniform  (does tapering help at all?)
  B taper_early vs C taper_late (does DIRECTION matter? same params, only order differs)
  D taper_steep vs B taper_early (steepness sensitivity)
Non-overlapping seed ranges = real effect.
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict


def main() -> None:
    arm_map = {
        "A_uniform_128": "A uniform_128 (anchor)",
        "B_taper_early": "B taper_early (TLM dir)",
        "C_taper_late": "C taper_late (wrong dir)",
        "D_taper_steep": "D taper_steep",
    }
    files = glob.glob("runs/sweep_still_taper/*.metrics.json")
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

    rows = {}
    order = ["A_uniform_128", "B_taper_early", "C_taper_late", "D_taper_steep"]
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
        better = label_a if a["val"][1] < b["val"][1] else label_b
        print(
            f"\n{label_a} vs {label_b}\n"
            f"  {label_a}: mean val_ppl={a['val'][1]:.4f} range [{ar[0]:.4f}, {ar[1]:.4f}]\n"
            f"  {label_b}: mean val_ppl={b['val'][1]:.4f} range [{br[0]:.4f}, {br[1]:.4f}]\n"
            f"  lower-mean winner: {better}\n"
            f"  -> {verdict}"
        )

    compare("B_taper_early", "B taper_early", "A_uniform_128", "A uniform")     # does tapering help?
    compare("B_taper_early", "B taper_early", "C_taper_late", "C taper_late")    # direction (headline)
    compare("D_taper_steep", "D taper_steep", "B_taper_early", "B taper_early")  # steepness


if __name__ == "__main__":
    main()
