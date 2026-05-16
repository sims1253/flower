#!/usr/bin/env python3
"""Write a dependency-free BPB-vs-wall-clock HTML report for a sweep directory."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _bpb_from_metric(data: dict[str, Any], metrics_path: Path) -> float | None:
    if "bpb" in data:
        return float(data["bpb"])
    comp_path = data.get("composite_ranker_json")
    if isinstance(comp_path, str):
        comp = _load(Path(comp_path))
        if comp:
            value = comp.get("metrics", {}).get("fineweb", {}).get("bpb")
            if value is not None:
                return float(value)
    sibling = metrics_path.parent / "variants" / metrics_path.stem.replace(".metrics", "") / "composite_ranker.json"
    comp = _load(sibling)
    if comp:
        value = comp.get("metrics", {}).get("fineweb", {}).get("bpb")
        if value is not None:
            return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/sweep_bpb_wallclock.html"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.directory.glob("*.metrics.json")):
        data = _load(path)
        if not data:
            continue
        bpb = _bpb_from_metric(data, path)
        if bpb is None:
            continue
        steps = int(data.get("steps", data.get("checkpoint_step", 0)))
        tps = float(data.get("tokens_per_sec", 0.0))
        tokens = steps * args.batch_size * args.seq_len
        wall_seconds = tokens / tps if tps > 0 and steps > 0 else 0.0
        rows.append(
            {
                "name": path.stem.replace(".metrics", ""),
                "variant": data.get("variant", "?"),
                "bpb": bpb,
                "wall_hours": wall_seconds / 3600.0,
                "steps": steps,
                "tokens_per_sec": tps,
            }
        )
    rows.sort(key=lambda r: (r["wall_hours"], r["bpb"]))

    if rows:
        max_x = max(r["wall_hours"] for r in rows) or 1.0
        min_y = min(r["bpb"] for r in rows)
        max_y = max(r["bpb"] for r in rows)
        span_y = max(max_y - min_y, 1e-6)
    else:
        max_x, min_y, span_y = 1.0, 0.0, 1.0

    points = []
    for r in rows:
        x = 50 + 700 * (r["wall_hours"] / max_x)
        y = 40 + 360 * ((r["bpb"] - min_y) / span_y)
        label = html.escape(r["name"])
        points.append(
            f'<circle cx="{x:.1f}" cy="{400 - y:.1f}" r="5"><title>{label}: {r["bpb"]:.5f} BPB, {r["wall_hours"]:.2f}h</title></circle>'
        )
        points.append(f'<text x="{x + 7:.1f}" y="{400 - y + 4:.1f}">{label}</text>')

    table = "\n".join(
        "<tr>"
        f"<td>{html.escape(r['name'])}</td><td>{html.escape(str(r['variant']))}</td>"
        f"<td>{r['bpb']:.5f}</td><td>{r['wall_hours']:.3f}</td><td>{r['steps']}</td><td>{r['tokens_per_sec']:.0f}</td>"
        "</tr>"
        for r in rows
    )
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Flower BPB vs wall-clock</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
svg {{ border: 1px solid #ccd5df; max-width: 100%; }}
circle {{ fill: #246bfe; opacity: 0.82; }}
text {{ font-size: 11px; }}
table {{ border-collapse: collapse; margin-top: 18px; }}
td, th {{ border-bottom: 1px solid #d9e2ec; padding: 6px 10px; text-align: right; }}
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {{ text-align: left; }}
</style>
<h1>BPB vs Estimated Wall-Clock</h1>
<svg width="820" height="430" viewBox="0 0 820 430" role="img">
<line x1="50" y1="400" x2="780" y2="400" stroke="#52606d"/>
<line x1="50" y1="20" x2="50" y2="400" stroke="#52606d"/>
<text x="360" y="424">estimated wall-clock hours</text>
<text x="4" y="24">BPB</text>
{"".join(points)}
</svg>
<table>
<tr><th>name</th><th>variant</th><th>BPB</th><th>wall h</th><th>steps</th><th>tok/s</th></tr>
{table}
</table>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(doc)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
