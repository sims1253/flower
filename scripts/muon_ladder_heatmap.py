#!/usr/bin/env python3
"""Create a simple HTML stability heatmap for sweep4_phase0_remuon outputs."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

PATTERN = re.compile(r"muon_lr(?P<lr>[^_]+)_ns(?P<ns>\d+)_warm(?P<warm>\d+k)")


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _bpb(data: dict[str, Any], path: Path) -> float | None:
    if "bpb" in data:
        return float(data["bpb"])
    comp_path = data.get("composite_ranker_json")
    if isinstance(comp_path, str):
        comp = _load(Path(comp_path))
        if comp:
            value = comp.get("metrics", {}).get("fineweb", {}).get("bpb")
            if value is not None:
                return float(value)
    sibling = path.parent / "variants" / path.stem.replace(".metrics", "") / "composite_ranker.json"
    comp = _load(sibling)
    if comp:
        value = comp.get("metrics", {}).get("fineweb", {}).get("bpb")
        if value is not None:
            return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/sweep4_phase0_muon_heatmap.html"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.directory.glob("*.metrics.json")):
        match = PATTERN.search(path.stem)
        if not match:
            continue
        data = _load(path)
        value = _bpb(data or {}, path) if data else None
        stable = data is not None and value is not None and data.get("loss") != "nan"
        rows.append({**match.groupdict(), "name": path.stem.replace(".metrics", ""), "stable": stable, "bpb": value})

    lrs = sorted({r["lr"] for r in rows})
    warms = sorted({r["warm"] for r in rows})
    ns_values = sorted({r["ns"] for r in rows}, key=int)
    cells = {(r["lr"], r["ns"], r["warm"]): r for r in rows}

    sections = []
    for ns in ns_values:
        trs = []
        for lr in lrs:
            tds = [f"<th>{html.escape(lr)}</th>"]
            for warm in warms:
                row = cells.get((lr, ns, warm))
                if not row:
                    tds.append("<td class='missing'>missing</td>")
                    continue
                cls = "stable" if row["stable"] else "failed"
                label = f"{row['bpb']:.5f}" if row["bpb"] is not None else "failed"
                tds.append(f"<td class='{cls}' title='{html.escape(row['name'])}'>{label}</td>")
            trs.append("<tr>" + "".join(tds) + "</tr>")
        header = "".join(f"<th>{html.escape(w)}</th>" for w in warms)
        sections.append(
            f"<h2>ns_steps={html.escape(ns)}</h2><table><tr><th>muon_lr</th>{header}</tr>{''.join(trs)}</table>"
        )

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Muon ladder stability</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
table {{ border-collapse: collapse; margin-bottom: 24px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: center; }}
td.stable {{ background: #d9f99d; }}
td.failed {{ background: #fecaca; }}
td.missing {{ background: #e2e8f0; }}
</style>
<h1>Sweep 4 Phase 0 Muon Stability</h1>
{"".join(sections)}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(doc)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
