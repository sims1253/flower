#!/usr/bin/env python3
"""Aggregate sweep .metrics.json files into a single ranked table.

Usage:
  uv run python scripts/aggregate_sweep_results.py runs/vast/sweep2/
  uv run python scripts/aggregate_sweep_results.py runs/vast/sweep2/ --sort val_perplexity --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _flatten_composite(data: dict[str, Any]) -> dict[str, float]:
    rank_inputs = data.get("rank_inputs", {})
    if not isinstance(rank_inputs, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in rank_inputs.items():
        try:
            out[f"rank_{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _attach_average_ranks(rows: list[dict[str, Any]]) -> None:
    rank_cols = sorted({k for row in rows for k in row if k.startswith("rank_")})
    if not rank_cols:
        return
    ranks_by_row: dict[int, list[int]] = {i: [] for i in range(len(rows))}
    for col in rank_cols:
        indexed = [(i, row[col]) for i, row in enumerate(rows) if isinstance(row.get(col), (int, float))]
        indexed.sort(key=lambda item: item[1])
        for rank, (i, _) in enumerate(indexed, start=1):
            ranks_by_row[i].append(rank)
    for i, ranks in ranks_by_row.items():
        if ranks:
            rows[i]["composite_avg_rank"] = round(sum(ranks) / len(ranks), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing *.metrics.json files")
    parser.add_argument("--sort", default="val_perplexity", help="Metric to sort by (asc)")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write the aggregated table as JSON")
    args = parser.parse_args()

    rows: list[dict] = []
    for path in sorted(args.directory.glob("*.metrics.json")):
        data = _load_json(path)
        if data is None:
            continue
        row = {
            "name": path.stem.replace(".metrics", ""),
            "variant": data.get("variant", "?"),
            "params_M": round(data.get("parameter_count", 0) / 1e6, 2),
            "train_loss": round(float(data.get("train_loss", float("nan"))), 4),
            "val_loss": round(float(data.get("val_loss", data.get("loss", float("nan")))), 4),
            "val_perplexity": round(float(data.get("val_perplexity", data.get("perplexity", float("nan")))), 2),
            "bpb": round(float(data.get("bpb", float("nan"))), 5),
            "tok_per_sec": round(float(data.get("tokens_per_sec", 0))),
            "vram_gb": round(float(data.get("gpu_memory_allocated", 0)) / 1e9, 2),
            "steps": int(data.get("steps", data.get("checkpoint_step", 0))),
            "seed": int(data.get("seed", 0)),
            "device": data.get("device", "?"),
        }
        composite_path = data.get("composite_ranker_json")
        if isinstance(composite_path, str):
            comp = _load_json(Path(composite_path))
            if comp is not None:
                row.update(_flatten_composite(comp))
                fineweb_bpb = comp.get("metrics", {}).get("fineweb", {}).get("bpb")
                if fineweb_bpb is not None:
                    row["bpb"] = round(float(fineweb_bpb), 5)
        if isinstance(data.get("composite_ranker"), dict):
            row.update(_flatten_composite(data["composite_ranker"]))
        rows.append(row)

    seen_names = {row["name"] for row in rows}
    for path in sorted(args.directory.glob("**/composite_ranker.json")) + sorted(
        args.directory.glob("*.composite.json")
    ):
        name = path.parent.name if path.name == "composite_ranker.json" else path.stem.replace(".composite", "")
        if name in seen_names:
            continue
        comp = _load_json(path)
        if comp is None:
            continue
        row = {
            "name": name,
            "variant": comp.get("variant", "?"),
            "params_M": 0,
            "train_loss": float("nan"),
            "val_loss": float("nan"),
            "val_perplexity": float("nan"),
            "bpb": round(float(comp.get("metrics", {}).get("fineweb", {}).get("bpb", float("nan"))), 5),
            "tok_per_sec": 0,
            "vram_gb": 0,
            "steps": 0,
            "seed": int(comp.get("seed", 0)),
            "device": "?",
        }
        row.update(_flatten_composite(comp))
        rows.append(row)

    _attach_average_ranks(rows)

    def sort_key(r: dict) -> float:
        v = r.get(args.sort)
        try:
            return float(v) if v is not None else float("inf")
        except (TypeError, ValueError):
            return float("inf")

    rows.sort(key=sort_key)

    if not rows:
        print(f"No metrics files found in {args.directory}")
        return

    cols = [
        "name",
        "variant",
        "params_M",
        "bpb",
        "composite_avg_rank",
        "val_loss",
        "val_perplexity",
        "train_loss",
        "tok_per_sec",
        "vram_gb",
        "steps",
        "seed",
    ]
    cols = [c for c in cols if any(c in r for r in rows)]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(f"{c:>{widths[c]}}" if c != "name" else f"{c:<{widths[c]}}" for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        line_parts = []
        for c in cols:
            v = r.get(c, "")
            if c == "name":
                line_parts.append(f"{v:<{widths[c]}}")
            else:
                line_parts.append(f"{v:>{widths[c]}}")
        print("  ".join(line_parts))

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote aggregated table to {args.json}")


if __name__ == "__main__":
    main()
