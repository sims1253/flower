#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from typing import Any


def _as_float(value: Any, default: float = 999.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def normalize_offer(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _field(row, "id", "ask_contract_id", "ID", "Ask", "ASK"),
        "gpu": _field(row, "gpu_name", "GPU", "GPU_NAME"),
        "num_gpus": _field(row, "num_gpus", "N", "Num", "GPUs"),
        "dph_total": _field(row, "dph_total", "dph_base", "dph", "$/hr", "DLP/$", "PRICE"),
        "reliability": _field(row, "reliability2", "reliability", "REL", "Reliability"),
        "inet_up": _field(row, "inet_up", "Inet_up", "Net_up", "UP"),
        "inet_down": _field(row, "inet_down", "Inet_down", "Net_down", "DOWN"),
    }


def parse_json(text: str) -> list[dict[str, Any]] | None:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        for key in ("offers", "results", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        return None
    return [row for row in data if isinstance(row, dict)]


def parse_json_lines(text: str) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    saw_json = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            return None
        saw_json = True
        if isinstance(item, dict):
            rows.append(item)
    return rows if saw_json else None


def parse_table(text: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header_idx = next((i for i, ln in enumerate(lines) if re.search(r"\bID\b", ln) and re.search(r"GPU", ln, re.I)), -1)
    if header_idx < 0:
        return []
    headers = re.split(r"\s{2,}", lines[header_idx].strip())
    rows: list[dict[str, Any]] = []
    for line in lines[header_idx + 1 :]:
        if set(line) <= {"-", " "}:
            continue
        parts = re.split(r"\s{2,}", line.strip(), maxsplit=max(len(headers) - 1, 0))
        if len(parts) < 2:
            continue
        rows.append(dict(zip(headers, parts, strict=False)))
    return rows


def parse_offers(text: str) -> list[dict[str, Any]]:
    for parser in (parse_json, parse_json_lines):
        rows = parser(text)
        if rows is not None:
            return rows
    return parse_table(text)


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    max_price = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    rows = parse_offers(sys.stdin.read())
    rows = [row for row in rows if _as_float(_field(row, "dph_total", "dph_base", "dph", "$/hr", "PRICE")) <= max_price]
    rows = sorted(rows, key=lambda r: _as_float(_field(r, "dph_total", "dph_base", "dph", "$/hr", "PRICE")))[:limit]
    for row in rows:
        print(json.dumps(normalize_offer(row), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
