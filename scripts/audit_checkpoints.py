"""Audit saved checkpoints under runs/ (Sweep 7 eval-validation E3).

For every `*.pt` checkpoint it: loads the payload, recovers the embedded config
(falling back to a sibling `config.yaml`), rebuilds the model, and attempts a
strict `load_state_dict`. Reports which checkpoints load cleanly and which are
stale/mismatched, plus whether the embedded variant matches the directory name
(the `vanilla_local.pt` filename in every variant dir is a known naming quirk —
the embedded config is the source of truth).

Usage:
    uv run python scripts/audit_checkpoints.py [--root runs] [--glob '**/*.pt']
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from flower.config import load_config
from flower.models import build_model


def _embedded_config(payload: dict[str, Any], ckpt: Path) -> dict[str, Any] | None:
    cfg = payload.get("config") if isinstance(payload, dict) else None
    if isinstance(cfg, dict):
        return cfg
    sibling = ckpt.parent / "config.yaml"
    if sibling.exists():
        loaded = yaml.safe_load(sibling.read_text())
        if isinstance(loaded, dict):
            return loaded
    return None


def audit_one(ckpt: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(ckpt), "status": "unknown"}
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
        row["status"] = "unreadable"
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    cfg_dict = _embedded_config(payload, ckpt)
    if cfg_dict is None:
        row["status"] = "no-config"
        return row

    try:
        cfg = load_config(None, cfg_dict)
        row["variant"] = cfg.model.variant
        row["dir"] = ckpt.parent.name
        row["variant_matches_dir"] = ckpt.parent.name.startswith(cfg.model.variant)
        row["step"] = payload.get("step") if isinstance(payload, dict) else None
        model = build_model(cfg.model)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            row["status"] = "shape-or-key-mismatch"
            row["missing_keys"] = len(missing)
            row["unexpected_keys"] = len(unexpected)
        else:
            row["status"] = "clean"
    except Exception as e:  # noqa: BLE001
        row["status"] = "load-error"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="runs")
    parser.add_argument("--glob", type=str, default="**/*.pt")
    parser.add_argument("--json", type=str, default=None, help="optional path to write the full report")
    args = parser.parse_args(argv)

    root = Path(args.root)
    checkpoints = sorted(root.glob(args.glob))
    rows = [audit_one(c) for c in checkpoints]

    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    print(f"Audited {len(rows)} checkpoint(s) under {root}/")
    for status, n in sorted(by_status.items()):
        print(f"  {status}: {n}")
    print()
    for r in rows:
        flag = "" if r["status"] == "clean" and r.get("variant_matches_dir", True) else "  <-- CHECK"
        extra = r.get("error", "")
        print(f"[{r['status']:>22}] {r.get('variant','?'):>20} step={r.get('step')} {r['path']}{flag} {extra}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
