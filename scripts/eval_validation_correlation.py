"""Cross-probe correlation pass (Sweep 7 eval-validation E4).

On already-trained FineWeb checkpoints, run the three recall discriminators and
check whether the on-manifold `text_recall_probe` and the in-distribution
`memory_ablation_probe` agree on the variant ranking — and confirm the synthetic
`mqar_probe` stays near-zero (the Phase A diagnosis). Eval-only; no training.

Usage:
    uv run python scripts/eval_validation_correlation.py \
        --root runs/sweep7_phase_a/variants --doc-limit 64
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from flower.config import load_config
from flower.eval import _load_checkpoint_model  # noqa: PLC2701 - internal reuse is intentional
from flower.models import build_model
from flower.probes.composite import (
    memory_ablation_probe,
    mqar_probe,
    needle_in_text_probe,
    text_recall_probe,
)
from flower.train import resolve_device, set_global_seed


def _variant_of(dir_name: str) -> str:
    # "hier_max_16_adamw_seed2" -> "hier_max_16_adamw"
    return dir_name.rsplit("_seed", 1)[0]


def _latest_checkpoint(variant_dir: Path) -> Path | None:
    """Pick the highest-step `*_step*.pt` (filenames vary by module, e.g.
    summary_memory_step10000.pt vs vanilla_local_step10000.pt)."""
    def step_of(p: Path) -> int:
        try:
            return int(p.stem.split("step")[-1])
        except ValueError:
            return -1

    ckpts = [p for p in variant_dir.glob("*.pt") if "step" in p.stem]
    return max(ckpts, key=step_of) if ckpts else None


def _config_for(ckpt: Path) -> dict[str, Any] | None:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config") if isinstance(payload, dict) else None
    if isinstance(cfg, dict):
        return cfg
    sib = ckpt.parent / "config.yaml"
    if sib.exists():
        loaded = yaml.safe_load(sib.read_text())
        if isinstance(loaded, dict):
            return loaded
    return None


def run_one(ckpt: Path, device: torch.device, doc_limit: int) -> dict[str, Any] | None:
    cfg_dict = _config_for(ckpt)
    if cfg_dict is None:
        return None
    cfg = load_config(None, cfg_dict)
    set_global_seed(int(cfg.training.seed))
    model = build_model(cfg.model).to(device).eval()
    try:
        _load_checkpoint_model(model, ckpt, device)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    text = text_recall_probe(model, cfg, device)
    mqar = mqar_probe(model, cfg, device)
    needle = needle_in_text_probe(model, cfg, device)
    ablation = memory_ablation_probe(model, cfg, device, doc_limit=doc_limit)
    return {
        "text_recall_bp": text["breaking_point"],
        "mqar_bp": mqar["breaking_point"],
        "needle_bp": needle["breaking_point"],
        "needle_curve": needle.get("capacity_curve", {}),
        "ablation_delta_bpb": ablation["delta_bpb"],
        # False when the variant's memory read could not be ablated
        # (memory_ablation_probe then reports NaN deltas) — carrying the flag
        # through lets the aggregation below exclude such rows instead of
        # letting NaN silently poison the mean and the rank agreement.
        "ablated": bool(ablation["ablated"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="runs/sweep7_phase_a/variants")
    parser.add_argument("--ckpt-name", type=str, default="*.pt")
    parser.add_argument("--doc-limit", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    per_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_dir in sorted(Path(args.root).iterdir()):
        if not variant_dir.is_dir():
            continue
        ckpt = _latest_checkpoint(variant_dir)
        if ckpt is None:
            continue
        res = run_one(ckpt, device, args.doc_limit)
        if res is None or "error" in (res or {}):
            print(f"skip {variant_dir.name}: {res.get('error') if res else 'no config'}")
            continue
        per_variant.setdefault(_variant_of(variant_dir.name), []).append(res)
        print(f"{variant_dir.name}: {res}")

    print("\n=== per-variant mean (n seeds) ===")
    summary: dict[str, Any] = {}
    for variant, rows in sorted(per_variant.items()):
        def agg(key: str, _rows=rows) -> tuple[float, float]:
            vals = [float(r[key]) for r in _rows]
            return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

        # Ablation rows are only meaningful when the probe actually patched
        # something (`ablated: true`). For unpatchable variants the probe
        # reports NaN deltas; a NaN through statistics.mean would propagate,
        # and through sorted() would silently scramble the rank-agreement
        # verdict below — so aggregate and rank only over measured rows.
        ab_rows = [r for r in rows if r.get("ablated")]
        tr = agg("text_recall_bp")
        nd = agg("needle_bp")
        ab = (
            agg("ablation_delta_bpb", ab_rows)
            if ab_rows
            else (float("nan"), 0.0)
        )
        summary[variant] = {
            "n": len(rows),
            "n_ablated": len(ab_rows),
            "text_recall_bp": tr,
            "needle_bp": nd,
            "ablation_delta_bpb": ab,
        }
        ab_str = f"{ab[0]:.4f}±{ab[1]:.4f}" if ab_rows else "unmeasured (read path unpatchable)"
        print(
            f"{variant:>22}  n={len(rows)}  "
            f"text_recall_bp={tr[0]:.1f}±{tr[1]:.1f}  "
            f"needle_bp={nd[0]:.2f}±{nd[1]:.2f}  "
            f"ablation_delta={ab_str}"
        )

    # Rank agreement: needle vs ablation, over the variants where ablation was
    # actually measured. Both rankings are computed on that same subset so the
    # comparison is apples-to-apples.
    measurable = [v for v in summary if summary[v]["n_ablated"] > 0]
    if len(measurable) >= 2:
        nd_rank = sorted(measurable, key=lambda v: summary[v]["needle_bp"][0])
        ab_rank = sorted(measurable, key=lambda v: summary[v]["ablation_delta_bpb"][0])
        print("\nneedle_bp ranking (low->high):     ", " < ".join(nd_rank))
        print("ablation_delta ranking (low->high):", " < ".join(ab_rank))
        print("rankings agree:" if nd_rank == ab_rank else "rankings DIFFER:", nd_rank == ab_rank)
    elif len(summary) >= 2:
        skipped = [v for v in summary if v not in measurable]
        print(f"\nrank agreement skipped: fewer than 2 variants have a measured "
              f"ablation delta (unmeasured: {', '.join(skipped) or 'none'})")

    if args.json:
        Path(args.json).write_text(json.dumps({k: str(v) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
