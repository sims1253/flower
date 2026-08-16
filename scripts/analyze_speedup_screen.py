#!/usr/bin/env python3
"""Compare loss traces across speedup-screen arms against the bf16 control.

WHY
  `scripts/bench_arms.py` answers "is it faster". This answers the question that
  actually gates shipping an FP8 arm: "does it train the same". Tensorwise FP8 is
  the coarse recipe (and the only fast one on sm_120 — rowwise measures 1.01x),
  so the numerical risk and the speedup are bought together and the loss trace is
  the only thing that can justify the purchase.

WHAT IT REPORTS
  For each arm, versus the control arm:
    - max |Δloss| over the aligned step grid, and the step where it occurs
    - Δ at the final logged step (has it converged back or is it drifting apart?)
    - val_bpb from the arm's metrics.json, and the delta

  The threshold is the SEED BAND: how much two runs differing only by seed differ
  in val_bpb. An arm inside that band is indistinguishable from a reseed.

  *** THE SEED BAND IS REGIME-SPECIFIC. *** The finished 450M runs give 0.0004
  (0.90234 vs 0.90270) — but that is measured at 10k steps, fully converged. A
  600-step screen with warmup_steps 500 is 83% LR ramp, where two seeds have had
  far less opportunity to converge toward each other, so its seed band is wider.
  Judging a 600-step delta against a 10k-step band compares across regimes and
  will label harmless precision offsets as regressions.

  Therefore: pass `--seed-dir` pointing at a same-length run of the control (and
  ideally the candidate) at a different seed, and this script measures the band
  instead of assuming one. `--seed-band` is only a fallback for when no such run
  exists, and it prints a warning saying so.

  Beware also that the absolute losses in a short screen do NOT predict the
  10k-step run. Divergence, not final loss, is what a screen detects.

USAGE
  PYTHONPATH=. uv run python scripts/analyze_speedup_screen.py \
    --run-dir runs/speedup_screen_450m \
    --seed-dir runs/speedup_screen_450m_seed1
  ... --control bf16_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_loss_trace(variant_dir: Path) -> dict[int, float]:
    """step -> train/loss from an arm's tensorboard events."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = variant_dir / "tensorboard"
    if not tb_dir.is_dir():
        return {}
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    if "train/loss" not in ea.Tags().get("scalars", []):
        return {}
    return {e.step: e.value for e in ea.Scalars("train/loss")}


def load_metrics(run_dir: Path, arm: str) -> dict:
    for candidate in sorted(run_dir.glob(f"{arm}*.metrics.json")):
        with candidate.open() as f:
            return json.load(f)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/speedup_screen_450m")
    ap.add_argument("--control", default="bf16_baseline")
    ap.add_argument(
        "--seed-dir",
        default=None,
        help="run dir of the SAME arms at a different seed; the band is measured from it (preferred)",
    )
    ap.add_argument(
        "--seed-band",
        type=float,
        default=None,
        help="fallback band when --seed-dir is unavailable. Must come from a run of the SAME length; "
        "the 10k-step value (0.0004) is wrong for a short screen.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    variants_dir = run_dir / "variants"
    if not variants_dir.is_dir():
        raise SystemExit(f"no variants/ under {run_dir} — has the screen been run?")

    arms = sorted(d.name for d in variants_dir.iterdir() if d.is_dir())
    control_dirs = [a for a in arms if a.startswith(args.control)]
    if not control_dirs:
        raise SystemExit(f"control arm {args.control!r} not found among {arms}")
    control = control_dirs[0]

    control_trace = load_loss_trace(variants_dir / control)
    if not control_trace:
        raise SystemExit(f"control arm {control} has no train/loss scalars")

    ctrl_metrics = load_metrics(run_dir, control)
    ctrl_bpb = ctrl_metrics.get("val_bpb")

    # Establish the threshold. Measuring it from a same-length reseed of the
    # control is the only defensible option; a band imported from a run of a
    # different length is comparing across regimes (see module docstring).
    seed_band = None
    band_source = ""
    reseed_deltas: dict[str, float] = {}
    if args.seed_dir:
        seed_dir = Path(args.seed_dir)
        for arm_name in {control, *(d.name for d in variants_dir.iterdir() if d.is_dir())}:
            base = load_metrics(run_dir, arm_name).get("val_bpb")
            other = load_metrics(seed_dir, arm_name).get("val_bpb")
            if base is not None and other is not None:
                reseed_deltas[arm_name] = abs(other - base)
        if control in reseed_deltas:
            seed_band = reseed_deltas[control]
            band_source = f"measured from {control} across seeds in {seed_dir}"
    if seed_band is None:
        seed_band = args.seed_band
        if seed_band is None:
            raise SystemExit(
                "no seed band available: pass --seed-dir (preferred) or --seed-band. "
                "Refusing to guess — a wrong band turns precision offsets into false regressions."
            )
        band_source = "SUPPLIED MANUALLY — verify it came from a run of this same length"

    print(f"control: {control}   ({len(control_trace)} logged points)")
    print(f"seed band: {seed_band:.5f} val_bpb  ({band_source})")
    if reseed_deltas:
        for arm_name, d in sorted(reseed_deltas.items()):
            print(f"    reseed delta  {arm_name:24} {d:+.5f}")
    print()
    print(f"{'arm':26} {'max|dloss|':>11} {'@step':>7} {'final dloss':>12} {'val_bpb':>9} {'d val_bpb':>10}  verdict")

    for arm in arms:
        trace = load_loss_trace(variants_dir / arm)
        metrics = load_metrics(run_dir, arm)
        bpb = metrics.get("val_bpb")

        if not trace:
            print(f"{arm:26} {'no trace':>11}")
            continue

        shared = sorted(set(trace) & set(control_trace))
        if not shared:
            print(f"{arm:26} {'no overlap':>11}")
            continue

        deltas = {s: trace[s] - control_trace[s] for s in shared}
        worst_step = max(deltas, key=lambda s: abs(deltas[s]))
        max_d = deltas[worst_step]
        final_d = deltas[shared[-1]]

        d_bpb = (bpb - ctrl_bpb) if (bpb is not None and ctrl_bpb is not None) else None

        if arm == control:
            verdict = "control"
        elif d_bpb is None:
            verdict = "no val_bpb"
        elif abs(d_bpb) <= seed_band:
            verdict = "PASS (within seed band)"
        elif abs(d_bpb) <= 2 * seed_band:
            verdict = "MARGINAL (1-2x band)"
        else:
            verdict = "FAIL (outside band)"

        bpb_s = f"{bpb:.5f}" if bpb is not None else "-"
        dbpb_s = f"{d_bpb:+.5f}" if d_bpb is not None else "-"
        print(
            f"{arm:26} {max_d:+11.4f} {worst_step:7d} {final_d:+12.4f} {bpb_s:>9} {dbpb_s:>10}  {verdict}"
        )

    print(
        "\nA growing |dloss| toward the end of the trace is the danger sign — that is "
        "divergence, not noise. A constant offset that does not widen is usually a "
        "harmless precision shift."
    )


if __name__ == "__main__":
    main()
