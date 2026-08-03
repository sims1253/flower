#!/usr/bin/env python3
"""Analyze the Still scaling sweep (sweep_still_scale).

Reads tensorboard logs for every variant x seed run, extracts per-variant
metrics (final loss, KL trajectory, best KL, KL convergence step), computes
statistical comparisons against the baseline (Welch's t-test + Cohen's d), and
produces:

  1. A ranked summary table (printed to stdout, sorted by final loss).
  2. A KL trajectory plot (mean +/- std across seeds), saved as PNG.
  3. A final-loss bar chart with error bars and significance markers, PNG.
  4. A JSON summary file capturing every metric and comparison.

Usage:
    python scripts/analyze_still_scale.py \
        --sweep-dir runs/sweep_still_scale \
        --output-dir reports/scale_analysis

The script is self-contained: scipy is used for the t-test p-value *if
available*, otherwise an exact p-value is computed via the regularized
incomplete beta function (Numerical Recipes) so the script runs in the default
project venv (which has numpy + matplotlib + tensorboard but no scipy).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

# tensorboard is a hard dependency for reading event logs.
from tensorboard.backend.event_processing import event_accumulator  # noqa: E402

# Type alias used in the statistics helpers below.
SequenceLike = Iterable[float]

# Candidate scalar tag names. The canonical Still training logs the KL loss under
# ``diagnostics/kl_loss`` (surfaced from the model's diagnostics dict). We also
# accept a couple of legacy/alternate names so the script stays robust.
LOSS_TAGS = ("train/loss", "loss")
VAL_LOSS_TAGS = ("validation/loss", "val/loss")
KL_TAGS = ("diagnostics/kl_loss", "train/kl_loss", "kl_loss")

# Significance thresholds for the asterisk markers.
P_THRESHOLD_STRONG = 0.01  # "**"
P_THRESHOLD_WEAK = 0.05  # "*"


# --------------------------------------------------------------------------- #
# Statistics: Welch's t-test + Cohen's d (scipy optional).
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-16) -> float:
    """Continued fraction expansion for the incomplete beta function.

    Lentz's method (Numerical Recipes, 3rd ed.). Used to obtain an exact Welch
    t-test p-value without scipy.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Symmetry factor: x^(a-1) (1-x)^(b-1) / B(a, b), computed in log-space.
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    # Use the symmetry relation I_x(a, b) = 1 - I_{1-x}(b, a) for stability.
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta) / b
    return 1.0 - front * _betacf(b, a, 1.0 - x)


def _t_distribution_two_tailed_p(t_stat: float, df: float) -> float:
    """Two-tailed p-value for Student's t given a statistic and dof.

    p = I_{df / (df + t^2)}(df/2, 1/2).
    """
    if not math.isfinite(df) or df <= 0:
        return float("nan")
    x = df / (df + t_stat * t_stat)
    return _regularized_incomplete_beta(0.5 * df, 0.5, x)


def welch_t_test(a: SequenceLike, b: SequenceLike) -> dict[str, float]:
    """Welch's two-sample t-test (unequal variances).

    Returns a dict with t_statistic, p_value (two-tailed), degrees_of_freedom
    (Welch-Satterthwaite) and the two sample means. Requires >= 2 samples in
    each group; otherwise returns NaNs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = {
        "t_statistic": float("nan"),
        "p_value": float("nan"),
        "degrees_of_freedom": float("nan"),
        "mean_a": float("nan"),
        "mean_b": float("nan"),
    }
    if a.size < 2 or b.size < 2:
        return out
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    se2_a = var_a / a.size
    se2_b = var_b / b.size
    denom = se2_a + se2_b
    if denom <= 0:
        # Identical, zero-variance samples: no detectable difference.
        out.update({"mean_a": mean_a, "mean_b": mean_b, "t_statistic": 0.0, "p_value": 1.0})
        return out
    t_stat = (mean_a - mean_b) / math.sqrt(denom)
    # Welch-Satterthwaite degrees of freedom.
    df = (denom * denom) / ((se2_a * se2_a) / max(a.size - 1, 1) + (se2_b * se2_b) / max(b.size - 1, 1))

    p_value = float("nan")
    # Prefer scipy when it is available (marginally more robust at the tails).
    try:
        from scipy import stats  # type: ignore[import-not-found]

        result = stats.ttest_ind(a, b, equal_var=False)
        p_value = float(result.pvalue)
        # Fall back to our value if scipy returns something non-finite.
        if not math.isfinite(p_value):
            p_value = _t_distribution_two_tailed_p(t_stat, df)
    except Exception:  # noqa: BLE001 - any scipy issue -> self-contained path.
        p_value = _t_distribution_two_tailed_p(t_stat, df)

    out.update(
        {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": float(df),
            "mean_a": mean_a,
            "mean_b": mean_b,
        }
    )
    return out


def cohens_d(a: SequenceLike, b: SequenceLike) -> float:
    """Cohen's d using the pooled standard deviation (Hedges-corrected optional).

    Positive d means group ``a`` has a larger mean than group ``b``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    pooled_std = math.sqrt((var_a + var_b) / 2.0)
    if pooled_std <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def significance_marker(p_value: float) -> str:
    """Return an asterisk marker for a two-tailed p-value."""
    if not math.isfinite(p_value):
        return "ns"
    if p_value < P_THRESHOLD_STRONG:
        return "**"
    if p_value < P_THRESHOLD_WEAK:
        return "*"
    return "ns"


# --------------------------------------------------------------------------- #
# Tensorboard loading + metric extraction.
# --------------------------------------------------------------------------- #
def _strip_seed_suffix(run_name: str) -> str:
    """Strip a trailing ``_seed<N>`` suffix to recover the base variant name."""
    if "_seed" in run_name:
        base, _, suffix = run_name.rpartition("_seed")
        if suffix.isdigit():
            return base
    return run_name


def find_run_dirs(sweep_dir: Path) -> list[tuple[str, Path]]:
    """Locate (run_name, tensorboard_log_dir) pairs for a sweep.

    Handles both the canonical sweep layout (``<sweep_dir>/variants/<run>/``) and
    a flat layout where run directories sit directly under ``sweep_dir``.
    """
    candidates: list[Path] = []
    variants_dir = sweep_dir / "variants"
    if variants_dir.is_dir():
        candidates = [p for p in sorted(variants_dir.iterdir()) if p.is_dir()]
    elif sweep_dir.is_dir():
        candidates = [p for p in sorted(sweep_dir.iterdir()) if p.is_dir()]

    runs: list[tuple[str, Path]] = []
    for run_dir in candidates:
        tb_dir = run_dir / "tensorboard"
        if tb_dir.is_dir():
            runs.append((run_dir.name, tb_dir))
            continue
        # Fall back: event files living directly in the run directory.
        if any(run_dir.glob("events.out.tfevents.*")):
            runs.append((run_dir.name, run_dir))
    return runs


def load_run_scalars(tb_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Load all scalar series from one tensorboard log directory.

    Returns a mapping of tag -> list of (step, value) tuples. Returns an empty
    dict if the directory holds no readable event files (handled gracefully).
    """
    try:
        ea = event_accumulator.EventAccumulator(
            str(tb_dir),
            size_guidance={event_accumulator.SCALARS: 0},  # 0 == load every scalar
        )
        ea.Reload()
    except Exception as exc:  # noqa: BLE001 - corrupt/partial logs are common.
        warnings.warn(f"Could not read tensorboard logs at {tb_dir}: {exc}", stacklevel=2)
        return {}

    scalars: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        try:
            events = ea.Scalars(tag)
        except Exception:  # noqa: BLE001 - some tags can fail to materialise.
            continue
        scalars[tag] = [(int(e.step), float(e.value)) for e in events]
    return scalars


def _pick_series(scalars: dict[str, list[tuple[int, float]]], tags: Iterable[str]) -> list[tuple[int, float]]:
    """Return the first available series among candidate tags, else []."""
    for tag in tags:
        series = scalars.get(tag)
        if series:
            return series
    return []


def _final_value(series: list[tuple[int, float]]) -> float:
    """Last value of a (step, value) series ordered by step."""
    if not series:
        return float("nan")
    ordered = sorted(series, key=lambda sv: sv[0])
    return float(ordered[-1][1])


def trim_warmup(series: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Drop the leading base-warmup segment of a series.

    During base warmup (``step < still_compact_from_step``) the StillLM logs
    ``kl_loss = 0.0`` and the metric is constant. Including those points would
    make ``best_kl`` (a minimum) collapse to 0 and push the KL trajectory off the
    relevant axis. We therefore trim the leading run of points that are equal
    (within a tiny tolerance) to the first value, up to the first change. If the
    whole series is flat or trimming would leave nothing, the original series is
    returned unchanged.
    """
    if len(series) < 2:
        return series
    ordered = sorted(series, key=lambda sv: sv[0])
    first = ordered[0][1]
    tol = 1e-9 + 1e-9 * abs(first)
    cut = 0
    for cut in range(len(ordered)):
        if abs(ordered[cut][1] - first) > tol:
            break
    else:
        # Never changed -> entirely flat; keep as-is so callers still see data.
        return series
    trimmed = ordered[cut:]
    return trimmed if len(trimmed) >= 2 else series


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing-edge moving average padded so the output length matches the input."""
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window) / window
    # 'same' keeps length, but the leading edge is biased toward the start; we
    # then trim to the valid (full-window) region when locating plateaus.
    return np.convolve(values, kernel, mode="same")


def kl_convergence_step(
    kl_series: list[tuple[int, float]],
    window: int = 10,
    rel_tol: float = 0.05,
) -> int | None:
    """Estimate the step at which the KL trajectory plateaus.

    Smooths the trajectory with a trailing moving average, then finds the
    earliest step whose smoothed value stays within ``rel_tol`` (relative) of the
    final smoothed value for the remainder of training. Returns ``None`` when the
    trajectory is too short to judge.
    """
    if len(kl_series) < 2 * window:
        if kl_series:
            return int(sorted(kl_series, key=lambda sv: sv[0])[-1][0])
        return None

    steps = np.array([s for s, _ in kl_series], dtype=float)
    vals = np.array([v for _, v in kl_series], dtype=float)
    order = np.argsort(steps)
    steps = steps[order]
    vals = vals[order]

    smoothed = moving_average(vals, window)
    final_val = float(smoothed[-1])
    if final_val == 0:
        return int(steps[-1])

    # Restrict to the valid (full-window) region so early edge effects don't
    # produce spurious "plateau" detections.
    valid = smoothed[window - 1 :]
    valid_steps = steps[window - 1 :]
    threshold = rel_tol * abs(final_val)
    within = np.abs(valid - final_val) <= threshold

    # Earliest index i such that every value from i onward is within tolerance.
    earliest = None
    for i in range(len(valid)):
        if within[i:].all():
            earliest = i
            break
    if earliest is None:
        return int(valid_steps[-1])
    return int(valid_steps[earliest])


def extract_run_metrics(scalars: dict[str, list[tuple[int, float]]]) -> dict[str, Any]:
    """Pull all per-run metrics from a single run's scalar dict."""
    loss_series = _pick_series(scalars, LOSS_TAGS)
    val_series = _pick_series(scalars, VAL_LOSS_TAGS)
    kl_series = _pick_series(scalars, KL_TAGS)

    # Trim the base-warmup segment from the KL trajectory: during warmup the KL
    # is logged as a flat 0.0 (compaction inactive), which would otherwise
    # dominate best_kl and the convergence/trajectory analysis.
    kl_trimmed = trim_warmup(kl_series)

    kl_values = [v for _, v in kl_trimmed]
    best_kl = min(kl_values) if kl_values else float("nan")
    best_kl_step = int(kl_trimmed[int(np.argmin(kl_values))][0]) if kl_values else None

    return {
        "final_loss": _final_value(loss_series),
        "final_val_loss": _final_value(val_series) if val_series else None,
        "final_kl": _final_value(kl_trimmed) if kl_trimmed else None,
        "best_kl": float(best_kl),
        "best_kl_step": best_kl_step,
        "convergence_step": kl_convergence_step(kl_trimmed),
        "loss_series": loss_series,
        "kl_series": kl_trimmed,
        "n_logged_steps": len(loss_series),
    }


def align_trajectories(
    series_list: list[list[tuple[int, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate multiple (step, value) series onto a common step grid.

    Returns (common_steps, values_matrix) where values_matrix has one row per
    series. Runs with no data contribute a row of NaNs and are excluded from the
    mean by ``np.nanmean``.
    """
    valid = [s for s in series_list if len(s) >= 2]
    if not valid:
        if series_list:
            steps = np.array([s for s, _ in series_list[0]], dtype=float)
            return steps, np.full((len(series_list), steps.size), np.nan)
        return np.array([]), np.array([])

    grid = np.unique(np.concatenate([np.array([s for s, _ in ser], dtype=float) for ser in valid]))
    rows = []
    for ser in series_list:
        if len(ser) < 2:
            rows.append(np.full(grid.shape, np.nan))
            continue
        steps = np.array([s for s, _ in ser], dtype=float)
        vals = np.array([v for _, v in ser], dtype=float)
        order = np.argsort(steps)
        rows.append(np.interp(grid, steps[order], vals[order]))
    return grid, np.array(rows)


# --------------------------------------------------------------------------- #
# Group aggregation + reporting.
# --------------------------------------------------------------------------- #
def aggregate_by_variant(
    run_metrics: dict[str, dict[str, Any]],
    baseline: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Aggregate per-run metrics by base variant name and compute baseline stats.

    Returns (by_variant, resolved_baseline).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for run_name, metrics in run_metrics.items():
        base = _strip_seed_suffix(run_name)
        groups.setdefault(base, []).append(metrics)

    # Resolve the baseline: explicit > a variant literally named "baseline"-ish.
    resolved_baseline = baseline
    if resolved_baseline not in groups:
        for name in groups:
            if "baseline" in name.lower():
                resolved_baseline = name
                break
        if resolved_baseline not in groups:
            resolved_baseline = sorted(groups)[0]

    baseline_losses = np.array(
        [m["final_loss"] for m in groups.get(resolved_baseline, []) if math.isfinite(m["final_loss"])],
        dtype=float,
    )

    by_variant: dict[str, dict[str, Any]] = {}
    for base, metrics_list in groups.items():
        losses = np.array([m["final_loss"] for m in metrics_list], dtype=float)
        losses = losses[np.isfinite(losses)]
        best_kls = np.array([m["best_kl"] for m in metrics_list if math.isfinite(m["best_kl"])], dtype=float)
        conv_steps = np.array(
            [m["convergence_step"] for m in metrics_list if m["convergence_step"] is not None],
            dtype=float,
        )
        final_kls = np.array([m["final_kl"] for m in metrics_list if math.isfinite(m["final_kl"])], dtype=float)

        # Aligned KL + loss trajectories across seeds (for plotting/JSON).
        kl_grid, kl_rows = align_trajectories([m["kl_series"] for m in metrics_list])
        loss_grid, loss_rows = align_trajectories([m["loss_series"] for m in metrics_list])

        # Statistical comparison against the baseline (Welch t-test + Cohen's d).
        comparison: dict[str, Any]
        if base == resolved_baseline:
            comparison = {
                "delta_loss": 0.0,
                "cohen_d": 0.0,
                "p_value": None,
                "significant": None,
                "note": "baseline",
            }
        else:
            test = welch_t_test(losses, baseline_losses)
            d = cohens_d(losses, baseline_losses)
            p = test["p_value"]
            comparison = {
                "delta_loss": float(losses.mean() - baseline_losses.mean())
                if losses.size and baseline_losses.size
                else float("nan"),
                "cohen_d": float(d),
                "p_value": float(p) if math.isfinite(p) else None,
                "significant": significance_marker(p) if math.isfinite(p) else None,
                "t_statistic": float(test["t_statistic"]) if math.isfinite(test["t_statistic"]) else None,
                "degrees_of_freedom": float(test["degrees_of_freedom"])
                if math.isfinite(test["degrees_of_freedom"])
                else None,
            }

        by_variant[base] = {
            "n_seeds": len(metrics_list),
            "n_seeds_with_loss": int(losses.size),
            "seeds": sorted(_extract_seed(run) for run in run_metrics if _strip_seed_suffix(run) == base),
            "final_loss_mean": float(losses.mean()) if losses.size else float("nan"),
            "final_loss_std": float(losses.std(ddof=1)) if losses.size > 1 else 0.0,
            "final_loss_min": float(losses.min()) if losses.size else float("nan"),
            "final_loss_max": float(losses.max()) if losses.size else float("nan"),
            "best_kl_mean": float(best_kls.mean()) if best_kls.size else float("nan"),
            "best_kl_std": float(best_kls.std(ddof=1)) if best_kls.size > 1 else 0.0,
            "final_kl_mean": float(final_kls.mean()) if final_kls.size else float("nan"),
            "convergence_step_mean": float(conv_steps.mean()) if conv_steps.size else float("nan"),
            "convergence_step_std": float(conv_steps.std(ddof=1)) if conv_steps.size > 1 else 0.0,
            "kl_trajectory_steps": kl_grid.tolist(),
            "kl_trajectory_mean": (np.nanmean(kl_rows, axis=0).tolist() if kl_rows.size else []),
            "kl_trajectory_std": (np.nanstd(kl_rows, axis=0, ddof=1).tolist() if kl_rows.shape[0] > 1 else []),
            "loss_trajectory_steps": loss_grid.tolist(),
            "loss_trajectory_mean": (np.nanmean(loss_rows, axis=0).tolist() if loss_rows.size else []),
            "vs_baseline": comparison,
        }

    return by_variant, resolved_baseline


def _extract_seed(run_name: str) -> int:
    """Pull the integer seed from a ``..._seedN`` run name (default 0)."""
    if "_seed" in run_name:
        _, _, suffix = run_name.rpartition("_seed")
        if suffix.isdigit():
            return int(suffix)
    return 0


def print_summary_table(
    by_variant: dict[str, dict[str, Any]],
    baseline: str,
    file=None,
) -> None:
    """Print the ranked summary table (best final loss first)."""
    out = file or sys.stdout
    ranked = sorted(
        by_variant.items(),
        key=lambda kv: kv[1]["final_loss_mean"] if math.isfinite(kv[1]["final_loss_mean"]) else math.inf,
    )

    header = (
        f"{'variant':<18} {'n':>3} {'loss_mean':>10} {'loss_std':>10} "
        f"{'best_kl':>10} {'conv_step':>10} {'Δloss':>9} {'d':>7} {'p':>9} {'sig':>4}"
    )
    print(file=out)
    print("=" * len(header), file=out)
    print(f"Still scaling sweep analysis  (baseline = {baseline})", file=out)
    print("=" * len(header), file=out)
    print(header, file=out)
    print("-" * len(header), file=out)
    for name, agg in ranked:
        comp = agg["vs_baseline"]
        p = comp["p_value"]
        d = comp["cohen_d"]
        delta = comp["delta_loss"]
        sig = comp["significant"]
        if sig is None:
            sig = "-" if name == baseline else "n/a"
        p_str = f"{p:.2e}" if isinstance(p, float) and math.isfinite(p) else "  n/a"
        d_str = f"{d:+.2f}" if math.isfinite(d) else "  n/a"
        delta_str = f"{delta:+.4f}" if math.isfinite(delta) else "  n/a"
        marker = " *" if name == baseline else ""
        print(
            f"{name:<18} {agg['n_seeds_with_loss']:>3} "
            f"{agg['final_loss_mean']:>10.4f} {agg['final_loss_std']:>10.4f} "
            f"{agg['best_kl_mean']:>10.4f} {agg['convergence_step_mean']:>10.0f} "
            f"{delta_str:>9} {d_str:>7} {p_str:>9} {sig:>4}{marker}",
            file=out,
        )
    print(
        "\nSignificance vs baseline: ** p<0.01, * p<0.05, ns not significant, "
        "n/a insufficient seeds (need >=2 per group).",
        file=out,
    )


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
def plot_kl_trajectory(
    by_variant: dict[str, dict[str, Any]],
    baseline: str,
    out_path: Path,
) -> None:
    """KL distillation loss over training, mean +/- std band per variant."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted = False
    # Plot baseline first (dashed) so other variants are visually on top.
    ordered = sorted(by_variant.items(), key=lambda kv: (kv[0] != baseline, kv[0]))
    for name, agg in ordered:
        steps = np.array(agg["kl_trajectory_steps"], dtype=float)
        mean = np.array(agg["kl_trajectory_mean"], dtype=float)
        if steps.size == 0 or mean.size == 0:
            continue
        is_base = name == baseline
        ax.plot(
            steps,
            mean,
            label=f"{name}{' (baseline)' if is_base else ''}",
            linewidth=2.0 if is_base else 1.6,
            linestyle="--" if is_base else "-",
            zorder=3 if is_base else 2,
        )
        std = np.array(agg["kl_trajectory_std"], dtype=float)
        if std.size == mean.size and np.any(np.isfinite(std)):
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
        plotted = True

    if not plotted:
        plt.close(fig)
        warnings.warn("No KL trajectory data available; skipping KL plot.", stacklevel=2)
        return

    ax.set_xlabel("Training step")
    ax.set_ylabel("KL distillation loss (diagnostics/kl_loss)")
    ax.set_title("Still scaling sweep: KL convergence trajectories")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved KL trajectory plot to {out_path}")


def plot_loss_comparison(
    by_variant: dict[str, dict[str, Any]],
    baseline: str,
    out_path: Path,
) -> None:
    """Final-loss bar chart with error bars and significance markers."""
    ranked = sorted(
        by_variant.items(),
        key=lambda kv: kv[1]["final_loss_mean"] if math.isfinite(kv[1]["final_loss_mean"]) else math.inf,
    )
    names = [name for name, _ in ranked]
    means = np.array([agg["final_loss_mean"] for _, agg in ranked], dtype=float)
    stds = np.array([agg["final_loss_std"] for _, agg in ranked], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ["#4c72b0" if name == baseline else "#55a868" for name in names]
    # Highlight variants that are worse than baseline in a warmer colour.
    base_mean = by_variant.get(baseline, {}).get("final_loss_mean", math.inf)
    for i, (name, agg) in enumerate(ranked):
        if math.isfinite(agg["final_loss_mean"]) and math.isfinite(base_mean) and name != baseline:
            colors[i] = "#c44e52" if agg["final_loss_mean"] > base_mean else "#55a868"

    ax.bar(range(len(names)), means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)

    # Annotate significance markers above each (non-baseline) bar.
    y_max = float(np.nanmax(means + stds)) if means.size else 1.0
    pad = 0.02 * (y_max if math.isfinite(y_max) and y_max > 0 else 1.0)
    for i, (name, agg) in enumerate(ranked):
        sig = agg["vs_baseline"].get("significant")
        if not sig or name == baseline:
            continue
        label = {"**": "**\np<0.01", "*": "*\np<0.05"}.get(sig, sig)
        ax.text(i, means[i] + stds[i] + pad, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Final training loss (train/loss, mean over seeds)")
    ax.set_title("Still scaling sweep: final loss by variant (error bars = std)")
    ax.grid(True, axis="y", alpha=0.3)

    # Legend for the colour coding.
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#4c72b0", alpha=0.85, label="baseline"),
        Patch(facecolor="#55a868", alpha=0.85, label="better than baseline"),
        Patch(facecolor="#c44e52", alpha=0.85, label="worse than baseline"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss comparison plot to {out_path}")


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the Still scaling sweep (tensorboard logs -> tables, plots, JSON).",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("runs/sweep_still_scale"),
        help="Directory containing the sweep output (variants/<run>/tensorboard).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/scale_analysis"),
        help="Where to write plots and the JSON summary.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="scale_baseline",
        help="Baseline variant name for statistical comparisons.",
    )
    parser.add_argument(
        "--convergence-rel-tol",
        type=float,
        default=0.05,
        help="Relative tolerance used to detect the KL plateau step.",
    )
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=10,
        help="Moving-average window (logged points) for KL plateau detection.",
    )
    args = parser.parse_args(argv)

    sweep_dir: Path = args.sweep_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_dir.is_dir():
        print(f"ERROR: sweep directory not found: {sweep_dir}", file=sys.stderr)
        return 1

    runs = find_run_dirs(sweep_dir)
    if not runs:
        print(f"ERROR: no run directories with tensorboard logs found under {sweep_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(runs)} run(s) under {sweep_dir}")

    # Per-run metric extraction (incomplete / corrupt runs are skipped gently).
    run_metrics: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for run_name, tb_dir in runs:
        scalars = load_run_scalars(tb_dir)
        if not scalars or not _pick_series(scalars, LOSS_TAGS):
            print(f"  [skip] {run_name}: no train/loss data in {tb_dir}")
            failed.append(run_name)
            continue
        metrics = extract_run_metrics(scalars)
        # Override convergence params from CLI for consistency.
        metrics["convergence_step"] = kl_convergence_step(
            metrics["kl_series"],
            window=args.convergence_window,
            rel_tol=args.convergence_rel_tol,
        )
        run_metrics[run_name] = metrics
        print(
            f"  [ok]   {run_name}: final_loss={metrics['final_loss']:.4f} "
            f"best_kl={metrics['best_kl']:.4f} "
            f"conv_step={metrics['convergence_step']} "
            f"({metrics['n_logged_steps']} logged steps)"
        )

    if not run_metrics:
        print("ERROR: no runs with usable metrics.", file=sys.stderr)
        return 1

    by_variant, resolved_baseline = aggregate_by_variant(run_metrics, args.baseline)
    if resolved_baseline != args.baseline:
        print(f"NOTE: requested baseline '{args.baseline}' not found; using '{resolved_baseline}' instead.")

    # 1. Ranked summary table to stdout.
    print_summary_table(by_variant, resolved_baseline)

    # 2. KL trajectory plot.
    plot_kl_trajectory(by_variant, resolved_baseline, output_dir / "kl_trajectory.png")

    # 3. Final-loss bar chart with error bars + significance markers.
    plot_loss_comparison(by_variant, resolved_baseline, output_dir / "loss_comparison.png")

    # 4. JSON summary file with every metric.
    ranking = sorted(
        by_variant,
        key=lambda v: by_variant[v]["final_loss_mean"] if math.isfinite(by_variant[v]["final_loss_mean"]) else math.inf,
    )
    summary = {
        "sweep_dir": str(sweep_dir),
        "output_dir": str(output_dir),
        "baseline": resolved_baseline,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_runs_total": len(runs),
        "n_runs_complete": len(run_metrics),
        "n_runs_skipped": len(failed),
        "skipped_runs": failed,
        "ranking_by_final_loss": ranking,
        "by_variant": by_variant,
        "per_run": {
            name: {k: v for k, v in m.items() if k not in {"loss_series", "kl_series"}}
            for name, m in run_metrics.items()
        },
    }
    json_path = output_dir / "scale_analysis_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved JSON summary to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
