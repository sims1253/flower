#!/usr/bin/env python
"""Analyze Still sweep results: compare variants, plot KL curves, summarize."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(output_dir: Path) -> list[dict]:
    results = []
    for f in sorted(output_dir.glob("*.metrics.json")):
        with f.open() as fh:
            data = json.load(fh)
            data["file"] = f.name
            results.append(data)
    return results


def load_tensorboard_scalars(output_dir: Path) -> dict[str, dict[str, list]]:
    """Load scalar logs from tensorboard event files."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return {}

    variants_dir = output_dir / "variants"
    if not variants_dir.exists():
        return {}

    all_scalars: dict[str, dict[str, list]] = {}
    for variant_dir in sorted(variants_dir.iterdir()):
        tb_dir = variant_dir / "tensorboard"
        if not tb_dir.exists():
            continue
        ea = event_accumulator.EventAccumulator(
            str(tb_dir),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
        scalars: dict[str, list] = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            scalars[tag] = [(e.step, e.value) for e in events]
        all_scalars[variant_dir.name] = scalars
    return all_scalars


def summarize_metrics(metrics: list[dict]) -> None:
    print("\n=== Final Metrics ===")
    print(f"{'variant':40s} {'loss':>8s} {'ppl':>8s} {'tok/s':>8s}")
    print("-" * 68)
    for m in sorted(metrics, key=lambda x: x.get("loss", 999)):
        name = m.get("file", "?").replace(".metrics.json", "")
        loss = m.get("loss", m.get("train_loss", float("nan")))
        ppl = m.get("perplexity", m.get("train_perplexity", float("nan")))
        tps = m.get("tokens_per_sec", 0)
        print(f"{name:40s} {loss:8.4f} {ppl:8.2f} {tps:8.0f}")


def plot_training_curves(scalars: dict[str, dict[str, list]], output_dir: Path) -> None:
    if not scalars:
        print("No tensorboard data found for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Group variants by base_name (strip _seedN)
    groups: dict[str, list[str]] = {}
    for name in scalars:
        base = name.rsplit("_seed", 1)[0] if "_seed" in name else name
        groups.setdefault(base, []).append(name)

    # Plot 1: Training loss
    ax = axes[0, 0]
    for base, names in sorted(groups.items()):
        all_steps = None
        all_losses = []
        for name in names:
            data = scalars[name].get("train/loss", [])
            if not data:
                continue
            steps = [d[0] for d in data]
            losses = [d[1] for d in data]
            all_losses.append(losses)
            if all_steps is None:
                all_steps = steps
        if all_losses and all_steps:
            mean_loss = np.mean(all_losses, axis=0)
            std_loss = np.std(all_losses, axis=0) if len(all_losses) > 1 else np.zeros_like(mean_loss)
            ax.plot(all_steps, mean_loss, label=base, linewidth=1.5)
            if len(all_losses) > 1:
                ax.fill_between(all_steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    # Plot 2: KL loss
    ax = axes[0, 1]
    for base, names in sorted(groups.items()):
        all_steps = None
        all_kls = []
        for name in names:
            data = scalars[name].get("train/kl_loss", [])
            if not data:
                continue
            steps = [d[0] for d in data]
            kls = [d[1] for d in data]
            all_kls.append(kls)
            if all_steps is None:
                all_steps = steps
        if all_kls and all_steps:
            mean_kl = np.mean(all_kls, axis=0)
            ax.plot(all_steps, mean_kl, label=base, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL Loss")
    ax.set_title("KL Distillation Loss")
    ax.legend(fontsize=7)

    # Plot 3: Learning rate
    ax = axes[1, 0]
    for base, names in sorted(groups.items()):
        for name in names[:1]:
            data = scalars[name].get("train/lr", [])
            if data:
                steps = [d[0] for d in data]
                lrs = [d[1] for d in data]
                ax.plot(steps, lrs, label=base, linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate")
    ax.legend(fontsize=7)

    # Plot 4: Validation loss
    ax = axes[1, 1]
    for base, names in sorted(groups.items()):
        all_steps = None
        all_vals = []
        for name in names:
            data = scalars[name].get("val/loss", [])
            if not data:
                continue
            steps = [d[0] for d in data]
            vals = [d[1] for d in data]
            all_vals.append(vals)
            if all_steps is None:
                all_steps = steps
        if all_vals and all_steps:
            mean_val = np.mean(all_vals, axis=0)
            ax.plot(all_steps, mean_val, label=base, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved training curves to {out_path}")


def compare_variants(metrics: list[dict]) -> None:
    """Group by base variant, compute mean +/- std across seeds."""
    groups: dict[str, list[dict]] = {}
    for m in metrics:
        name = m.get("file", "").replace(".metrics.json", "")
        base = name.rsplit("_seed", 1)[0] if "_seed" in name else name
        groups.setdefault(base, []).append(m)

    print("\n=== Variant Comparison (mean +/- std across seeds) ===")
    print(f"{'variant':40s} {'loss_mean':>10s} {'loss_std':>10s} {'n_seeds':>8s}")
    print("-" * 72)
    for base, ms in sorted(groups.items()):
        losses = [m.get("loss", m.get("train_loss", float("nan"))) for m in ms]
        mean = np.mean(losses)
        std = np.std(losses) if len(losses) > 1 else 0.0
        print(f"{base:40s} {mean:10.4f} {std:10.4f} {len(ms):8d}")


def main(argv: list[str] | None = None) -> None:
    output_dir = Path(argv[0] if argv else "runs/sweep_still")

    metrics = load_metrics(output_dir)
    if metrics:
        summarize_metrics(metrics)
        compare_variants(metrics)
    else:
        print(f"No metrics found in {output_dir}")

    scalars = load_tensorboard_scalars(output_dir)
    if scalars:
        plot_training_curves(scalars, output_dir)
    else:
        print("No tensorboard data found.")

    # Write summary JSON
    summary = {
        "variants": [
            {
                "name": m.get("file", "").replace(".metrics.json", ""),
                "loss": m.get("loss", m.get("train_loss")),
                "perplexity": m.get("perplexity", m.get("train_perplexity")),
                "tokens_per_sec": m.get("tokens_per_sec"),
            }
            for m in metrics
        ]
    }
    summary_path = output_dir / "analysis_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
