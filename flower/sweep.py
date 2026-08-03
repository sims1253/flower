from __future__ import annotations

import argparse
import json
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from flower.train import train


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of two dictionaries without mutating inputs."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_sweep(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    sweep_path = Path(path)
    with sweep_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("sweep"), dict):
        raise ValueError("Sweep file must contain a 'sweep' mapping")

    sweep = raw["sweep"]
    defaults = sweep.get("defaults", {})
    variants = sweep.get("variants", [])
    if not isinstance(defaults, dict):
        raise ValueError("sweep.defaults must be a mapping")
    if not isinstance(variants, list):
        raise ValueError("sweep.variants must be a list")

    name = str(sweep.get("name") or sweep_path.stem)
    expanded: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"sweep.variants[{index}] must be a mapping")
        variant_name = variant.get("name") or variant.get("model", {}).get("variant")
        if not variant_name:
            raise ValueError(f"sweep.variants[{index}] must define name or model.variant")
        overrides = {k: v for k, v in variant.items() if k != "name"}
        config = deep_merge(defaults, overrides)
        expanded.append({"name": str(variant_name), "config": config})
    return name, expanded


def select_variants(variants: list[dict[str, Any]], names: str | None, limit: int | None) -> list[dict[str, Any]]:
    selected = variants
    if names:
        requested = [name.strip() for name in names.split(",") if name.strip()]
        by_name = {variant["name"]: variant for variant in variants}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"Unknown sweep variants: {', '.join(missing)}")
        selected = [by_name[name] for name in requested]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        selected = selected[:limit]
    return selected


def parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    seeds = [int(seed.strip()) for seed in raw.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def expand_seed_variants(variants: list[dict[str, Any]], cli_seeds: list[int] | None = None) -> list[dict[str, Any]]:
    """Expand sweep variants into one concrete trial per seed.

    The config field `training.seeds` is intentionally treated as a launcher
    directive, not as something `train()` consumes. Single-seed sweeps keep their
    original names for backwards-compatible output paths.
    """
    expanded: list[dict[str, Any]] = []
    for variant in variants:
        config = deepcopy(variant["config"])
        training = config.setdefault("training", {})
        raw_seeds = cli_seeds if cli_seeds is not None else training.get("seeds")
        if raw_seeds is None:
            raw_seeds = [training.get("seed", 0)]
        seeds = [int(seed) for seed in raw_seeds]
        if len(seeds) == 1:
            seed_config = deepcopy(config)
            seed_config.setdefault("training", {})["seed"] = seeds[0]
            expanded.append({"name": variant["name"], "base_name": variant["name"], "seed": seeds[0], "config": seed_config})
            continue
        for seed in seeds:
            seed_config = deepcopy(config)
            seed_config.setdefault("training", {})["seed"] = seed
            expanded.append(
                {
                    "name": f"{variant['name']}_seed{seed}",
                    "base_name": variant["name"],
                    "seed": seed,
                    "config": seed_config,
                }
            )
    return expanded


def write_variant_config(config: dict[str, Any], directory: Path, name: str) -> Path:
    config_path = directory / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def run_sweep(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run experiment variants from a sweep YAML file.")
    parser.add_argument("--config", required=True, help="Sweep YAML containing sweep.defaults and sweep.variants")
    parser.add_argument("--steps", type=int, default=None, help="Override training.steps for every variant")
    parser.add_argument("--device", type=str, default=None, help="Override training.device for every variant")
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many selected variants")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated variant names to run")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds; overrides training.seeds")
    parser.add_argument("--output-dir", type=str, default="runs/sweep", help="Directory for metrics and summary JSON")
    parser.add_argument("--smoke", action="store_true", help="Use train smoke settings for quick validation")
    args = parser.parse_args(argv)

    sweep_name, variants = load_sweep(args.config)
    selected = select_variants(variants, args.variants, args.limit)
    selected = expand_seed_variants(selected, parse_seeds(args.seeds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "sweep": sweep_name,
        "config": str(Path(args.config)),
        "output_dir": str(output_dir),
        "started_at": time.time(),
        "variants": [],
    }

    with tempfile.TemporaryDirectory(prefix="flower-sweep-") as tmp:
        tmpdir = Path(tmp)
        for variant in selected:
            name = variant["name"]
            config = deepcopy(variant["config"])
            config.setdefault("training", {})["metrics_json"] = str(output_dir / f"{name}.metrics.json")
            config["training"].setdefault("log_backend", "tensorboard")
            config["training"]["output_dir"] = str(output_dir / "variants" / name)
            if args.steps is not None:
                config["training"]["steps"] = args.steps
            if args.device is not None:
                config["training"]["device"] = args.device

            variant_config_path = write_variant_config(config, tmpdir, name)
            train_args = ["--config", str(variant_config_path)]
            if args.steps is not None:
                train_args.extend(["--steps", str(args.steps)])
            if args.device is not None:
                train_args.extend(["--device", args.device])
            if args.smoke:
                train_args.append("--smoke")
            try:
                metrics = train(train_args)
                summary["variants"].append(
                    {
                        "name": name,
                        "base_name": variant.get("base_name", name),
                        "seed": variant.get("seed", config["training"].get("seed")),
                        "metrics_json": config["training"]["metrics_json"],
                        "metrics": metrics,
                        "status": "completed",
                    }
                )
            except Exception as exc:
                print(f"[sweep] VARIANT {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
                summary["variants"].append(
                    {
                        "name": name,
                        "base_name": variant.get("base_name", name),
                        "seed": variant.get("seed", config["training"].get("seed")),
                        "metrics_json": config["training"]["metrics_json"],
                        "metrics": {},
                        "status": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary["finished_at"] = time.time()
    summary["variant_count"] = len(summary["variants"])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run_sweep()
