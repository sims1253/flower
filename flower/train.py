from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import torch

from flower.config import load_config
from flower.data import token_batches, validation_token_batches
from flower.models import build_model
from flower.models.base import count_parameters
from flower.optim import build_optimizer


class ScalarLogger(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class NoOpSummaryWriter:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "events.out.tfevents.fallback").touch()

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def build_summary_writer(log_dir: Path) -> ScalarLogger:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        return NoOpSummaryWriter(log_dir)

    return SummaryWriter(log_dir=str(log_dir))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: torch.nn.Module, batches, steps: int) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(steps):
            input_ids = next(batches)
            out = model(input_ids, labels=input_ids)
            loss = out["loss"]
            if loss is None:
                raise RuntimeError("loss was not computed")
            total_loss += float(loss.detach().cpu()) * input_ids.numel()
            total_tokens += input_ids.numel()
    if was_training:
        model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "val_loss": mean_loss,
        "val_perplexity": math.exp(min(mean_loss, 20.0)),
        "val_tokens": total_tokens,
        "val_batches": steps,
    }


def initialize_lr_schedule(optims: list[torch.optim.Optimizer]) -> None:
    for opt in optims:
        for group in opt.param_groups:
            group.setdefault("base_lr", group["lr"])


def lr_multiplier(step: int, warmup_steps: int, schedule: str) -> float:
    if schedule == "constant" or warmup_steps <= 0:
        return 1.0
    if schedule == "linear_warmup":
        return min(1.0, step / max(1, warmup_steps))
    raise ValueError(f"Unknown training.lr_schedule: {schedule}")


def apply_lr_schedule(optims: list[torch.optim.Optimizer], step: int, warmup_steps: int, schedule: str) -> None:
    scale = lr_multiplier(step, warmup_steps, schedule)
    for opt in optims:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * scale


def log_learning_rates(writer: ScalarLogger, optims: list[torch.optim.Optimizer], step: int) -> None:
    for opt_index, opt in enumerate(optims):
        opt_name = opt.__class__.__name__.lower()
        for group_index, group in enumerate(opt.param_groups):
            writer.add_scalar(f"train/lr/optimizer_{opt_index}_group_{group_index}", group["lr"], step)
            writer.add_scalar(f"train/lr/{opt_name}_group_{group_index}", group["lr"], step)


def train(argv: list[str] | None = None) -> dict[str, float | int | str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-backend", choices=["none", "tensorboard"], default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--validation-split", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "muon"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    overrides = {"model": {}, "training": {}, "data": {}}
    if args.variant:
        overrides["model"]["variant"] = args.variant
    if args.steps is not None:
        overrides["training"]["steps"] = args.steps
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        overrides["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.device:
        overrides["training"]["device"] = args.device
    if args.metrics_json:
        overrides["training"]["metrics_json"] = args.metrics_json
    if args.output_dir:
        overrides["training"]["output_dir"] = args.output_dir
    if args.log_backend:
        overrides["training"]["log_backend"] = args.log_backend
    if args.validation_steps is not None:
        overrides["training"]["validation_steps"] = args.validation_steps
    if args.validation_interval is not None:
        overrides["training"]["validation_interval"] = args.validation_interval
    if args.validation_split is not None:
        overrides["data"]["validation_split"] = args.validation_split
    if args.optimizer is not None:
        overrides["training"]["optimizer"] = args.optimizer
    if args.seed is not None:
        overrides["training"]["seed"] = args.seed
    if args.smoke:
        overrides["model"].update(
            {
                "vocab_size": 256,
                "d_model": 32,
                "num_heads": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "max_seq_len": 32,
                "local_window": 8,
                "memory_slots": 4,
                "flow_steps": 1,
            }
        )
        overrides["data"].update({"dataset": "synthetic", "sequence_length": 32, "synthetic_vocab_size": 256})
        overrides["training"].setdefault("steps", 2)
        overrides["training"].setdefault("batch_size", 2)

    cfg = load_config(args.config, overrides)
    set_global_seed(int(cfg.training.seed))
    device = resolve_device(cfg.training.device)
    model = build_model(cfg.model).to(device)
    optim_or_list = build_optimizer(model, cfg.training)
    optims = optim_or_list if isinstance(optim_or_list, list) else [optim_or_list]
    initialize_lr_schedule(optims)
    batches = token_batches(cfg.data, cfg.training.batch_size, device)
    val_batches = (
        validation_token_batches(cfg.data, cfg.training.batch_size, device)
        if cfg.training.validation_steps > 0
        else None
    )
    output_dir = Path(cfg.training.output_dir)
    writer: ScalarLogger | None = None
    if cfg.training.log_backend == "tensorboard":
        writer = build_summary_writer(output_dir / "tensorboard")
    elif cfg.training.log_backend not in {"none", ""}:
        raise ValueError(f"Unknown training.log_backend: {cfg.training.log_backend}")

    # Resume from latest checkpoint if one exists in `output_dir`. Survives bid
    # displacement: the disk persists when a vast bid instance is stopped (not
    # destroyed), so re-launching the same trial picks up where it left off.
    resume_step = 0
    if cfg.training.save_checkpoints and not args.smoke and output_dir.exists():
        ckpts = sorted(
            output_dir.glob(f"{cfg.model.variant}_step*.pt"),
            key=lambda p: int(p.stem.split("step")[-1]),
        )
        if ckpts:
            latest = ckpts[-1]
            print(f"[resume] loading checkpoint {latest}")
            payload = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            for opt, opt_state in zip(optims, payload.get("optimizers", []), strict=False):
                opt.load_state_dict(opt_state)
            resume_step = int(payload.get("step", 0))
            if "rng_state" in payload:
                torch.set_rng_state(payload["rng_state"].cpu())
            if "cuda_rng_state" in payload and device.type == "cuda":
                torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu())
            print(f"[resume] resuming from step {resume_step + 1} (checkpoint at step {resume_step})")
    initialize_lr_schedule(optims)

    start = time.perf_counter()
    last_log_time = start
    last_log_tokens = 0
    log_interval = max(1, min(cfg.training.eval_interval, cfg.training.steps))
    last_loss = 0.0
    tokens = 0

    model.train()
    try:
        for step in range(resume_step + 1, cfg.training.steps + 1):
            accum_steps = max(1, int(cfg.training.gradient_accumulation_steps))
            for opt in optims:
                opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accum_steps):
                input_ids = next(batches)
                out = model(input_ids, labels=input_ids)
                loss = out["loss"]
                if loss is None:
                    raise RuntimeError("loss was not computed")
                (loss / accum_steps).backward()
                step_loss += float(loss.detach().cpu())
                tokens += input_ids.numel()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            apply_lr_schedule(optims, step, cfg.training.warmup_steps, cfg.training.lr_schedule)
            for opt in optims:
                opt.step()
            last_loss = step_loss / accum_steps
            should_log = step == 1 or step % log_interval == 0 or step == cfg.training.steps
            if writer is not None and should_log:
                now = time.perf_counter()
                interval_elapsed = max(now - last_log_time, 1e-9)
                tokens_per_sec = (tokens - last_log_tokens) / interval_elapsed
                writer.add_scalar("train/loss", last_loss, step)
                writer.add_scalar("train/perplexity", math.exp(min(last_loss, 20.0)), step)
                writer.add_scalar("train/lr", optims[0].param_groups[0]["lr"], step)
                log_learning_rates(writer, optims, step)
                writer.add_scalar("throughput/tokens_per_sec", tokens_per_sec, step)
                writer.flush()
                last_log_time = now
                last_log_tokens = tokens
            if (
                val_batches is not None
                and cfg.training.validation_interval > 0
                and step % cfg.training.validation_interval == 0
            ):
                val_metrics = evaluate(model, val_batches, cfg.training.validation_steps)
                if writer is not None:
                    writer.add_scalar("validation/loss", float(val_metrics["val_loss"]), step)
                    writer.add_scalar("validation/perplexity", float(val_metrics["val_perplexity"]), step)
                    writer.flush()
            if cfg.training.save_checkpoints and step % cfg.training.checkpoint_interval == 0 and not args.smoke:
                output_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "step": step,
                    "config": asdict(cfg),
                    "model": model.state_dict(),
                    "optimizers": [opt.state_dict() for opt in optims],
                    "rng_state": torch.get_rng_state(),
                }
                if device.type == "cuda":
                    payload["cuda_rng_state"] = torch.cuda.get_rng_state()
                ckpt_path = output_dir / f"{cfg.model.variant}_step{step}.pt"
                # Atomic write: tmp file then rename, so a kill mid-write doesn't corrupt.
                tmp_path = ckpt_path.with_suffix(".pt.tmp")
                torch.save(payload, tmp_path)
                tmp_path.rename(ckpt_path)
                # Keep only the latest 2 checkpoints to bound disk usage.
                old_ckpts = sorted(
                    output_dir.glob(f"{cfg.model.variant}_step*.pt"),
                    key=lambda p: int(p.stem.split("step")[-1]),
                )
                for old in old_ckpts[:-2]:
                    old.unlink(missing_ok=True)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    elapsed = max(time.perf_counter() - start, 1e-9)
    metrics = {
        "variant": cfg.model.variant,
        "steps": cfg.training.steps,
        "loss": last_loss,
        "train_loss": last_loss,
        "perplexity": float(torch.exp(torch.tensor(last_loss)).item()),
        "train_perplexity": float(torch.exp(torch.tensor(last_loss)).item()),
        "tokens_per_sec": tokens / elapsed,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "seed": int(cfg.training.seed),
        "gpu_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    if val_batches is not None:
        metrics.update(evaluate(model, val_batches, cfg.training.validation_steps))
        metrics["validation_split"] = (
            cfg.data.validation_split if cfg.data.dataset != "synthetic" else "synthetic_validation"
        )
    if cfg.training.composite_eval:
        from flower.probes.composite import run_composite_eval

        composite_path = (
            Path(cfg.training.composite_eval_json)
            if cfg.training.composite_eval_json
            else Path(cfg.training.output_dir) / "composite_ranker.json"
        )
        composite = run_composite_eval(model, cfg, device=device)
        composite_path.parent.mkdir(parents=True, exist_ok=True)
        composite_path.write_text(json.dumps(composite, indent=2, sort_keys=True))
        metrics["composite_ranker_json"] = str(composite_path)
    if cfg.training.metrics_json:
        metrics_path = Path(cfg.training.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    train()
