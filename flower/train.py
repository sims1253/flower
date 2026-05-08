from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Protocol

import torch
from torch.optim import AdamW

from flower.config import load_config
from flower.data import token_batches, validation_token_batches
from flower.models import build_model
from flower.models.base import count_parameters


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
    return {"val_loss": mean_loss, "val_perplexity": math.exp(min(mean_loss, 20.0)), "val_tokens": total_tokens, "val_batches": steps}

def train(argv: list[str] | None = None) -> dict[str, float | int | str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-backend", choices=["none", "tensorboard"], default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--validation-split", type=str, default=None)
    args = parser.parse_args(argv)

    overrides = {"model": {}, "training": {}, "data": {}}
    if args.variant:
        overrides["model"]["variant"] = args.variant
    if args.steps is not None:
        overrides["training"]["steps"] = args.steps
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
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
    if args.smoke:
        overrides["model"].update({"vocab_size": 256, "d_model": 32, "num_heads": 4, "num_layers": 1, "ffn_dim": 64, "max_seq_len": 32, "local_window": 8, "memory_slots": 4, "flow_steps": 1})
        overrides["data"].update({"dataset": "synthetic", "sequence_length": 32, "synthetic_vocab_size": 256})
        overrides["training"].setdefault("steps", 2)
        overrides["training"].setdefault("batch_size", 2)

    cfg = load_config(args.config, overrides)
    device = resolve_device(cfg.training.device)
    model = build_model(cfg.model).to(device)
    optim = AdamW(model.parameters(), lr=cfg.training.lr)
    batches = token_batches(cfg.data, cfg.training.batch_size, device)
    val_batches = validation_token_batches(cfg.data, cfg.training.batch_size, device) if cfg.training.validation_steps > 0 else None
    output_dir = Path(cfg.training.output_dir)
    writer: ScalarLogger | None = None
    if cfg.training.log_backend == "tensorboard":
        writer = build_summary_writer(output_dir / "tensorboard")
    elif cfg.training.log_backend not in {"none", ""}:
        raise ValueError(f"Unknown training.log_backend: {cfg.training.log_backend}")

    start = time.perf_counter()
    last_log_time = start
    last_log_tokens = 0
    log_interval = max(1, min(cfg.training.eval_interval, cfg.training.steps))
    last_loss = 0.0
    tokens = 0

    model.train()
    try:
        for step in range(1, cfg.training.steps + 1):
            input_ids = next(batches)
            optim.zero_grad(set_to_none=True)
            out = model(input_ids, labels=input_ids)
            loss = out["loss"]
            if loss is None:
                raise RuntimeError("loss was not computed")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optim.step()
            last_loss = float(loss.detach().cpu())
            tokens += input_ids.numel()
            should_log = step == 1 or step % log_interval == 0 or step == cfg.training.steps
            if writer is not None and should_log:
                now = time.perf_counter()
                interval_elapsed = max(now - last_log_time, 1e-9)
                tokens_per_sec = (tokens - last_log_tokens) / interval_elapsed
                writer.add_scalar("train/loss", last_loss, step)
                writer.add_scalar("train/perplexity", math.exp(min(last_loss, 20.0)), step)
                writer.add_scalar("train/lr", optim.param_groups[0]["lr"], step)
                writer.add_scalar("throughput/tokens_per_sec", tokens_per_sec, step)
                writer.flush()
                last_log_time = now
                last_log_tokens = tokens
            if val_batches is not None and cfg.training.validation_interval > 0 and step % cfg.training.validation_interval == 0:
                val_metrics = evaluate(model, val_batches, cfg.training.validation_steps)
                if writer is not None:
                    writer.add_scalar("validation/loss", float(val_metrics["val_loss"]), step)
                    writer.add_scalar("validation/perplexity", float(val_metrics["val_perplexity"]), step)
                    writer.flush()
            if step % cfg.training.checkpoint_interval == 0 and not args.smoke:
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_dir / f"{cfg.model.variant}_step{step}.pt")
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
        "gpu_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    if val_batches is not None:
        metrics.update(evaluate(model, val_batches, cfg.training.validation_steps))
        metrics["validation_split"] = cfg.data.validation_split if cfg.data.dataset != "synthetic" else "synthetic_validation"
    if cfg.training.metrics_json:
        metrics_path = Path(cfg.training.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    train()
