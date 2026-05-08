from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from flower.config import load_config
from flower.data import token_batches
from flower.models import build_model

UNUSED_PARAMETER_VARIANTS = {"fa_sm", "fa_fm"}

def resolve_find_unused_parameters(value: str, variant: str) -> bool:
    if value == "auto":
        return variant in UNUSED_PARAMETER_VARIANTS
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("find_unused_parameters must be one of: auto, true, false")


def _distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _aggregate_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu())


def _aggregate_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return float(tensor.cpu())


def run_benchmark(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = argparse.ArgumentParser(description="Synthetic DDP throughput benchmark")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--per-gpu-batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--find-unused-parameters", choices=["auto", "true", "false"], default="auto")
    args = parser.parse_args(argv)

    rank, local_rank, world_size = _distributed_env()
    use_ddp = world_size > 1
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when --device cuda is used")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device(args.device)
        backend = "gloo"

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    overrides = {
        "model": {"variant": args.variant},
        "data": {"dataset": "synthetic"},
        "training": {"batch_size": args.per_gpu_batch_size, "log_backend": "none"},
    }
    cfg = load_config(args.config, overrides)
    find_unused_parameters = resolve_find_unused_parameters(args.find_unused_parameters, cfg.model.variant)
    if cfg.training.find_unused_parameters is not None and args.find_unused_parameters == "auto":
        find_unused_parameters = bool(cfg.training.find_unused_parameters)
    model = build_model(cfg.model).to(device)
    if use_ddp:
        ddp_kwargs: dict[str, Any] = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, find_unused_parameters=find_unused_parameters, **ddp_kwargs)

    optim = AdamW(model.parameters(), lr=cfg.training.lr)
    batches = token_batches(cfg.data, args.per_gpu_batch_size, device)
    total_steps = args.warmup_steps + args.steps
    measured_tokens = 0
    last_loss = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    start = 0.0
    for step in range(1, total_steps + 1):
        input_ids = next(batches)
        optim.zero_grad(set_to_none=True)
        out = model(input_ids, labels=input_ids)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("loss was not computed")
        loss.backward()
        trainable_parameters = model.module.parameters() if isinstance(model, DistributedDataParallel) else model.parameters()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, cfg.training.grad_clip)
        optim.step()
        last_loss = float(loss.detach().cpu())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step == args.warmup_steps:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
        elif step > args.warmup_steps:
            measured_tokens += input_ids.numel() * world_size

    elapsed = max(time.perf_counter() - start, 1e-9)
    local_tokens_per_sec = measured_tokens / elapsed
    tokens_per_sec = _aggregate_max(local_tokens_per_sec, device)
    max_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    metrics: dict[str, Any] = {
        "variant": cfg.model.variant,
        "per_gpu_batch_size": args.per_gpu_batch_size,
        "global_batch_size": args.per_gpu_batch_size * world_size,
        "world_size": world_size,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "find_unused_parameters": find_unused_parameters,
        "tokens_per_sec": tokens_per_sec,
        "max_gpu_memory_allocated": int(_aggregate_max(float(max_mem), device)),
        "loss": _aggregate_mean(last_loss, device),
    }

    if rank == 0:
        if args.metrics_json:
            metrics_path = Path(args.metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(json.dumps(metrics, indent=2, sort_keys=True))
    if use_ddp:
        dist.destroy_process_group()
    return metrics if rank == 0 else None


if __name__ == "__main__":
    run_benchmark()
