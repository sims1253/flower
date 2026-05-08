from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from flower.config import load_config
from flower.data import token_batches
from flower.models import build_model
from flower.models.base import count_parameters
from flower.train import resolve_device


@torch.no_grad()
def evaluate(argv: list[str] | None = None) -> dict[str, float | int | str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--metrics-json", type=str, default=None)
    args = parser.parse_args(argv)

    overrides = {"model": {}, "training": {"device": args.device}, "data": {}}
    if args.variant:
        overrides["model"]["variant"] = args.variant
    if args.smoke:
        overrides["model"].update({"vocab_size": 256, "d_model": 32, "num_heads": 4, "num_layers": 1, "ffn_dim": 64, "max_seq_len": 32, "local_window": 8, "memory_slots": 4, "flow_steps": 1})
        overrides["data"].update({"dataset": "synthetic", "sequence_length": 32, "synthetic_vocab_size": 256})
        overrides["training"]["batch_size"] = 2

    if args.metrics_json:
        overrides["training"]["metrics_json"] = args.metrics_json
    cfg = load_config(args.config, overrides)
    device = resolve_device(cfg.training.device)
    model = build_model(cfg.model).to(device).eval()
    batches = token_batches(cfg.data, cfg.training.batch_size, device)
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()
    for _ in range(args.batches):
        batch = next(batches)
        out = model(batch, labels=batch)
        total_loss += float(out["loss"].cpu())
        total_tokens += batch.numel()
    loss = total_loss / args.batches
    metrics = {
        "variant": cfg.model.variant,
        "loss": loss,
        "perplexity": math.exp(loss),
        "tokens_per_sec": total_tokens / max(time.perf_counter() - start, 1e-9),
        "parameter_count": count_parameters(model),
        "device": str(device),
        "gpu_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    if cfg.training.metrics_json:
        metrics_path = Path(cfg.training.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    evaluate()
