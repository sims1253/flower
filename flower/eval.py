from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from flower.config import ExperimentConfig, load_config
from flower.data import build_tokenizer, fineweb_validation_documents, token_batches
from flower.models import build_model
from flower.models.base import count_parameters
from flower.train import resolve_device, set_global_seed


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _checkpoint_config(path: Path) -> dict[str, Any] | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config")
    return config if isinstance(config, dict) else None


def _model_has_sdp_bias(model: torch.nn.Module) -> bool | None:
    """Whether the model's SDPCrossAttention modules have bias, for the MHA
    state-dict remap. Returns None if the model has no such module (non-summary /
    non-bloom), which the remap treats as "keep bias" (the MHA default)."""
    from flower.models.memory import SDPCrossAttention

    for module in model.modules():
        if isinstance(module, SDPCrossAttention):
            return module.q_proj.bias is not None
    return None


def _load_checkpoint_model(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> int | None:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    state = payload.get("model", payload)
    # S14 Opportunity 2 Part A: bloom_memory's K-hash ModuleList became a single
    # `hash_weights` Parameter. Remap legacy `hashes.{i}.weight` checkpoints so
    # old artifacts (sweep5/7/13 bloom runs) still load. No-op for new-format /
    # non-bloom state_dicts.
    from flower.models.bloom_memory import remap_legacy_bloom_state_dict
    from flower.models.memory import remap_legacy_mha_state_dict

    state = remap_legacy_bloom_state_dict(state)
    # S14 Opportunity: summary_memory / bloom_memory replaced their
    # nn.MultiheadAttention perceiver with a compile-clean SDPCrossAttention.
    # Remap legacy in_proj_*/out_proj.* MHA keys to the new q/k/v/out_proj
    # projection layout. `bias` is taken from the target model's SDPCrossAttention
    # modules (nn.MultiheadAttention always had bias, but SDPCrossAttention
    # respects config.use_bias, so a use_bias=False checkpoint must drop them).
    # No-op for new-format / non-summary / non-bloom state_dicts.
    bias = _model_has_sdp_bias(model)
    state = remap_legacy_mha_state_dict(state, bias=bias)
    model.load_state_dict(state)
    step = payload.get("step")
    return int(step) if step is not None else None


def _bootstrap_ratio_ci(
    numerators: list[float],
    denominators: list[float],
    *,
    scale: float = 1.0,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float] | None:
    pairs = [(n, d) for n, d in zip(numerators, denominators, strict=False) if d > 0]
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        n_total = 0.0
        d_total = 0.0
        for _ in pairs:
            n, d = pairs[rng.randrange(len(pairs))]
            n_total += n
            d_total += d
        values.append(scale * n_total / max(d_total, 1e-12))
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return lo, hi


@torch.no_grad()
def evaluate_batches(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    batches_count: int,
) -> dict[str, float | int]:
    batches = token_batches(cfg.data, cfg.training.batch_size, device, seed=int(cfg.training.seed))
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()
    for _ in range(batches_count):
        batch = next(batches)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            input_ids, labels = batch
        else:
            input_ids, labels = batch, batch
        out = model(input_ids, labels=labels)
        total_loss += float(out["loss"].cpu())
        total_tokens += int((labels[:, 1:] != -100).sum().cpu())
    loss = total_loss / max(batches_count, 1)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "tokens_per_sec": total_tokens / max(time.perf_counter() - start, 1e-9),
        "eval_tokens": total_tokens,
        "eval_batches": batches_count,
    }


@torch.no_grad()
def evaluate_documents(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    doc_limit: int | None = None,
    bootstrap: bool = False,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    if cfg.data.dataset in {"synthetic", "mqar"}:
        batch_metrics = evaluate_batches(model, cfg, device, batches_count=max(1, doc_limit or 10))
        loss = float(batch_metrics["loss"])
        bpb = loss / math.log(2.0)
        metrics: dict[str, Any] = {
            **batch_metrics,
            "bpb": bpb,
            "raw_bytes": int(batch_metrics["eval_tokens"]),
            "validation_docs": int(batch_metrics["eval_batches"]),
        }
        if bootstrap:
            metrics["loss_ci95"] = [loss, loss]
            metrics["bpb_ci95"] = [bpb, bpb]
        return metrics

    encoder = build_tokenizer(cfg.data.tokenizer)
    eval_seq_len = getattr(cfg.data, "eval_seq_len", None)
    max_len = eval_seq_len or cfg.data.sequence_length or cfg.model.max_seq_len
    doc_nlls: list[float] = []
    doc_pred_tokens: list[float] = []
    doc_bytes: list[float] = []
    start = time.perf_counter()

    for text in fineweb_validation_documents(cfg.data, limit=doc_limit):
        raw_bytes = len(text.encode("utf-8", errors="replace"))
        token_ids = encoder.encode(text)
        if len(token_ids) < 2 or raw_bytes <= 0:
            continue
        nll = 0.0
        pred_tokens = 0
        for offset in range(0, len(token_ids), max_len):
            chunk = token_ids[offset : offset + max_len]
            if len(chunk) < 2:
                continue
            batch = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
            out = model(batch, labels=batch)
            loss = out["loss"]
            if loss is None:
                raise RuntimeError("loss was not computed")
            count = batch.shape[1] - 1
            nll += float(loss.detach().cpu()) * count
            pred_tokens += count
        if pred_tokens > 0:
            doc_nlls.append(nll)
            doc_pred_tokens.append(float(pred_tokens))
            doc_bytes.append(float(raw_bytes))

    total_nll = sum(doc_nlls)
    total_pred_tokens = sum(doc_pred_tokens)
    total_bytes = sum(doc_bytes)
    loss = total_nll / max(total_pred_tokens, 1.0)
    bpb = total_nll / math.log(2.0) / max(total_bytes, 1.0)
    metrics = {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "bpb": bpb,
        "raw_bytes": int(total_bytes),
        "eval_tokens": int(total_pred_tokens),
        "validation_docs": len(doc_nlls),
        "tokens_per_sec": total_pred_tokens / max(time.perf_counter() - start, 1e-9),
    }
    if bootstrap:
        loss_ci = _bootstrap_ratio_ci(
            doc_nlls,
            doc_pred_tokens,
            resamples=bootstrap_samples,
            seed=int(cfg.training.seed),
        )
        bpb_ci = _bootstrap_ratio_ci(
            doc_nlls,
            doc_bytes,
            scale=1.0 / math.log(2.0),
            resamples=bootstrap_samples,
            seed=int(cfg.training.seed),
        )
        if loss_ci is not None:
            metrics["loss_ci95"] = list(loss_ci)
            metrics["perplexity_ci95"] = [math.exp(min(v, 20.0)) for v in loss_ci]
        if bpb_ci is not None:
            metrics["bpb_ci95"] = list(bpb_ci)
    return metrics


@torch.no_grad()
def sliding_window_loss(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    window_size: int,
    stride: int,
    device: torch.device,
) -> float:
    """Mean per-token loss with overlapping (sliding) windows.

    Each window of `window_size` tokens is run through the model; the loss
    for tokens in the window is accumulated. Overlapping windows let every
    token be scored with near-full backward context, giving a more accurate
    bits-per-byte measurement than non-overlapping chunks. Use for final
    evaluations, not during-training monitoring (more forward passes).
    """
    del device  # token_ids already lives on device; kept for API symmetry.
    T = token_ids.numel()
    total_loss = 0.0
    total_tokens = 0
    # If the sequence is shorter than a full window, score it as a single
    # window of size T (CE then predicts T-1 tokens).
    effective_window = min(window_size, T)
    for start in range(0, max(1, T - window_size + 1), stride):
        window = token_ids[start : start + effective_window].view(1, effective_window)
        out = model(window, labels=window)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("loss was not computed")
        count = effective_window - 1
        total_loss += float(loss.detach().cpu()) * count
        total_tokens += count
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def sliding_window_document_loss(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    window_size: int,
    stride: int,
    doc_limit: int | None = None,
) -> dict[str, Any]:
    """Sliding-window evaluation mirroring `evaluate_documents`'s shape.

    Uses overlapping windows (stride < window_size) so every token is scored
    with near-full backward context. Returns a point estimate only (no
    bootstrap CI), keeping it cheaper to reason about than the chunk path.
    """
    if cfg.data.dataset in {"synthetic", "mqar"}:
        batch_metrics = evaluate_batches(model, cfg, device, batches_count=max(1, doc_limit or 10))
        loss = float(batch_metrics["loss"])
        bpb = loss / math.log(2.0)
        return {
            **batch_metrics,
            "bpb": bpb,
            "raw_bytes": int(batch_metrics["eval_tokens"]),
            "validation_docs": int(batch_metrics["eval_batches"]),
        }

    encoder = build_tokenizer(cfg.data.tokenizer)
    doc_nlls: list[float] = []
    doc_pred_tokens: list[float] = []
    doc_bytes: list[float] = []
    start = time.perf_counter()

    for text in fineweb_validation_documents(cfg.data, limit=doc_limit):
        raw_bytes = len(text.encode("utf-8", errors="replace"))
        token_ids = encoder.encode(text)
        if len(token_ids) < 2 or raw_bytes <= 0:
            continue
        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        nll = sliding_window_loss(
            model, ids_tensor, window_size=window_size, stride=stride, device=device
        ) * max(len(token_ids) - 1, 0)
        pred_tokens = max(len(token_ids) - 1, 0)
        if pred_tokens > 0:
            doc_nlls.append(nll)
            doc_pred_tokens.append(float(pred_tokens))
            doc_bytes.append(float(raw_bytes))

    total_nll = sum(doc_nlls)
    total_pred_tokens = sum(doc_pred_tokens)
    total_bytes = sum(doc_bytes)
    loss = total_nll / max(total_pred_tokens, 1.0)
    bpb = total_nll / math.log(2.0) / max(total_bytes, 1.0)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "bpb": bpb,
        "raw_bytes": int(total_bytes),
        "eval_tokens": int(total_pred_tokens),
        "validation_docs": len(doc_nlls),
        "tokens_per_sec": total_pred_tokens / max(time.perf_counter() - start, 1e-9),
    }


@torch.no_grad()
def evaluate(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--metric", choices=["loss", "ppl", "bpb", "all"], default="all")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--doc-limit", type=int, default=None)
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--composite", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument(
        "--eval-mode",
        choices=["chunk", "sliding"],
        default="chunk",
        help="chunk: non-overlapping document chunks (default, reproduces prior runs). "
        "sliding: overlapping windows so every token gets near-full backward context.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride for --eval-mode sliding. Defaults to 64 (or window_size//4).",
    )
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    overrides: dict[str, Any] = {"model": {}, "training": {"device": args.device}, "data": {}}
    if args.variant:
        overrides["model"]["variant"] = args.variant
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
        overrides["training"]["batch_size"] = 2

    if args.metrics_json:
        overrides["training"]["metrics_json"] = args.metrics_json

    if args.config is None and checkpoint is not None:
        checkpoint_cfg = _checkpoint_config(checkpoint)
        if checkpoint_cfg is None:
            raise ValueError("--config is required for checkpoints that do not contain an embedded config")
        merged = _deep_merge(checkpoint_cfg, overrides)
        cfg = load_config(None, merged)
    else:
        cfg = load_config(args.config, overrides)

    set_global_seed(int(cfg.training.seed))
    device = resolve_device(cfg.training.device)
    model = build_model(cfg.model).to(device).eval()
    checkpoint_step = None
    if checkpoint is not None:
        checkpoint_step = _load_checkpoint_model(model, checkpoint, device)

    use_doc_eval = args.metric in {"bpb", "all"} or args.ci
    if args.eval_mode == "sliding":
        eval_seq_len = getattr(cfg.data, "eval_seq_len", None)
        window_size = eval_seq_len or cfg.data.sequence_length or cfg.model.max_seq_len
        stride = args.stride or 64
        metrics = sliding_window_document_loss(
            model,
            cfg,
            device,
            window_size=window_size,
            stride=stride,
            doc_limit=args.doc_limit,
        )
    else:
        metrics = (
            evaluate_documents(
                model,
                cfg,
                device,
                doc_limit=args.doc_limit,
                bootstrap=args.ci,
                bootstrap_samples=args.bootstrap_samples,
            )
            if use_doc_eval
            else evaluate_batches(model, cfg, device, batches_count=args.batches)
        )
    metrics.update(
        {
            "variant": cfg.model.variant,
            "parameter_count": count_parameters(model),
            "device": str(device),
            "seed": int(cfg.training.seed),
            "gpu_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
    )
    if checkpoint is not None:
        metrics["checkpoint"] = str(checkpoint)
    if checkpoint_step is not None:
        metrics["checkpoint_step"] = checkpoint_step

    if args.composite:
        from flower.probes.composite import run_composite_eval

        metrics["composite_ranker"] = run_composite_eval(model, cfg, device=device, doc_limit=args.doc_limit)

    if cfg.training.metrics_json:
        metrics_path = Path(cfg.training.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    evaluate()
