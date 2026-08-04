from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import torch

from flower.config import load_config
from flower.data import token_batches, validation_token_batches
from flower.models import build_model
from flower.models.base import count_parameters
from flower.optim import build_optimizer
from flower.shm_guard import start_shm_watchdog


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    return batch, batch


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


def configure_vram_limit(device: torch.device, fraction: float = 0.85) -> None:
    """Cap the CUDA caching allocator so an oversized batch raises a hard OOM
    instead of silently spilling into host RAM.

    On WSL2 the WDDM memory manager does NOT raise OutOfMemoryError when the
    GPU runs out — it spills into shared host RAM over PCIe and throughput
    collapses by an order of magnitude. A run that looks very slow is usually a
    run that overshot VRAM. Capping the allocator at `fraction` of total VRAM
    forces a real OOM (which the batch-size comments in the configs and the
    bench scripts already size against), so the failure mode is loud. No-op on
    CPU. The default 0.85 leaves headroom for fragmentation and for the small
    non-tensor allocations PyTorch makes outside the caching allocator.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)
    except (RuntimeError, ValueError):
        # Already set (e.g. a parent process configured it) or unsupported —
        # never let the guard itself crash a run.
        pass


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


def configure_precision(precision: str, device: torch.device) -> torch.dtype | None:
    """Apply global precision settings; return the autocast dtype (None = off).

    fp32 leaves everything at PyTorch defaults so pre-sweep-13 runs reproduce.
    tf32/bf16 both enable TF32 matmul kernels; bf16 additionally autocasts the
    forward/backward. bf16 needs no GradScaler — its exponent range matches fp32,
    so gradients do not underflow the way fp16 does. Master weights stay fp32.
    """
    if precision not in {"fp32", "tf32", "bf16"}:
        raise ValueError(f"training.precision must be fp32|tf32|bf16, got {precision!r}")
    if precision == "fp32":
        return None
    torch.set_float32_matmul_precision("high")
    if precision == "tf32":
        return None
    if device.type != "cuda":
        print("[precision] bf16 requested but device is not CUDA; running fp32")
        return None
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("training.precision=bf16 but this GPU does not support bf16")
    return torch.bfloat16


def autocast_ctx(device: torch.device, dtype: torch.dtype | None):
    """Autocast context for `dtype`, or a no-op when precision is fp32/tf32."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def evaluate(
    model: torch.nn.Module,
    batches,
    steps: int,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
    bytes_per_token: float | None = None,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = device or torch.device("cpu")
    with torch.no_grad():
        for _ in range(steps):
            input_ids, labels = unpack_batch(next(batches))
            with autocast_ctx(device, amp_dtype):
                out = model(input_ids, labels=labels)
            loss = out["loss"]
            if loss is None:
                raise RuntimeError("loss was not computed")
            token_count = int((labels[:, 1:] != -100).sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * max(token_count, 1)
            total_tokens += token_count
    if was_training:
        model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    metrics: dict[str, float | int] = {
        "val_loss": mean_loss,
        "val_perplexity": math.exp(min(mean_loss, 20.0)),
        "val_tokens": total_tokens,
        "val_batches": steps,
    }
    if bytes_per_token:
        # Perplexity is tokenizer-dependent; bits-per-byte is not. Emit both so
        # runs on different tokenizers stay comparable.
        metrics["val_bpb"] = mean_loss / math.log(2.0) / bytes_per_token
    return metrics


def initialize_lr_schedule(optims: list[torch.optim.Optimizer]) -> None:
    for opt in optims:
        for group in opt.param_groups:
            group.setdefault("base_lr", group["lr"])


def lr_multiplier(
    step: int,
    warmup_steps: int,
    schedule: str,
    total_steps: int = 0,
    decay_frac: float = 0.2,
    final_frac: float = 0.0,
) -> float:
    """LR multiplier at `step` (1-indexed).

    `constant` and `linear_warmup` are the legacy schedules — note that
    linear_warmup never decays, it holds the peak LR to the last step.
    `wsd` warms up, holds, then decays linearly to `final_frac` over the last
    `decay_frac` of training. `cosine` decays from the end of warmup to
    `final_frac` at the final step.
    """
    if schedule not in {"constant", "linear_warmup", "wsd", "cosine"}:
        raise ValueError(f"Unknown training.lr_schedule: {schedule}")
    if schedule == "constant":
        return 1.0
    warm = 1.0 if warmup_steps <= 0 else min(1.0, step / max(1, warmup_steps))
    if schedule == "linear_warmup":
        return warm
    if total_steps <= 0:
        return warm
    if schedule == "wsd":
        decay_steps = max(1, int(round(total_steps * decay_frac)))
        decay_start = total_steps - decay_steps
        if step <= decay_start:
            return warm
        progress = min(1.0, (step - decay_start) / decay_steps)
        return warm * (1.0 - progress * (1.0 - final_frac))
    # cosine: decay across the post-warmup span.
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return warm * (final_frac + (1.0 - final_frac) * cosine)


def apply_lr_schedule(
    optims: list[torch.optim.Optimizer],
    step: int,
    warmup_steps: int,
    schedule: str,
    total_steps: int = 0,
    decay_frac: float = 0.2,
    final_frac: float = 0.0,
) -> None:
    scale = lr_multiplier(step, warmup_steps, schedule, total_steps, decay_frac, final_frac)
    for opt in optims:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * scale


def log_learning_rates(writer: ScalarLogger, optims: list[torch.optim.Optimizer], step: int) -> None:
    for opt_index, opt in enumerate(optims):
        opt_name = opt.__class__.__name__.lower()
        for group_index, group in enumerate(opt.param_groups):
            writer.add_scalar(f"train/lr/{opt_name}_{opt_index}_group_{group_index}", group["lr"], step)


def update_attention_windows(model: torch.nn.Module, step: int, cfg) -> None:
    """Expand attention windows over training (S2, window warmup).

    Linearly ramps every attention module's local_window from
    cfg.model.attn_warmup_start to cfg.model.local_window over
    cfg.model.attn_warmup_steps steps, then holds at local_window.
    No-op when attn_warmup_steps == 0 (the default: use local_window always).
    """
    if getattr(cfg.model, "attn_warmup_steps", 0) == 0:
        return
    target = cfg.model.local_window
    if step >= cfg.model.attn_warmup_steps:
        target = cfg.model.local_window
    else:
        frac = step / max(1, cfg.model.attn_warmup_steps)
        target = int(round(cfg.model.attn_warmup_start + frac * (cfg.model.local_window - cfg.model.attn_warmup_start)))
    for module in model.modules():
        lw = getattr(module, "local_window", None)
        if lw is not None and lw != target:
            module.local_window = target
            # Invalidate the FlexAttention block-mask cache (S1).
            if hasattr(module, "_cached_block_mask"):
                module._cached_block_mask = None


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
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "muon", "aurora"])
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
        overrides["data"].update(
            {"dataset": "synthetic", "sequence_length": 32, "synthetic_vocab_size": 256, "eval_seq_len": 32}
        )
        overrides["training"].setdefault("steps", 2)
        overrides["training"].setdefault("batch_size", 2)

    cfg = load_config(args.config, overrides)
    set_global_seed(int(cfg.training.seed))
    device = resolve_device(cfg.training.device)
    configure_vram_limit(device)
    amp_dtype = configure_precision(cfg.training.precision, device)
    model = build_model(cfg.model).to(device)
    # Optimizers are always built on the eager module: torch.compile returns a
    # wrapper whose .parameters() are the same objects, but keeping the eager
    # reference makes state_dict keys stable across compiled/uncompiled runs.
    optim_or_list = build_optimizer(model, cfg.training)
    eager_model = model
    if cfg.training.compile_model:
        # The diagnostics walk uses dir()/getattr over every submodule, which
        # Dynamo cannot trace; leaving it on would graph-break every forward.
        # Variant-specific scalars are unavailable under compile as a result.
        if getattr(eager_model, "collect_module_diagnostics", False):
            eager_model.collect_module_diagnostics = False
            print("[compile] module diagnostics disabled (untraceable by Dynamo)")
        model = torch.compile(model, mode=cfg.training.compile_mode, dynamic=False)
    optims = optim_or_list if isinstance(optim_or_list, list) else [optim_or_list]
    # S12.4: EMA weight averaging for evaluation.
    ema_model: torch.nn.Module | None = None
    if getattr(cfg.training, "ema_decay", 0.0) > 0:
        import copy

        ema_model = copy.deepcopy(eager_model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    initialize_lr_schedule(optims)
    batches = token_batches(cfg.data, cfg.training.batch_size, device, seed=int(cfg.training.seed))
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
            payload = torch.load(latest, map_location=device, weights_only=True)
            # S14 Opportunity 2 Part A: bloom_memory's K-hash ModuleList became a
            # single `hash_weights` Parameter. Remap legacy checkpoints so an
            # in-flight bloom run resumed after the upgrade keeps its learned
            # hashes. No-op for new-format / non-bloom state_dicts.
            from flower.models.bloom_memory import remap_legacy_bloom_state_dict
            from flower.models.memory import remap_legacy_mha_state_dict

            state = remap_legacy_bloom_state_dict(payload["model"])
            # S14 Opportunity: summary_memory / bloom_memory replaced their
            # nn.MultiheadAttention perceiver with a compile-clean
            # SDPCrossAttention. Remap legacy MHA in_proj_*/out_proj.* keys to
            # the new q/k/v/out_proj layout. `bias` comes from the config:
            # nn.MultiheadAttention always had bias, but SDPCrossAttention
            # respects use_bias, so a use_bias=False checkpoint must drop them.
            # No-op for new-format state_dicts.
            state = remap_legacy_mha_state_dict(state, bias=getattr(cfg.model, "use_bias", True))
            eager_model.load_state_dict(state)
            for opt, opt_state in zip(optims, payload.get("optimizers", []), strict=False):
                opt.load_state_dict(opt_state)
            resume_step = int(payload.get("step", 0))
            if "rng_state" in payload:
                torch.set_rng_state(payload["rng_state"].cpu())
            if "cuda_rng_state" in payload and device.type == "cuda":
                torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu())
            print(f"[resume] resuming from step {resume_step + 1} (checkpoint at step {resume_step})")
    # Fast-forward the data stream past batches already consumed before resume.
    # Synthetic/MQAR generators use their own RNG seeded at construction time,
    # so without this a resumed run replays identical data from step 0.
    if resume_step > 0:
        accum_steps = max(1, int(cfg.training.gradient_accumulation_steps))
        for _ in range(resume_step * accum_steps):
            try:
                next(batches)
            except StopIteration:
                break
    initialize_lr_schedule(optims)

    start = time.perf_counter()
    last_log_time = start
    last_log_tokens = 0
    log_interval = max(1, min(cfg.training.eval_interval, cfg.training.steps))
    last_loss = 0.0
    tokens = 0

    model.train()
    shm_watchdog = start_shm_watchdog()
    try:
        for step in range(resume_step + 1, cfg.training.steps + 1):
            update_attention_windows(eager_model, step, cfg)
            if hasattr(eager_model, "set_step"):
                eager_model.set_step(step)
            accum_steps = max(1, int(cfg.training.gradient_accumulation_steps))
            for opt in optims:
                opt.zero_grad(set_to_none=True)
            # Accumulate the loss on-device: the previous float(...cpu()) here
            # forced a host sync once per micro-step (8x per optimizer step at
            # accum=8), serialising the pipeline for a number only read at
            # logging time.
            step_loss = torch.zeros((), device=device)
            last_diagnostics: dict[str, Any] = {}
            for _ in range(accum_steps):
                input_ids, labels = unpack_batch(next(batches))
                # S9: Token Superposition Training phase 1 — compress to bags.
                tst_phase_1 = False
                if getattr(cfg.training, "tst_enabled", False):
                    phase_1_steps = int(cfg.training.steps * float(getattr(cfg.training, "tst_phase_ratio", 0.0)))
                    tst_phase_1 = step <= phase_1_steps
                if tst_phase_1:
                    from flower.data import compress_to_bags

                    try:
                        input_ids = compress_to_bags(input_ids, int(cfg.training.tst_bag_size))
                    except Exception:
                        tst_phase_1 = False
                with autocast_ctx(device, amp_dtype):
                    out = model(input_ids, labels=labels)
                    loss = out["loss"]
                    if loss is None:
                        raise RuntimeError("loss was not computed")
                (loss / accum_steps).backward()
                step_loss += loss.detach()
                tokens += input_ids.numel()
                last_diagnostics = out.get("diagnostics", {}) or {}
            torch.nn.utils.clip_grad_norm_(eager_model.parameters(), cfg.training.grad_clip)
            apply_lr_schedule(
                optims,
                step,
                cfg.training.warmup_steps,
                cfg.training.lr_schedule,
                total_steps=cfg.training.steps,
                decay_frac=cfg.training.lr_decay_frac,
                final_frac=cfg.training.lr_final_frac,
            )
            for opt in optims:
                opt.step()
            if ema_model is not None:
                with torch.no_grad():
                    decay = float(cfg.training.ema_decay)
                    for ema_p, model_p in zip(ema_model.parameters(), eager_model.parameters()):
                        ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)
            last_loss = float(step_loss) / accum_steps
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
                # Surface any scalar fields the model put in its diagnostics dict
                # (variant-specific signals like hamiltonian_energy_drift_mean,
                # frequency_decay_mean_mag, etc.). Skip non-scalar entries such as
                # parameter_count (constant) and config (nested dict).
                for key, value in last_diagnostics.items():
                    if key in {"parameter_count", "config"}:
                        continue
                    if isinstance(value, bool):
                        continue
                    # 0-d tensors are the norm here: models keep diagnostics on
                    # device so the forward never syncs, and the host transfer
                    # happens once, here, only on logging steps.
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 0:
                            writer.add_scalar(f"diagnostics/{key}", float(value.detach()), step)
                        continue
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"diagnostics/{key}", float(value), step)
                writer.flush()
                last_log_time = now
                last_log_tokens = tokens
            if (
                val_batches is not None
                and cfg.training.validation_interval > 0
                and step % cfg.training.validation_interval == 0
            ):
                val_metrics = evaluate(
                    ema_model if ema_model is not None else model,
                    val_batches, cfg.training.validation_steps, device, amp_dtype,
                    bytes_per_token=cfg.data.bytes_per_token,
                )
                if writer is not None:
                    writer.add_scalar("validation/loss", float(val_metrics["val_loss"]), step)
                    writer.add_scalar("validation/perplexity", float(val_metrics["val_perplexity"]), step)
                    if "val_bpb" in val_metrics:
                        writer.add_scalar("validation/bpb", float(val_metrics["val_bpb"]), step)
                    writer.flush()
            if cfg.training.save_checkpoints and step % cfg.training.checkpoint_interval == 0 and not args.smoke:
                output_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "step": step,
                    "config": asdict(cfg),
                    "model": eager_model.state_dict(),
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
        shm_watchdog.stop()
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
        "parameter_count": count_parameters(eager_model),
        "device": str(device),
        "seed": int(cfg.training.seed),
        "gpu_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    if val_batches is not None:
        metrics.update(
            evaluate(
                ema_model if ema_model is not None else model,
                val_batches, cfg.training.validation_steps, device, amp_dtype,
                bytes_per_token=cfg.data.bytes_per_token,
            )
        )
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
        composite = run_composite_eval(eager_model, cfg, device=device)
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
