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
from flower.models.base import count_parameters, prebuild_attention_masks
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

    When ``attn_warmup_quantize > 0`` the ramp is quantised: the target window
    only changes every ``quantize`` steps, so a warmup of N steps produces at
    most ``ceil(N / quantize)`` distinct windows instead of N. Each distinct
    window forces a FlexAttention ``create_block_mask`` recompile (its
    ``mask_mod`` closure captures the window), so without quantising the ramp
    exhausts torch.compile's recompile limit and flex falls back to the eager
    dense path. Quantising keeps the recompile count under the limit (default
    8) while preserving the ramp's training-dynamics benefit. The final window
    is always ``local_window``; ``quantize`` is a step-stride (number of steps
    between window changes), not a window-value grid.
    """
    if getattr(cfg.model, "attn_warmup_steps", 0) == 0:
        return
    target = cfg.model.local_window
    if step >= cfg.model.attn_warmup_steps:
        target = cfg.model.local_window
    else:
        # `quantize` is a step-stride: only advance the ramp every `quantize`
        # steps, holding the window constant in between. This bounds the number
        # of distinct windows (= create_block_mask recompiles) to
        # ceil(warmup_steps / quantize). Without it the per-step ramp exhausts
        # torch.compile's recompile limit. Snap the *effective step* down to
        # the nearest quantize boundary so each plateau is constant.
        quantize = int(getattr(cfg.model, "attn_warmup_quantize", 0) or 0)
        eff_step = step if quantize <= 1 else (step // quantize) * quantize
        frac = eff_step / max(1, cfg.model.attn_warmup_steps)
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
    configure_vram_limit(device, fraction=getattr(cfg.training, "vram_fraction", 0.85))
    amp_dtype = configure_precision(cfg.training.precision, device)
    model = build_model(cfg.model).to(device)
    # FP8 linear conversion must happen BEFORE build_optimizer: swapping in
    # Float8Linear rebinds the weight Parameters, so an optimizer built first
    # would hold references to the discarded originals and silently train
    # nothing. It must also precede torch.compile so the compiled graph sees
    # the FP8 modules. See flower/precision.py for the measurement behind the
    # recipe choice and the guardrails.
    from flower.precision import maybe_convert_fp8

    model, fp8_info = maybe_convert_fp8(model, cfg.training, device)
    # Optimizers are always built on the eager module: torch.compile returns a
    # wrapper whose .parameters() are the same objects, but keeping the eager
    # reference makes state_dict keys stable across compiled/uncompiled runs.
    optim_or_list = build_optimizer(model, cfg.training)
    eager_model = model
    # CUDAGraph (reduce-overhead) + gradient accumulation is a known
    # torch.compile limitation: the captured graph pins the `.grad` tensor
    # addresses, but accumulation reuses them across micro-steps and the graph
    # replay then reads a buffer that a later replay has already overwritten
    # ("accessing gradient tensor output of CUDAGraphs that has been
    # overwritten... when a .grad tensor is allocated during CUDAGraph capture").
    # The error's own suggested fix is to allocate stable `.grad` buffers
    # *outside* CUDAGraph capture before the compiled backward runs, so the
    # graph captures their (now-stable) addresses. We do that by directly
    # assigning `p.grad = torch.zeros_like(p)` before the training loop (see
    # below) and keeping those buffers allocated across the run
    # (zero_grad(set_to_none=False) below). With accum == 1 this is unnecessary,
    # so we skip it. See pytorch/pytorch#169545 — standard workarounds
    # (mark_step_begin, cloning) do not fix it; only pre-allocated buffers do.
    use_persistent_grads = (
        cfg.training.compile_model
        and cfg.training.compile_mode == "reduce-overhead"
        and int(cfg.training.gradient_accumulation_steps) > 1
        and device.type == "cuda"
    )
    if cfg.training.compile_model:
        # The diagnostics walk uses dir()/getattr over every submodule, which
        # Dynamo cannot trace; leaving it on would graph-break every forward.
        # Variant-specific scalars are unavailable under compile as a result.
        if getattr(eager_model, "collect_module_diagnostics", False):
            eager_model.collect_module_diagnostics = False
            print("[compile] module diagnostics disabled (untraceable by Dynamo)")
        # Eagerly build every flex-attention BlockMask before compiling so the
        # compiled forward only reads the cached masks. Building them inside the
        # graph mutates module state, which cudagraph mode flags as tensor
        # aliasing across the per-layer reads ("accessing tensor output of
        # CUDAGraphs that has been overwritten"). No-op when flex is off.
        if device.type == "cuda":
            prebuild_attention_masks(eager_model, cfg.data.sequence_length, device)
            # S9 TST phase 1 runs at the COMPRESSED length T/s, which needs its
            # own BlockMask. Prebuilding it here keeps mask construction out of
            # the compiled graph for both phases (building it inside mutates
            # module state, which cudagraph mode flags as tensor aliasing).
            if getattr(cfg.training, "tst_enabled", False):
                bag = max(1, int(getattr(cfg.training, "tst_bag_size", 1)))
                if bag > 1:
                    prebuild_attention_masks(eager_model, cfg.data.sequence_length // bag, device)
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
        # Materialise the parameter lists once for the multi-tensor EMA update
        # below. Pairing is positional and safe because ema_model is a deepcopy
        # of eager_model, so .parameters() yields the same structure in the same
        # order. The explicit length check turns any future divergence (e.g. a
        # variant that adds parameters after the copy) into a loud failure here
        # rather than a whole run of silently mis-paired weight averaging.
        #
        # Caching `.data` is safe because every optimizer in flower/optim.py
        # mutates parameters in place (`p.data -= ...`); none rebinds `p.data`
        # to a new tensor, which would leave these references stale.
        ema_params = [p.data for p in ema_model.parameters()]
        model_params = [p.data for p in eager_model.parameters()]
        if len(ema_params) != len(model_params):
            raise RuntimeError(
                f"EMA/model parameter count mismatch ({len(ema_params)} vs {len(model_params)})"
            )
        # The EMA copy runs validation forwards eagerly (it is NOT wrapped in
        # torch.compile, unlike `model` below). A flex-attention forward that
        # misses the fused compiled kernel falls back to the unfused math path,
        # which materialises the full (B, H, T, T) score matrix per layer. At
        # the 450M config (seq 8192, 16 heads) that is ~8 GB per allocation and
        # OOMs on top of the compiled training model's ~24 GB of live inductor
        # workspaces. On WSL2/WDDM the overshoot does not raise a clean
        # OutOfMemoryError — it spills to host shared memory and the next CUDA
        # kernel fails with `cudaErrorUnknown` (the bloom_memory 450M sweep-13
        # step-1000 crash, surfaced at currentStreamCaptureStatusMayInitCtx).
        # `_load_flex_attention_compiled` is the codebase's existing primitive
        # for exactly this (used when activation_checkpointing triggers an
        # eager recompute): a standalone-compiled fused flex kernel that never
        # materialises the dense scores. Forcing `_flex_needs_compile=True` on
        # the EMA copy's flex attention drops the eval peak from OOM (>32 GB)
        # to ~11.5 GB, matching the compiled model. No-op when flex is off
        # (use_flex is False) — there is no dense-score fallback to avoid.
        from flower.models.base import CausalSelfAttention

        for module in ema_model.modules():
            if isinstance(module, CausalSelfAttention) and getattr(module, "use_flex", False):
                module._flex_needs_compile = True
    initialize_lr_schedule(optims)
    batches = token_batches(cfg.data, cfg.training.batch_size, device, seed=int(cfg.training.seed))
    eval_bs = cfg.training.eval_batch_size or cfg.training.batch_size
    val_batches = (
        validation_token_batches(cfg.data, eval_bs, device)
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
    # Scalar-logging cadence. Historically this was derived from `eval_interval`,
    # which is 1000 in the production configs — so a 10,000-step run recorded
    # only 11 loss points. That is far too coarse to see a loss spike, a grad-norm
    # excursion, or any of the instabilities the Muon literature's stability
    # fixes (MuonClip and friends) exist to address: at 1000-step resolution an
    # instability and a recovery are one indistinguishable sample. `log_interval`
    # now decouples the two; 0 keeps the legacy behaviour so old configs
    # reproduce their logging exactly.
    explicit_log_interval = int(getattr(cfg.training, "log_interval", 0) or 0)
    log_interval = max(
        1,
        min(
            explicit_log_interval or cfg.training.eval_interval,
            cfg.training.steps,
        ),
    )
    last_loss = 0.0
    tokens = 0
    # Gradient-norm statistics, accumulated on device between log points so the
    # per-step cost is one add and one compare (no host sync). Reset each time
    # they are logged, so each point summarises its own interval rather than the
    # whole run to date.
    grad_norm_sum = torch.zeros((), device=device)
    grad_clipped_sum = torch.zeros((), device=device)
    grad_norm_max = torch.zeros((), device=device)
    grad_stat_steps = 0

    model.train()
    shm_watchdog = start_shm_watchdog()
    # CUDAGraph + grad-accum warmup (see `use_persistent_grads` above):
    # pre-allocate a zeroed `.grad` tensor for every trainable parameter *before*
    # CUDAGraph capture, so the graph pins those (now-stable) addresses instead
    # of allocating fresh `.grad` tensors during capture (which then get
    # overwritten on the next replay — the "accessing gradient tensor output of
    # CUDAGraphs that has been overwritten" error). This is the fix the error
    # message itself suggests ("preallocating zeroed .grad tensors") and the only
    # one that works in torch 2.13 (mark_step_begin / cloning do not — see
    # pytorch/pytorch#169545). We allocate directly rather than via an eager
    # backward so flex-attention stays on its fused compiled kernel (the eager
    # backward would materialise the T x T scores matrix and can OOM at long
    # context). The buffers are kept across the run via zero_grad(set_to_none=
    # False) below. Skipped unless the combination is active, so default-mode
    # and accum==1 runs are unaffected. Cost: +1x params worth of `.grad`
    # memory, which is already the steady-state cost of any backward.
    if use_persistent_grads:
        for p in eager_model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        print("[compile] persistent .grad buffers allocated for reduce-overhead + grad accumulation")
    try:
        for step in range(resume_step + 1, cfg.training.steps + 1):
            update_attention_windows(eager_model, step, cfg)
            if hasattr(eager_model, "set_step"):
                eager_model.set_step(step)
            accum_steps = max(1, int(cfg.training.gradient_accumulation_steps))
            for opt in optims:
                # Keep pre-allocated `.grad` buffers (stable addresses for
                # CUDAGraph replay) when reduce-overhead + accumulation is on;
                # otherwise free them (the standard, lower-memory path).
                opt.zero_grad(set_to_none=not use_persistent_grads)
            # Accumulate the loss on-device: the previous float(...cpu()) here
            # forced a host sync once per micro-step (8x per optimizer step at
            # accum=8), serialising the pipeline for a number only read at
            # logging time.
            step_loss = torch.zeros((), device=device)
            last_diagnostics: dict[str, Any] = {}
            for _ in range(accum_steps):
                input_ids, labels = unpack_batch(next(batches))
                # S9: Token Superposition Training phase 1 — compress to bags.
                #
                # BOTH inputs and labels must be bagged. Previously only the
                # inputs were, which left `labels` 2-D against a 3-D input and
                # crashed the forward ("too many values to unpack") — and the
                # model call sits outside the try/except below, so the guard did
                # not catch it either. `tst_enabled: true` was unusable.
                #
                # The model averages each bag's embeddings (one position per
                # bag, so T/s positions instead of T — the source of the
                # speedup) and scores against the next bag's token SET via
                # multi-hot CE. See CausalLM._multi_hot_cross_entropy.
                tst_phase_1 = False
                if getattr(cfg.training, "tst_enabled", False):
                    phase_1_steps = int(cfg.training.steps * float(getattr(cfg.training, "tst_phase_ratio", 0.0)))
                    tst_phase_1 = step <= phase_1_steps
                if tst_phase_1:
                    from flower.data import compress_to_bags

                    bag = int(cfg.training.tst_bag_size)
                    input_ids = compress_to_bags(input_ids, bag)
                    labels = compress_to_bags(labels, bag)
                with autocast_ctx(device, amp_dtype):
                    out = model(input_ids, labels=labels)
                    loss = out["loss"]
                    if loss is None:
                        raise RuntimeError("loss was not computed")
                (loss / accum_steps).backward()
                step_loss += loss.detach()
                tokens += input_ids.numel()
                last_diagnostics = out.get("diagnostics", {}) or {}
            # `clip_grad_norm_` already computes the total pre-clip gradient
            # norm; it was being discarded. It is the single most informative
            # stability signal available for free — a run that is going unstable
            # shows grad-norm excursions and a rising clip rate well before the
            # loss curve visibly breaks. Accumulated ON DEVICE and only synced at
            # logging time, so this adds no per-step host sync.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                eager_model.parameters(), cfg.training.grad_clip
            )
            grad_norm_sum += grad_norm.detach()
            grad_clipped_sum += (grad_norm.detach() > cfg.training.grad_clip).to(grad_norm.dtype)
            grad_norm_max = torch.maximum(grad_norm_max, grad_norm.detach())
            grad_stat_steps += 1
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
                    # Multi-tensor form of the same two ops the per-parameter
                    # loop ran, in the same order, so it is BIT-EXACT against
                    # previous runs (verified). The loop issued 2 kernel launches
                    # per parameter (284 for this 142-tensor model) every step
                    # for pure-bandwidth work; _foreach_ groups them.
                    #
                    # Deliberately NOT torch._foreach_lerp_, which computes the
                    # algebraically identical e + (1-d)(m - e) with different
                    # rounding — measured ~5e-7 drift in fp32. The EMA feeds
                    # eval weights, and this codebase's runs are compared at
                    # 0.0004 BPB resolution, so a free bit-exact form is worth
                    # preferring over a marginally faster inexact one.
                    torch._foreach_mul_(ema_params, decay)
                    torch._foreach_add_(ema_params, model_params, alpha=1.0 - decay)
            should_log = step == 1 or step % log_interval == 0 or step == cfg.training.steps
            if should_log:
                # `float(step_loss)` is a host sync; paying it every step would
                # reintroduce exactly the sync the on-device loss accumulator
                # above exists to avoid. The value is only read at logging
                # time, and `should_log` includes the final step, so the
                # closing metrics always see the last value.
                last_loss = float(step_loss) / accum_steps
            if writer is not None and should_log:
                now = time.perf_counter()
                interval_elapsed = max(now - last_log_time, 1e-9)
                tokens_per_sec = (tokens - last_log_tokens) / interval_elapsed
                writer.add_scalar("train/loss", last_loss, step)
                writer.add_scalar("train/perplexity", math.exp(min(last_loss, 20.0)), step)
                writer.add_scalar("train/lr", optims[0].param_groups[0]["lr"], step)
                log_learning_rates(writer, optims, step)
                # Stability signals. `grad_norm_max` and `grad_clip_frac` are the
                # two that move first when training is going wrong: a rising clip
                # fraction means the optimizer is being throttled more and more
                # often, and a max far above the mean means isolated spikes that
                # a mean would hide. Both are interval statistics, reset below.
                if grad_stat_steps > 0:
                    writer.add_scalar(
                        "train/grad_norm", float(grad_norm_sum) / grad_stat_steps, step
                    )
                    writer.add_scalar("train/grad_norm_max", float(grad_norm_max), step)
                    writer.add_scalar(
                        "train/grad_clip_frac", float(grad_clipped_sum) / grad_stat_steps, step
                    )
                    grad_norm_sum.zero_()
                    grad_clipped_sum.zero_()
                    grad_norm_max.zero_()
                    grad_stat_steps = 0
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
    # Record the FP8 layout (recipe, how many Linears were converted) so a run's
    # numerical setup is readable from its own metrics rather than only from the
    # config that launched it. Empty dict when fp8_linear is off.
    metrics.update(fp8_info)
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
        # Run the composite on the SAME weights the final val metrics above used.
        # Previously this always passed `eager_model` (raw weights) even when
        # ema_decay > 0, so the val metrics that sort sweep decisions described
        # the EMA model while the composite ranked the raw one — two different
        # models in the same decision. The EMA copy is an eager deepcopy that is
        # never compiled, so run_composite_eval's model.eval()/train() handling
        # applies to it unchanged (it stays in eval() throughout, like any
        # eval-only model).
        composite_model = ema_model if ema_model is not None else eager_model
        composite_weights = "ema" if ema_model is not None else "raw"
        composite = run_composite_eval(composite_model, cfg, device=device)
        composite["eval_weights"] = composite_weights
        composite_path.parent.mkdir(parents=True, exist_ok=True)
        composite_path.write_text(json.dumps(composite, indent=2, sort_keys=True))
        metrics["composite_ranker_json"] = str(composite_path)
        metrics["composite_eval_weights"] = composite_weights
    if cfg.training.metrics_json:
        metrics_path = Path(cfg.training.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    train()
