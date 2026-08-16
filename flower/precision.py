"""FP8 linear-layer training conversion (torchao).

WHY TENSORWISE AND NOT ROWWISE
  Measured on the RTX 5090 (sm_120) at the 450M `vanilla_matched` block shape
  (d_model 1280, ffn hidden 3392, batch 2 x seq 8192), full fwd+bwd, compiled:

      bf16                18.87 ms/block
      fp8 tensorwise      13.61 ms/block   1.39x
      fp8 rowwise         18.72 ms/block   1.01x

  Rowwise is the numerically safer recipe everywhere else, but on this GPU the
  per-row scale computation costs back exactly what the FP8 GEMM wins, so it is
  pointless here. Tensorwise is the only recipe that buys anything, which means
  the speed and the numerical risk are bought together and the guardrails below
  are not optional.

  For context on why this is the lever at all: the baseline profile
  (docs/profiling/baseline_profile.md) puts cutlass GEMM at 62.1% of kernel time
  for this config — the FFN gate/up/down and the Q/K/V/O projections. Those are
  exactly the matmuls this converts.

WHY NOT FP4
  NVFP4 and MXFP4 were measured on the same GPU and are not viable: nvfp4 runs
  at 1.02x bf16 with 13.9% relative error, mxfp4 at 0.49x (slower than bf16).
  Consumer Blackwell exposes FP4 tensor cores but there is no fast sm_120 GEMM
  path behind them in torch 2.13 / torchao 0.18, and torchao 0.18 has no MX
  training path at all (only `inference_workflow`). There is nothing to build on.

GUARDRAILS
  `keep_bf16_blocks` leaves the first and last N transformer blocks in bf16.
  The first block sees raw embeddings and the last feeds the LM head; both carry
  the widest activation ranges and are the usual sources of low-precision
  divergence. The LM head itself is never converted — it is tied to the
  embedding (see CausalLM.__init__), so quantizing it would perturb the
  embedding table through the tie, and `fp8_lm_head` already covers the eval-time
  head matmul separately.

  Pair this with `model.smooth_swiglu` (docs/training-speedups.md S10), which
  exists specifically to narrow the SwiGLU activation range that otherwise makes
  FP8 FFNs diverge.

ORDERING
  Conversion swaps `nn.Linear` for `Float8Linear`, which rebinds the weight
  Parameter objects. It must therefore run BEFORE `build_optimizer` (so the
  optimizer sees the final parameters) and BEFORE `torch.compile`.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

# Layers whose dims are not both divisible by 16 cannot use the FP8 GEMM.
_FP8_DIM_ALIGNMENT = 16


def _block_index(fqn: str) -> int | None:
    """Return the transformer block index a parameter FQN sits under, or None.

    Names look like `blocks.7.ff.gate`. Anything not under `blocks.<int>` (the
    LM head, MTP heads, memory-mechanism modules hung off the model root) has no
    block index and is left alone by the block-range guardrail.
    """
    parts = fqn.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def fp8_module_filter(num_layers: int, keep_bf16_blocks: int):
    """Build the `module_filter_fn` deciding which Linears become FP8.

    Returns a callable `(module, fqn) -> bool` as torchao's
    `convert_to_float8_training` expects.
    """

    def should_convert(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        # FP8 GEMM alignment requirement. Sub-16 dims would silently fall back
        # to a slow path (or error), so exclude them rather than find out at
        # step 1 of a 17-hour run.
        if module.in_features % _FP8_DIM_ALIGNMENT or module.out_features % _FP8_DIM_ALIGNMENT:
            return False
        idx = _block_index(fqn)
        if idx is None:
            # Not inside a transformer block: LM head, MTP heads, and any
            # model-level projection. Left in bf16 — see module docstring.
            return False
        if keep_bf16_blocks > 0:
            if idx < keep_bf16_blocks or idx >= num_layers - keep_bf16_blocks:
                return False
        return True

    return should_convert


def _fp8_config(recipe: str, use_fast_accum: bool = False):
    """Build the torchao `Float8LinearConfig` for a recipe name.

    torchao's recipes fast-accum only the *forward* GEMM —
    `Float8LinearConfig`'s class default is `gemm_config_output=Float8GemmConfig(use_fast_accum=True)`
    with the two backward GEMMs (dgrad, wgrad) left at fp32 accumulation.
    `use_fast_accum=True` flips those two as well, covering ~2/3 of the
    `_scaled_mm` work per converted Linear. On Ada this flag was the
    difference between full- and half-rate FP8; on sm_120 it may be a no-op —
    bench_arms decides. It changes numerics (reduced-precision K
    accumulation), so a positive throughput result still needs the quality
    screen. `Float8GemmConfig` is frozen, so the flag replaces fields rather
    than mutating the shared class-default instances.
    """
    from torchao.float8 import Float8LinearConfig
    from torchao.float8.config import Float8GemmConfig, Float8LinearRecipeName

    recipes = {
        "tensorwise": Float8LinearRecipeName.TENSORWISE,
        "rowwise": Float8LinearRecipeName.ROWWISE,
    }
    if recipe not in recipes:
        raise RuntimeError(f"training.fp8_recipe must be one of {sorted(recipes)}, got {recipe!r}")
    config = Float8LinearConfig.from_recipe_name(recipes[recipe])
    if use_fast_accum:
        config = dataclasses.replace(
            config,
            gemm_config_grad_input=Float8GemmConfig(use_fast_accum=True),
            gemm_config_grad_weight=Float8GemmConfig(use_fast_accum=True),
        )
    return config


def convert_model_to_fp8(
    model: nn.Module,
    *,
    recipe: str = "tensorwise",
    keep_bf16_blocks: int = 1,
    use_fast_accum: bool = False,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Convert eligible `nn.Linear` layers to FP8 training in place.

    Returns `(model, info)` where `info` records what was converted, so the
    caller can log it into the run metrics — a run's precision layout must be
    recoverable from its artifacts, not only from the config that launched it.

    Raises ImportError if torchao is unavailable and RuntimeError for an
    unknown recipe: a silent fall back to bf16 would make an FP8-labelled run
    secretly a bf16 run, which is worse than failing.
    """
    try:
        from torchao.float8 import convert_to_float8_training

        config = _fp8_config(recipe, use_fast_accum)
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "training.fp8_linear requires torchao. Install it with `uv add torchao`."
        ) from e

    num_layers = len(getattr(model, "blocks", []))
    if num_layers == 0:
        raise RuntimeError(
            "convert_model_to_fp8 found no `blocks` on the model; the block-range "
            "guardrail cannot be applied. Refusing to convert blindly."
        )

    eligible = [
        fqn
        for fqn, mod in model.named_modules()
        if fp8_module_filter(num_layers, keep_bf16_blocks)(mod, fqn)
    ]
    total_linear = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))

    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=fp8_module_filter(num_layers, keep_bf16_blocks),
    )

    info = {
        "fp8_recipe": recipe,
        "fp8_keep_bf16_blocks": keep_bf16_blocks,
        "fp8_use_fast_accum": use_fast_accum,
        "fp8_converted_linears": len(eligible),
        "fp8_total_linears": total_linear,
    }
    if verbose:
        print(
            f"[fp8] recipe={recipe} converted {len(eligible)}/{total_linear} Linear layers "
            f"(blocks {keep_bf16_blocks}..{num_layers - keep_bf16_blocks - 1} of {num_layers}; "
            f"first/last {keep_bf16_blocks} and the tied LM head stay bf16)",
            flush=True,
        )
    return model, info


class fp8_weight_cache:
    """Cache each FP8 layer's quantized weight across accumulation microsteps.

    *** MEASURED A NET LOSS AT THIS MODEL'S SHAPE. NOT WIRED IN. ***

      Compiled, accum=16, d_model 1280 / hidden 3392:

          activations per microstep      no cache     cached     result
          N = 4096  (unrepresentative)   227.4 ms    193.7 ms    1.174x
          N = 16384 (the real shape)     374.0 ms    384.2 ms    0.973x

      The real config runs batch 2 x seq 8192 = **16384** rows per microstep. At
      that ratio the activation-side quantization so dominates the weight side
      that the saving vanishes, while the cached path's own overhead (holding
      the cast weight live across microsteps, and a second compiled graph that
      inductor fuses less well) costs ~2.7%.

      The N=4096 row is kept deliberately as a warning: it is the same code
      showing +14.8%, and shipping on it would have been a regression. The
      benefit of this optimization is entirely a function of the
      weight:activation ratio, so any measurement at the wrong batch/sequence
      size inverts the answer.

      An earlier back-of-envelope estimate put the ceiling at ~0.7% of step
      time; the overhead simply exceeds that. Kept, tested, and documented
      because this is an idea that will occur to someone again.

    THE WASTE IT TARGETS (real, just not worth recovering this way)
      With dynamic scaling, torchao re-computes `amax(weight)` and re-casts the
      weight to e4m3 on EVERY forward. Under `gradient_accumulation_steps: 16`
      that is 16 identical quantizations per optimizer step, because the weight
      does not change until the step. The FP8 profile puts total quantization
      overhead at ~4.6% of CUDA time (137 ms quantize + 132 ms amax per step),
      of which the weight side is roughly 17% by element count — so eliminating
      15/16 of it recovers on the order of 0.7% of step time.

      torchao 0.18 offers no delayed scaling (only `dynamic`/`disabled`), so
      this cannot be amortized the supported way.

    WHY A CONTEXT MANAGER AND NOT AUTOMATIC INVALIDATION
      The obvious design — cache and invalidate when the weight changes — is a
      trap here. `torch.Tensor._version` does NOT track `p.data -= ...`, which
      is exactly how `flower/optim.py`'s Muon applies cautious weight decay
      (`p.data -= cautious_wd * lr * mask * p.data`). Verified directly:

          p.data -= x   -> _version unchanged
          p.add_(x)     -> _version incremented

      A version-counter cache would therefore serve STALE quantized weights
      after some optimizer updates and not others, with no error — training
      would just quietly get worse. So invalidation is explicit instead: the
      cache exists only inside this context, which wraps the accumulation loop
      and is exited before `optimizer.step()`.

    FAIL-LOUD
      On exit, a cheap sampled fingerprint of every cached weight is re-checked.
      If any weight changed while the cache was live, this raises rather than
      letting a silent correctness bug through. That converts the dangerous
      failure mode into an immediate, obvious one. `verify=False` skips it.

    Usage (see flower/train.py):
        with fp8_weight_cache(model):
            for _ in range(accum):
                model(...).backward()
        optimizer.step()
    """

    # Sample stride for the fingerprint: enough elements to catch any real
    # update, few enough that the check is free next to a training step.
    _FINGERPRINT_STRIDE = 4096

    def __init__(self, model: nn.Module, *, verify: bool = True) -> None:
        self.model = model
        self.verify = verify
        self._layers: list = []
        self._orig_forwards: dict = {}
        self._fingerprints: dict = {}

    @staticmethod
    def _fingerprint(w) -> float:
        flat = w.detach().reshape(-1)
        return float(flat[:: fp8_weight_cache._FINGERPRINT_STRIDE].float().sum())

    def __enter__(self):
        try:
            from torchao.float8.config import ScalingType
            from torchao.float8.float8_linear import Float8Linear
            from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
            from torchao.float8.float8_training_tensor import GemmInputRole
        except ImportError:
            return self  # torchao absent: no FP8 layers exist, nothing to cache

        for mod in self.model.modules():
            if not isinstance(mod, Float8Linear):
                continue
            cfg = mod.config
            # Only tensorwise dynamic weight scaling is cacheable here. Under
            # axiswise/rowwise the grad_input cast may use a different weight
            # config than the output cast, so one cached tensor cannot serve
            # both. Tensorwise is the recipe this codebase uses (rowwise
            # measured 1.01x on sm_120, i.e. pointless), so this covers it.
            if cfg.cast_config_weight.scaling_type is ScalingType.DISABLED:
                continue
            if cfg.cast_config_weight != cfg.cast_config_weight_for_grad_input:
                continue

            w_fp8_t = hp_tensor_to_float8_dynamic(
                mod.weight.t(),
                cfg.cast_config_weight.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=cfg.cast_config_weight.scaling_granularity,
                round_scales_to_power_of_2=cfg.round_scales_to_power_of_2,
            )

            from torchao.float8.float8_linear import matmul_with_hp_or_float8_args

            def cached_forward(x, _mod=mod, _w=w_fp8_t):
                # Mirrors Float8Linear.forward, but hands in the already-cast
                # weight. torchao's `tensor_already_casted_to_fp8` check then
                # skips re-quantizing it — a supported path (it is how fp8
                # all-gathered weights are consumed under FSDP).
                if torch.is_autocast_enabled():
                    x = x.to(torch.get_autocast_gpu_dtype())
                out = matmul_with_hp_or_float8_args.apply(
                    x, _w, _mod.linear_mm_config, _mod.config
                )
                if _mod.bias is not None:
                    out = out + _mod.bias.to(out.dtype)
                return out

            self._orig_forwards[mod] = mod.forward
            mod.forward = cached_forward
            self._layers.append(mod)
            if self.verify:
                self._fingerprints[mod] = self._fingerprint(mod.weight)
        return self

    def __exit__(self, exc_type, exc, tb):
        changed = []
        for mod in self._layers:
            mod.forward = self._orig_forwards[mod]
            if self.verify and exc_type is None:
                if self._fingerprint(mod.weight) != self._fingerprints[mod]:
                    changed.append(mod)
        self._layers.clear()
        self._orig_forwards.clear()
        self._fingerprints.clear()
        if changed:
            raise RuntimeError(
                f"fp8_weight_cache: {len(changed)} weight(s) were modified while the "
                "cache was active, so the cached FP8 weights were stale. The cache "
                "must not span an optimizer step. This is raised rather than "
                "silently training on stale weights."
            )
        return False


def maybe_convert_fp8(model: nn.Module, training_cfg, device) -> tuple[nn.Module, dict]:
    """Apply FP8 conversion if `training.fp8_linear` is set; otherwise no-op.

    Single entry point shared by flower/train.py and scripts/profile_step.py (and
    anything else that reproduces the training wiring) so a benchmark can never
    silently measure a different precision layout than a real run. Returns
    `(model, info)` with an empty info dict when FP8 is off.

    MUST be called after `build_model(...).to(device)` and BEFORE
    `build_optimizer` and `torch.compile` — see module docstring.
    """
    if not getattr(training_cfg, "fp8_linear", False):
        return model, {}
    if device.type != "cuda":
        raise RuntimeError("training.fp8_linear requires CUDA")
    if training_cfg.precision != "bf16":
        raise RuntimeError(
            "training.fp8_linear requires training.precision: bf16 "
            f"(FP8 layers consume bf16 activations), got {training_cfg.precision!r}"
        )
    return convert_model_to_fp8(
        model,
        recipe=training_cfg.fp8_recipe,
        keep_bf16_blocks=int(training_cfg.fp8_keep_bf16_blocks),
        use_fast_accum=bool(getattr(training_cfg, "fp8_use_fast_accum", False)),
    )
