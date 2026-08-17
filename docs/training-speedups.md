# Flower Training Pipeline Speedups

Implementation spec for throughput and sample-efficiency improvements.

Each section is self-contained: the file(s) to modify, the current code, the target code, and a validation step. Implement sections in order — later ones depend on earlier ones.

**Context**: Flower is a memory-augmented transformer research project. The architecture comparison (memory variants vs vanilla) is the core research output. Throughput and optimizer improvements are SAFE (they help all variants equally). Architecture modifications are DANGEROUS (they change information pathways and confound memory mechanism comparisons). This document only contains SAFE improvements, with explicit notes where validation is needed.

**Source for all improvements**: KellerJordan/modded-nanogpt speedrun (records #1-#55) and Track 3 optimization benchmark (records #1-#46). https://github.com/KellerJordan/modded-nanogpt

> **Measured results live in [`profiling/speedup_results.md`](profiling/speedup_results.md).**
> This file is the *spec*; that file records what was actually measured on the
> RTX 5090 at the 450M config, including several items here that turned out to be
> dead ends. Read it before implementing anything below. In particular it
> supersedes:
>
> - **Section 13's FP4 plan** — nvfp4 measures 1.02x bf16 with 13.9% error and
>   mxfp4 measures 0.49x on sm_120; there is no fast FP4 GEMM for consumer
>   Blackwell, and torchao 0.18 has no MX/NVFP4 training path at all. FP8
>   tensorwise (1.30x at model scale) is the whole of the low-precision win.
>   Section 13's Transformer Engine suggestion is also moot — TE is not installed
>   and torchao covers the working path.
> - **Section 14 Opportunity 1 / S14-5a (Newton-Schulz)** — still not worth doing,
>   but *not* because the optimizer is free. `baseline_profile.md`'s "~0 ms,
>   <1% of step" was a subtraction artifact; measured directly it is 134 ms/step.
>   It is skippable because the batching is already optimal (4 shape groups, zero
>   singletons) and it is ~2.8% at production accum.
> - **Section 14 Opportunity 5 / S14-5b (fused linear CE)** — was *broken* under
>   `torch.compile` on torch 2.13 and is now fixed; it is a ~1.1 GB memory win,
>   not a speed win.

---

## Section 1: FlexAttention Migration (HIGH PRIORITY)

### Why

Flower currently materializes a dense `(1, 1, T, T)` attention mask tensor per layer and passes it to `F.scaled_dot_product_attention`. At seq=2048 this is ~16MB/layer in fp32. At seq=32K (the target for memory mechanism experiments) it becomes ~4GB/layer — this will OOM on the 5090's 32GB.

FlexAttention (PyTorch 2.5+) compiles the mask pattern into the attention kernel without ever materializing the full T×T matrix. The modded-nanogpt speedrun went from dense causal at seq=1024 to FlexAttention at seq=64K and got FASTER (record #12, KoszarskyB).

Reference: https://pytorch.org/docs/stable/nn.attention.flex_attention.html
Record #12: https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-11-19_FlexAttention

### Current Code

File: `flower/models/base.py`, class `CausalSelfAttention` (line ~226)

The forward method (line ~281) does:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    q, k, v = self.qkv_heads(x)
    seq_len = x.shape[1]
    keep = causal_mask(seq_len, x.device, self.local_window).view(1, 1, seq_len, seq_len)
    if self.kernel_bias == "rbf":
        attn_mask = self._rbf_bias(seq_len, x.device, q.dtype).expand(...)
        attn_mask = attn_mask.masked_fill(~keep, torch.finfo(q.dtype).min)
    else:
        attn_mask = keep
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).contiguous().view(x.shape)
    return self.out(out)
```

The function `causal_mask` is imported and creates a dense boolean tensor.

### Target Implementation

Replace SDPA + materialized mask with `torch.nn.attention.flex_attention.flex_attention`.

**Step 1**: Create a mask function using `create_block_mask`:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def _make_causal_local_mask(local_window: int | None, seq_len: int, device, dtype):
    """Create a compiled block mask for causal + optional sliding-window attention."""
    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        if local_window is not None:
            # Sliding window: only attend to tokens within local_window positions
            window = (q_idx - kv_idx) < local_window
            return causal & window
        return causal

    return create_block_mask(
        mask_mod,
        B=None,  # flex_attention broadcasts over batch
        H=None,  # broadcasts over heads
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
```

**Step 2**: Modify `CausalSelfAttention.forward` to use flex_attention:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    q, k, v = self.qkv_heads(x)
    seq_len = x.shape[1]

    if self.kernel_bias == "rbf":
        # RBF bias needs an additive score_mod, not just a mask.
        # Use flex_attention's score_mod parameter for this.
        # See flex_attention docs for score_mod signature.
        rbf_scale = self.rbf_log_scale.exp().to(device=q.device, dtype=q.dtype)
        def rbf_score_mod(score, b, h, q_idx, kv_idx):
            dist = (q_idx - kv_idx).float()
            return score - rbf_scale[h] * (dist ** 2) / (self.local_window ** 2)
        block_mask = _make_causal_local_mask(self.local_window, seq_len, q.device, q.dtype)
        out = flex_attention(q, k, v, score_mod=rbf_score_mod, block_mask=block_mask)
    else:
        block_mask = _make_causal_local_mask(self.local_window, seq_len, q.device, q.dtype)
        out = flex_attention(q, k, v, block_mask=block_mask)

    out = out.transpose(1, 2).contiguous().view(x.shape)
    return self.out(out)
```

**Step 3**: Cache the block mask. `create_block_mask` is expensive (it compiles). The mask only changes when seq_len or local_window changes. Add a cache:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, ...):
        ...
        self._cached_block_mask = None
        self._cached_seq_len = 0
        self._cached_window = None

    def _get_block_mask(self, seq_len, device, dtype):
        if (self._cached_block_mask is None or
            seq_len != self._cached_seq_len or
            self.local_window != self._cached_window):
            self._cached_block_mask = _make_causal_local_mask(
                self.local_window, seq_len, device, dtype
            )
            self._cached_seq_len = seq_len
            self._cached_window = self.local_window
        return self._cached_block_mask
```

**Step 4**: Apply `flex_attention = torch.compile(flex_attention)` once at module level or in the model constructor. FlexAttention benefits from compilation:

```python
# At top of base.py, after imports:
_flex_attention_compiled = torch.compile(flex_attention)
```

Then use `_flex_attention_compiled` in forward.

### Important Compatibility Notes

- **torch.compile interaction**: If `compile_model=True` in TrainingConfig, the block mask creation and flex_attention call will be inside the compiled region. This is fine — flex_attention is designed for this. But `create_block_mask` itself should be called OUTSIDE the compiled forward (in a precompute step or with caching). The cache above handles this.
- **Grad accumulation**: No interaction. FlexAttention works normally with backward.
- **`still_lm.py`**: This file re-implements the block forward to intercept the KV cache (line ~290-360). It currently calls `self.attn.qkv_heads()` and then does its own attention. It will also need to use flex_attention if it currently uses SDPA. Check `still_lm.py` for any `scaled_dot_product_attention` or `causal_mask` calls.
- **Variant attention modules**: `flow_attention.py`, `hamiltonian_attention.py`, `memory.py`, `partitioned_memory.py` may have their own attention calls. These variants override `CausalSelfAttention` — check each for SDPA usage and migrate if they use materialized masks.
- **Other model files**: Search for `scaled_dot_product_attention` and `causal_mask` across the entire codebase. Every call site needs migration.
- **RBF kernel bias**: The current RBF implementation materializes a dense (B, H, T, T) tensor. With flex_attention, use `score_mod` instead — it's applied inside the kernel without materialization.

### Validation

```bash
# Run existing tests to verify equivalence
uv run pytest tests/test_nextgen_pipeline.py -x -q

# Quick smoke test: train 10 steps and check loss is reasonable
uv run python -m flower.train --config configs/smoke.yaml --max-steps 10

# Verify attention output matches old implementation on a small input:
# Create a 2-layer model, run both old SDPA and new flex_attention on the
# same random input, assert outputs are within 1e-5.
```

---

## Section 2: Attention Window Warmup Schedule

### Why

Start training with a short attention window and expand it over the first portion of training. Tokens processed during warmup use a smaller attention matrix (faster), and the model learns local patterns before needing long-range context.

Record #13 (fernbear): https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-11-24_WindowWarmup

This is a training schedule, not an architecture change. The final model has the same architecture; only the training trajectory differs.

### Current Code

File: `flower/models/base.py`, function `layer_attn_windows` (referenced in tests at line ~567).

File: `flower/config.py`, `ModelConfig` has:
```python
attn_window_schedule: list[int | None] | None = None
```

This is a static per-layer schedule. We need a DYNAMIC schedule that changes the attention window over training steps.

### Target Implementation

**Step 1**: Add config fields to `ModelConfig` in `flower/config.py`:

```python
# Attention window warmup: start with a small window and expand over training.
# The window grows linearly from `attn_warmup_start` to `local_window` over
# `attn_warmup_steps` training steps. After that, it stays at `local_window`.
# Set attn_warmup_steps=0 to disable (default).
attn_warmup_start: int = 256
attn_warmup_steps: int = 0  # 0 = disabled, use local_window from step 0
```

**Step 2**: In `flower/train.py`, before each training step, update the attention modules' local_window:

```python
def update_attention_windows(model, step, cfg):
    """Expand attention windows over training according to warmup schedule."""
    if cfg.model.attn_warmup_steps == 0:
        return
    if step >= cfg.model.attn_warmup_steps:
        target_window = cfg.model.local_window
    else:
        frac = step / cfg.model.attn_warmup_steps
        target_window = int(cfg.model.attn_warmup_start + 
                          frac * (cfg.model.local_window - cfg.model.attn_warmup_start))
    
    for module in model.modules():
        if hasattr(module, 'local_window') and module.local_window is not None:
            if module.local_window != target_window:
                module.local_window = target_window
                module._cached_block_mask = None  # invalidate cache
```

Call this at the top of the training loop, before the forward pass.

**Step 3**: If using `torch.compile`, window changes will invalidate the compiled graph. Two options:
- Option A: Disable compile during warmup, enable after warmup completes.
- Option B: Use a fixed set of window sizes (e.g. 256, 512, 1024, 2048) and switch between them, accepting a recompile at each switch.
- Option C (simplest): Only use window warmup without compile, or set warmup_steps low enough that the recompile cost is acceptable.

For prototyping, Option C is fine. For production runs, use flex_attention (Section 1) which handles arbitrary masks without recompilation.

### Validation

Run a short training (100 steps) with warmup_steps=50, local_window=1024, attn_warmup_start=128. Log the effective window size each step and verify throughput increases as the window grows.

---

## Section 3: FP8 LM Head

### Why

The `lm_head` projection (d_model → vocab_size) is one of the largest single matmuls in the model. At 350M params with vocab=50K and d_model=1024, it's a 1024×50000 matrix multiply per token. FP8 matmul runs ~2x faster than BF16 on Blackwell (RTX 5090) with negligible quality impact on logit computation.

Record #19 (YouJiacheng): https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-01-13_Fp8LmHead

### Current Code

File: `flower/models/base.py`, class `CausalLM` (line ~330):

```python
self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
self.head.weight = self.token.weight  # tied embeddings
```

And in forward (line ~426):
```python
logits = self.head(self.ln(x))
```

File: `flower/train.py` — loss is computed as:
```python
loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
```

### Target Implementation

**Step 1**: Add FP8 support to the head computation. Use `torch.float8_e4m3fn` for the matmul, then cast logits back to bf16/fp32 for the loss.

```python
# In CausalLM.forward, replace:
#   logits = self.head(self.ln(x))
# with:

x_normed = self.ln(x)
if self.config.fp8_lm_head and x_normed.dtype == torch.bfloat16:
    # FP8 matmul for the head projection
    # abs-amax row scales; DIVIDE by the scale before the e4m3 cast so each
    # row fills the +-448 range -- _scaled_mm multiplies the scales back.
    scale_a = x_normed.abs().amax(dim=-1, keepdim=True).float() / 448.0
    scale_b = self.head.weight.abs().amax(dim=-1, keepdim=True).float() / 448.0
    x_fp8 = (x_normed.float() / scale_a).to(torch.float8_e4m3fn)
    w_fp8 = (self.head.weight.float() / scale_b).to(torch.float8_e4m3fn)
    logits = torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=scale_a.t().contiguous() if scale_a.shape[0] == 1 else scale_a,
        scale_b=scale_b.view(1, -1).contiguous(),
        out_dtype=torch.bfloat16,
    )
else:
    logits = self.head(x_normed)
```

**Step 2**: Add config flag to `ModelConfig` in `flower/config.py`:

```python
# Use FP8 matmul for the lm_head projection (EVAL ONLY -- _scaled_mm has no
# backward). Requires BF16 activations + CUDA (sm_89+). Quality impact: ~3-4%
# norm-relative logit error vs the bf16 head (e4m3 quantization), pinned by
# tests/test_training_speedups.py::test_fp8_head_matches_bf16_head.
fp8_lm_head: bool = False
```

### Important Notes

- **Tied embeddings**: Flower ties `self.head.weight = self.token.weight`. The FP8 cast happens on a VIEW of the embedding weights — it does not modify the actual embedding parameters. The master weights stay BF16.
- **`torch._scaled_mm`**: This is the low-level FP8 matmul. It requires per-tensor or per-row scale factors. The code above computes scales from the actual activation/weight magnitudes. For production, precompute the weight scale once (it changes slowly during training).
- **torch.compile compatibility**: `torch._scaled_mm` is compile-compatible.
- **Fallback**: On GPUs without FP8 support (Ampere and older), `torch._scaled_mm` with FP8 dtypes will error. The `if self.config.fp8_lm_head` guard plus the dtype check handles this.
- **Alternative**: If `torch._scaled_mm` is too fiddly, the simpler approach is to use `torch.float8_e4m3fn` autocast via `torch.autocast('cuda', dtype=torch.float8_e4m3fn)`. However, this is less mature and may not use the fastest kernel path.

### Validation

```bash
# Run the precision benchmark script to compare
uv run python scripts/bench_precision.py

# The flag is eval-only, so train loss is identical with it on or off;
# compare EVAL loss instead. NOTE: the original implementation of this flag
# was broken (unscaled cast -> logits ~3e-5x too small -> eval loss exactly
# ln(vocab)); any eval result recorded with fp8_lm_head=True before the fix
# is invalid. The corrected path is pinned against the bf16 head by
# tests/test_training_speedups.py::test_fp8_head_matches_bf16_head.
```

---

## Section 4: BF16 Cross-Entropy Loss

### Why

Flower currently computes cross-entropy in FP32 (the default for `F.cross_entropy`). Computing it in BF16 saves a cast and uses faster reduction kernels. The quality impact is negligible for training (the gradient signal is dominated by the forward pass quantization, not the loss reduction precision).

Record #37 (Gusarich): https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-09-27_BF16CE

### Current Code

File: `flower/models/base.py`, line ~429:

```python
loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
```

### Target Implementation

Simply cast logits to bf16 before cross-entropy if the config requests it:

```python
# In CausalLM.forward, replace the loss computation:
if labels is not None:
    loss_dtype = torch.bfloat16 if getattr(self.config, 'bf16_cross_entropy', False) else logits.dtype
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).to(loss_dtype),
        labels[:, 1:].reshape(-1),
    )
```

Add config flag:

```python
# Compute cross-entropy loss in BF16 instead of FP32. Tiny speedup.
bf16_cross_entropy: bool = False
```

### Validation

Train 100 steps with and without. Compare final val loss. Difference should be <0.001.

---

## Section 5: NorMuon Optimizer Update

### Why

NorMuon normalizes the Muon update direction to unit norm before applying the learning rate. This addresses a known Muon instability: the Newton-Schulz orthogonalization doesn't guarantee a normalized output, and the update norm can drift. NorMuon is a one-line change in the optimizer.

Paper: arXiv:2510.05491 (NorMuon)
Track 3 records #8-#10: consistently beats raw Muon by ~2-5% in step efficiency.

### Current Code

File: `flower/optim.py`, class `Muon` (line ~81). The `step()` method applies Newton-Schulz orthogonalization to the momentum buffer, then scales by learning rate.

### Target Implementation

After the Newton-Schulz iteration produces the orthogonalized update `X`, normalize it:

```python
# In Muon.step(), after the Newton-Schulz iteration produces X:
# Current: the update is X scaled by learning rate and aspect-ratio scaling
# Add: normalize X to unit Frobenius norm before scaling

X = X / (X.norm() + eps)  # NorMuon normalization
```

Add a config flag:

```python
# NorMuon: normalize the Muon update to unit Frobenius norm before LR scaling.
# arXiv:2510.05491. Addresses update-norm drift in Newton-Schulz orthogonalization.
norm_update: bool = False  # add to TrainingConfig
```

Route the flag through `build_optimizer()` in `optim.py` to the Muon constructor.

### Validation

Run the existing Muon ladder test or a short sweep comparing `norm_update=False` vs `True`. The NorMuon variant should reach the same val loss in ~2-5% fewer steps. Check that it doesn't change architecture rankings (run vanilla vs. best-memory with both optimizers and confirm the ranking holds).

---

## Section 6: Cautious Weight Decay

### Why

Standard weight decay applies uniformly to all parameters. Cautious Weight Decay (CWD) only decays weights where the optimizer update is already shrinking them (i.e., where `update * weight > 0`). This avoids fighting the optimizer on coordinates where it's trying to grow the weight.

Track 3 records #43, #50, #46: consistently improves final loss by ~0.001-0.003.

### Current Code

File: `flower/optim.py`. Muon currently applies weight decay as a standard L2 penalty in the optimizer step. Check how `weight_decay` is applied in the `Muon.step()` and `AdamW.step()` methods.

### Target Implementation

**For Muon (2D weights)**:

```python
# In Muon.step(), after computing the final update but before applying it:
if self.cautious_wd > 0:
    # Only decay where update is already shrinking the weight
    # update direction: w_new = w - lr * update
    # "shrinking" means: sign(update) == sign(weight) at that coordinate
    cautious_mask = (update * p.data > 0).to(update.dtype)
    p.data -= self.cautious_wd * lr * cautious_mask * p.data
```

**For AdamW (1D/embedding weights)**:

```python
# In the AdamW step, apply CWD instead of standard weight decay:
if self.cautious_wd > 0:
    cautious_mask = (update * p.data > 0).to(update.dtype)
    p.data -= self.cautious_wd * lr * cautious_mask * p.data
elif self.weight_decay > 0:
    # Standard weight decay (fallback)
    p.data -= self.weight_decay * lr * p.data
```

Add config:

```python
# Cautious Weight Decay: only decay weights where the optimizer update is
# already shrinking them. Track 3 record #43, #46.
# When set, replaces standard weight_decay for Muon params.
cautious_wd: float = 0.0  # 0.0 = disabled. Try 0.025 for Muon params.
```

### Validation

Run a short sweep comparing standard WD vs CWD at matched values. CWD should give equal or slightly better final loss. Verify it doesn't change architecture rankings.

---

## Section 7: Distributed Training Optimizations (FOR RENTED MULTI-GPU)

### Why

When renting an 8x H100/B200 node for a final training run, distributed communication becomes a bottleneck. These optimizations are only relevant for multi-GPU training (not the 5090).

Records #22-24: gradient all_reduce → reduce_scatter, compute/comm overlap.

### Current State

File: `flower/distributed_benchmark.py` — uses DDP with `dist.all_reduce`. The training loop in `train.py` uses standard DDP.

### Target Implementation

**Step 1**: Switch from DDP to FSDP (Fully Sharded Data Parallel). FSDP shards model parameters across GPUs, reducing memory per GPU and using `reduce_scatter` instead of `all_reduce` for gradients.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    # Prefetch next layer's params during current layer's compute
    forward_prefetch=True,
)
```

**Step 2**: Overlap gradient communication with backward computation. PyTorch 2.5+ does this automatically with FSDP if `torch.compile` is used. Alternatively, use FSDP's built-in backward prefetch:

```python
from torch.distributed.fsdp import BackwardPrefetch
model = FSDP(model, backward_prefetch=BackwardPrefetch.BACKWARD_PRE, ...)
```

**Step 3**: Use `reduce_scatter` directly (FSDP does this by default). No code change needed beyond switching to FSDP.

### Important Notes

- **FSDP + torch.compile**: Works in PyTorch 2.5+. Compile the inner model, then wrap with FSDP.
- **FSDP + Muon**: The Muon optimizer's Newton-Schulz iteration operates on full weight matrices. Under FSDP, weights are sharded. You need to either (a) use FSDP's `gather_fqn_params` to unshard before the optimizer step, or (b) run the Newton-Schulz on sharded matrices (incorrect). The cleanest approach: use FSDP with `ShardingStrategy.SHARD_GRAD_OP` (also called "ZeRO-2"), which only shards gradients and optimizer states, keeping full parameters on each GPU. This avoids the Muon sharding problem entirely but uses more memory.

### Validation

Run `distributed_benchmark.py` on 2+ GPUs. Compare throughput (tokens/sec) before and after FSDP migration. Verify loss curves match single-GPU training.

---

## Section 8: Multi-Token Prediction (OPTIONAL — FINAL RUNS ONLY)

### Why

Multi-Token Prediction (MTP) predicts multiple future tokens from each position, not just the next one. This gives each position multiple gradient signals, improving sample efficiency by ~1.5-2x.

Record #53 (varunneal): https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-12-22_MultiTokenPrediction

Note: TST (Token Superposition Training, arXiv:2605.06546) is a DIFFERENT method from MTP — it targets wall-clock throughput via input compression, not auxiliary prediction heads. TST has its own section (Section 9) because it is orthogonal to MTP and better suited to Flower. See Section 9.

### When to Use

ONLY for final "maximum quality" runs where you've already selected your winning architecture. Do NOT use during architecture sweeps — MTP changes the loss surface and makes comparisons to published baselines difficult.

### Current Code

File: `flower/models/base.py`, `CausalLM.forward` (line ~426):
```python
logits = self.head(self.ln(x))
loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
```

### Target Implementation

Add extra prediction heads for tokens t+2, t+3, etc:

```python
class CausalLM(nn.Module):
    def __init__(self, config, blocks):
        ...
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight  # tied for t+1
        # MTP heads for t+2, t+3, ... (untied, smaller LR)
        self.mtp_heads = nn.ModuleList([
            nn.Linear(config.d_model, config.vocab_size, bias=False)
            for _ in range(config.mtp_extra_heads)
        ]) if getattr(config, 'mtp_extra_heads', 0) > 0 else None
```

In forward:
```python
logits = self.head(self.ln(x))  # t+1 prediction (standard)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))

if self.mtp_heads is not None:
    for i, mtp_head in enumerate(self.mtp_heads):
        offset = i + 2  # t+2, t+3, ...
        mtp_logits = mtp_head(x_normed)
        mtp_loss = F.cross_entropy(
            mtp_logits[:, :-(offset)].reshape(-1, mtp_logits.size(-1)),
            labels[:, offset:].reshape(-1),
        )
        loss = loss + mtp_weight * mtp_loss
```

Add config:
```python
# Multi-Token Prediction: predict N extra future tokens. Only for final runs.
# Sample efficiency gain ~1.5x but changes loss surface (incomparable to baselines).
mtp_extra_heads: int = 0  # 0 = disabled. Try 2-3 for final runs.
mtp_weight: float = 0.5   # weight for auxiliary MTP losses
```

### Validation

Only validate after all other sections are implemented. Train with and without MTP at 350M. MTP should reach the same val loss in ~30% fewer tokens. Final quality should be equal or better.

---

## Implementation Order and Priority

**Phase 1 — Throughput wins (do first, zero risk to research):**
1. **Section 1 (FlexAttention)** — HIGH. Enables seq=32K. Required for long-context experiments.
2. **Section 4 (BF16 CE)** — LOW effort, free throughput.
3. **Section 3 (FP8 LM Head)** — MEDIUM. Free throughput on Blackwell.
4. **Section 2 (Window Warmup)** — MEDIUM. Free sample efficiency.

**Phase 2 — Optimizer and eval improvements (validate, then adopt):**
5. **Section 12.1 (QK-Norm)** — Already implemented. Just enable in config.
6. **Section 5 (NorMuon)** — LOW effort. Validate it doesn't change architecture rankings.
7. **Section 6 (CWD)** — LOW effort. Same validation as NorMuon.
8. **Section 12.2 (Orthogonal Init)** — LOW effort. Pairs with Muon.
9. **Section 12.3 (Hybrid NS)** — LOW effort. DeepSeek-V4's Muon refinement.
10. **Section 12.4 (EMA Eval)** — LOW effort. Free quality boost at eval time.
11. **Section 12.5 (Sliding-Window Eval)** — LOW effort. Better BPB measurement.

**Phase 3 — Scale-up (for the 600M+ runs):**
12. **Section 13 (Scale-Up Config)** — The 600M / 32K / mixed-precision target config.
13. **Section 10 (Smooth-SwiGLU)** — Needed for FP8/FP4 training stability.
14. **Section 11 (Precision Routing)** — FP4 FFN + FP8 attention + BF16 memory.

**Phase 4 — Final quality runs (after architecture is selected):**
15. **Section 9 (TST)** — HIGH priority for final runs. 2.5x wall-clock, validated at 600M, final model = baseline arch.
16. **Section 8 (MTP)** — Additive with TST. D=1 recommended for Flower's scale.
17. **Section 7 (FSDP)** — Only for rented multi-GPU runs.

## Section 9: Token Superposition Training / TST (HIGH PRIORITY — FINAL RUNS)

### Why

TST is a two-phase training method that compresses input tokens into "bags" during phase 1 (processing s-fold more data per FLOP), then reverts to standard next-token prediction in phase 2. The final model is architecturally identical to a standard NTP baseline.

Reference: arXiv:2605.06546 (Nous Research)

TST is **distinct from MTP** (Section 8): MTP adds auxiliary prediction heads but processes the same tokens per FLOP. TST changes tokens-per-FLOP via input compression. The TST paper explicitly states: "MTP and its variants do not increase training-time throughput... TST occupies a different point in the design space... we view TST as orthogonal to auxiliary-loss methods, and combining the two is a natural direction for future work."

### Why TST is better suited to Flower than MTP

1. **Directly validated at 600M** (Flower's target scale) by the original paper. MTP's usefulness is known to degrade below 1B.
2. **Final model is architecturally identical to baseline.** Phase 2 reverts to standard NTP. This preserves comparability for Flower's architecture-comparison research. MTP permanently changes the loss surface.
3. **2.5x wall-clock speedup** is bigger and more reliable than MTP's ~1.5x sample efficiency gain.
4. **Drop-in**: no architecture, optimizer, tokenizer, or data pipeline changes needed.
5. **Orthogonal to MTP**: third-party experiments (AOMTS series, HuggingFace) confirm TST+MTP gains are additive. At 100M/3k steps: base=2.287 nats, MTP=1 alone=2.276 (-0.011), TST alone=2.214 (-0.073), TST+MTP=1=2.205 (-0.083, best).

### How It Works

**Phase 1 (superposition)**: Average embeddings of contiguous s-grams into "s-tokens" (e.g. s=4 means 4 consecutive tokens become 1 input position). This compresses the input s-fold. Predict the next bag of s tokens with multi-hot cross-entropy (MCE) — a single head, order-independent loss.

**Phase 2 (recovery)**: Revert to standard next-token prediction. The model quickly recovers token-level competence and surpasses an equal-FLOP NTP baseline.

### Target Implementation

**Step 1**: Add config fields to `TrainingConfig` in `flower/config.py`:

```python
# Token Superposition Training (arXiv:2605.06546)
# Phase 1: compress s consecutive tokens into bags, train with MCE loss
# Phase 2: revert to standard NTP
# bag_size: number of tokens per bag (4-8 recommended)
# step_ratio: fraction of total steps spent in phase 1 (0.2-0.4)
tst_enabled: bool = False
tst_bag_size: int = 4
tst_phase_ratio: float = 0.3  # fraction of total steps for phase 1
```

**Step 2**: In the data pipeline (`flower/data.py`), implement the bag compression for phase 1:

```python
def compress_to_bags(token_ids: torch.Tensor, bag_size: int) -> torch.Tensor:
    """Compress consecutive tokens into bags by averaging embeddings.
    Input: (B, T) token IDs
    Output: (B, T // bag_size) token ID bags (each position = bag_size original tokens)
    """
    B, T = token_ids.shape
    T_compressed = T // bag_size
    # Truncate to multiple of bag_size
    token_ids = token_ids[:, :T_compressed * bag_size]
    return token_ids.view(B, T_compressed, bag_size)  # (B, T', s)
```

**Step 3**: In the model forward pass, during phase 1, embed all tokens in each bag and average:

```python
# In CausalLM.forward, during TST phase 1:
if tst_phase_1_active:
    # input_ids shape: (B, T_compressed, bag_size)
    embeddings = self.token(input_ids)  # (B, T_compressed, bag_size, d_model)
    x = embeddings.mean(dim=2)  # (B, T_compressed, d_model) — bag embedding
    # ... standard transformer blocks ...
    # Loss: multi-hot cross-entropy over the bag_size target tokens
```

**Step 4**: Implement the multi-hot cross-entropy loss for phase 1:

```python
def multi_hot_cross_entropy(logits, target_bags, vocab_size):
    """MCE loss: predict the SET of tokens in the next bag.
    logits: (B, T, vocab_size)
    target_bags: (B, T, bag_size) token IDs
    """
    B, T, s = target_bags.shape
    # Create multi-hot target: 1 for each token in the bag
    multi_hot = torch.zeros(B, T, vocab_size, device=logits.device)
    for i in range(s):
        multi_hot.scatter_(2, target_bags[:, :, i:i+1], 1.0)
    # Normalize to distribution
    multi_hot = multi_hot / multi_hot.sum(dim=-1, keepdim=True).clamp(min=1)
    # Cross-entropy with multi-hot target distribution
    loss = -(multi_hot * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    return loss
```

**Step 5**: In `flower/train.py`, switch between TST phase 1 and NTP phase 2 based on step count:

```python
if cfg.training.tst_enabled:
    phase_1_steps = int(cfg.training.steps * cfg.training.tst_phase_ratio)
    tst_phase_1_active = step < phase_1_steps
else:
    tst_phase_1_active = False
```

### Important Notes

- **Compute-bound regime**: TST trades compute for data — it processes s-fold more tokens but at s-fold lower cost per token. This is favorable when you are compute-bound (which Flower is, on both 5090 and rented hardware). If you are data-limited (not enough unique tokens), TST is less useful.
- **Compatibility with everything**: TST doesn't touch the architecture, optimizer, attention, or memory mechanisms. It's purely a training schedule + data compression. Safe to combine with all other sections in this doc.
- **Compatibility with MTP**: TST and MTP are orthogonal and additive (AOMTS experiments). Use TST for the throughput win, optionally add MTP heads in phase 2 for additional sample efficiency.
- **Phase 2 recovery**: The model needs a brief recovery period when switching from MCE to NTP. The paper shows this converges quickly (a few hundred steps). The `tst_phase_ratio` of 0.3 means the last 70% of training is standard NTP.

### Validation

```bash
# 1. Equivalence: train 350M with and without TST (same total compute).
#    TST should reach equal or lower val loss at the same wall-clock time.

# 2. Phase transition: monitor val loss around the phase 1→2 switch.
#    Expect a brief spike then rapid recovery to below phase-1 trajectory.

# 3. Architecture comparison: run vanilla vs best-memory with TST.
#    Confirm memory mechanism ranking holds (TST shouldn't change relative rankings).
```

---

## Section 10: Smooth-SwiGLU for FP8/FP4 Quantization (FOR LOW-PRECISION RUNS)

### Why

SwiGLU is quadratic in its inputs: when the gate and value weights align (which happens naturally over long training), activation magnitudes can spike far beyond what the rest of the network produces. These outliers are tolerable in BF16 but **destabilize FP8/FP4 training** — the low-precision quantization scales can't accommodate the dynamic range, and the training diverges.

Smooth-SwiGLU is a mathematically equivalent reformulation that factors out a per-channel scale *before* quantization and rescales *after* the down-projection. The function is identical; the numerical stability under low precision is dramatically better.

Reference: arXiv:2409.12517 (Peng et al., "Scaling FP8 training to trillion-token LLMs")
Result: FP8 training on 7B Llama2 diverged with vanilla SwiGLU; converged with Smooth-SwiGLU, matching BF16 quality with +34% throughput. Also stabilizes BF16 training at large scale even without quantization.

### When to Use

Enable this flag when running with FP8 or FP4 FFN layers (the mixed-precision strategy discussed in the scaling analysis). For pure BF16 training it's optional but harmless — it adds no compute, only a scale factor.

### Current Code

File: `flower/models/base.py`, class `FeedForward`. Flower has GELU and SwiGLU modes. The SwiGLU path looks like:

```python
# Pseudocode from current SwiGLU implementation:
# gate = self.w_gate(x)        # Linear
# up   = self.w_up(x)          # Linear
# act  = F.silu(gate)          # Activation
# fused = act * up             # Element-wise gate
# out  = self.w_down(fused)    # Linear
```

### Target Implementation

Apply a per-channel maximum-abs scale to the `up` projection output *before* the element-wise multiply, carry the scale tensor through, and rescale the down-projection output.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout, config, ...):
        ...
        self.smooth_swiglu = getattr(config, 'smooth_swiglu', False)

    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)

        if self.ffn_activation == "swiglu":
            act = F.silu(gate)
            if self.smooth_swiglu:
                # Per-channel scale: prevents SwiGLU outlier amplification
                # under FP8/FP4. Mathematically equivalent to standard SwiGLU.
                # up shape: (B, T, ffn_dim)
                up_scale = up.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
                up_scaled = up / up_scale          # normalize before quant
                fused = act * up_scaled             # gate * scaled_value
                out = self.w_down(fused)             # (B, T, d_model)
                out = out * up_scale                 # rescale after down-proj
            else:
                fused = act * up
                out = self.w_down(fused)
        else:
            # GELU path unchanged
            ...

        return self.drop(out)
```

### Important Notes

- **Mathematical equivalence**: `w_down(silu(gate) * (up/s)) * s == w_down(silu(gate) * up)`. The scale factor cancels through the linearity of `w_down`. The only change is numerical: the intermediate `fused` tensor has a smaller dynamic range.
- **FP8 interaction**: When `up` is cast to FP8 for the element-wise multiply, the per-channel scaling ensures the values are in [-1, 1] range, which FP8 E4M3 handles well. Without scaling, outlier channels can hit 448 (FP8 E4M3 max), causing overflow → NaN.
- **Fusion**: The scale computation (`up.abs().amax`) adds a reduction, but it can be fused into the preceding `w_up` matmul kernel via `torch.compile`. The rescale multiplies into `w_down`'s output. Net overhead: near zero under compile.
- **Alternative (simpler)**: Instead of per-token scaling (reduction over `ffn_dim`), use a **running per-channel scale** over the `ffn_dim` dimension, updated every N steps. This avoids the reduction entirely but is an approximation. The per-token version above is exact and preferred.

### Validation

```bash
# 1. Equivalence test (BF16, no quantization):
#    Run forward pass on random input with smooth_swiglu=True and =False.
#    Assert outputs are within 1e-4 of each other (should be near-identical).

# 2. FP8 stability test:
#    Train 500 steps with FP8 FFN layers + smooth_swiglu=True.
#    Confirm no NaN/Inf in loss. Compare to same run with smooth_swiglu=False
#    (which should diverge after ~200 steps per the paper).
```

---

## Section 12: SAFE Additions from Frontier Models (Qwen3, DeepSeek-V4, Parameter Golf)

These are low-risk, high-value improvements that don't change information pathways. All validated by at least two independent frontier labs.

### 12.1: QK-Norm (RMSNorm on Q and K per head)

**Why**: Prevents attention-logit explosion during training. Two independent frontier labs (Qwen3, DeepSeek-V4) both converged on this. Zero-parameter (negligible: per-head scale), zero pathway change. Flower already has a `qk_norm` config option — just ensure it's enabled by default in the 600M config.

**Implementation**: Already implemented in `flower/models/base.py` as `HeadRMSNorm`. Just set `qk_norm: True` in the 600M config.

References: Qwen3 (arXiv:2505.09388), DeepSeek-V4 (arXiv:2606.19348), Dehghani et al. 2023.

### 12.2: Orthogonal Weight Initialization

**Why**: Parameter Golf found that `nn.init.orthogonal_()` on all weight matrices accelerates early convergence. Synergizes with Muon — the Newton-Schulz step preserves orthogonality, so starting from an orthogonal point keeps the optimizer in its ideal operating regime.

**Implementation**: In `CausalLM._apply_scaled_init()`, replace the normal init for 2D weights with orthogonal init:

```python
if isinstance(module, nn.Linear) and module.weight.ndim == 2:
    if getattr(module, "_is_residual_out", False):
        nn.init.orthogonal_(module.weight)
        module.weight.data *= depth_scale  # scale down for residual outputs
    else:
        nn.init.orthogonal_(module.weight)
```

Add config flag: `orthogonal_init: bool = False`

Reference: OpenAI Parameter Golf winning submissions (#164+).

### 12.3: Hybrid Newton-Schulz Schedule for Muon

**Why**: DeepSeek-V4 uses a 2-stage Newton-Schulz iteration: first 8 iterations with the standard coefficients (3.4445, -4.7750, 2.0315) for rapid convergence, last 2 with (2, -1.5, 0.5) to pin singular values at exactly 1. This gives better numerical stability than the current single-coefficient approach.

**Implementation**: In `flower/optim.py`, `Muon.step()`, modify the Newton-Schulz call:

```python
def newton_schulz_2stage(G, steps=10, eps=1e-7):
    """2-stage NS: rapid converge (8 steps), then pin to 1 (2 steps)."""
    # Stage 1: standard coefficients for rapid convergence
    a1, b1, c1 = 3.4445, -4.7750, 2.0315
    # Stage 2: pin singular values to 1
    a2, b2, c2 = 2.0, -1.5, 0.5
    
    split = steps - 2  # last 2 steps use stage 2
    X = G.bfloat16() / (G.norm() + eps)
    if X.size(0) > X.size(1):
        X = X.T
    
    for i in range(steps):
        a, b, c = (a1, b1, c1) if i < split else (a2, b2, c2)
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if X.size(0) > X.size(1):
        X = X.T
    return X.to(G.dtype)
```

Add config flag: `ns_schedule: str = "standard"` → `"standard"` | `"hybrid"`

Reference: DeepSeek-V4 (arXiv:2606.19348).

### 12.4: EMA Weight Averaging for Evaluation

**Why**: Evaluate with an exponential moving average of the weights (decay=0.997) rather than the final weights. Parameter Golf found consistent quality gains at zero training cost. The EMA smooths out late-training noise.

**Implementation**: In `flower/train.py`, maintain an EMA copy of the model weights:

```python
import copy

# After model creation:
if cfg.training.ema_decay > 0:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

# After each optimizer step:
if cfg.training.ema_decay > 0:
    with torch.no_grad():
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(cfg.training.ema_decay).add_(model_p.data, alpha=1 - cfg.training.ema_decay)

# At eval time, use ema_model instead of model
```

Add config: `ema_decay: float = 0.0` (0.0 = disabled, 0.997 = recommended)

Reference: OpenAI Parameter Golf (#401).

### 12.5: Sliding-Window Evaluation

**Why**: During evaluation, use overlapping windows (stride=64) so every token gets near-full context from both sides. Parameter Golf showed large measured-BPB gains. This is an eval-only improvement — it doesn't change the model or training at all.

**Implementation**: In the eval pipeline, instead of evaluating on non-overlapping chunks:

```python
def sliding_window_eval(model, token_ids, window_size, stride):
    """Evaluate with overlapping windows for better BPB measurement.
    Each token is scored using a window centered on it (causal: looks backward).
    """
    T = token_ids.size(1)
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, T - window_size, stride):
        window = token_ids[:, start:start + window_size]
        with torch.no_grad():
            out = model(window, labels=window)
        total_loss += out["loss"].item() * window_size
        total_tokens += window_size
    return total_loss / total_tokens
```

Note: this is more expensive than standard eval (more forward passes) but gives a more accurate BPB measurement. Use for final evaluations, not during-training monitoring.

Reference: OpenAI Parameter Golf winning submissions.

---

## Section 13: Scale-Up Configuration (600M, Long Context, Mixed Precision)

### Why

The core research finding from the scaling analysis: Flower's memory mechanisms showed <2% BPB difference at 25M params because (a) the model was too small to form the circuits that memory mechanisms support, and (b) seq=2048 full attention already gives every token access to all context — there is nothing for external memory to do.

Two variables must change simultaneously for memory mechanisms to show measurable signal:

1. **Model size**: Must cross ~500M where capacity for multi-head induction circuits and specialized attention patterns emerges. Below this, architectural innovations are typically <2% BPB difference regardless of mechanism quality.

2. **Sequence length**: Must exceed the sliding window by ≥4x. With a 2048 sliding window, seq must be ≥8K minimum, ideally 16K-32K. Below this ratio, local attention covers most token dependencies and external memory adds no information.

The third variable — precision — is what makes long sequences *affordable*. Training at seq=32K in BF16 costs 16x more attention FLOPs than seq=8K. Using FP4 for the FFN layers (60% of compute) and FP8 for attention (15-30% of compute at long context) makes seq=32K training fit within a 1-2 day rental budget on an 8x H100/B200 node.

### Target: The Configuration

This section defines the target model/training configuration for Flower's publishable run. It is NOT a code change to make — it is a config specification to implement in `flower/config.py` and the sweep YAML files, plus the precision routing logic.

#### Model Size: 600M

Rationale from the compute analysis:
- At 600M on 8x H100 (€200-300 rental): Chinchilla-optimal (12B tokens) in ~6 hours. Overtrained 50B tokens in ~1 day. Matches Pythia-1B and approaches SmolLM-360M quality.
- At 600M on the 5090: ~5 days for Chinchilla-optimal, feasible for prototyping and eval suite validation.
- 600M is the smallest size where the emergence literature (Phase-Transitional Scaling, GPT-2 124M-774M validation) shows capability phase transitions becoming measurable.

Target architecture (approximate, adjust to keep param count at ~600M):
```python
d_model: 1024
num_heads: 16
head_dim: 64
num_layers: 28  # deeper than 25M config to match Llama-style aspect ratio
vocab_size: 50000  # or custom tokenizer size
```

#### Sequence Length: 32K with 2048 Sliding Window

This is the regime where memory mechanisms become load-bearing. With seq=32K and window=2048, tokens in the first 2K of the sequence are invisible to tokens past position 22K unless the memory carries that information forward.

```python
sequence_length: 32768
local_window: 2048
# attn_window_schedule: interleave full-attention layers among windowed ones
# e.g. [2048, 2048, None, 2048, 2048, None, ...] for ~4:1 window:full ratio
```

Note: this ratio (window:full) is itself a research variable for Flower. The MAD paper (Poli et al., arXiv:2403.17844) recommends testing hybrid ratios. The consensus ratio for linear/recurrent hybrids at scale is 3:1 (75% subquadratic, 25% full attention).

#### Precision Routing: Per-Layer Mixed Precision

The key insight: precision is not a global setting. Different parts of the model have different sensitivity to quantization. The precision routing should be:

| Component                         | Precision     | Why                                                             |
|-----------------------------------|---------------|-----------------------------------------------------------------|
| FFN up/gate/down (SwiGLU)         | FP4 (NVFP4)   | Biggest matmuls, least sensitive. 6.7x BF16 throughput on B300. |
| FFN (with Smooth-SwiGLU scaling)  | FP4           | Section 9 scaling prevents SwiGLU outlier divergence.          |
| Attention Q/K/V/O projections     | FP8           | Attention scores sensitive to precision. FP8 is the floor.     |
| Attention softmax                  | FP32          | Always high precision.                                          |
| Memory write path (Flower-specific) | BF16        | MUST keep high precision. Routing decisions need dynamic range. |
| Embedding / LM head               | BF16          | Vocab projection is unstable in low precision.                  |
| Optimizer states (master weights)  | FP32          | Always FP32 master weights.                                     |
| Cross-entropy loss                | BF16          | Section 4.                                                       |

### Target Implementation: Precision Routing

**Step 1**: Add precision config to `ModelConfig` in `flower/config.py`:

```python
# Mixed-precision per-layer routing for long-context training.
# Global setting: use 'bf16' for all layers if not doing low-precision training.
# For mixed precision, specify per-component precision:
ffn_precision: str = "bf16"       # bf16 | fp8 | fp4 (NVFP4)
attn_precision: str = "bf16"      # bf16 | fp8  (do not use fp4 for attention)
memory_precision: str = "bf16"    # bf16 only — Flower memory path must stay high precision
head_precision: str = "bf16"      # bf16 | fp8  (lm_head matmul)
```

**Step 2**: Implement precision casting in the forward pass. The cleanest approach is a `PrecisionCast` context manager or explicit casts at each component boundary:

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        target_dt = {
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
        }[self.config.attn_precision]

        x_cast = x.to(target_dt) if x.dtype != target_dt else x
        q, k, v = self.qkv_heads(x_cast)

        # Softmax always FP32
        # ... (attention computation, FP32 softmax, output in target_dt)

        out = self.out(out.to(x.dtype))  # cast back to residual stream dtype
        return out


class FeedForward(nn.Module):
    def forward(self, x):
        if self.config.ffn_precision == "fp4":
            # Cast inputs to FP4 for the matmuls
            # Use Smooth-SwiGLU (Section 9) to control outlier magnitudes
            ...
        elif self.config.ffn_precision == "fp8":
            # Use torch._scaled_mm for FP8 matmuls
            ...
        else:
            # BF16 (current behavior)
            ...
```

**Step 3**: For FP4 FFN layers on Blackwell (sm_120+), use the NVFP4 format via `torch._scaled_mm` with block-scaled FP4 tensors. The NVFP4 format uses two-level scaling (FP8 micro-blocks on FP4 values) to maintain near-FP8 accuracy.

```python
def fp4_linear(x, weight):
    """NVFP4 linear layer via torch._scaled_mm.
    Requires Blackwell (sm_120+) or Hopper (sm_90+) with Transformer Engine.
    """
    x_fp4 = x.to(torch.float8_e4m3fn)  # placeholder; NVFP4 needs block scaling
    w_fp4 = weight.to(torch.float8_e4m3fn)

    # NVFP4 uses two-level scaling: per-tensor FP32 scale + per-block FP8 scale
    # See: torch.nn.functional.fp4_linear (PyTorch 2.7+ on Blackwell)
    # Or: transformer_engine.pytorch.fp4_linear

    return torch._scaled_mm(
        x_fp4, w_fp4.t(),
        scale_a=compute_block_scale(x),
        scale_b=compute_block_scale(weight),
        out_dtype=torch.bfloat16,
    )
```

Note: NVFP4 training is still experimental as of mid-2026. The most mature path is via NVIDIA's Transformer Engine library (`transformer_engine`), which handles the block scaling and autocasting automatically:

```python
# Using Transformer Engine (preferred for FP4/FP8 training):
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# Replace nn.Linear with te.Linear:
# te.Linear automatically handles FP8/FP4 autocasting
# when the model is wrapped in a TE forward context
```

### Important Notes

- **Memory write path precision**: The memory write mechanism (bloom hashes, surprise gates, slot addressing) involves continuous-valued routing decisions that depend on representational differences between slots. Low precision (FP8/FP4) can collapse these differences:
  - Bloom memory hash functions could collapse (distinct hashes become indistinguishable)
  - Gate values near 0 vs near 0.1 are hard to distinguish in FP4 (3 mantissa bits)
  - The gradient signal for write addressing is already weak — low precision could kill it entirely
  - **Rule**: Keep the entire memory write/read path in BF16. Only quantize the standard transformer layers (FFN, attention projections) that the memory mechanism wraps around.
- **Validation via ablation**: Whether FP8/FP4 degrades memory mechanism signal is itself a publishable finding. Test it: run vanilla vs. best-memory at (BF16 everything) vs. (FP4 FFN + BF16 memory). If the ranking holds under low precision, that's a robustness result. If it breaks, that's a precision-sensitivity result.
- **5090 vs rented hardware**: FP4 (NVFP4) is available on the 5090 (sm_120, Blackwell). Prototype the precision routing there. FP8 is available on both H100 and B200. FP4 is only on Blackwell (B200/B300, and the 5090).
- **torch.compile interaction**: `torch._scaled_mm` and TE linear layers are compile-compatible. The precision casts should be outside the compiled region (applied as config-time module attributes, not per-forward casts).
- **Sequence length warmup**: Do not start training at seq=32K. Use a sequence length schedule (Section 2's window warmup, extended to sequence length warmup). Start at seq=4096, expand to 8K, 16K, 32K over the first 20-30% of training. This matches the speedrun's batch size schedule (record #46) and YaRN-style length extension (record #31).

### Validation

```bash
# 1. 600M model fits in 32GB at seq=32K with precision routing:
#    Build the model, forward+backward one batch, check VRAM usage.
#    Target: < 28GB on the 5090 (leave headroom for fragmentation).

# 2. Precision routing correctness:
#    Run 100 steps at each precision tier:
#    a. BF16 everywhere (baseline)
#    b. FP8 FFN + FP8 attention + BF16 memory
#    c. FP4 FFN + FP8 attention + BF16 memory
#    Compare val loss. b should be within 0.01 of a. c should be within 0.02 of a.

# 3. Memory mechanism signal at scale:
#    Run vanilla vs best-memory variant at 600M, seq=32K, mixed precision.
#    The memory variant should show a BPB improvement >0.03 over vanilla.
#    (At 25M/seq=2048 this was <0.02. At 600M/seq=32K it should be measurable.)

# 4. Training stability:
#    Monitor for NaN/Inf in loss, especially during precision transitions.
#    SwiGLU outliers (Section 9) are the most common failure mode.
```

### Compute Budget Reference

Based on the rental cost analysis (mid-2026 neocloud rates, Vast.ai/RunPod/Lambda):

| Run                              | 5090 (BF16)  | 8x H100 (BF16)  | 8x B300 (mixed FP4) |
|----------------------------------|-------------|-----------------|---------------------|
| 600M @ 12B tok (Chinchilla)      | ~5 days     | ~6h (~€130)     | ~2.5h (~€50)        |
| 600M @ 50B tok (overtrained)     | ~21 days    | ~25h (~€540)    | ~10h (~€210)        |
| 600M @ 100B tok (well-overtrained)| ~42 days   | ~50h (~€1,080)  | ~20h (~€420)        |

These are approximate. Actual throughput depends on MFU, which depends on model size, seq length, precision, and how well-tuned the training loop is.

---

## Section 14: Custom Training Kernels (Triton/CUDA)

### Why

Stock PyTorch + cuBLAS + FlashAttention already handle 80-90% of a standard transformer's FLOPs near-optimally. But Flower has several custom operations that are NOT standard PyTorch ops and execute as chains of small kernel launches with HBM round-trips between them. Fusing these into single Triton kernels eliminates launch overhead and intermediate memory traffic.

The expected speedups are modest (3-10% total training time) — not the 100x inference speedups from Starling — because the model is compute-bound on the large matmuls (FFN, attention, embeddings) which cuBLAS already handles well. But some of these kernels are unique to Flower's architecture and could be mentioned as engineering contributions in a paper.

**Skill reference**: Use the `small-transformer-research` skill's approach for Triton kernel development. The 5090 is sm_120 (Blackwell) — Triton generates excellent code for this architecture.

### Opportunity 1: Fused Newton-Schulz Orthogonalization for Muon (HIGHEST PAYOFF)

**Current bottleneck**: `flower/optim.py`, function `_zeropower_via_newtonschulz5` (line ~55). Each Newton-Schulz iteration runs:
```
A = X @ X.T                    # matmul 1
B = b * A + c * (A @ A)        # matmul 2 (A@A) + 2 element-wise ops
X = a * X + B @ X              # matmul 3 + element-wise op
```
That's 3 matmuls × 5 iterations = **15 separate cuBLAS kernel launches per 2D parameter per optimizer step**. Each launch reads/writes the full matrix from HBM. At 600M params with ~400 2D weight matrices, that's 6000 matmul launches per step.

**Fused kernel target**: A single Triton kernel that:
1. Loads X into shared memory or registers
2. Runs all 5 Newton-Schulz iterations in-register, keeping A and B as intermediate tiles
3. Fuses the initial normalization (`X = X / (X.norm() + eps)`) into iteration 0
4. Fuses the final transposition and dtype cast
5. Writes only the final result to HBM

**Challenge**: The matmuls are over the full weight matrix (e.g. 1024×4096 for a 600M model). This won't fit in shared memory. The kernel needs to operate in tiles (block_matmul pattern). The Newton-Schulz polynomial is applied to the full matrix, so tiling requires careful coordination — each tile needs the full A = X@X.T, which depends on all other tiles.

**Alternative approach (simpler)**: Instead of a fully fused kernel, fuse just the element-wise operations within each iteration:
- Current: `B = b * A + c * (A @ A)` is 3 ops (matmul + mul + add)
- Fused: a single Triton kernel that takes A, computes `b * A + c * A²`, and returns B
- This avoids 2 intermediate HBM writes per iteration (10 total across 5 iterations)

And fuse the normalization + first matmul launch:
- Current: `x = x / norm; x = x @ x.T` is 2 ops
- Fused: a single "normalize-then-matmul" kernel

**Expected speedup**: 1.5-3x on the optimizer step (currently ~5-10% of total step time). Net wall-clock: **3-8% total training speedup**.

**Priority**: HIGH. This runs on every parameter at every step. The kernel pattern (iterative matrix polynomial) is well-defined. No published fused Triton kernel for this exists.

### Opportunity 2: Fused Bloom Memory Routing (RESEARCH-SPECIFIC)

**Current bottleneck**: `flower/models/bloom_memory.py`, method `_bloom_route` (line ~82). Per layer per step:
1. K independent `nn.Linear` projections (K=4, loop at line 86) → K kernel launches
2. K softmax operations over memory_slots dimension → K kernel launches
3. Stack + mean → 2 kernel launches
4. Diagnostic entropy + KL computation (lines 94-99) → ~5 kernel launches

Total: ~5K + 7 kernel launches per layer per step. With K=4 and 28 layers: ~644 launches per step.

**Two-part fix**:

Part A (easy, do immediately) — **DONE** (see `flower/models/bloom_memory.py`, `tests/test_bloom_memory.py`, commit on branch `training-speedups`): Replace the Python loop over hashes with a single batched matmul. Instead of `nn.ModuleList` of K separate Linear layers, store the hashes as a single `(K, d_model, memory_slots)` weight tensor and compute all K projections in one `torch.einsum('bd,kds->bks', x, weights)`:

```python
# Instead of K separate nn.Linear:
self.hash_weights = nn.Parameter(torch.randn(K, d_model, memory_slots) * 0.05)

def _bloom_route(self, items):
    # items: (B, P, D)
    # Single batched matmul: (B, P, D) @ (K, D, S) -> (K, B, P, S)
    logits = torch.einsum('bpd,kds->kbps', items, self.hash_weights) / temp
    return logits.softmax(dim=-1).mean(dim=0)
```

This reduces K kernel launches to 1. No Triton needed.

Part B (Triton kernel): Fuse the softmax + mean + diagnostics into a single kernel. Input: `(K, B, P, S)` logits. Output: `(B, P, S)` averaged routing plan + scalar diagnostics. The kernel computes per-slot softmax across K hashes in-register, accumulates the mean, and optionally computes entropy/KL only when `compute_diagnostics=True`.

**Expected speedup**: Part A alone gives ~50% reduction in bloom routing kernel launches. Part B adds another ~20%. Since bloom routing is ~5-10% of total compute, net wall-clock: **2-5% total training speedup**.

**Priority**: MEDIUM. The batched matmul (Part A) is a 10-minute code change with no kernel writing. Do it first.

### Opportunity 3: Analytical Titans Surprise Gradient (RESEARCH CONTRIBUTION)

**Current bottleneck**: `flower/models/titans_mac.py`, method `_surprise_update` (line ~85). Per layer per step:
1. Creates a computation graph for the inner loss (requires_grad on memory leaf)
2. Forward pass through inner MSE retrieval loss
3. `torch.autograd.grad()` to get the surprise signal
4. Graph is created and destroyed every step

The autograd graph creation/destruction overhead is significant — it's the same overhead that makes `torch.autograd.functional.jacobian` slow compared to analytical gradients.

**The analytical shortcut**: The surprise signal is the gradient of MSE retrieval loss w.r.t. memory slots. This has a closed form:

```
Given:
  scores_s = <M_s, key> / sqrt(D)     for each slot s
  w_s = softmax(scores)_s              attention weight on slot s
  predicted = sum_s w_s * M_s
  loss = MSE(predicted, target)

∂loss/∂M_s = 2 * w_s * (predicted - target) / D
           + 2 * (w_s * (1-w_s) * <predicted-target, M_s>) / (D * sqrt(D)) * key
```

The first term is the direct gradient; the second is the through-softmax contribution. A custom kernel computing this directly (2 matmuls + element-wise) avoids the entire autograd machinery.

**Expected speedup**: Could halve the compute of the Titans variant specifically. But Titans is one variant among many.

**Priority**: LOW for speedup purposes. **HIGH as a research contribution** — computing Titans-style surprise without autograd is a novel optimization that nobody has published.

### Opportunity 4: Gated Diagnostics (immediate, no kernel needed)

Throughout Flower's memory modules, diagnostic computations (entropy, KL, routing stats, write magnitudes) are computed every forward pass via `self.last_diag_*` attributes. These are gathered by the diagnostic walker in `CausalLM.forward` (base.py:442-462).

Under `torch.compile`, diagnostics are already disabled (base.py:296-299). But under eager mode, they run every step. Gate them behind a `step % log_interval == 0` check:

```python
# In each memory module, replace unconditional diagnostic computation:
if self.training and step % 50 != 0:
    # Skip diagnostics except every 50 steps
    self.last_diag_bloom_routing_entropy = float('nan')
    self.last_diag_bloom_hash_divergence = float('nan')
else:
    # ... compute diagnostics ...
```

**Expected speedup**: 1-3% in eager mode. Zero under compile.

**Priority**: LOW, but zero-effort.

### Opportunity 5: Off-the-Shelf Fused Kernels (DO BEFORE WRITING CUSTOM)

Before writing any custom Triton kernels, adopt these existing libraries. They provide the biggest kernel-level wins with the least effort, and NVIDIA's own training pipeline uses them exclusively (NVIDIA does NOT write custom per-model CUDA kernels — they route everything through Transformer Engine).

Reference: arXiv:2509.25149 (NVFP4 pretraining — confirms NVIDIA delegates all kernel work to TE)
Reference: Nemotron training pipeline analysis (docs/research/nemotron-training-techniques.md)

#### 5a: Turbo-Muon Newton-Schulz Triton Kernels (DROP-IN for Opportunity 1)

This replaces the custom Newton-Schulz kernel proposed in Opportunity 1 with an existing, tested implementation.

**Source**: `tboissin/newton_schulz_triton` (HuggingFace Kernels), repo: `thib-s/flash-newton-schulz`
**Paper**: Boissin et al. 2025

Three fused Triton kernels replace the 15-separate-matmul chain:
- `ns_line_1(X)`: fused `A = X @ X.T` (Gram matrix)
- `ns_line_2(A, alpha, beta)`: fused `B = b*A + c*A@A` (4th-order polynomial)
- `ns_line_3(B, X, a)`: fused `C = a*X + B@X` (NS update step)

AOL preconditioning is fused into the first iteration, saving one NS iteration (4 vs 5). Reported: **2.8x faster** than baseline Muon NS, same convergence.

**Integration**: Replace `_zeropower_via_newtonschulz5()` in `optim.py`:
```python
from kernels import get_kernel
ns_kernel = get_kernel("tboissin/newton_schulz_triton")
```

**Priority**: P0. This is the single highest-leverage kernel change. 1 day of integration work, 2-3x optimizer step speedup.

**STATUS (2026-08-04): the launch-overhead half is DONE via batched `bmm` instead of the Triton kernel.** Profiling showed the optimizer step was launch-bound, not compute-bound (wall-clock 2.7× GPU self-CUDA; GPU starved by per-param Python dispatch). A fused single-matrix Triton kernel still launches once per param, so it would not have moved that wall-clock. Batched `bmm` over same-shape param groups (`flower/optim.py`, `muon_ns_batched: true`) collapses the launches and measured **2.6× full-step / 8.5× optimizer-step** speedup at the d512/L8/seq8192 shape — see `NEXT_IDEAS.md` section 5 update. The compute-fusion half (this Triton kernel) is deferred until profiling shows matmuls themselves, not launches, dominate; at current shapes they do not.

#### 5b: Liger-Kernel (Fused Linear CE, RMSNorm, SwiGLU, RoPE)

**STATUS (2026-08-04): FusedLinearCrossEntropy is DONE** (config flag
`model.fused_linear_ce`, default off). `liger-kernel>=0.8.1` was already a
dependency. The fused path never materializes the `(B*T, vocab)` logits tensor
during training; numerically exact vs eager (fp32 loss diff 4.8e-7, tied-weight
grad diff 2.2e-8). Measured memory at the 450M/seq=8192/vocab=16K shape:
**−0.83 GB (−4.1%)** — real but smaller than the >30% projected, because logits
are ~1 GB of a 20 GB budget at that ratio (the saving scales with B*T*vocab).
seq=32768 runs under fused CE but still OOMs on the 413M model via the SDPA
attention *mask*, not the head — fused CE is necessary-but-not-sufficient for
seq=32K and must pair with the compiled FlexAttention path (the bake-off
default). Full measurement and rationale: `NEXT_IDEAS.md` §6. RMSNorm/SwiGLU/
RoPE kernels remain P1 (separate task).

Open-source Triton kernel library for LLM training. ~20% total throughput, ~60% memory reduction.

**Source**: `link-org/Liger-Kernel` (arXiv:2410.10989)

Most impactful kernels for Flower:

- **FusedLinearCrossEntropy**: Chunks the lm_head projection + CE loss to avoid materializing full logits. At vocab=50K, seq=32K, this saves **gigabytes of activation memory** — potentially the difference between fitting seq=32K on the 5090 or not. Directly relevant to Flower's tied-embedding head.
- **FusedRMSNorm**: ~7x faster, ~3x less memory than standard RMSNorm
- **FusedSwiGLU**: Recomputes activation in backward, ~1.6x memory reduction
- **FusedRoPE**: ~8x faster

**Integration**: Drop-in replacements for `nn.RMSNorm`, `F.silu`, etc. FSDP-compatible.
```python
from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP, LigerLinearCrossEntropy
```

**Priority**: P0 for FusedLinearCrossEntropy (unlocks seq=32K), P1 for the rest.

#### 5c: Transformer Engine (FP4/FP8 Without Hand-Rolled Scaling)

NVIDIA's training infrastructure library. This is what Nemotron, the NVFP4 paper, and all NVIDIA training pipelines use for low-precision training.

**Source**: `NVIDIA/TransformerEngine` (pip install transformer-engine)

For Flower, TE replaces the hand-rolled FP8/FP4 scaling code proposed in Sections 3 and 11 with a maintained, NVIDIA-backed implementation:

- `te.Linear`: Auto-routes Fprop/Dgrad/Wgrad to FP8 or NVFP4 Tensor Cores with correct scaling
- Handles 2D block scaling (16x16 weights), amax history, scale factor bookkeeping automatically
- NVFP4 support: Random Hadamard Transform + stochastic rounding + per-block scaling — the full recipe from arXiv:2509.25149
- Fused kernels for LayerNormLinear, RMSNorm, etc.
- Runs on RTX 5090 (sm_120) — community-validated 20-30% speedup at ~560M params

**Critical stability lessons from Nemotron for Flower's precision routing**:
- Keep first 4 + last 4 GEMMs in BF16 under FP8/FP4 (Nemotron-H finding)
- Keep FP32 gradient reductions for the output/shared layer even under low precision (Nemotron 3 Ultra diverged at 8T tokens when this was violated)
- FP8 recipes may not generalize below 8B params — validate carefully at 350-600M

**Integration**: Replace `nn.Linear` with `te.Linear` for FFN and attention layers. Memory write path stays as standard `nn.Linear` in BF16.

**Priority**: P1. Adopts the full NVIDIA low-precision training stack. Reduces FP4/FP8 implementation to config changes rather than custom scaling code. However, TE adds a dependency and changes the model's module structure — validate that it doesn't break Flower's memory mechanism code paths.

#### 5d: Fused AdamW (for embedding/head/1D params)

**Source**: `anviit/triton-llm-kernels`

Fuses weight update + first moment + second moment + weight decay in a single kernel. 3.45x faster than PyTorch AdamW at 50M params.

Relevant to Flower's AdamW path (embeddings, 1D params, head) which runs alongside Muon.

**Priority**: P2. Smaller payoff than Muon fusion since AdamW handles fewer parameters, but still meaningful.

#### 5e: Nemotron Precision Recipes (STABILITY LESSONS)

These are specific precision-routing lessons from NVIDIA's Nemotron training pipeline (Nemotron-H 8B/56B, Nemotron 3 Ultra 550B). They validate and extend Flower's Section 13 precision routing with hard-won production findings.

Reference: Nemotron-H (arXiv:2504.03624), Nemotron 3 Ultra (tech report 2026)
Full analysis: `docs/research/nemotron-training-techniques.md`

**Lesson 1: Keep first 4 + last 4 GEMMs in BF16 under FP8/FP4**

Nemotron-H found that the first and last transformer blocks are most sensitive to quantization noise. Even when all middle layers run FP8, the first 4 blocks and last 4 blocks should stay BF16.

Flower's Section 13 currently only keeps the LM head and memory path in BF16. Extend this: keep the first and last N blocks fully in BF16 when using FP8/FP4. Config:

```python
# Number of blocks at start and end to keep in BF16 during mixed-precision training.
# Nemotron-H uses 4 on each side. At Flower's 28 layers, 3-4 is appropriate.
bf16_guard_blocks: int = 0  # 0 = disabled. Set to 3-4 when using FP8/FP4.
```

**Lesson 2: FP32 gradient reductions for output/shared layers**

Nemotron 3 Ultra hit a divergence at ~8T tokens when they reduced the output-layer gradient accumulation precision from FP32 to BF16. The MTP auxiliary loss contribution was lost in BF16's 7 mantissa bits. Fix: keep FP32 gradient reductions for the output layer even under FP4/FP8 elsewhere.

This is critical for Flower if using MTP (Section 8) + low precision: the MTP heads' gradients must be accumulated in FP32.

**Lesson 3: FP8 recipes may not generalize below 8B params**

Nemotron-H explicitly states: "We found it very important to do verification on a minimum of 8B parameters when constructing our FP8 recipe, as results with smaller models did not generalize."

At Flower's 350-600M scale, FP8/FP4 training is riskier than at 8B+. The precision progression should be:
1. BF16 everywhere (safe default, use for all architecture comparisons)
2. FP8 for FFN+attention linears only (via TE), validate loss curve matches BF16
3. FP4 for FFN only (via TE NVFP4BlockScaling), only for final scaling runs

Do NOT use FP8/FP4 for architecture bake-off comparisons — the precision noise could interact with memory mechanism signal in unpredictable ways.

**Lesson 4: MXFP8 on Blackwell (5090-only)**

Blackwell adds MXFP8 (Microscaling FP8): per-32-value block scaling instead of per-tensor. Uses E4M3 for all values (no E5M2 needed because block scaling handles dynamic range locally). More precise than per-tensor FP8 and avoids the "flush small values to zero" problem.

TE recipe: `MXFP8BlockScaling(fp_format=Format.E4M3)`

Only available on Blackwell (5090, B200, B300). Not on H100. At Flower's small scale, MXFP8 may give better accuracy than per-tensor FP8 — test it as an intermediate step between BF16 and FP4.

**Lesson 5: Async checkpointing**

Nemotron-H uses non-blocking checkpoint saves. PyTorch 2.3+ has `torch.distributed.checkpoint.state_dict_saver.async_save`. For Flower's multi-day 5090 runs or rented runs, this avoids the ~30-60s GPU stall per checkpoint. Low priority but free for multi-day runs.

```python
# In train.py checkpoint save:
import torch.distributed.checkpoint as dcp
dcp.save(state_dict, checkpoint_id=ckpt_path, planner=dcp.DefaultSavePlanner())
# PyTorch 2.3+ with async_save=True avoids blocking the training loop
```

---

## Section 15: Evaluation Probes for the 600M Scale-Up

### Why

Flower's existing evaluation suite was designed for 25M models where standard benchmarks (MMLU, HellaSwag) are pure noise. At 600M with seq=32K, the eval suite must measure three things: (1) whether the model has improved in absolute terms, (2) whether memory mechanisms now show measurable differences, and (3) whether the capabilities that emerge at this scale interact with the memory architecture.

The key principle from the emergence literature: use CONTINUOUS metrics (Brier score, token edit distance, per-token probability) alongside discrete metrics (accuracy, exact match). Discontinuous metrics can create false emergence signals. Continuous metrics reveal whether underlying improvements are smooth or phase-transitional.

Reference: Schaeffer et al. NeurIPS 2023 ("Are Emergent Abilities a Mirage?")
Reference: Poli et al. arXiv:2403.17844 (MAD — synthetic probes that predict scaling)

### The Evaluation Pipeline

The eval suite should run at three tiers, ordered by frequency:

#### Tier 1: Every-step monitoring (cheap, always on)

| Metric | What it measures | Implementation |
|--------|-----------------|----------------|
| **Training loss** | Next-token prediction quality | Already logged |
| **Validation BPB** | Bits-per-byte on held-out FineWeb — cross-model comparable | Already computed; ensure it uses held-out data, not train data |
| **Token throughput** | Tokens/sec — detects pipeline bottlenecks | Already logged; compute MFU from it |
| **MFU** | Model FLOPs Utilization = achieved / peak FLOPs | `MFU = (6 * N * tokens_per_sec) / peak_TFLOPS`. Should be 35-50% on 5090, 45-55% on H100 |

#### Tier 2: Every N steps (every 500-1000 steps, ~2-5 min each)

These are the probes that actually measure memory mechanism quality. **CRITICAL**: Flower's prior sweeps showed val-loss is nearly blind to memory mechanisms. These probes are the only way to see whether memory is helping.

| Probe | What it measures | Why it matters for Flower | Metric type |
|------|-----------------|--------------------------|-------------|
| **MQAR (Multi-Query Associative Recall)** | Multiple concurrent key-value retrievals from context | THE single best small-scale proxy for memory mechanism quality. Rank-correlated with compute-optimal perplexity at scale (Poli et al., MAD paper). Must be run at seq=32K to stress the memory. | Accuracy + Brier score |
| **Induction/copy head test** | Whether the model can copy patterns from earlier in the context | Detects whether induction heads have formed. Score >50% = functional induction heads. Must be tested at multiple distances (512, 2K, 8K, 16K tokens back). | Accuracy at each distance |
| **Selective copy probe** | Copy specific marked tokens from a long pattern, skip others | MAD discriminant — separates models that can compress/retrieve from those that can't. | Accuracy |
| **BLiMP minimal pairs** | Grammaticality judgments | >80% = reasonable syntactic competence. Flower's bloom_memory showed BLiMP collapse to chance at 25M — this tracks whether the architecture preserves grammaticality at scale. | Accuracy |
| **Long-range dependency probe** | Can the model use information from 16K-32K tokens ago? | Directly tests whether the memory mechanism is carrying information across the sliding-window boundary. This is Flower's core research question made into a probe. | Accuracy + per-token probability |

**MQAR implementation**: Generate synthetic sequences with multiple key-value pairs distributed across the context. At a query position, the model must retrieve the value associated with a key that appeared thousands of tokens earlier. Measure accuracy as a function of (a) number of KV pairs (difficulty) and (b) distance to the key (memory range). Compare vanilla_local vs. memory variants.

**Long-range dependency probe**: Insert a specific fact/word early in a long document (position 0-2K), then query it at the end (position 28K-32K). The model can only answer correctly if (a) full attention layers reach back that far, or (b) the memory mechanism carried the information forward. Run this with sliding-window-only (no full attention layers) to isolate the memory mechanism's contribution.

#### Tier 3: End-of-training evaluation (once, ~30-60 min)

| Probe | What it measures | Why |
|------|-----------------|-----|
| **FineWeb-Edu held-out BPB** | Final perplexity comparison to published models | Compare to Pythia-410M, SmolLM-360M, Pythia-1B |
| **tinyBenchmarks** (tinyMMLU, tinyHellaSwag, tinyARC) | 100-example subsets that correlate with full benchmarks | Use via lm-evaluation-harness with `--num_fewshot 0`. Directional signal only at 600M. |
| **Loss-by-position curve** | Per-token loss as a function of position in sequence | If memory helps, loss should be lower at positions FAR from any full-attention layer. Plot loss vs. distance-to-last-full-attention-layer. This directly visualizes the memory mechanism's contribution. |
| **Memory ablation delta** | BPB gap between memory-enabled and memory-disabled (vanilla_local) | At 25M this was <0.02. At 600M/seq=32K, the hypothesis is >0.03. If it's still <0.02, the memory architecture doesn't work at this scale either. |
| **Spectral entropy H̃ (if doing grokking/algorithmic probes)** | Normalized spectral entropy of representation covariance | If Flower runs algorithmic probes (modular arithmetic), H̃ crossing ~0.61 predicts generalization (arXiv:2604.13123). Advanced; skip unless doing grokking experiments. |

### The Metrics That Matter Most

If you can only add three things to the eval pipeline, add these:

1. **MQAR at seq=32K** — this is the single probe most likely to show a difference between memory variants. It's what the MAD paper validated as rank-correlated with compute-optimal scaling.

2. **Loss-by-position curve** — this is Flower's contribution made visible. If you plot per-token loss as a function of distance to the last full-attention layer, and the memory variant has lower loss at high distances, that's the key result. Nobody has published this visualization for compressed-memory architectures.

3. **Induction/copy at multiple distances** — this shows whether the model formed the circuits that make memory useful. If induction copy works at 512 tokens back but fails at 8K tokens back in the vanilla model, but works at 8K in the memory model, that's your mechanism working as designed.

### Implementation Notes

- **MQAR/induction/selective-copy probes**: These are synthetic data probes. They don't need a separate dataset — generate them on-the-fly during eval. Keep them short enough to run in <2 minutes total.
- **Loss-by-position**: Requires modifying the eval loop to record per-position loss instead of averaging. Add a `return_per_position_loss=True` flag to the eval function.
- **BLiMP**: Download the 67-paradigm minimal pairs dataset. Run as zero-shot grammaticality judgment (which sentence is more likely?).
- **tinyBenchmarks**: Use `lm-evaluation-harness` (pip installable). Run with `--num_fewshot 0` and `--tasks tiny_mmlu,tiny_hellaswag,tiny_arc`.
- **Continuous metrics**: For MQAR and induction probes, report both accuracy AND Brier score (mean squared error of the probability assigned to the correct token). Brier score catches improvements that accuracy misses (a model going from 49% to 51% accuracy might have improved its probability calibration significantly without crossing the 50% threshold).

### What NOT to Bother With at 600M

- **Full MMLU/HellaSwag/ARC**: Still noise below 1B. tinyBenchmarks gives the same directional signal in 1/100th the compute.
- **Chain-of-thought evaluation**: Requires >10B to be meaningful.
- **HumanEval/code evaluation**: Requires >3B to produce syntactically valid code consistently.
- **MT-Bench/AlpacaEval**: Chat benchmarks; need SFT first, irrelevant for base model research.

---

## What NOT to Implement (Architecture Changes That Confound Memory Research)

These are from the modded-nanogpt speedrun but should NOT be added to Flower because they change the model's information pathways:

- **Logit softcap** (records #18, #54): overfit to the 3.28 loss target
- **U-Net skip connections** (record #11): creates alternative information pathways, makes memory mechanisms redundant
- **Drop first MLP/attn layer** (records #30, #35): overfit to 124M param count
- **Value embeddings** (record #14): provides a shortcut that could make external memory unnecessary
- **Sparse attention gate** (record #28): adds a routing mechanism that interacts with memory mechanisms
- **ReLU² activation** (record #5): throughput hack at 124M; SwiGLU is better at Flower's scale
- **Smear module** (record #34): micro-optimization specific to the speedrun setup
- **Partial Key Offset, Bigram hash, Learnable XSA**: speedrun-specific micro-optimizations
- **DyT (Dynamic Tanh) as normalization replacement**: matches RMSNorm quality but the speedup vanishes under torch.compile. Adds α-tuning complexity. Tested in modded-nanogpt and diverged to NaN under Muon. Not worth it.
- **Tanh as FFN activation**: no evidence it beats SwiGLU in language models. SinGLU (tanh/sin-based GLU variant) beat SwiGLU on ViT but not on LM tasks.
- **SmearGate / BigramHash** (Parameter Golf): add shortcut token-pair pathways that compete with memory's role of carrying cross-window context.
- **mHC (Manifold-Constrained Hyper-Connections)** (DeepSeek-V4): widens residual stream and adds parallel residual pathways. Directly competes with memory mechanisms as an alternative info-routing channel. The Birkhoff-polytope stability constraint is interesting if Flower ever tests expanded residual streams, but the mechanism itself confounds memory comparisons.
- **CSA/HCA (Compressed/Heavily Compressed Attention)** (DeepSeek-V4): these ARE compression/memory mechanisms. Testing them alongside Flower's bloom/summary memory would conflate two compression designs.

## Research Variable Candidates (test as dedicated ablation axes, NOT defaults)

These ideas change information pathways and therefore MUST NOT be default settings. But they intersect Flower's research question directly and could be publishable findings if tested as separate ablation axes.

- **XSA (Exclusive Self Attention)** [arXiv:2603.09078, Apple; Parameter Golf #65]: After standard attention output `y_i`, subtracts the projection onto the self-value `v_i`: `z = y - (y·v̂)v̂`. This forces attention to capture ONLY contextual (orthogonal-to-self) information — the model cannot pass self-information through attention and must find another channel. Validated up to 2.7B; gains GROW with sequence length. **Theoretical alignment with Flower's thesis**: if attention can't carry self-information, the model may become MORE dependent on external memory → larger memory signal. Hypothesis: XSA + memory mechanisms are synergistic (memory fills the gap XSA creates). Test as: {vanilla, XSA} × {no-memory, bloom-memory, summary-memory}. This is a 2×3 grid that could be a paper on its own.
- **Partial RoPE** (16/64 or 64 dims): Zero-parameter. Apply rotary PE to only a subset of head dims; rest attend position-invariant. Minor ablation candidate.

The principle: throughput and optimizer improvements are safe. Architecture modifications are dangerous.

---

## Section 15: S14 training-step speedups (implemented) — CUDA graphs, warmup fix, activation checkpointing

Three training-step improvements, each validated on the RTX 5090 (torch 2.13.0+cu130, sm_120). Full measurement detail and rationale: `NEXT_IDEAS.md` §8–§10.

### 15.1 CUDA graphs (`compile_mode: reduce-overhead`) — works, shape-dependent

Switching `compile_mode` from `default` to `reduce-overhead` enables CUDA graphs (replays a captured GPU-side graph instead of re-launching each op), eliminating per-op CPU launch overhead. **Not a 1-line flip**: `reduce-overhead` + `gradient_accumulation_steps > 1` crashes with a known PyTorch bug (pytorch/pytorch#169545). Fix implemented in `flower/train.py`: pre-allocate persistent `.grad` buffers before CUDAGraph capture + `zero_grad(set_to_none=False)`. This is the only workaround that works in torch 2.13 (mark_step_begin / cloning do not). Correctness: loss matches `default` within 2e-5.

Measured (real `flower.train`, real fineweb_edu, 80 steps):

| config | shape | default | reduce-overhead | speedup |
|--------|-------|---------|-----------------|---------|
| 100m longctx | d768/L14, seq8192, b4/accum2, flex | 77k tok/s | 83k tok/s | **+7.7%** |
| 100m phase0 | d768/L14, seq1024, b16/accum4, SDPA | 90k tok/s | 84k tok/s | **-7.0%** |
| bloom bake-off | d512/L8, seq8192, b4/accum2, flex | 212k tok/s | 208k tok/s | **-2.0%** |

**Verdict:** reduce-overhead wins only on long-context compute-bound shapes; it is slower on short-sequence / high-launch-rate shapes. Flipped on `sweep13_100m_longctx_phase0.yaml` only; measured A/B before adopting on any other config.

### 15.2 Attention-window warmup recompile bug — fixed (latent; warmup off in production)

`attn_warmup_steps > 0` + `flex_attention` + compile caused `create_block_mask` to recompile once per distinct window value (flex's `mask_mod` closes over the window integer), exhausting torch.compile's recompile limit (8) and falling back to the eager dense-flex path (throughput collapsed to 16k tok/s). Fix: `ModelConfig.attn_warmup_quantize` — a step-stride that holds the window constant every N steps, bounding recompiles to `ceil(warmup_steps/quantize)`. With `quantize=8`: 8→3 recompiles, 0 limit hits, loss unchanged, legacy per-step ramp (`quantize=0`) preserved. All production configs have `attn_warmup_steps=0`, so this unblocks future warmup+compile use.

### 15.3 Activation checkpointing — the seq=32K enabler

`ModelConfig.activation_checkpoint: bool = False` wraps each transformer block in `torch.utils.checkpoint(..., use_reentrant=False)` during training, cutting activation memory from O(num_layers) to O(1). Correctness: loss **bit-identical** with dropout>0 (RNG state preserved).

**Non-obvious blocker, fixed:** checkpoint + FlexAttention are incompatible (pytorch/pytorch#147879) — the recompute forward falls back to flex's eager dense path and OOMs. Fix: compile `flex_attention` as a standalone callable (`_load_flex_attention_compiled` in `base.py`) so the fused kernel runs during recompute too.

Measured memory:

| shape | no-checkpoint | checkpoint | result |
|-------|--------------|-----------|--------|
| 100M, seq8192, b4 | 16.45 GB | 6.97 GB | **-58%** |
| 100M, seq32768, b1 | OOM (29.8 GB fwd) | 11.09 GB | **fits** |
| 100M, seq32768, b2 | OOM | 13.75 GB | fits with headroom |

**seq=32K now fits on the 5090** — the Section 13 gate (memory mechanisms at long context) is open on local hardware. Throughput cost ~25-33% (one extra forward per backward); favourable when memory is the binding constraint. Off by default so seq≤8K throughput runs are unaffected.
