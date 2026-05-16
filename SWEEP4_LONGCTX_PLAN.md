# Sweep 4 — Phase 1 long-context retrofit + RoPE

## Context

The Sweep 4 Phase 1 memory bake-off is currently configured for seq_len=512,
batch=128 (`configs/sweep4_phase1_memory_bake_off.yaml`). Memory architectures
(summary_memory, titans_mac, flow_meanflow, flow_pma) gain most of their value
when memory slots have many tokens to compress over. At 512 tokens × 16 slots
each slot sees only ~32 tokens before being overwritten — barely enough to
test the architectural hypothesis.

The decision: bump Phase 1 to **seq_len=2048, batch=32** (preserves
tokens-per-step ≈ 65k) and adopt **RoPE** in place of the absolute position
embedding so the trained models are length-extendable (target post-training:
5–50k context via YaRN or position interpolation).

This is an explore-phase change. Do not change anything that isn't necessary
to support 2048-seq + RoPE. In particular: do not touch the optimizer, the
tokenizer, the memory architectures themselves, or the methodology fixes
already in place.

---

## Files to modify

### 1. `flower/models/base.py` — replace absolute position embedding with RoPE

Two edits.

**Edit A: Add RoPE rotation in `CausalSelfAttention.forward`.**

Add a `RotaryEmbedding` helper class near the top of the file:

```python
class RotaryEmbedding(nn.Module):
    """Standard RoPE (Su et al. 2021).

    Caches cos/sin tables up to max_seq_len. The base frequency (default 10000)
    controls the position-encoding period; YaRN/PI length-extension at the end
    of training will rescale this, but during training keep base=10000.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (T, head_dim/2)
        # Buffers: (1, 1, T, head_dim) for broadcasting over (B, H, T, head_dim).
        self.register_buffer("cos", freqs.cos().repeat_interleave(2, dim=-1).view(1, 1, max_seq_len, head_dim), persistent=False)
        self.register_buffer("sin", freqs.sin().repeat_interleave(2, dim=-1).view(1, 1, max_seq_len, head_dim), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Interleaved layout: rotate pairs (x[..., 2i], x[..., 2i+1]).
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, H, T, head_dim). cos/sin cached at max_seq_len, slice to T.
        t = q.shape[-2]
        cos = self.cos[..., :t, :].to(dtype=q.dtype)
        sin = self.sin[..., :t, :].to(dtype=q.dtype)
        return (q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin)
```

Update `CausalSelfAttention.__init__` to build a `RotaryEmbedding` whose
`head_dim = config.d_model // config.num_heads` and `max_seq_len =
config.max_seq_len`. Store it as `self.rope`.

Update `CausalSelfAttention.forward` to apply RoPE to `q` and `k` (NOT `v`)
right after the `_split` call, before computing scores. Concretely:

```python
q, k, v = self._split(q), self._split(k), self._split(v)
q, k = self.rope(q, k)
mask = causal_mask(...)
...
```

The RBF-bias branch must also use the rotated q/k. Keep `scaled_dot_attention`
unchanged.

**Edit B: Remove the absolute position embedding from `CausalLM`.**

In `CausalLM.__init__`, delete `self.pos = nn.Embedding(config.max_seq_len,
config.d_model)`. In `CausalLM.forward`, replace:

```python
pos = torch.arange(input_ids.shape[1], device=input_ids.device)
x = self.token(input_ids) + self.pos(pos).unsqueeze(0)
```

with simply:

```python
x = self.token(input_ids)
```

### 2. Custom-forward variants — same pos-embed removal

The following variants have their own `forward` that bypasses `CausalLM`. They
each currently maintain their own `self.pos = nn.Embedding(...)`. Remove it
and stop adding it to `x`:

- `flower/models/engram_lite.py` — `EngramLiteLM.__init__` / `.forward`
- `flower/models/flow_meanflow.py` — `MeanFlowMemoryLM.__init__` / `.forward`
- `flower/models/flow_pma.py` — `FlowPMALM.__init__` / `.forward`

Do NOT modify:
- `flower/models/fla_layer.py` — FLA's GatedDeltaNet handles positions internally via short-conv + delta rule. Adding RoPE on top would double-position-encode.
- `flower/models/flow_attention.py`, `flow_attn_flow.py`, `flow_attn_summary.py` — legacy Sweep-1 variants, not in the Phase 1 bake-off. Leaving them on absolute pos embed is fine.
- `flower/models/titans_mac.py` — already uses `CausalLM` base, gets the fix free from Edit B.
- `flower/models/summary_memory.py`, `flower/models/phase_memory.py`, `flower/models/linear_memory.py`, `flower/models/partitioned_memory.py`, `flower/models/vanilla.py` — all use `CausalLM` base, get the fix free.

### 3. `flower/config.py` — add RoPE config field

Add to `ModelConfig`:

```python
# Sweep 4: RoPE base frequency. 10000 is the standard; raising it to 50000 or
# 500000 supports longer post-training context extension via YaRN/PI without
# retraining from scratch. Keep 10000 for the Phase 1 bake-off.
rope_base: float = 10000.0
```

Pass `config.rope_base` to `RotaryEmbedding` in `CausalSelfAttention.__init__`.

### 4. `configs/sweep4_phase1_memory_bake_off.yaml` — long-context settings

In `sweep.defaults`:

```yaml
data:
  sequence_length: 2048   # was 512
training:
  batch_size: 32           # was 128 — preserves ~65k tokens/step
model:
  max_seq_len: 2048        # was 512
  local_window: 128        # was 64 — scale with seq_len; keep ratio ≈ 1/16
  rope_base: 10000.0
```

Leave `memory_slots: 16` unchanged — the whole point is to give each slot more
tokens to compress.

### 5. `configs/sweep4_phase0_remuon.yaml` — also bump to 2048

Same `sequence_length`, `max_seq_len`, `batch_size`, `local_window` changes as
Phase 1. The Muon ladder should be tested at the same regime Phase 1 will run
at, otherwise the chosen Muon config won't transfer.

### 6. `tests/test_shapes.py` — extend max_seq_len in the test config

The existing test uses `max_seq_len=16`. RoPE works at any even head_dim and
any T ≤ max_seq_len, so the test should keep passing. But verify by running
both the CPU and CUDA variants after the change. If any test fails because
RoPE's cached cos/sin tables don't cover a seq_len, that's a real bug — fix
it by sizing the cache to `max(max_seq_len, T_test)`.

---

## What will break and how to handle it

### Old checkpoints will not resume

Every existing checkpoint under `runs/local_5090/` was trained with an
absolute pos embedding parameter that no longer exists. `model.load_state_dict`
will fail with "unexpected keys: ['pos.weight']" (or per-variant equivalent).

This is expected. Do not add backwards-compatibility shims. The roadmap
already calls for fresh runs after the bug fix anyway; the bug-fix reruns
(`runs/local_5090/fixed_bug_reruns*`) are the last set of checkpoints worth
preserving, and those are at seq=512 so they wouldn't have been usable for
the Phase 1 bake-off regardless.

Action: do not delete old checkpoints, but expect Phase 1 to start from
scratch.

### Probe pattern lengths

`flower/probes/composite.py`'s `induction_copy_probe` uses
`pattern_len = max(4, min(32, cfg.model.max_seq_len // 4))`. At
max_seq_len=2048 this caps at 32 (unchanged behavior). `associative_recall_probe`
uses `seq_len = min(cfg.model.max_seq_len, max(32, pairs*2+8))`. At
max=2048, pairs=8, this gives seq=24 (unchanged behavior).

Neither probe needs modification. BLiMP-mini sentences fit easily in any
seq_len ≥ 100.

### Activation memory at seq=2048, d=384, batch=32, 6 layers

Rough budget: ~12 × 32 × 2048 × 384 × 6 ≈ 1.8 GB bf16 activations. Well
within a 5090's 32 GB. If a real run OOMs, the first knob to turn is
gradient checkpointing on the FFN (cheap, ~30% memory savings), not seq_len
or batch.

### FLA tokens-per-second at seq=2048

FLA's GatedDeltaNet chunked Triton kernel was designed for long sequences;
expect it to be RELATIVELY faster vs. softmax attention at seq=2048 than at
seq=512. This may shift the wall-clock plot in FLA's favor compared to a
seq=512 run. That is the correct measurement, not a confound.

---

## Verification

After all edits, run in order:

1. **Tests pass** —
   ```
   uv run python -m pytest tests/test_shapes.py tests/test_causal.py tests/test_flow_inverse.py -x -q
   ```
   All CPU variants and the CUDA fla_gdn variant should pass.

2. **Smoke train each Phase 1 variant** (synthetic data, tiny model) —
   ```
   for v in summary_memory engram_lite flow_meanflow flow_pma titans_mac fla_gdn; do
     uv run python -m flower.train --variant $v --smoke || echo "FAILED: $v"
   done
   ```
   All six should print a metrics block and exit 0.

3. **Smoke composite eval each variant** —
   ```
   for v in summary_memory engram_lite flow_meanflow flow_pma titans_mac fla_gdn; do
     uv run python -m flower.eval --variant $v --smoke --composite || echo "FAILED: $v"
   done
   ```
   All six should produce a `composite_ranker` block.

4. **Real-scale single-variant sanity** (NOT the full sweep — just confirm
   the new config loads and runs ~100 steps on the 5090):
   ```
   uv run python -m flower.train --config configs/sweep4_phase1_memory_bake_off.yaml \
     --variant summary_memory --steps 100 --output-dir runs/longctx_sanity
   ```
   Should report tokens/sec ≥ 5000 on a 5090 and not OOM. If it OOMs,
   reduce `batch_size` to 16 in the config and re-test before launching the
   full sweep.

5. **Confirm RoPE is actually being applied.** Quick sanity check:
   ```python
   import torch
   from flower.config import ModelConfig
   from flower.models.base import CausalSelfAttention
   cfg = ModelConfig(d_model=64, num_heads=4, max_seq_len=128)
   attn = CausalSelfAttention(cfg, local_window=16)
   assert hasattr(attn, "rope"), "RoPE module missing"
   assert attn.rope.cos.shape[-2] >= 128, "RoPE cache too short"
   x = torch.randn(2, 32, 64)
   out = attn(x)
   assert out.shape == x.shape
   print("RoPE wired correctly.")
   ```
   Run this as `uv run python -c "..."` and confirm.

---

## Out of scope (do NOT touch in this change)

- The MeanFlow/OT-CFM training objective in `flow_meanflow.py`.
- The Titans gradient-surprise mechanism in `titans_mac.py`.
- The composite ranker probes in `flower/probes/composite.py`.
- Optimizer choice or LR schedules.
- Tokenizer (still custom 4K BPE on FineWeb-Edu).
- The Muon optimizer or `flower/optim.py`.
- Dataset choice — still FineWeb-Edu. The dataset swap is exploit-phase work.
- Length-extension (YaRN / PI) — that's a post-training step. Not now.

---

## Commit message suggestion

```
Sweep 4 Phase 1: RoPE + seq=2048 retrofit

Replace absolute position embeddings with RoPE (base=10000) in CausalLM and
in the three custom-forward variants (engram_lite, flow_meanflow, flow_pma).
Bump Phase 0 + Phase 1 sweep configs from seq=512/batch=128 to seq=2048/
batch=32, scaling local_window 64→128. fla_gdn unchanged (FLA handles
positions internally). Old checkpoints with pos.weight will not load —
Phase 1 starts fresh.
```
