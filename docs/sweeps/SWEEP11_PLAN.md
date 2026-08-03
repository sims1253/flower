# Sweep 11 — Tapered compactor budgets (does the TLM "wider-early" result apply to KV compaction?)

## TL;DR

Vary the compactor's per-layer `d_latent` across the 6 layers (wider early, narrower late) under a fixed total trainable-param budget, and test whether the TLMs paper's "later layers reinforce rather than transform, so they need less capacity" finding transfers to KV-cache compaction. Param-matched within ±0.82%, 3 seeds, held-out `val_perplexity`, decisive non-overlapping-range comparison. Three arms are the core: **uniform (anchor) vs taper-early (predicted win) vs taper-late (wrong-direction control, predicted loss)**.

## Hypothesis

Under a fixed total compactor budget, **wider-early / narrower-late `d_latent` allocation beats uniform**, and the *direction* matters more than the amount: taper-late should be *worse* than uniform. Mechanism (from Bayat, TLMs): early layers must compress more diverse, less-abstracted information and so need more representational capacity; later layers operate on already-summarised latents and mostly refine, so they need less. This was shown for transformer MLP width; here we ask whether the same allocation principle governs the **KV-compactor's latent bottleneck**. The compactor is a Perceiver whose `d_latent` is the width of its latent bank and all internal projections (`still.py:117-160`), so it is the direct analogue of per-layer MLP width. A positive result reproduces TLMs in a new subsystem (compaction, not FFN); a tie says compaction obeys a different capacity law; a clean taper-late loss reproduces the TLM directionality finding. All three are publishable-ish.

## 1. Why this is cheap and orthogonal

- The sweep10 infrastructure (param-matched arms, 3 seeds, held-out val, `scripts/sweep10_analyze.py`) is reused verbatim.
- Same frozen 12.2M base model (d_model 384, 6 heads → head_dim 64, 6 layers).
- The ONLY code change is letting `d_latent` be a per-layer schedule instead of a scalar — a small, backward-compatible addition (see §2).
- No flow, no spectral, no OT — pure standard `StillCompactor`. Isolates the taper question.

## 2. Code change required (per-layer `d_latent`)

### Current state (the gap)

- `still_d_latent` is a single scalar in `ModelConfig` (`flower/config.py:106`).
- `build_still_model` reads it once (`still_lm.py:623`) and threads it to `StillLM.__init__` as `d_latent=dl` (`still_lm.py:646`).
- `StillLM.__init__` builds one compactor per layer but passes the **same** `d_latent=dl` to every layer (`still_lm.py:160`).
- The existing `layer_adaptive` / `_pyramid_budget` mechanism (`still_lm.py:45-59`, `config.py:111`) only varies `compact_len` per layer — it does **not** vary `d_latent`. So this is genuinely new surface, not a re-skin of PyramidKV.

### New config fields (`flower/config.py`, ~3 lines added near line 120)

```
still_d_latent_schedule: list[int] | None = None   # per-layer d_latent; None = uniform (current)
still_d_latent_taper: str | None = None            # "early_cosine" | "late_cosine" | None (convenience)
```

Two ways to specify the schedule; the explicit list wins:
- **Explicit list** (`still_d_latent_schedule`): e.g. `[152, 146, 134, 120, 108, 104]`. Length must equal `num_layers`; entries must be even (RoPE constraint, `still.py:53`). All the arms below use this form — most transparent and least error-prone.
- **Convenience shorthand** (`still_d_latent_taper`): `"early_cosine"` / `"late_cosine"` expands to the canonical schedule `[152,146,134,120,108,104]` or its reverse. Saves typing in the YAML for the two main arms. (Optional; the explicit list is sufficient and is what the param audit below validates.)

### Constructor change (`still_lm.py`)

In `StillLM.__init__`:

1. Accept the schedule. Replace the single `dl` (line 126) with a per-layer list when provided. Sketch:
   - After line 126 (`dl = d_latent or 2 * head_dim`), compute:
     - `d_latent_schedule_cfg = getattr(config, "still_d_latent_schedule", None)` (or thread as a new `__init__` kwarg from `build_still_model`).
     - If set: validate `len == num_layers`, all even, all ≥ 2; `layer_d_latents = list(schedule)`.
     - Else: `layer_d_latents = [dl] * num_layers` (current behavior).
   - Store `self.layer_d_latents = layer_d_latents` (mirrors the existing `self.layer_compact_lens` at line 152).
2. In the per-layer loop (line 154-183), change `d_latent=dl` (line 160) to `d_latent=layer_d_latents[i]`.

In `build_still_model` (`still_lm.py:646-673`): read `still_d_latent_schedule` via `getattr` (line ~624, next to `d_latent`) and pass it to `StillLM(...)`.

### Identity-init constraint (important — load-bearing)

`StillCompactor._init_identity` (`still.py:321-357`) hard-codes the identity output heads as `key_proj.weight[:d, :d] = eye(d)` and `val_proj.weight[:d, d:2*d] = eye(d)` (lines 356-357). This is only valid when **`d_latent == 2 * head_dim`** (i.e. 128 here). For any other `d_latent`, the slice assignment throws `RuntimeError` (verified: `Target sizes [64,16]. Tensor sizes [64,64]`). Sweep10 already hit this: arm C used `d_latent=192` and the standard path silently built it with `identity_init=True` only because... actually it would have crashed too — **the taper arms with `d_latent ≠ 128` on layers 0-4 (and all of arms B/C/D) MUST set `identity_init=False` for those layers.** This is a required part of the change, not optional:

- In the per-layer loop: `identity_init = (layer_d_latents[i] == 2 * head_dim)`.
- This is safe: identity-init is a training-stability nicety, not a correctness requirement, and sweep10's arms C/E already trained fine with non-identity inits (d_latent 192/144).

### Backward compatibility

Default `still_d_latent_schedule = None` → `layer_d_latents = [dl]*num_layers` → identical to today. The existing `still_d_latent` scalar still works unchanged. No existing config breaks; `pytest tests/test_shapes.py` stays green.

### Lines of code

~15-20 lines total: 2 config fields, ~6 lines in `StillLM.__init__` (schedule resolution + the `identity_init` guard + the loop change), ~3 lines in `build_still_model`. No changes to `still.py` (the compactor itself already takes `d_latent` and `identity_init` as kwargs).

## 3. Param audit (real, computed by instantiating `StillCompactor`)

Base geometry: `num_kv_heads=6, head_dim=64, compact_len=64, num_blocks=2, num_layers=6` (the sweep10 defaults). Param count per layer vs `d_latent` (all `identity_init` off for d≠128):

| d_latent | per-layer params |
|----------|-----------------|
| 76  | 147,744 |
| 84  | 171,360 |
| 104 | 237,120 |
| 108 | 251,424 |
| 120 | 296,640 |
| 128 | 328,704 |
| 134 | 353,760 |
| 138 | 370,944 |
| 146 | 406,464 |
| 152 | 434,112 |
| 162 | 482,112 |
| 172 | 532,512 |

Per-layer params grow roughly **quadratically** in `d_latent`, so a symmetric cosine taper *overshoots* the uniform total. The schedules below were chosen by grid search to land inside ±2.5%.

### Arm totals

| arm | schedule (per-layer d_latent) | total trainable compactor | diff vs uniform-128 |
|-----|-------------------------------|---------------------------|---------------------|
| **A uniform_128** (anchor) | `[128,128,128,128,128,128]` | **1,972,224** | 0.00% |
| **B taper_early** (cosine 152→104) | `[152,146,134,120,108,104]` | **1,979,520** | **+0.37%** |
| **C taper_late** (wrong dir = B reversed) | `[104,108,120,134,146,152]` | **1,979,520** | **+0.37%** |
| **D taper_steep** (cosine 172→76) | `[172,162,138,108,84,76]` | **1,956,096** | **−0.82%** |

All four arms are within ±0.82% of the anchor. **Critically, B and C have byte-identical totals** (per-layer params are a pure function of `d_latent`, so reversing the schedule reverses the per-layer counts but sums to the same total). This is the cleanest possible direction control: the only thing that differs between B and C is *which layer* gets the width.

## 4. Arms

| arm | purpose | schedule | predicted outcome |
|-----|---------|----------|-------------------|
| **A_uniform_128** | anchor / null control (identical to sweep10 arm A) | `[128]*6` | baseline val_ppl ≈ 2.848 |
| **B_taper_early** | the hypothesis (TLM direction) | `[152,146,134,120,108,104]` | **wins** if TLM transfers |
| **C_taper_late** | wrong-direction control (TLM: ~1 ppl worse) | `[104,108,120,134,146,152]` | **loses**; same params as B |
| **D_taper_steep** | steepness probe | `[172,162,138,108,84,76]` | exploratory steepness axis |

Decisive comparisons:
- **B vs A** — does tapering help at all?
- **B vs C** — does *direction* matter (the TLM signature)? Same params, only order differs. **Headline test.**
- **D vs B** — steepness sensitivity (secondary).

## 5. Config draft (`configs/sweep_still_taper.yaml`)

```yaml
sweep:
  name: sweep_still_taper

  defaults:
    model:
      vocab_size: 4096
      d_model: 384
      num_heads: 6
      num_layers: 6
      ffn_dim: 1536
      max_seq_len: 1024
      local_window: 256
      rope_base: 10000.0
      dropout: 0.0
      still_compact_len: 64
      still_num_blocks: 2
      still_d_latent: 128
      still_kl_topk: 200
      still_kl_weight: 1.0
      still_ce_weight: 0.1
      still_compact_from_step: 1500
      still_kl_temperature: 1.0
      still_base_warmup_steps: 1500

    data:
      dataset: fineweb_edu
      tokenizer: "custom:tokenizers/fineweb_4k.json"
      sequence_length: 1024
      eval_seq_len: 1024

    training:
      # SWEEP10 LESSON: std_7M (d_latent=192) spiked to ~32GB and crashed at
      # batch 32x2. Taper arm D has d_latent=172 on layer 0. KEEP batch 16x4.
      batch_size: 16
      gradient_accumulation_steps: 4
      steps: 6500
      lr: 0.003
      warmup_steps: 200
      lr_schedule: linear_warmup
      grad_clip: 1.0
      eval_interval: 500
      checkpoint_interval: 1000
      save_checkpoints: true
      device: auto
      seed: 0
      seeds: [0, 1, 2]
      log_backend: tensorboard
      composite_eval: false
      optimizer: muon
      muon_lr: 0.003
      muon_momentum: 0.95
      validation_steps: 20

  variants:
    - name: A_uniform_128
      model: { variant: still }

    - name: B_taper_early
      model:
        variant: still
        still_d_latent_schedule: [152, 146, 134, 120, 108, 104]

    - name: C_taper_late
      model:
        variant: still
        still_d_latent_schedule: [104, 108, 120, 134, 146, 152]

    - name: D_taper_steep
      model:
        variant: still
        still_d_latent_schedule: [172, 162, 138, 108, 84, 76]
```

## 6. Verification steps (hard-gate before launching)

1. **Param audit (no training).** Instantiate `StillLM` per arm via `build_still_model`, print `diagnostics["compactor_parameter_count"]`. Hard gate: all four within ±2.5% of 1,972,224.
2. **Identity-init guard check.** Arms B/C/D must build without the `RuntimeError` from `still.py:357` — confirm `identity_init=False` for layers where `d_latent != 128`.
3. **Shape sanity.** Forward pass on dummy `(B=2, T=128)`, assert `logits.shape == (2,128,4096)`, loss finite.
4. **Pytest.** `pytest tests/test_shapes.py -q`.
5. **30-step smoke** on arm D (widest early layer, d_latent=172): confirm `val_perplexity` in metrics, peak VRAM < 12GB at batch 16x4.

### GPU-memory note (explicit, per the sweep10 lesson)

Sweep10's `C_std_7M` arm crashed repeatedly during the compaction-phase memory spike at batch 32x2. The fix was batch 16×4, which held peak VRAM ~8-10GB. Taper arm D has `d_latent=172` on layer 0 (close to the crashing 192), so the same risk applies. **All arms use batch 16×4; do not raise it.**

## 7. Analysis plan

Copy `scripts/sweep10_analyze.py` → `scripts/sweep11_analyze.py` with new `arm_map` and `compare()` calls for B-vs-A, B-vs-C, D-vs-B.

## 8. What "interesting" looks like

1. **Taper-early clearly wins (B < A).** Free win — reproduces TLM wider-early result in a *different subsystem* (KV compaction, not FFN).
2. **Taper-late clearly loses (C > A) and B beats C.** Confirms *directionality* — the capacity law is specifically early>late. B-vs-C is the cleanest possible test (identical params, only order differs).
3. **Tie.** Compaction does *not* obey the reinforce-not-transform pattern — the Perceiver's cross-attention distributes compression work more uniformly than an FFN. Null but discriminating.

## 9. Cost

4 arms × 3 seeds × 6500 steps = 12 runs, all standard (no flow overhead). Cheap.
