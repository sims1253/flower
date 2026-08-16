# Implemented but never switched on — audit, August 2026

Every `ModelConfig` / `TrainingConfig` / `DataConfig` field cross-referenced
against every file in `configs/`. A field listed here is implemented in the
model or trainer, validated, in most cases test-covered — and set by no config
that has ever run.

**Why this list is worth keeping.** The NorMuon precedent: it sat in
`flower/optim.py` as a switched-off "implemented" feature from S5 until the 1500
step Muon screen finally ran it, at which point it measured **+0.517 val_bpb** —
a global Frobenius normalisation that had been silently cutting the Muon LR ~36x.
Implemented-and-unrun is not the same as working. Everything below is a
hypothesis with code attached, not a feature.

Method: `dataclasses.fields()` over each config class, regex for `^\s*#?\s*<name>:`
across `configs/*.yaml`. Fields whose default is what every config wants
(`output_dir`, `text_field`, `memory_param_patterns`, `muon_ns_batched`) are
excluded as uninteresting.

---

## Dead: validated config with no consumer

These parse, validate, reject bad values — and are read by nothing outside
`flower/config.py`. Setting `ffn_precision: fp8` today validates cleanly and has
no effect. That is worse than not having the option.

| field | where it lives |
|---|---|
| `ffn_precision` | `config.py:322`, validated `config.py:344` |
| `attn_precision` | `config.py:323`, validated `config.py:346` |
| `memory_precision` | `config.py:324`, validated `config.py:348` (bf16-locked by design) |
| `head_precision` | `config.py:325`, validated `config.py:354` |
| `bf16_guard_blocks` | `config.py:326` — no validation, no consumer, no test |

The per-module precision idea they encode was superseded: `flower/precision.py`
does the FP8 conversion by module *class* with `fp8_keep_bf16_blocks` for the
first/last-block margin, which is the thing that actually shipped and measured
1.305x. These five are the fossil of the earlier design. Either delete them or
wire `bf16_guard_blocks` to the margin logic it duplicates.

---

## Live and plausibly useful, never run

### Multi-Token Prediction — `mtp_extra_heads`, `mtp_weight`

Untied auxiliary heads predicting t+2, t+3, ... on the already-normed hidden
state; the main tied head still does t+1. Wired into **both** loss paths,
including the Liger fused-CE one (`flower/models/base.py:1131` and `:1147`).
Test-covered. Default 0 heads.

This is the highest-value unused item, for two reasons. It is a sample-efficiency
play on an axis the project has now shown is live, and
`docs/research/frontier-speedups-2026-08.md` records that MTP gains are
**additive with TST** — which is the 2.5x throughput item already in flight
(`configs/tst_screen_450m.yaml`). Cost is extra `vocab_size x d_model` heads
(~21M params each at 16K vocab / d1280) plus their CE, which is not free at
accum 16.

### Attention window warmup — `attn_warmup_steps`, `attn_warmup_start`, `attn_warmup_quantize`

Linearly ramps `local_window` from `attn_warmup_start` to `local_window`
(`flower/train.py:240`). The tricky part is already solved and documented: each
distinct window forces a FlexAttention `create_block_mask` recompile because
`mask_mod` closes over the window, so an unquantised ramp exhausts
`torch.compile`'s recompile limit and silently drops flex to the eager dense
path. `attn_warmup_quantize` is a step-stride that bounds the recompile count.

Two distinct reasons to want it, worth not conflating:
- **Throughput**: a smaller window early is cheaper attention for free.
- **Research**: `frontier-speedups` §4 records the SWAX finding that *shorter*
  windows improve long-context performance by forcing reliance on memory — and
  flags that window 2048 at seq 8192 may be too generous for Flower's memory
  thesis to be testable. A ramp is one way to probe that.

### `fp8_lm_head`

Implemented in `precision.py` / `base.py:702`, test-covered. Note
`docs/profiling/speedup_results.md` states the tied LM head is deliberately never
converted, so read that guardrail before enabling — the flag may exist for the
eval-time path only.

### `orthogonal_init`

`base.py:761`, `_apply_orthogonal_init` at `:794`, test-covered. Cheap one-shot
A/B; no strong prior either way.

### `data.num_workers`, `data.prefetch_factor`

Newly plumbed (the old code accepted and silently dropped them on the training
path). Deliberately not worth raising: `speedup_results.md` measures the loader
at **1,079,869 tok/s against a model consuming ~55K**, i.e. 20x headroom. Listed
only so the next person does not re-derive that. Raising them also changes data
*order*, so runs at different worker counts are seed-comparable, not
bit-comparable.

---

## Research-axis variants, never run

Implemented model options belonging to the memory/architecture research axis
rather than to throughput. Listed for completeness; each is a new arm, and
`docs/training-speedups.md` is explicit that architecture changes confound the
memory-mechanism comparison.

`num_memory_banks` + `bank_router_temperature` (`partitioned_memory.py`, **no
test coverage** — the only item here with none), `titans_analytical_surprise`
(`titans_mac.py:159`; note its perf test currently fails on a contended box,
measuring 0.68x rather than a speedup), `noise_std`
(`hamiltonian_attention.py:88`), `deep_shallow_flow` (`flow_attention.py:26`),
`meanflow_ot_cfm` / `meanflow_ot_epsilon` / `meanflow_ot_iters` /
`meanflow_loss_weight` (`flow_meanflow.py:137`), `still_ot_reg_weight` /
`still_key_velocity_hidden` / `still_val_velocity_hidden` (`still_lm.py:793`),
`flow_step_size`, `summary_style`, `memory_kernel_bias`,
`memory_update_frequency`, `local_attn_rbf_scale`, `orthogonal_eps`.

---

## How to measure each of these

Benchmarks are written and smoke-tested; none have been run on the GPU, which
was occupied by the 450M job throughout. Nothing below has a number yet.

| what | tool | why that tool |
|---|---|---|
| MTP throughput + VRAM | `scripts/bench_arms.py --config configs/perf_bench_450m.yaml` | arms `mtp_1_head`, `mtp_2_heads`, `mtp_2_heads_unfused` |
| `fp8_lm_head` | same | arm `fp8_lm_head` |
| window cost curve | same | arms `window_512/1024/4096` — the ramp's price list |
| **window warmup itself** | `scripts/bench_attn_warmup.py` | bench_arms *cannot* measure it (see below) |
| **optimizer cost** (CM, Aurora, per-head) | `scripts/bench_optimizer_step.py` | bench_arms *cannot resolve* it (see below) |
| quality of any of it | `configs/comp_muon_screen_450m.yaml` + an MTP screen | none of the above measure quality |

Two of these need their own tool rather than a bench_arms arm, and the reasons
are worth keeping because both look like they should be arms:

**The optimizer is below bench_arms' noise floor.** At accum 16 it is ~2.8% of
the step while this box drifts ~2% run-to-run, so bench_arms cannot resolve even
a *doubling* of optimizer cost. `bench_optimizer_step.py` times
`optimizer.step()` alone and converts back to percent-of-step via `--step-ms`.
It builds the real model (post-`maybe_convert_fp8`) rather than synthetic
matrices, because Muon's cost is driven by the parameter *shape histogram* — it
batches same-shape matrices into one Newton-Schulz, and the 450M model has 4
shape groups with zero singletons.

*Early signal, CPU only, not predictive of GPU:* Aurora measured **2.4x** the
control's optimizer time. Consistent with its implementation — 12 NS steps per
matrix per rebalancing pass, and its docstring explains why it stays on the
per-matrix path while Muon batches. Worth confirming on GPU before the screen,
since 2.4x of 2.8% is ~4% of step.

**The window warmup only exists across steps.** A bench that builds a model and
times steps sees the *final* window and silently reports the control.
`bench_attn_warmup.py` drives the real `update_attention_windows` on a compiled
model. Its primary output is not throughput but *recompile safety*:
`local_window` is a plain Python int, so Dynamo guards on it and every distinct
window is a fresh recompile; past `cache_size_limit` (default 8) Dynamo stops
compiling and flex drops to the eager dense path — slower *and* the OOM-prone
one. `--plan-only` computes the schedule with no GPU. It already flags the
obvious settings as unsafe:

```
window 512 -> 2048 over 2000 steps, quantize 250
distinct windows: 9   dynamo cache_size_limit: 8
  RISK: ... Raise --quantize to >= 286
```

Measured separately: 5 window changes on a *1-layer* model produced 14 unique
Dynamo graphs. Note that `cache_size_limit` is per code object while
`unique_graphs` counts frames across the whole model, so this does *not* show
2.8 recompiles of one frame — the distinct-window count is a planning proxy and
the script's per-plateau `+N graphs` column is the real per-frame number.

**Checked and NOT a problem:** `update_attention_windows` nulls
`_cached_block_mask` without rebuilding it, and `_get_or_build_block_mask`
raises under `is_compiling()` on a null cache. That reads like a crash waiting
to happen. It is not — verified across a full ramp, the mask rebuilds and
`_cached_window` tracks `local_window` correctly. Do not "fix" it.

---

## Also found

`tests/test_training_speedups.py::test_norm_update_produces_unit_norm_update_direction`
was asserting the **pre-fix** NorMuon semantics (unit Frobenius norm) and
contradicting `tests/test_newton_schulz.py`'s
`test_normuon_equalises_row_scales_without_changing_step_size`, which asserts the
corrected redistribution rule. The NorMuon fix had landed without updating this
test. Rewritten to assert norm preservation plus row-RMS equalisation.
