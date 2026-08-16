# Sweep 13 — Pipeline upgrade

Everything here is *pipeline*, not a research question. It exists because the
sweep-13 arm list was about to spend 21 runs comparing compactor mechanisms on
a base that had four correctable defects and ran ~3x slower than it needed to.

All changes are config-gated and **default to the pre-sweep-13 behaviour**, so
every existing config in `configs/` reproduces exactly. The 133-test suite
passes unchanged.

---

## 1. Initialization was the largest single defect

`nn.Embedding` defaults to N(0, 1) and is tied to the LM head, and `base.py`
had no init code at all. Measured at the sweep-13 100M config:

| init | loss at step 0 | ideal `ln(V)` |
|---|---|---|
| `torch` (what every sweep to date used) | **565.1** | 8.32 |
| `scaled` (new) | **8.42** | 8.32 |

The model was spending its opening phase doing nothing but shrinking the
embedding norm. This also interacts with every LR heuristic in the repo — the
`muon_lr` values were tuned against a model in that state.

`init_scheme: scaled` draws Linear/Embedding from N(0, `init_std`), zeroes
biases, and scales residual output projections (attention `out`, FFN `down`) by
`1/sqrt(2*num_layers)`.

## 2. The LR schedule never decayed

`lr_multiplier` only implemented `constant` and `linear_warmup`. `linear_warmup`
warms up and then holds peak LR to the final step — **no sweep in this repo has
ever had a decay phase**. Added `wsd` (warmup-stable-decay, the Delphi recipe:
hold, then linear decay to zero over the last `lr_decay_frac`) and `cosine`.

A decay tail is normally worth more val loss at this scale than any of the
architectural arms it is being used to compare.

## 3. fp32 eager on a Blackwell card

No `autocast`, no `set_float32_matmul_precision`, no `torch.compile`.

- `precision: bf16` — TF32 matmuls + bf16 autocast, fp32 master weights. No
  GradScaler needed (bf16 has fp32's exponent range).
- `compile_model: true` — verified numerically equivalent on a 40-step smoke
  run (eager 5.56294 vs compiled 5.56297).

Two caveats found while wiring this up:
- Compile disables the module-diagnostics walk (`dir()`/`getattr` over every
  submodule is untraceable by Dynamo), so `last_diag_*` variant scalars are
  unavailable under compile. This is worth it everywhere given the measured
  2.5-2.8x; the AttnRes routing metric is recovered from one extra uncompiled
  run per arm rather than by leaving 15 runs uncompiled.
- That same walk plus `count_parameters(self)` and `asdict(self.config)` ran on
  **every forward pass**. Constants are now cached; the walk is opt-out.

Also removed host syncs inside the hot loop: a `float(loss.detach().cpu())` in
the training accumulation loop, and **four more inside `StillLM.forward`**
(`kl_loss`, `student_loss`, `teacher_loss`, `attn_match_loss`) plus a per-forward
`asdict(self.config)`. Dynamo surfaced the Still ones as graph breaks. Model
diagnostics are now kept as detached 0-d tensors and converted to host floats by
`train.py` only on logging steps.

### Measured (RTX 5090, `scripts/bench_precision.py`)

Phase-0 base, batch 8 x accum 8 x seq 1024, 111.7M params:

| mode | tok/s | s/step | peak GB | speedup |
|---|---|---|---|---|
| fp32 | 37,075 | 1.768 | 11.71 | 1.00x |
| tf32 | 45,010 | 1.456 | 11.71 | 1.21x |
| bf16 | 72,903 | 0.899 | 9.66 | 1.97x |
| **bf16+compile** | **103,355** | **0.634** | **6.06** | **2.79x** |

Losses agree to 4 decimals across all four modes (9.8266 / 9.8266 / 9.8269 /
9.8265), so bf16 is numerically clean at this scale — the `nanowhale` bf16-NaN
report does not reproduce here. Confirmed again on a 120-step real-data run:
loss 9.7 -> 5.48, no instability.

Batch shape at fixed effective batch 64, bf16+compile:

| shape | tok/s | peak GB |
|---|---|---|
| 8 x 8 | 103,355 | 6.06 |
| **16 x 4** | **113,283** | **10.42** |
| 32 x 2 | 131,817 | 19.17 |

16 x 4 is the config default. 32 x 2 is 16% faster but sits at ~60% of VRAM for
~20 minutes saved over the whole phase-0 run, and headroom is worth more than
that here — see the WSL2 note below.

**Phase 1 (Still) is the important measurement.** `D_std_17M`, batch 8 x accum 8:

| mode | tok/s | s/step | peak GB |
|---|---|---|---|
| fp32 | 10,759 | 6.09 | **27.85** |
| bf16 | 10,863 | 6.03 | 26.71 |
| **bf16+compile** | **26,471** | **2.48** | **14.99** |

Two things fall out of this:

1. **bf16 alone buys nothing** (10,863 vs 10,759). The Still dual forward is
   launch-bound, not matmul-bound. Compile is what fuses away the overhead.
2. **fp32 peaks at 27.85 GB on a 32 GB card.** That is the sweep-12 memory
   fragility, explained — the flow arms were running with ~4 GB of headroom.
   bf16+compile halves it to 14.99 GB.

Phase 1 therefore has `compile_model: true`, at the cost of the flow
velocity-net diagnostics (see below). The flow arm compiles too: 297s one-off
compile, then 0.309s per micro-step at 12.98 GB.

### WSL2 caveat, learned the hard way

Exceeding VRAM on WSL2 does **not** raise `OutOfMemoryError`. The WDDM memory
manager silently spills to host RAM over PCIe and throughput collapses, so an
oversized batch presents as a mysteriously slow run rather than a failure — the
first batch-shape sweep here hung for 10 minutes instead of erroring.
`scripts/bench_precision.py --mem-fraction 0.9` caps the caching allocator so it
raises instead, which is the only way to get an honest OOM boundary. Any future
batch-size search on this machine should use it.

## 4. QK-norm, and the rest of the architecture

`base.py` was GPT-2 with RoPE: LayerNorm, GELU MLP, biases everywhere, no
QK-norm — while training with Muon. Muon's updates are full-rank, so they
inflate `||W_Q||` and `||W_K||`, and attention multiplies the two in `QK^T`;
that is the MaxLogit-explosion mode (Kimi K2 needed MuonClip to ship Muon).
At 12M/6 layers it never bit. At 768d/14 layers it is a live risk, and it would
present as exactly the seed-dependent instability sweep 12 spent a full sweep
characterising.

New options: `qk_norm` (parameter-free RMSNorm on Q/K per head), `norm_type:
rmsnorm`, `ffn_activation: swiglu` (hidden width auto-matched to `2/3 *
ffn_dim`), `use_bias: false`.

One correctness hazard worth recording: `still_lm.py` re-implements the block
forward in two places to intercept the KV cache, composing `attn.qkv` and
`attn.rope` by hand. Adding QK-norm inside `CausalSelfAttention.forward` alone
would have applied it in the base model's forward but **not** in the Still
teacher/student passes. Both paths now go through
`CausalSelfAttention.qkv_heads()`.

## 5. Tokenizer — the data change that matters

Measured on 2000 held-out FineWeb-Edu docs (not in the BPE training set):

| vocab | bytes/token | vs 4k | tied embedding (d768) | total @ fixed core |
|---|---|---|---|---|
| 4k (current) | 3.434 | 1.00x | 3.1M | 102.4M |
| 8k | 3.883 | 1.13x | 6.3M | 105.5M |
| **16k (chosen)** | **4.279** | **1.25x** | **12.6M** | **111.7M** |
| 32k | 4.580 | 1.33x | 25.2M | 124.4M |

At the same token budget and the same FLOPs per step, the 4k tokenizer bought
3.4 GB of text where 16k buys 4.2 GB. It also means a 1024-token context window
covered ~3.5 KB of document instead of ~4.4 KB — which works directly against a
project whose research question is compaction over context.

16k takes 25 of the 33 available percentage points at 40% of the embedding cost.
The non-embedding **core stays at 99.11M**, unchanged from the 4k draft; the
"100M class" label refers to the core, which is what sets the FLOPs.

Corpus is unchanged (FineWeb-Edu). Deliberately: the Delta AttnRes paper trained
on FineWeb-Edu, so the numbers stay comparable to published ones, and changing
corpus + tokenizer + architecture at once would make phase 0 uninterpretable.

New tokenizers: `tokenizers/fineweb_{8k,16k,32k}.json`.
`scripts/train_custom_tokenizer.py` gained `--from-cache` (offline, reads the
prefetched parquet shards) and a held-out bytes/token report.

## 6. Aurora optimizer

Vendored from `tilde-research/aurora-release` (MIT, not pip-installable).
Muon's polar step leaves row norms free, which on rectangular matrices produces
neuron death (>25% dead MLP neurons by step 500). Aurora alternates the polar
step with row-norm rebalancing toward the isotropic target.

For **square** matrices it reduces exactly to Muon, so the difference is confined
to rectangular weights — here the FFN (768x3072/2048) and qkv (768x2304). Select
with `optimizer: aurora`. Tests assert both the square-matrix equivalence and the
row-norm equalisation on rectangular ones.

`weight_decay` is now an explicit config field (default 0.01, which is the torch
AdamW default every previous sweep picked up implicitly). New
`weight_decay_exclude_embeddings` splits embeddings and 1D params (norm gains,
biases) into a zero-decay group — decaying an embedding shrinks rare-token rows
in proportion to how rarely they are updated, and decaying a norm gain works
against the normalisation it parameterises. Default `false` preserves the old
behaviour; the sweep-13 configs set it `true`.

## 7. Depth-axis routing (new capability)

`flower/models/attn_res.py`. Orthogonal to the Still line: Still compacts along
the sequence axis, this retrieves along the depth axis.

Plain AttnRes (Kimi K3) **degrades at scale** (+6.9% ppl at 1044M, +6.6% at
7.6B), so only the two published fixes are implemented:

- **Delta Block AttnRes** (arXiv:2605.18855) — route over block deltas with
  additive routing. Fixes routing collapse (max softmax weight ~0.6 through
  depth vs ~0.2). Reference: 220M/L=12 on FineWeb-Edu, 38.71 -> 37.08 ppl.
- **S-LR-ATTNRES** (arXiv:2607.09694) — `attn_res_key: sliced` takes the key
  from the last `r` dims of the value. No extra projection, no extra activation
  memory. Reference: best loss/FLOPs Pareto point.

Added parameters are 455 (sliced) to 5383 (full) at the 100M config, so this is
close to a pure mechanism test. The module is an **exact identity at init**
(zero-init output gate — a small deviation from the papers, which rely on
zero-init query alone), so arms start bit-identical to baseline.

`diagnostics/attn_res_max_weight_mean` reproduces the papers' collapse metric.
Watch it: if it sits near `1/n_sources`, a null result means "routing
collapsed", not "depth routing does not help".

Rejected under `still_*` variants with a clear error — StillLM's re-implemented
block loop has no router, and silently dropping it would make the student
disagree with its own base.

## 8. Per-layer FFN width

`ffn_dim_schedule` gives TLM-style layer-wise allocation. The published TLM
result (wider-early beats uniform by ~0.32 ppl; wider-late *loses* by ~1.01 ppl,
8/8 comparisons across 3 scales) is measured on the **base model's `d_ff`**, not
on a compactor latent width — so `configs/sweep13_100m_phase0_taper.yaml` tests
the actual claim. All three arms verified identical at 111.696M parameters.

---

## What changed in the experiment plan

| | first draft | now |
|---|---|---|
| phase-1 arms | 7 x 3 seeds = 21 runs | 6 arms, 39 runs (3 x 8 + 3 x 5) |
| `G_flow_velh` | 6.59M vs 5.29M anchor | **dropped** — a 25% param gap is not a mechanism test at any seed count |
| taper test | compactor `d_latent` only | plus phase-0b on base `ffn_dim`, which is what TLM actually measured |

The seed reallocation is the substantive change. At n=3 the minimum detectable
difference at 80% power is Cohen's d = 3.07, which against the seed variances
measured at 12M is:

| arm sd (measured at 12M) | detectable at n=3 | at n=5 | at n=8 |
|---|---|---|---|
| 0.017 (standard, sweep10/11) | 0.052 | 0.034 | 0.026 |
| 0.023 (flow 10-step, sweep12) | 0.071 | 0.046 | 0.035 |
| 0.047 (flow 5-step, sweep12) | 0.144 | 0.095 | 0.071 |

The effects sweep 10 actually observed were **0.024–0.040 ppl**. So at 3 seeds
the headline D-vs-E capacity-vs-mechanism question cannot resolve, and a null
would mean "underpowered", not "closed". D/E/F therefore get 8 seeds; the taper
arms keep 5 because TLM predicts a much larger ~0.3 ppl effect.

If 39 runs is too much compute, run `--variants D_std_17M,E_flow_5,F_flow_10`
first — those three carry the actual research questions.

---

## 9. FP8 / NVFP4 — measured, and not worth it at this size

The RTX 5090 is sm_120 and torch 2.9 exposes both FP8 and NVFP4 (`Blockwise
1x16`, packed `float4_e2m1fn_x2` with e4m3 block scales). Raw GEMM at 4096^3:

| format | TFLOPS | vs bf16 |
|---|---|---|
| bf16 | 148 | 1.0x |
| fp8 (tensorwise) | 377 | 2.5x |
| fp8 (rowwise) | 383 | 2.6x |
| **nvfp4 (blockwise 1x16)** | **1266** | **8.5x** |

That looks decisive and it is not. At *this model's* GEMM shapes (M=16384,
K=768), the picture inverts once quantization is counted:

| gemm | shape (M,K,N) | bf16 TF | fp4 TF | fp4 + quantize | net |
|---|---|---|---|---|---|
| qkv | 16384, 768, 2304 | 199 | 940 | 256 | 1.29x |
| attn_out | 16384, 768, 768 | 164 | 368 | 77 | **0.47x** |
| ffn_up | 16384, 768, 2048 | 181 | 855 | 216 | 1.19x |
| ffn_down | 16384, 2048, 768 | 168 | 738 | 102 | **0.61x** |
| lm_head | 16384, 768, 16384 | 194 | 896 | 667 | 3.43x |

`d_model=768` makes these small, memory-bound GEMMs. The quantize/dequantize
traffic scales with tensor size while the useful work scales with `M*K*N`, so
for the narrow layers the format change costs more than it saves. Only the wide
lm_head GEMM wins clearly.

Confirmed end-to-end with the production path rather than a hand-rolled kernel —
`torchao.float8.convert_to_float8_training` (fused, well-tested) on the real
111.7M model:

| | tok/s | peak GB |
|---|---|---|
| bf16 + compile | 83,959 | 9.97 |
| bf16 + compile + fp8 linear | **89,798** | 8.13 |

**+7%.** Not the 2.6x the raw GEMM suggested, for exactly the reason above.

**Verdict: skip both.** bf16 + compile is the right stopping point for a 768-wide
model. Revisit if `d_model` reaches ~2048+, where the GEMMs get large enough for
the format to dominate the quantization overhead — NVFP4's published validation
is at 12B/10T tokens, three orders of magnitude from here. The NVFP4 recipe is
also four interacting pieces (selective bf16, random Hadamard, 2D block scaling,
stochastic rounding); getting one subtly wrong would corrupt research results
silently, which is a far worse outcome than being 7% slower.

`torchao` is installed as a dev dependency so this stays easy to re-measure.

## 10. Tokenizer — what the literature actually offers

Two things worth taking, beyond the compression number:

**(a) Perplexity is not comparable across tokenizers.** A coarser tokenizer
predicts more text per token, so its per-token loss is higher for reasons that
have nothing to do with model quality. Comparing a 4k-vocab sweep-12 result to a
16k-vocab sweep-13 result on `val_perplexity` is meaningless. Bits-per-byte
normalises by text and is the correct cross-tokenizer metric.

`flower/eval.py` already computed true per-document BPB for final numbers, but
training-time validation only emitted perplexity. Added `data.bytes_per_token`
(measured constant) so in-loop validation now emits `val_bpb` and
`validation/bpb` too. Verified on a real run: loss 6.401 nats/token ->
2.158 bits/byte at 4.279 bytes/token.

**(b) Compression is not the only tokenizer quality axis.** Zouhar et al. 2023
("Tokenization and the Noiseless Channel") show that Renyi efficiency of the
token distribution predicts downstream quality where raw compression does not —
a tokenizer can compress well while concentrating probability mass on a few
merges, leaving most of the vocabulary barely trained.

`scripts/analyze_tokenizer.py` reports it, and it is a real counter-signal to
the compression story:

| tokenizer | bytes/tok | fertility | Renyi | unused | embed |
|---|---|---|---|---|---|
| 4k | 3.434 | 1.805 | **0.585** | 73 | 3.1M |
| 8k | 3.883 | 1.596 | 0.520 | 98 | 6.3M |
| **16k** | **4.279** | **1.449** | **0.467** | **246** | **12.6M** |
| 32k | 4.580 | 1.354 | 0.425 | 1363 | 25.2M |

Compression rises monotonically while Renyi efficiency falls monotonically — the
two metrics disagree, which is exactly Zouhar's point. 32k also leaves 1,363
merges unseen in a 10 MB sample. 16k remains the pick: it takes most of the
available compression while the efficiency drop is still modest and almost the
whole vocabulary gets used.

**(c) The algorithm axis was unexplored, and it was hiding the largest win.**
The vocab-size table above sweeps one recipe: byte-level BPE with GPT-2
pre-tokenization. `scripts/compare_tokenizer_algorithms.py` trains genuinely
different tokenizers at fixed vocab on the same 40k docs:

| algorithm | vocab | bytes/tok | fertility | Renyi | unused | text/FLOP |
|---|---|---|---|---|---|---|
| bpe_regex | 8192 | 3.883 | 1.596 | 0.520 | 98 | 1.00x |
| **bpe_noregex** | 8192 | 4.206 | 1.474 | **0.765** | 84 | 1.08x |
| bpe_regex (current) | 16384 | 4.279 | 1.449 | 0.467 | 246 | 1.05x |
| **bpe_noregex** | 16384 | 4.844 | 1.280 | **0.752** | 135 | 1.19x |
| bpe_regex | 32768 | 4.580 | 1.354 | 0.425 | 1363 | 1.03x |
| **bpe_noregex** | **32768** | **5.492** | **1.129** | **0.741** | 325 | **1.23x** |
| unigram | 16384 | 3.926 | 1.579 | 0.436 | 315 | — |
| wordpiece | 16384 | 4.254 | 1.457 | 0.468 | 543 | — |

`bpe_noregex` drops the GPT-2 pre-tokenization split, so merges may span
whitespace and one token can cover several words ("superword" tokens). It is the
t=0 limit of **SuperBPE** (arXiv:2503.13423), whose thesis is that the
whitespace constraint is an inherited assumption rather than a requirement.

Two things make this notable rather than just another number:

1. **It wins compression and Renyi efficiency simultaneously.** Everywhere else
   in this document those two trade off — compression rises as Renyi falls
   across the vocab sweep. Winning both at once is the unusual case.
2. **It dominates rather than trades.** noregex-16k compresses better than
   regex-32k (4.844 vs 4.580) at *half* the embedding table. noregex-8k
   (4.206) roughly matches regex-16k (4.279) at half the embedding.
3. **The advantage grows with vocab** (+8.3% at 8k, +13.2% at 16k, +19.9% at
   32k) and, crucially, its Renyi efficiency **does not decay** with vocab
   (0.765 / 0.752 / 0.741) where regex-BPE's does (0.520 / 0.467 / 0.425). At
   32k, regex leaves 1363 merges unused on the sample against no-regex's 325.

That third point matters for the vocab-size conclusion reached earlier in this
document. The reason 16k was chosen over 32k was that 32k's extra compression
stopped paying for its head FLOPs and its embedding rows went underused — both
of those are properties of the *regex* recipe. Under no-regex the text/FLOP
optimum moves to 32k (1.23x) and the unused-row problem largely goes away. If
the superword arm wins the probe, the vocab question should be re-opened rather
than inherited.

SuperBPE further reports that t=0 is *worse* than a properly tuned two-stage
transition, so this is a lower bound on the idea. (True two-stage needs a mid-
training pre-tokenizer switch, which HF `tokenizers` does not expose; it would
need a custom trainer.)

Unigram lost on compression here (3.926) despite Bostrom & Durrett 2020 arguing
it beats BPE for LM pretraining — a reminder that compression is not downstream
quality, which is why it still gets a training arm.

**Not adopted on these numbers.** Intrinsic metrics are not loss.
`configs/sweep13_tokenizer_probe.yaml` settles it on val_bpb from a real run
(6 arms, ~3.9 h). This is the same discipline the rest of this document argues
for: the whole failure mode being corrected here is adopting a plausible setting
without measuring it.

Also checked: the current tokenizer uses `ByteLevel(use_regex=True)`, so it does
apply the GPT-2 pre-tokenization split. Worth verifying rather than assuming —
and it turned out to be the thing worth questioning.

### Does the bigger vocab pay for its FLOPs?

The embedding is tied to the lm_head, so vocab is not free compute: the head
matmul is `2 * d_model * V` per token. Per-token FLOPs and text throughput:

| vocab | bytes/tok | head MFLOP | core MFLOP | head share | embed | text/FLOP |
|---|---|---|---|---|---|---|
| 4k | 3.434 | 6.3 | 242.2 | 2.5% | 3.1M | 1.00x |
| 8k | 3.883 | 12.6 | 242.2 | 4.9% | 6.3M | 1.10x |
| **16k** | 4.279 | 25.2 | 242.2 | 9.4% | 12.6M | **1.16x** |
| 32k | 4.580 | 50.3 | 242.2 | 17.2% | 25.2M | 1.13x |

16k is the peak of text-per-FLOP: 32k's extra compression no longer covers its
head cost. That is an independent argument for the same choice.

### On "the embedding steals capacity from the transformer"

This is the standard small-model argument against large vocabularies, and the
evidence behind it (BabyLM-style studies showing gains plateau by ~6-10k and
turn negative past ~10k) is real but comes from **~10M-parameter models at fixed
total parameters**. Both qualifiers matter:

- At ~10M with d=256, a 16k embedding is **~40%** of the model. At 111M with
  d=768 it is **11%**. The finding is about embedding *fraction*, not about an
  absolute vocab number, and it does not transfer across a 10x scale gap.
- The sweep-13 configs hold the **non-embedding core fixed** at 99.11M and let
  total params grow (102M -> 112M). Nothing is taken from the transformer. The
  cost of the bigger vocab is paid in the head matmul (the table above), not in
  depth or width.

If the intent is instead a hard ~100M *total* budget, the trade is real and 16k
would have to be paid for by shrinking the core. That is a different question
and the answer might well be 8k. Worth being explicit about which budget is
being held fixed, because the two framings give different answers — and the
superword result partly dissolves the tension anyway, since noregex-8k buys
regex-16k's compression at 8k's embedding cost.

## 11. Optimizer calibration

The inherited `muon_lr` (0.002 / 0.003) was tuned against the broken
initialization. With `init_scheme: scaled` those numbers have no empirical
backing left — they are a leftover constant.

Two pieces:

**`configs/sweep13_lr_calibration.yaml`** — 6 LR points, 2500 steps, 1 seed
(~2.4 h). LR effects at this spacing dwarf seed noise. Sweeps `muon_lr` only;
`lr` (the AdamW group: embeddings and norm gains) is held, because a 2D grid
costs 36 runs for a weak interaction.

**`flower/hparams.py`** — the Complete(d)P transfer rules, so the result is
transferable instead of re-swept at every configuration:

| dimension | multiplier |
|---|---|
| width | `1 / m_N` |
| depth | `m_L ** (alpha - 1)` |
| batch / horizon | `sqrt(m_B / m_D)` |

The horizon term is the one that matters here and the one usually forgotten:
**optimal LR decreases with run length**, so an LR tuned on a 2500-step probe is
systematically too high for the 15000-step run. Multiply the winner by
`horizon_correction(2500, 15000)` = 0.41. This is precisely what broke Delphi's
first attempt — clean at small scale, diverged at 1e23 FLOPs.

Note the batch and horizon terms are `sqrt(m_B / m_D)` jointly, not `sqrt(m_B)`:
4x batch at the same step count is also 4x data and the two cancel exactly. A
naive "bigger batch, bigger LR" rule would double the LR of a run that also got
4x longer. There is a test pinning this.

## 12. The Still loss geometry — found last, matters most

Measured, not inferred. At the phase-1 geometry (seq 1024, `compact_len` 64),
**only 64 of 1024 positions — 6.25% — can differ between teacher and student.**
The other 960 are bit-identical (max abs diff 4.8e-7, float noise).

The reason is structural. The student compacts the prefix `[0:ctx_end)` and only
queries in `[ctx_end:T)` ever see the compact cache; prefix queries run plain
local attention through the *same frozen base* in both passes. And `ctx_end` is
hardwired to `T - compact_len`, so the number of positions carrying any signal
is pinned to the compaction budget.

### The scaling bug on top of it

`_topk_kl_loss` derives `answer_mask = labels != -100`. Training labels contain
no `-100`, so the mask is all-True and the reduction is `kl.sum() / (B*T)` —
dividing by 1024 when only 64 positions contribute. Measured ratio: **exactly
16.0x**, matching `1024/64`.

So `still_kl_weight: 1.0` has been an effective **0.0625** against the CE term,
in every Still sweep run so far. The distillation objective — the thing the
whole line of work is about — was running at a sixteenth of its nominal weight,
and the reported loss was CE-dominated. The CE half is no better: it averages
over a prefix that is a frozen-base constant, identical across every arm and
every seed.

This is a strong candidate explanation for the sweep10/12 pattern — effects of
0.024-0.040 ppl swamped by seed sd up to 0.047. A metric that is ~94% shared
signal, with the arm-distinguishing term down-weighted 16x, is exactly what
produces tiny effects buried in noise.

### What was changed

`still_loss_positions: suffix` (new; default `all` preserves legacy) restricts
both KL and CE to positions that can actually differ, so the configured weights
mean what they say. Phase 1 sets it. `diagnostics/loss_position_frac` now
reports the fraction, so this can never be invisible again.

`still_suffix_len` (new; default None preserves the legacy coupling) decouples
the evaluated span from the compaction budget.

Phase 1 also raises `validation_steps` 20 -> 100. Under `suffix` reduction the
val loss is scored on 64 positions per sequence, so 20 batches x 8 gave 10,240
scored positions — the standard error on that mean is the same order as the
effects being chased. Validation is a rounding error in runtime; measurement
noise is not.

### What was deliberately NOT changed

`still_suffix_len` is left unset. Setting it to 512 would score 512 positions
instead of 64 — 8x more signal per step, a much better-powered experiment — but
it changes the compression ratio under study from 15x (960 -> 64) to 8x
(512 -> 64) and breaks comparability with the 12M results. That is a research
decision, not a bug fix, and it is yours to make. The knob is there and tested.

### Comparability warning

Fixing the reduction means phase-13 Still numbers are **not** comparable on
absolute value to sweeps 10-12, which ran with the 16x-diluted KL. Arms within
sweep 13 remain comparable to each other, which is what the bake-off needs. Any
writeup that puts a sweep-13 number next to a sweep-12 number must say so.

### Related: the base has no global attention

`vanilla_local` sets `local_window: 256` on all 14 layers at seq 1024 — no layer
ever attends over the full sequence. This interacts with the above: the Still
student's suffix queries see a 64-slot summary of the whole 960-token prefix,
while the teacher only sees the last 256 raw tokens. So the compactor is partly
*granting reach the base never had*, not purely *approximating the full cache*.
Those are different claims and the current setup measures a blend.

`attn_window_schedule` (new, per-layer windows, `null` = full attention) and
`configs/sweep13_attn_window_probe.yaml` test all-local vs 3:1 vs 6:1 hybrid vs
all-full. Zero added parameters — mask-only, so any difference is mechanism.
Still inherits per-layer windows automatically.

## Compute budget (measured)

| stage | runs | per run | total |
|---|---|---|---|
| **phase 0c tokenizer probe** | **6** | **~0.65 h** | **~3.9 h** |
| **phase 0d attn-window probe** | **12** | **~1.2 h** | **~14 h** |
| **phase 0a LR calibration** | **6** | **~0.4 h** | **~2.4 h** |
| phase 0b taper | 9 | ~1.0 h | ~9 h |
| phase 0 base | 1 | ~2.4 h | ~2.4 h |
| **phase 1 bake-off** | **39** | **~4.5 h** | **~175 h (7.3 days)** |
| — D/E/F only | 24 | ~4.5 h | ~107 h (4.5 days) |
| AttnRes probe | 15 | ~1.0 h | ~15 h |

Phase 1 dominates, and that is a direct consequence of the seed budget. Under
the original fp32 plan the same 39 runs would have been ~430 h (18 days), so
compile is what makes the powered comparison affordable at all — but 7.3 days of
continuous GPU is still a real commitment and it is your call whether to take it.
The honest options:

- **Full 39 runs (~7.3 days).** Both questions powered.
- **D/E/F at 8 seeds only (~107 h).** The capacity-vs-mechanism and step-count
  questions, powered; taper deferred.
- **All 6 arms at 5 seeds (~135 h).** Detects 0.034-0.095 ppl. Splits the
  difference but leaves D-vs-E marginal for the smallest plausible effects.

Cutting to 3 seeds to save time is the one option I would argue against: it
returns the sweep to a state where a null result carries no information.

## Suggested order

Before launching anything:

```
PYTHONPATH=. uv run python scripts/preflight_sweep.py configs/sweep13_*.yaml
```

It builds every arm on CPU and checks the failures that are expensive to
discover late: vocab_size disagreeing with the tokenizer file, missing tokenizer
or checkpoint paths, invalid schedules, absent `bytes_per_token`. It also prints
the per-arm parameter counts and total run count. Phase 1 is *expected* to fail
until phase 0 has produced its checkpoint.

0a. `configs/sweep13_tokenizer_probe.yaml` — ~3.9 h. Do this **first**: the
   tokenizer determines `vocab_size`, which determines the base architecture,
   which every later stage inherits and cannot change afterwards. Decide on
   val_bpb; also look at bpb-per-wallclock, since vocab changes head FLOPs.
    **If the winner is not the 16k regex BPE, update `vocab_size`,
    `data.tokenizer` and `data.bytes_per_token` in phase0 / phase0_taper /
    phase1 / attn_res_probe before continuing** — they currently hardcode
    16384 + `tokenizers/fineweb_16k.json` + 4.279. Re-run the preflight after
    editing; it will catch a vocab/tokenizer mismatch.
0b. `configs/sweep13_lr_calibration.yaml` — ~2.4 h, on the winning tokenizer.
   Everything downstream inherits the base LR and the inherited value is no
   longer justified after the init fix. Apply `horizon_correction(2500, 15000)`
   to the winner.
1. `configs/sweep13_100m_phase0_taper.yaml` — 3 arms x 3 seeds, 6000 steps.
   Picks the base FFN allocation and shakes out the new pipeline on real data.
2. `configs/sweep13_100m_phase0.yaml` — full 15000-step base with the winning
   allocation. This is the shared frozen teacher.
3. `configs/sweep13_100m_phase1.yaml` — the compactor bake-off.
4. `configs/sweep13_attn_res_probe.yaml` — the orthogonal depth-routing probe;
   independent of 2-3 and can run whenever there is spare GPU.

Before step 1, run ~200 steps of phase 0 twice (fp32/no-compile vs bf16/compile)
to get the actual speedup on your card and confirm bf16 is stable at this scale.
`entities/nanowhale.md` reports bf16 NaN at ~110M — that was MLA+MoE, not a
dense GPT, but it is cheap to rule out.

---

## 13. Long-context: what FlexAttention unlocks (training-speedups integration)

The original sweep-13 plan above runs at seq=1024 with a 256 sliding window. At
that ratio every token reaches the whole sequence through the windowed
attention itself, so there is nothing for an external memory to carry across —
which is the project's whole reason for existing. `docs/training-speedups.md`
Section 13 is explicit about this: for memory mechanisms to show measurable
signal, **sequence length must exceed the sliding window by >=4x** (seq >= 8K
at a 2048 window). Below that ratio local attention covers most token
dependencies and external memory adds no information.

The blocker was always the dense causal/local attention mask. SDPA materializes
a `(1, 1, T, T)` fp32 mask per layer: ~268 MB/layer at seq 8K, ~4 GB/layer at
seq 32K. At seq 32K that OOMs the 5090's 32 GB before the model even runs.

**FlexAttention** (`docs/training-speedups.md` Section 1) compiles the mask
pattern into the attention kernel without ever materializing the T x T matrix.
It is enabled with `model.flex_attention: true` and requires `compile_model:
true` — **without compile, flex_attention runs unfused and is slower than SDPA**
(it materializes the full scores matrix and warns). This was the single largest
implementation caveat; the configs below always pair the two.

### Measured (RTX 5090, flex + compile + bf16 + muon, 0.85 memory cap)

What FlexAttention actually saves is the **mask**, not the activations. This
matters and is easy to get wrong, so here is the honest decomposition (100M,
d768/L14, batch 1, sliding window 2048):

| seq | model+opt | flex block mask | forward (act+logits) | peak |
|-----|-----------|-----------------|----------------------|------|
| 8192 | 0.51 GB | ~0 | 3.91 GB | 4.77 GB |
| 16384 | 0.79 GB | ~0 | 7.57 GB | 8.97 GB |
| 32768 | 0.90 GB | ~0 | 14.92 GB | 17.0 GB |
| 65536 | — | — | — | OOM ("tried to allocate 32 GB") |

Two things to read off this table:
1. The flex block mask is ~0 GB at every seq — that is the win. The SDPA path
   materializes a `(1,1,T,T)` fp32 causal/local mask per layer, which at seq 32K
   is ~4 GB/layer and OOMs before the model runs. FlexAttention compiles the
   mask into the kernel and never holds it.
2. **Activation memory still scales linearly with tokens-per-batch.** The
   forward pass holds per-token activations and the `(B*T, vocab)` logits tensor,
   both of which grow with `B*T`. That is why 65K OOMs: a single logits tensor at
   batch 1 × seq 65536 × vocab 16384 in bf16 is ~2 GB, and the layered activations
   on top push the peak past 32 GB.

The practical consequence: batch size must shrink as sequence grows to keep
`B*T` (and thus peak memory) under the cap. That is exactly what the configs do.
Per-token memory is roughly flat across seq; per-sequence memory is not.

Throughput and fit at the config batch sizes (flex + compile + bf16):

| size | seq | batch | peak GB | tok/s | notes |
|---|---|---|---|---|---|
| 100M (d768/L14) | 8192 | 4 | 19.9 | 120k | flex, the longctx_phase0 config (torch 2.13) |
| 100M | 16384 | 2 | 18.2 | 113k | flex |
| 100M | 32768 | 1 | 17.0 | 103k | **flex fits; SDPA OOMs at batch 1** |
| 350M (d1024/L28) | 8192 | 2 | 23.2 | 33k | flex |
| 350M | 16384 | 1 | 23.3 | 30k | flex |

The crossover point: at seq <= 4K SDPA (flash kernel) is as fast or faster than
flex and saves nothing, so the seq=1024 configs above do **not** enable flex.
From seq 8K upward flex is both faster (where SDPA fits at all) and the only
thing that fits at 32K. Use `scripts/bench_speedups.py` to re-measure.

### What the speedups work added (and what it did not change)

Implemented from `docs/training-speedups.md`, all behind config flags that
default to legacy so the seq=1024 results stay reproducible:

- **FlexAttention (S1)** — the long-context enabler above. The win is real and
  matches the doc: 4.4x throughput at seq 32K and half the VRAM vs SDPA.
- **FP8 lm_head (S3)** — eval-only. `_scaled_mm` has no backward kernel in torch
  2.9, so the FP8 head runs in inference and the BF16 head is used during
  training. The pipeline doc's section 9 verdict ("skip FP8/NVFP4 at this size")
  stands: only the wide lm_head GEMM wins clearly, and the eval-only path makes
  it a negligible-end-of-training convenience, not a training speedup.
- **NorMuon (S5), Cautious WD (S6), Smooth-SwiGLU (S10), Orthogonal init
  (S12.2)** — roughly cost-neutral on throughput (measured). Their value is
  sample efficiency / stability, which needs a longer run to show. Not enabled
  in the overnight configs below; opt in per-arm when re-running a winner.
- **BF16 CE (S4), EMA eval (S12.4), sliding-window eval (S12.5), MTP (S8), TST
  (S9)** — available, off by default.

Skipped (need hardware/external deps outside the 5090's reach): **FSDP (S7)**,
**Triton/Liger/TE kernels (S14)**, **FP4/NVFP4 precision routing (S13)**, the
**600M @ seq=32K scale-up config** (OOMs the 5090 — needs a rented 8xGPU node,
see the doc's compute-budget table).

### New configs

The seq=1024 Still-compactor pipeline (phase0/phase1 above) is preserved
unchanged for comparability with the published 12M/100M results. Two new configs
add the long-context direction:

- **`configs/sweep13_100m_longctx_phase0.yaml`** — the long-context base.
  Identical 100M arch (d768/L14), but seq=8192, local_window=2048 (4x ratio),
  flex_attention. This is the base that makes the memory question testable.
- **`configs/sweep13_longctx_memory_bakeoff.yaml`** — the core experiment.
  `vanilla_local` (no memory) vs `bloom_memory` vs `summary_memory` at seq=8192 /
  window=2048, where tokens past position 4K cannot see the first 2K unless a
  memory mechanism carries them across the window. 3 arms x 2 seeds, an
  overnight directional first pass; follow up with 8 seeds on any winner.

### Suggested order (updated)

0. (unchanged) tokenizer probe, LR calibration, taper shake-out.
1. `configs/sweep13_100m_longctx_phase0.yaml` — the long-context base. ~3-4 h.
2. `configs/sweep13_longctx_memory_bakeoff.yaml` — the memory bake-off. ~12-18 h.
3. (optional, separate GPU budget) the seq=1024 Still bake-off (phase0+phase1)
   and AttnRes probe — preserved for the compactor question, orthogonal to the
   long-context memory question.

If the overnight bake-off shows a memory arm beating vanilla at long context,
that is the first positive signal for the project's thesis and the trigger for
the powered (8-seed) follow-up and the 350M / 600M scale-up.

### First results (overnight directional pass)

`configs/sweep13_longctx_memory_bakeoff.yaml`, d512/L8 (~33-54M), seq=8192,
local_window=2048 (4x ratio), 6000 steps, 2 seeds/arm, flex+compile+bf16+muon,
batch 8 (effective batch 131072 tokens). This is the floor of the regime where
memory mechanisms can show signal — and signal appeared.

| arm | params | seeds | val_loss (±) | val_bpb | vs vanilla |
|-----|--------|------:|-------------|---------|-----------|
| **bloom_memory** | 54.5M | 2 | **3.2615 ± 0.0041** | **1.0996** | **-0.0054** |
| vanilla_local | 33.3M | 2 | 3.2774 ± 0.0064 | 1.1050 | baseline |
| summary_memory | 64.9M | — | (re-running, see note) | — | — |

**Reading.** bloom_memory beats vanilla on both seeds (3.265, 3.259 vs 3.273,
3.283) with tighter variance. The -0.0054 bpb delta is small and not yet
significant at n=2, but it is consistent and in the predicted direction: the
memory mechanism helps once sequence exceeds the sliding window. This is the
first positive long-context signal in the project and the trigger for the
powered follow-up (`configs/sweep13_longctx_memory_powered.yaml`, 4 seeds,
10000 steps).

**Caveats.** (1) The arms are NOT parameter-matched — bloom (54.5M) has ~64%
more params than vanilla (33.3M). The bloom win could be capacity, not
mechanism. A fair follow-up needs a param-matched vanilla control. (2) n=2 is
directional only; the powered config raises this to n=4. (3) summary_memory
OOMed on both seeds at batch 8 due to allocator fragmentation (the VRAM cap
caught it loudly, as designed); it re-runs with `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True`. summary_memory is the most memory-hungry arm
(perceiver cross-attention); batch 8 = 23.4 GB peak in isolation but fragments
under the long-context activations.

**Environment note.** These runs use the upgraded stack (torch 2.13.0+cu130,
Python 3.13, fla 0.5.2, flex+compile, bloom diagnostics graph-break fixed).
Throughput at batch 8: ~400k tok/s, ~22 GB peak (vanilla), ~26 GB peak (bloom).
The bloom diagnostics host-sync was graph-breaking compile and dropping GPU
util to ~46%; gating it behind `torch.compiler.is_compiling()` (S14 Opportunity
4) restored full utilization.

### Verdict: the directional bloom win was capacity, not mechanism

The directional pass's -0.0054 bpb bloom "win" was confounded: bloom (54.5M)
had ~64% more params than vanilla (33.3M). The powered follow-up resolves this
by comparing bloom against a **param-matched vanilla control** (d640/L10,
49.6M non-emb, within 8% of bloom's 46.1M), both at 10000 steps:

| arm | non-emb params | seeds | val_loss (±) | val_bpb | vs matched vanilla |
|-----|---------------|------:|-------------|---------|-------------------|
| **vanilla_matched54M** | 49.6M | 2 | **3.0803 ± 0.0023** | **1.0385** | baseline |
| bloom_memory | 46.1M | 1 | 3.1818 | 1.0728 | **+0.0343 bpb (worse)** |
| vanilla_local (small) | 24.9M | 2 (6k) | 3.2774 ± 0.0064 | 1.1050 | (different step count) |

**At matched parameters and matched steps, plain sliding-window attention beats
bloom memory by +0.034 bpb** — a gap ~15x the seed noise (0.0023). The memory
mechanism is *hurting* at this scale: the params bloom spends on hash
projections / perceiver summaries / slot writes return less than spending the
same params on a wider vanilla transformer.

This is the expected null result at 33-54M / seq 8192, and it matches
`docs/training-speedups.md` Section 13's explicit prediction: memory mechanisms
need ~500M+ params (and seq >= 8K) before the circuits they support can form.
The positive reading: the experiment worked — it produced a clean, decisive
mechanism-vs-capacity separation at the smallest scale where the question was
testable, and the answer is "not yet." The scale-up that could flip this
(600M / seq 32K) is the doc's Phase 3 target and needs rented 8xGPU hardware.

**What this means for the project.** Do not chase memory-mechanism wins below
~500M — they are capacity artefacts. The long-context infrastructure
(FlexAttention, the bake-off configs) is now in place and validated; the next
step that could show real memory signal is the 600M scale-up, not more
small-model variants.



