# Vocabulary size — the compute-optimal-vocab axis is dead at 450M (CPU screen)

Date: 2026-08-14. Companion probe to the 2026-08 training-speedup review,
which flagged vocab size as the largest untouched *structural* throughput
lever (bigger vocab → more bytes/token → fewer transformer passes per byte of
training text, against a bigger tied head). This is the cheap CPU kill-test
that review specified, run before spending any GPU time: train BPE at several
vocab sizes on identical data and price the trade. **The axis dies on the
decision rule; no GPU screen was run.**

Tool: `scripts/compare_vocab_sizes.py` (regenerates everything below in
~13 CPU-minutes):

```bash
uv run python scripts/compare_vocab_sizes.py \
    --cache data_cache/sample/10BT --train-docs 20000 --report-docs 2000 \
    --vocabs 4096 8192 16384 24576 32768 \
    --out tokenizers/vocab_sweep --reference tokenizers/fineweb_16k.json
```

## Method

Byte-level BPE with the exact `train_custom_tokenizer.py` recipe (ByteLevel
pre-tokenization, no specials, byte alphabet seeded), fitted on the **same**
20,000 FineWeb-Edu docs (sample-10BT, local cache) for every vocab size,
evaluated on the **same** 2,000 held-out docs. A per-size run of
`train_custom_tokenizer.py` would not have been a controlled comparison — its
held-out set is "the first N docs of the source", which moves with source
order.

**Anchor:** the production `fineweb_16k.json` evaluated on these held-out
docs scores 4.274 bytes/token against the 4.279 recorded in every config
(0.1% apart), and matches the freshly-trained sweep 16K exactly. The corpus
is representative and the recipe reproduces.

## Result

| vocab | bytes/token | Δ vs 16K | tokens/word | Renyi α=2.5 | net @ h=5% | net @ h=8% | tied params @ d1280 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 3.426 | −19.8% | 1.807 | 0.585 | **−16.7%** | −14.7% | 5.2M |
| 8192 | 3.880 | −9.2% | 1.596 | 0.520 | **−6.9%** | −5.4% | 10.5M |
| **16384** | **4.274** | — | 1.449 | 0.467 | — | — | 21.0M |
| 24576 | 4.462 | +4.4% | 1.388 | 0.442 | **+1.9%** | +0.4% | 31.5M |
| 32768 | 4.574 | +7.0% | 1.354 | 0.426 | **+1.9%** | −0.9% | 41.9M |

The net column is the wall-clock-per-byte estimate
`net(v) = (bpt(v)/bpt(16K)) / (1 + h·(v/16384 − 1))`, where `h` is the tied
head's share of step time. The head GEMMs are ~5.1% of step FLOPs
(6·21M·262k tokens / 6·413M·262k) and run in **bf16** (deliberately never
FP8-converted — tied embeddings), while the converted blocks run FP8 at
~1.39×, so 5–8% of step *time* is the plausible band.

**Verdict: the best point anywhere on the grid is +1.9%** (24,576 and 32,768
at the optimistic h=5%; at h=8% the 32K vocab is net negative). The review's
decision rule kills anything under ~3–4% (inside run-to-run drift) and
required ~8%+ to queue the GPU quality screen. Neither is met. **16K stays.**

Two independent signals agree the marginal merges are low-value:

- **The compression curve is strongly concave**: +13.3% (4K→8K), +10.2%
  (8K→16K), +4.4% (16K→24K), +2.5% (24K→32K). 16K sits on the knee.
- **Renyi efficiency falls monotonically with vocab** (0.585 → 0.426): each
  added tier is used less evenly, i.e. increasingly dead vocabulary
  (Zouhar et al. 2023 — the metric with actual evidence of predicting
  downstream quality).

The smaller-vocab direction loses too: 8,192 nets −5.4% to −6.9% because the
compression loss dwarfs the head saving. 16K is at the flat optimum of this
family, in both directions.

Incidentally, this converges with the recently reverse-engineered ~16K vocab
of a frontier lab: whatever their inference-side reasons, 16K is also what
training-throughput economics selects for this corpus at this scale.

## Caveats (what this does and does not rule out)

- **Throughput only.** Quality effects of vocab size (fertility, embedding
  LR retuning) were not screened — but a quality-only rescue would have to
  be worth more than the ~2% throughput ceiling to matter at matched BPB,
  and there is no prior for that at Flower's scale.
- **The h band bounds the upside.** Even h=0 (impossible: the head matmul
  exists) caps the 32K gain at +7.0%. FP8-converting the head would only
  push h to ~3.6% → +3.3% at 32K, still under threshold, and it buys the
  tied-embedding quantization risk the guardrail exists to avoid.
- **Corpus-specific.** The knee is a property of FineWeb-Edu English text. A
  different domain mix (heavy code, multilingual) moves it — re-run the
  script if the data mix ever changes materially; it costs 13 CPU-minutes,
  not a GPU screen.

Artifacts: `tokenizers/vocab_sweep/` (gitignored; tokenizers + results.json
regenerable with the command above).
