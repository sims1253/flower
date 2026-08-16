# Tokenizer candidates — the 16K recipe is at its ceiling, except one algorithm change

Date: 2026-08-14. Follow-up to `vocab_size_results.md` (which killed the
vocab-*size* axis). Question here: is the production tokenizer — byte-level
BPE with the recipe from 2026-05 (sweep4 era), merges fitted on only 20k
docs, regenerated 2026-08-04 — beatable by (1) more merge-training data,
(2) a different algorithm at the same vocab, or (3) an off-the-shelf
tokenizer?

Tool: `scripts/compare_tokenizer_candidates.py`. All candidates evaluated on
the SAME 2,000 held-out docs (the split every tokenizer script here uses).
The `tokenizers/algo/*` artifacts were trained 2026-08-03 by
`compare_tokenizer_algorithms.py` at its defaults (40k docs, same split) —
that run's numbers were never recorded anywhere; this doc records them.

## Result

| candidate | vocab | merge-fit docs | bytes/token | Δ (= net¹) | tokens/word | Renyi α=2.5 |
|---|---:|---:|---:|---:|---:|---:|
| **bpe_regex (production)** | 16384 | 20,000 | **4.274** | — | 1.449 | 0.467 |
| bpe_regex (refit) | 16384 | 40,000 | 4.279 | +0.1% | 1.447 | 0.467 |
| bpe_regex (refit) | 16384 | 100,000 | 4.279 | +0.1% | 1.447 | 0.467 |
| bpe_regex (refit) | 16384 | 500,000 | 4.280 | +0.1% | 1.447 | 0.467 |
| **bpe_noregex** | 16384 | 40,000 | **4.844** | **+13.3%** | 1.278 | 0.752 |
| unigram | 16384 | 40,000 | 3.926 | −8.1% | 1.577 | 0.436 |
| wordpiece | 16384 | 40,000 | 4.254 | −0.5% | 1.455 | 0.468 |
| gpt2 (pretrained) | 50257 | WebText | 4.626 | +8.2% → net² **−1.9%/−7.1%** | 1.339 | 0.406 |

¹ At the same vocab the head doesn't grow, so net wall-clock-per-byte = Δ
bytes/token directly.
² gpt2's vocab is 3.1× ours, so its raw +8.2% goes through the
`vocab_size_results.md` economics (head shares h=5%/8%) and lands **net
negative**.

## Verdicts

**1. The tokenizer is not undertrained.** 25× more merge-training data
(20k → 500k docs) moves bytes/token by +0.1% — the merge set saturated before
20k docs. The May-era 20k-doc fit is at the recipe's ceiling. Nothing to fix
here.

**2. Off-the-shelf tokenizers lose structurally.** gpt2 compresses +8.2%
better per token, but its advantage arrives bundled with a 50K vocab, and the
vocab-size screen already measured that trade as net-negative at this model
scale (the knee is at 16K). A production tokenizer's "optimization" is real
but is mostly vocab size — which we've priced. (Also: gpt2 has the *lowest*
Renyi efficiency of the table, 0.406 — big vocabs are used least evenly.)

**3. unigram / wordpiece: dead** (−8.1% / −0.5% at matched vocab; matches the
"compression and downstream quality disagree" caveat in
`compare_tokenizer_algorithms.py`'s docstring — but compression is the axis
that is pure throughput, and they lose it).

**4. bpe_noregex is the one candidate that clears every bar: +13.3%
bytes/token at the SAME vocab — zero parameter cost.** Dropping the
pre-tokenizer's regex constraint lets merges cross whitespace ("superword"
tokens). This is the t=0 limit of SuperBPE (arXiv:2503.13423), which reports
~14% fewer tokens for English at 8B scale — our +13.3% at 450M-scale data
matches it. Unlike the vocab-size axis, none of the gain is spent on head
cost: net wall-clock-per-byte = +13.3% if BPB holds. Fertility 1.449 → 1.278
(longer, fewer tokens — per-token loss rises, BPB normalises it), and Renyi
0.467 → 0.752 (the vocabulary is used *far* more evenly — superword tokens
carry frequent word pairs, so no dead-vocabulary concern).

**Caveats before adopting:**
- SuperBPE's paper finds t=0 is the *crude* version — a two-stage schedule
  (regex-constrained merges first, superword merges after a transition point)
  is the refined recipe. If t=0 screens well, the two-stage variant is the
  follow-up, not the ceiling.
- A tokenizer swap breaks comparability with every existing run — same class
  of decision as the sweep-13 4k→16k move (SWEEP13_PIPELINE.md §5), which was
  made deliberately and documented. New baseline, new era.
- Research-axis interaction: at a fixed token window (512/2048), noregex sees
  ~13% more *bytes* of history. Within any architecture comparison, all arms
  must share the tokenizer (the repo rule already).

## Screen result (2026-08-16, runs/tokenizer_screen_450m): FAILS — the compression does not convert

`configs/tokenizer_screen_450m.yaml`, two arms on the full production FP8
stack, identical except `data.tokenizer` / `bytes_per_token`, 1500 steps,
seed 0. Final (end-of-run EMA) eval:

| arm | val_bpb | val_loss (nats/tok) | tok/s (whole-run) | bytes/s |
|---|---:|---:|---:|---:|
| `tok_bpe_control` | **1.13845** | 3.377 | 53,432 | 228.6k |
| `tok_noregex` | 1.17099 (**+0.033**) | 3.932 | 56,890 | 275.6k (+20.5%) |

**+0.033 val_bpb vs a ~0.01 reseed band at this length — 3-4x outside noise.**
The +13.3% bytes/token raised per-token loss by +16.4%, MORE than the
compression gained, so per-byte loss got worse, not better. The in-loop
validation (step 1400) showed an even larger gap (+0.064); the interesting
detail is that *train* loss per byte slightly favoured noregex (0.774 vs
0.785 nats/byte) while held-out per-byte loss strongly disfavoured it — a
generalisation gap on the superword representation, consistent with rare
long tokens being undertrained at this horizon (warmup is 500 of 1500 steps).

The throughput half of the bet was real — +20.5% bytes/s at matched config —
but the quality gap is far beyond anything equal-compute evaluation could
recover (~19% more data buys ~0.01 BPB at this scale, not 0.033).

**This is SuperBPE's own warning materialized** (their paper: t=0 is worse
than a transition-point schedule), and it raises the burden for the two-stage
variant: the transition schedule would need to recover >0.03 BPB, more than
the entire compression headroom at this scale. Absent a strong external
prior, the axis is closed at 450M; the production 16K regex BPE stays.

With this, the 16K tokenizer is validated on every axis: vocab size (the
knee is at 16K), merge-corpus size (saturated by 20k docs), algorithm
(challengers all lose — unigram, wordpiece, superword-t0), and off-the-shelf
(large-vocab economics). No further tokenizer work is queued.

Historical: the screen was launched via `scripts/queue_tokenizer_screen.sh`
behind the window-10k decision run.

Artifacts: `tokenizers/vocab_sweep/candidates_results.json` (gitignored,
regenerable); the noregex tokenizer itself is
`tokenizers/algo/bpe_noregex_16384.json` (regenerate with
`scripts/compare_tokenizer_algorithms.py --vocab 16384 --algos bpe_noregex`).
