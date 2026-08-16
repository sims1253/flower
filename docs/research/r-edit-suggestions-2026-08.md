# R Edit Suggestions — Project Design (2026-08-16)

Design doc for an open, R-specialized next-edit-suggestion model, served local
(ggml/llama.cpp, default) or remote (hosted, optionally paid), shipped as
editor extensions for VS Code, Positron, and (pending API verification) Zed.

**Status:** proposal. Nothing built; this captures the plan and the kill-tests
that decide whether to build it. Inspired by Zed's Zeta/Zeta2; the thesis we
take from them is *the data flywheel matters more than the model*.

**Training hardware:** RTX 5090 (32GB), flower stack
**Inference targets:** laptop CPU (local path), 5090/rented GPU (remote path)
**Model scale:** ~150–450M from-scratch, or 0.5–7B fine-tuned open base

---

## 1. PRODUCT THESIS

- R is underserved by general models. Failures are semantic, not stylistic:
  NA/NaN/NULL propagation, vectorized semantics, tidyverse NSE/data-masking,
  S3/S4/R6 dispatch, 1-based indexing, `<-` vs `=`. Generic code transfer
  doesn't fix these; R is starved in pretraining mixes.
- **Wedge: local-first and private.** R's center of gravity is pharma,
  clinical trials, biostatistics — environments where cloud autocomplete is
  compliance-blocked. "Small, local, R-only, code never leaves your machine"
  is a pitch no general vendor serves, and it is only available *because* the
  model is small.
- **Hybrid deployment, both first-class:**
  - local: model runs on the user's machine, zero telemetry by default;
  - remote: hosted inference (could be paid — open weights + paid convenience
    is a proven combo; local fallback always exists);
  - opt-in data sharing: accepted/rejected edits shared only with explicit
    consent (mirrors Zed's opt-in "Training Data Collection"). This is the
    consented version of the flywheel Zeta2 has, grown over time into the moat.

## 2. THE TASK

Adopt Zed's public edit-prediction spec rather than inventing a format:

- **Inputs:** file prefix/suffix around cursor, recent edits as in-context
  examples, LSP symbol/type context for nearby identifiers (Zeta2's addition).
- **Output:** a small edit (median a few lines), within a few hundred ms.
- **Reference artifacts (open):**
  - Zeta1 weights: `huggingface.co/zed-industries/zeta` — a fine-tuned
    **Qwen2.5-Coder-7B**; training data and code open-sourced under
    zed-industries on GitHub.
  - Zed's edit-prediction data spec (published with the Zeta announcement).
  - Zeta2: weights public, ~100k-example opt-in dataset **not** released;
    +30% acceptance over Zeta1.

Note what this means for us: Zeta's recipe needed a 7B base. Our niche allows
(or demands) smaller — that size gap *is* the local/latency product.

## 3. SERVING (ggml / llama.cpp)

- **Runtime:** llama.cpp (ggml) everywhere.
  - GGUF = single file (weights + tokenizer + arch config) → trivial
    first-run download from HF.
  - Local: embedded in the extension as a sidecar (node-llama-cpp) or a
    bundled `llama-server` the extension spawns.
  - Remote: `llama-server` on the 5090 / rented GPU with continuous batching;
    same GGUF artifact, so local ⇄ remote is a config flip, not a port.
- **Hard constraint this imposes:** the shipped model must be a *vanilla
  Llama-style decoder* with a standard-serializable tokenizer (GGUF/HF
  formats). Flower's memory/flow attention variants are off the ship path —
  flower trains vanilla too, so this costs nothing in the trainer. A custom
  R-optimized BPE is fine; it just has to export to standard formats.
- **Quantization / size budget:** 150M int8 ≈ ~200MB, 450M int8 ≈ ~500MB,
  450M q4 ≈ ~250MB. All are shippable downloads.
- **Context budget is the real local cost:** prefill dominates on CPU.
  - Mitigations: KV prefix caching across keystrokes (file prefix changes
    slowly; llama.cpp supports prefix reuse), cap context at 2–8k locally,
    offer full context when remote.
  - The day-one latency gate (§9) measures this before any training happens.

## 4. EDITOR TARGETS

| Target | Mechanism | Notes |
|---|---|---|
| **VS Code** | `inlineCompletionProvider` API (Copilot's mechanism) | coexists with R extension; MS marketplace |
| **Positron** | same (VS Code fork; "almost all VS Code extensions compatible") | **publishes via Open VSX, not MS marketplace** — publish both. Built-in R support (no vscode-r dependency). Arguably the *primary* audience: it's Posit's data-science IDE. |
| **Zed** | OPEN QUESTION: verify whether the extension API exposes edit-prediction hooks vs inline completion only | Zeta is built-in there; extension story for custom predictors needs checking |
| RStudio | — | effectively closed. Skip. |

Positioning note: Positron ships a first-party "Positron Assistant" with
completions. Ours is the local/private/R-specialized complement, not a
general assistant — different trust and cost story.

## 5. DATA PLAN

### 5.1 Bootstrap from public git (no users needed)

- **Source:** r-universe mirrors every CRAN package (~20k) as git; plus
  Bioconductor, tidyverse/r-lib GitHub orgs.
- **Example construction, per Zeta's spec:**
  - diff hunk = the edit (target output);
  - file state at parent commit = prefix/suffix context;
  - prior hunks in the same commit/session = "recent edits" examples;
  - LSP enrichment: replay `{languageserver}` at the parent commit to emit
    symbol/type context for nearby identifiers — Zeta2's conditioning,
    generated offline, no users required.
- **Filtering (the actual work):** R/Rmd files only; small diffs;
  human-looking coding commits — drop version bumps, `cran-submit` churn,
  bot activity, generated files. Expect ≥95% discard. Raw pool is plausibly
  millions of candidates; Zeta2 curated ~100k. Volume is not the constraint;
  intent fidelity is — "would a developer have accepted this edit at this
  moment" is only approximated by finished diffs.
- **Temporal split:** train on commits < date X; evaluate on edits > X from
  held-out packages. Never random-split (leaks author style).

### 5.2 The consented flywheel (later)

Local mode never transmits code. Opt-in shares accepted/rejected edits.
Remote-inference users opt in more naturally. Over time this becomes real
acceptance signal — the thing Zeta2 has that git mining can't give us.

## 6. MODEL ROUTES

**Route A — from-scratch, ~150–450M vanilla decoder on flower.**
- Full ownership: weights, tokenizer, exact size chosen for local latency.
- Flower synergies: vanilla arch already in the registry; bpe_noregex result
  (+13.3% bytes/token on edu prose at zero param cost) should be even better
  on R's `%>%` `|>` `<-` `$` `[[` token soup — refit on an R-heavy code
  corpus; FP8+Muon stack at ~57k tok/s means 10BT ≈ 2 days of pretraining;
  two-stage: code-heavy pretrain mix including lots of R → SFT on edit pairs.
- Missing pieces: code-corpus data pipeline (flower consumes FineWeb-Edu
  today), tokenizer refit, GGUF export script.

**Route B — SFT an Apache-2.0 open base (Qwen coder 0.5–7B class).**
- Quality-per-FLOP winner; Zeta1 *is* the existence proof (Qwen2.5-Coder-7B
  fine-tuned on edit data).
- Less control over vocab/size; multilingual embedding table is dead weight
  for R; 7B-class is marginal for laptop CPU.

**Decision:** Route B first as validation (days, fits the 5090); commit to
Route A only if the product is real — A is the one with ownership, size
control, and the tokenizer story.

## 7. EVAL

- **Primary:** held-out-commit edit replay — exact-match and prefix-match
  edit accuracy, plus end-to-end latency. This is the number that predicts
  acceptance.
- **Sanity:** MultiPL-E humaneval-r (161 problems) — general R codegen, not
  the core task, but tracks R competence.
- **Baselines:** (0) copy-from-context trivial baseline; (1) small open coder
  zero-shot; (2) **Zeta1 itself on R edit examples** — it's open and 7B, so
  day one we can quantify exactly how bad the general-purpose SOTA is on R
  edits → measures the niche before spending a training FLOP.
- **Latency measured in llama.cpp**, not PyTorch, on both paths (laptop CPU,
  5090).

## 8. LICENSING

- Route A: we own the weights (MIT or CC-BY, pick at release).
- Route B: base must be Apache-2.0/MIT (Qwen coder smalls qualify).
- CRAN corpus is mostly MIT/GPL: training on public code and shipping weights
  is standard practice, but license-filter the corpus deliberately (drop
  restrictive-licensed packages).
- Extension code MIT; publish to Open VSX (Positron) *and* MS marketplace
  (VS Code).

## 9. SEQUENCING & KILL-TESTS

| Stage | Work | Kill criterion |
|---|---|---|
| 0 (day one) | Latency gate: run off-the-shelf GGUFs (e.g. Qwen-coder 0.5B, SmolLM-class) in llama.cpp on laptop CPU at 2–8k ctx with prefix caching; measure p50/p95 | no config lands near ~300ms → local-only dies; pivot remote-first |
| 0b | Niche gate: eval Zeta1 on ~50 held-out R edits vs Python edits | R gap too small → weak thesis, stop |
| 1 | Data probe: mine ~20 r-universe repos; hand-audit 100 constructed examples | <50% usable after filtering → data cost underestimated |
| 2 | Route B validation: SFT a small Apache base on mined pairs; eval on held-out commits | no meaningful delta over baselines → the specialization isn't learnable from this data; stop |
| 3 | Commit: Route A — tokenizer refit, corpus build, GGUF export, extension skeleton, Positron+VS Code publishing | — |
| 4 | Pretrain (2–6 weeks wall-clock incl. corpus) → SFT → alpha; opt-in telemetry design | — |

Stage 0/0b/1 together are ~a weekend and require zero training.

## 10. RISKS & OPEN QUESTIONS

- Commit diffs approximate finished intent, not keystroke-in-the-moment
  intent — weaker signal than Zed's live collection (their core advantage).
- Zed extension hooks: unverified.
- Positron Assistant overlap: positioning must be sharp (local/private/R-only).
- CPU prefill at long context may cap local quality → context budget may
  decide model design (shorter context, better prefix caching).
- Solo bandwidth: model + data + eval + extension + runtime is a lot; the
  stages above are deliberately separable, and stages 0–2 are throwaway-safe.

## REFERENCES

- Zeta2 announcement: https://zed.dev/blog/zeta2 (+30% acceptance, data flywheel thesis)
- Zeta1: https://zed.dev/blog/edit-prediction — open-source, open-data model
- Zeta1 weights (Qwen2.5-Coder-7B fine-tune): https://huggingface.co/zed-industries/zeta
- Zed edit-prediction docs: https://zed.dev/docs/ai/edit-prediction
- Zed opt-in data policy: https://zed.dev/docs/ai/ai-improvement
- Positron extension compatibility: https://positron.posit.co/extensions.html (Open VSX, not MS marketplace)
- Positron migration notes: https://positron.posit.co/migrate-vscode.html
- MultiPL-E (humaneval-r, 161 problems): https://github.com/nuprl/MultiPL-E
- Flower stack context: `docs/research/frontier-speedups-2026-08.md`, `docs/profiling/tokenizer_candidates_results.md`
