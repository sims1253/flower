from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F

from flower.config import ExperimentConfig
from flower.data import build_tokenizer, fineweb_validation_documents
from flower.eval import evaluate_documents


def _probe_vocab(cfg: ExperimentConfig) -> tuple[int, int]:
    # Avoid the first few token ids because tokenizers often reserve them for
    # special/control tokens. Synthetic configs can still use the full range.
    if cfg.data.dataset in {"synthetic", "mqar"}:
        return 0, min(cfg.model.vocab_size, cfg.data.synthetic_vocab_size)
    return min(8, cfg.model.vocab_size - 1), cfg.model.vocab_size


def _eval_seq_len(cfg: ExperimentConfig) -> int:
    """Return the effective eval sequence length for probes (Sweep 7 A1)."""
    return getattr(cfg.data, "eval_seq_len", None) or cfg.model.max_seq_len


def _long_context_batch_size(requested: int, seq_len: int, *, token_budget: int = 8192) -> int:
    """Cap probe microbatches so eval_seq_len probes do not dominate VRAM.

    The attention path is dense in T x T even when `local_window` is set, so
    memory scales roughly with batch * seq_len^2. A token budget of 8192 means
    eval_seq_len=4096 probes run at batch=2, while short smoke probes keep their
    original batch sizes.
    """
    return max(1, min(requested, token_budget // max(1, seq_len)))


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def induction_copy_probe(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    batches: int = 16,
    batch_size: int = 16,
) -> dict[str, float | int]:
    lo, hi = _probe_vocab(cfg)
    seq_cap = _eval_seq_len(cfg)
    pattern_len = max(4, seq_cap // 3)
    seq_len = pattern_len * 3
    batch_size = _long_context_batch_size(batch_size, seq_len)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.training.seed) + 101)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for _ in range(batches):
        pattern = torch.randint(lo, hi, (batch_size, pattern_len), generator=gen, dtype=torch.long)
        filler = torch.randint(lo, hi, (batch_size, pattern_len), generator=gen, dtype=torch.long)
        seq = torch.cat([pattern, filler, pattern], dim=1).to(device)
        logits = model(seq)["logits"]
        # Score from the FIRST REPEATED token onward (logit at 2p predicts the
        # token at 2p+1). The previous `start = 2p - 1` also scored the
        # prediction of the repeated block's FIRST token — which induction
        # cannot know: its only evidence is one earlier copy of a random
        # pattern, so that position sits at the chance floor and pinned even a
        # perfect inductive model's accuracy to (p-1)/p.
        start = pattern_len * 2
        pred_logits = logits[:, start:-1, :].reshape(-1, logits.shape[-1])
        labels = seq[:, start + 1 :].reshape(-1)
        loss = F.cross_entropy(pred_logits, labels, reduction="sum")
        total_loss += float(loss.cpu())
        total_correct += int((pred_logits.argmax(dim=-1) == labels).sum().cpu())
        total += int(labels.numel())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "tokens": total,
    }


def _unique_token_rows(
    lo: int,
    hi: int,
    batch_size: int,
    count: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """(batch_size, count) token ids sampled WITHOUT replacement per row.

    Each row's `count` ids are distinct (a per-row `randperm` slice, the same
    path mqar_probe has always used). WITH-replacement sampling lets the same
    key appear twice in an associative-recall row, which makes the queried
    pair ambiguous — at the byte vocab (256) and 8 pairs, ~11% of rows had a
    duplicated key and were unanswerable even for a perfect model. Falls back
    to WITH-replacement randint when the vocabulary is smaller than `count`
    (uniqueness is impossible there).
    """
    vocab = hi - lo
    if vocab >= count:
        return torch.stack(
            [torch.randperm(vocab, generator=gen, dtype=torch.long)[:count] + lo for _ in range(batch_size)]
        )
    return torch.randint(lo, hi, (batch_size, count), generator=gen, dtype=torch.long)


@torch.no_grad()
def associative_recall_probe(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    batches: int = 16,
    batch_size: int = 16,
    pairs: int = 8,
) -> dict[str, float | int]:
    lo, hi = _probe_vocab(cfg)
    seq_len = max(32, _eval_seq_len(cfg))
    batch_size = _long_context_batch_size(batch_size, seq_len)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.training.seed) + 202)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for _ in range(batches):
        # Unique keys: a duplicated key would make the queried pair ambiguous
        # (two candidate answers for one query). Unique values keep parity with
        # mqar_probe's sampling.
        keys = _unique_token_rows(lo, hi, batch_size, pairs, gen)
        vals = _unique_token_rows(lo, hi, batch_size, pairs, gen)
        query_idx = torch.randint(0, pairs, (batch_size,), generator=gen)
        kv = torch.stack([keys, vals], dim=-1).reshape(batch_size, pairs * 2)
        query_key = keys[torch.arange(batch_size), query_idx].unsqueeze(1)
        answer = vals[torch.arange(batch_size), query_idx].unsqueeze(1)
        pad_len = max(0, seq_len - kv.shape[1] - 2)
        delay = torch.randint(lo, hi, (batch_size, pad_len), generator=gen, dtype=torch.long)
        seq = torch.cat([kv, delay, query_key, answer], dim=1).to(device)
        logits = model(seq)["logits"][:, -2, :]
        labels = seq[:, -1]
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss.cpu())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().cpu())
        total += int(labels.numel())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "examples": total,
    }


def _monotone_breaking_point(curve: dict[str, float], *, threshold: float = 0.5) -> int:
    """Largest level N such that every level <= N is MEASURED and reached `threshold`.

    The previous rule ("largest passing level") reported 64 for a non-monotonic
    curve that passed 16, failed 32, and passed 64 — overstating capacity on
    exactly the dip the curve exists to locate. A capacity breaking point is a
    prefix property: a model supports N pairs only if it supports every load up
    to N. A NaN level (unmeasurable at that length) caps the prefix at the last
    confirmed level: extending past an unconfirmed gap would claim capacity
    that was never measured. (Today the unmeasurable condition — the sequence
    no longer fits `eval_seq_len` — is monotone in the level, so nothing
    follows a gap; this stays conservative if that ever changes.)
    """
    breaking_point = 0
    for level in sorted(curve, key=int):
        acc = curve[level]
        if isinstance(acc, float) and math.isnan(acc):
            break
        if acc >= threshold:
            breaking_point = int(level)
        else:
            break
    return breaking_point


@torch.no_grad()
def mqar_probe(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    batches: int = 8,
    batch_size: int = 8,
    num_pairs_list: tuple[int, ...] = (16, 32, 64, 128),
    query_fraction: float = 0.25,
    delay_modes: tuple[str, ...] = ("short", "long"),
) -> dict[str, Any]:
    """Multi-query associative recall (MQAR) capacity curve.

    Plants N key→value pairs then queries a fraction of them at the end.
    Returns accuracy at each num_pairs level and a breaking_point scalar
    (largest num_pairs such that every level <= it reached accuracy >= 0.5 —
    see `_monotone_breaking_point`).
    """
    lo, hi = _probe_vocab(cfg)
    seq_cap = _eval_seq_len(cfg)
    batch_size = _long_context_batch_size(batch_size, seq_cap)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.training.seed) + 777)

    capacity_curve: dict[str, dict[str, float]] = {}
    full_vocab_curve: dict[str, dict[str, float]] = {}
    breaking_points: dict[str, int] = {}

    def resolve_delay(mode: str, seq_needed_without_delay: int) -> int | None:
        max_delay = seq_cap - seq_needed_without_delay
        if max_delay < 0:
            return None
        if mode == "short":
            return min(max_delay, max(0, int(getattr(cfg.model, "local_window", 0) or 0) // 2))
        if mode == "long":
            return max_delay
        raise ValueError(f"Unknown MQAR delay mode: {mode}")

    for mode in delay_modes:
        mode_curve: dict[str, float] = {}
        mode_full_vocab_curve: dict[str, float] = {}
        for num_pairs in num_pairs_list:
            num_queries = max(1, int(num_pairs * query_fraction))
            # Sequence: [k1,v1,...,kN,vN] + [delay] + [q1,a1,q2,a2,...]
            kv_len = num_pairs * 2
            qa_len = num_queries * 2
            delay_tokens = resolve_delay(mode, kv_len + qa_len)
            if delay_tokens is None:
                mode_curve[str(num_pairs)] = float("nan")
                mode_full_vocab_curve[str(num_pairs)] = float("nan")
                continue

            total_candidate_correct = 0
            total_full_vocab_correct = 0
            total = 0
            for _ in range(batches):
                keys = _unique_token_rows(lo, hi, batch_size, num_pairs, gen)
                vals = _unique_token_rows(lo, hi, batch_size, num_pairs, gen)
                kv = torch.stack([keys, vals], dim=-1).reshape(batch_size, kv_len)

                query_indices = torch.stack(
                    [torch.randperm(num_pairs, generator=gen)[:num_queries] for _ in range(batch_size)]
                )
                query_keys = keys[torch.arange(batch_size).unsqueeze(1), query_indices]
                query_vals = vals[torch.arange(batch_size).unsqueeze(1), query_indices]

                delay = (
                    torch.randint(lo, hi, (batch_size, delay_tokens), generator=gen, dtype=torch.long)
                    if delay_tokens > 0
                    else kv.new_empty(batch_size, 0)
                )

                # Interleave query key/answer pairs: [q1,a1,q2,a2,...]
                qa = torch.stack([query_keys, query_vals], dim=-1).reshape(batch_size, qa_len)
                seq = torch.cat([kv, delay, qa], dim=1).to(device)

                logits = model(seq)["logits"]
                ans_start = kv_len + delay_tokens + 1
                for qi in range(num_queries):
                    pos = ans_start + qi * 2 - 1  # logit at position before the answer
                    step_logits = logits[:, pos, :]
                    label = seq[:, pos + 1]
                    full_vocab_pred = step_logits.argmax(dim=-1)
                    candidate_logits = step_logits.gather(1, vals.to(device))
                    candidate_pred = candidate_logits.argmax(dim=-1)
                    label_idx = query_indices[:, qi].to(device)
                    total_full_vocab_correct += int((full_vocab_pred == label).sum().cpu())
                    total_candidate_correct += int((candidate_pred == label_idx).sum().cpu())
                    total += batch_size

            acc = total_candidate_correct / max(total, 1)
            full_vocab_acc = total_full_vocab_correct / max(total, 1)
            mode_curve[str(num_pairs)] = acc
            mode_full_vocab_curve[str(num_pairs)] = full_vocab_acc
        capacity_curve[mode] = mode_curve
        full_vocab_curve[mode] = mode_full_vocab_curve
        breaking_points[mode] = _monotone_breaking_point(mode_curve)

    return {
        "capacity_curve": capacity_curve,
        "full_vocab_capacity_curve": full_vocab_curve,
        "breaking_points": breaking_points,
        "breaking_point": breaking_points.get("long", max(breaking_points.values(), default=0)),
    }


def _real_token_pool(cfg: ExperimentConfig, *, max_docs: int = 8, max_tokens: int = 8192) -> torch.Tensor:
    """Pool of in-vocabulary token ids drawn from real validation text.

    The synthetic `mqar_probe` samples ids in `0..synthetic_vocab_size`, which are
    off-manifold for a FineWeb-trained model (Phase A: all such variants floored at
    breaking-point 0). This pool restricts the recall task to tokens the model has
    actually seen, so the probe measures real recall ability rather than reaction to
    alien ids. Falls back to the synthetic range for synthetic/mqar datasets.
    """
    if cfg.data.dataset in {"synthetic", "mqar"}:
        lo, hi = _probe_vocab(cfg)
        return torch.arange(lo, hi, dtype=torch.long)
    encoder = build_tokenizer(cfg.data.tokenizer)
    ids: list[int] = []
    for text in fineweb_validation_documents(cfg.data, limit=max_docs):
        ids.extend(encoder.encode(text))
        if len(ids) >= max_tokens:
            break
    if not ids:
        lo, hi = _probe_vocab(cfg)
        return torch.arange(lo, hi, dtype=torch.long)
    uniq = torch.unique(torch.tensor(ids[:max_tokens], dtype=torch.long))
    return uniq[uniq < cfg.model.vocab_size]


@torch.no_grad()
def text_recall_probe(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    batches: int = 8,
    batch_size: int = 8,
    num_pairs_list: tuple[int, ...] = (16, 32, 64, 128),
    query_fraction: float = 0.25,
    delay_modes: tuple[str, ...] = ("short", "long"),
    token_pool: torch.Tensor | None = None,
) -> dict[str, Any]:
    """On-manifold multi-query associative recall.

    Identical task structure to `mqar_probe` (plant N key→value pairs, query a
    fraction at the end, candidate-set scoring) but keys/values/delay are sampled
    from REAL in-vocabulary tokens (`_real_token_pool`) so a FineWeb-trained model
    is evaluated on its training manifold. This is the discriminator the Sweep-7
    eval-validation gate cares about: it can run on FineWeb checkpoints directly.
    """
    pool = (token_pool if token_pool is not None else _real_token_pool(cfg)).to(dtype=torch.long)
    pool_size = int(pool.numel())
    seq_cap = _eval_seq_len(cfg)
    batch_size = _long_context_batch_size(batch_size, seq_cap)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.training.seed) + 1337)

    capacity_curve: dict[str, dict[str, float]] = {}
    breaking_points: dict[str, int] = {}

    def resolve_delay(mode: str, seq_needed_without_delay: int) -> int | None:
        max_delay = seq_cap - seq_needed_without_delay
        if max_delay < 0:
            return None
        if mode == "short":
            return min(max_delay, max(0, int(getattr(cfg.model, "local_window", 0) or 0) // 2))
        if mode == "long":
            return max_delay
        raise ValueError(f"Unknown text-recall delay mode: {mode}")

    for mode in delay_modes:
        mode_curve: dict[str, float] = {}
        for num_pairs in num_pairs_list:
            num_queries = max(1, int(num_pairs * query_fraction))
            kv_len = num_pairs * 2
            qa_len = num_queries * 2
            delay_tokens = resolve_delay(mode, kv_len + qa_len)
            # Need 2*num_pairs distinct tokens (keys disjoint from values) from the pool.
            if delay_tokens is None or pool_size < 2 * num_pairs:
                mode_curve[str(num_pairs)] = float("nan")
                continue

            total_correct = 0
            total = 0
            for _ in range(batches):
                idx = torch.stack(
                    [torch.randperm(pool_size, generator=gen)[: 2 * num_pairs] for _ in range(batch_size)]
                )
                keys = pool[idx[:, :num_pairs]]
                vals = pool[idx[:, num_pairs:]]
                kv = torch.stack([keys, vals], dim=-1).reshape(batch_size, kv_len)

                query_indices = torch.stack(
                    [torch.randperm(num_pairs, generator=gen)[:num_queries] for _ in range(batch_size)]
                )
                query_keys = keys[torch.arange(batch_size).unsqueeze(1), query_indices]
                query_vals = vals[torch.arange(batch_size).unsqueeze(1), query_indices]

                if delay_tokens > 0:
                    delay = pool[torch.randint(0, pool_size, (batch_size, delay_tokens), generator=gen)]
                else:
                    delay = kv.new_empty(batch_size, 0)

                qa = torch.stack([query_keys, query_vals], dim=-1).reshape(batch_size, qa_len)
                seq = torch.cat([kv, delay, qa], dim=1).to(device)

                logits = model(seq)["logits"]
                ans_start = kv_len + delay_tokens + 1
                for qi in range(num_queries):
                    pos = ans_start + qi * 2 - 1
                    step_logits = logits[:, pos, :]
                    candidate_pred = step_logits.gather(1, vals.to(device)).argmax(dim=-1)
                    label_idx = query_indices[:, qi].to(device)
                    total_correct += int((candidate_pred == label_idx).sum().cpu())
                    total += batch_size

            acc = total_correct / max(total, 1)
            mode_curve[str(num_pairs)] = acc
        capacity_curve[mode] = mode_curve
        breaking_points[mode] = _monotone_breaking_point(mode_curve)

    return {
        "capacity_curve": capacity_curve,
        "breaking_points": breaking_points,
        "breaking_point": breaking_points.get("long", max(breaking_points.values(), default=0)),
        "pool_size": pool_size,
    }


def _continuation_nll(
    model: torch.nn.Module, prefix_ids: list[int], cont_ids: list[int], device: torch.device
) -> float:
    """Mean NLL of `cont_ids` conditioned on `prefix_ids` (teacher-forced).

    Only the continuation tokens are scored (labels `-100` over the prefix), so
    this measures how strongly the model predicts the candidate value given the
    in-context needle — a natural-language likelihood, on the FineWeb manifold.
    """
    if not cont_ids:
        return float("inf")
    ids = prefix_ids + cont_ids
    batch = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    labels = batch.clone()
    labels[:, : len(prefix_ids)] = -100
    out = model(batch, labels=labels)
    loss = out["loss"]
    if loss is None:
        raise RuntimeError("loss was not computed")
    return float(loss.cpu())


# Natural-language key/value pools for the needle probe. Disjoint so a planted
# key never collides with a value. Multi-token under the 4k BPE tokenizer is fine
# because scoring is by continuation NLL, not single-token argmax.
_NEEDLE_KEYS: tuple[str, ...] = (
    "river", "garden", "market", "winter", "engine", "letter", "island", "harvest",
    "candle", "anchor", "planet", "forest", "window", "doctor", "copper", "thunder",
)
_NEEDLE_VALUES: tuple[str, ...] = (
    "mountain", "silver", "yellow", "velvet", "apple", "seven", "blue", "ocean",
    "iron", "maple", "amber", "north", "quartz", "ember", "willow", "frost",
)


@torch.no_grad()
def needle_in_text_probe(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    trials: int = 16,
    num_pairs_list: tuple[int, ...] = (1, 2, 4, 8),
    depth_modes: tuple[str, ...] = ("early", "late"),
    filler_docs: int = 16,
) -> dict[str, Any]:
    """On-manifold associative recall: facts planted in real FineWeb prose.

    Each trial writes `N` natural-language facts ("The secret word for X is Y.")
    separated by real validation text, then queries one fact ("The secret word
    for X is") and scores which candidate value Y has the lowest continuation NLL.
    Both the token distribution AND the sentence structure live on the FineWeb
    manifold, unlike `mqar_probe`/`text_recall_probe` whose synthetic layout is
    off-manifold (E4: those floor at 0 on every FineWeb checkpoint). `depth`
    controls where the queried fact sits: "early" (start, easy) vs "late" (just
    before the query, recency) — and the filler length puts memory under pressure.

    For synthetic/mqar datasets the probe is a no-op (returns empty curves), since
    it requires a real text tokenizer and validation corpus.
    """
    if cfg.data.dataset in {"synthetic", "mqar"}:
        return {"capacity_curve": {}, "breaking_points": {}, "breaking_point": 0, "skipped": True}

    encoder = build_tokenizer(cfg.data.tokenizer)
    seq_cap = _eval_seq_len(cfg)
    rng = torch.Generator(device="cpu").manual_seed(int(cfg.training.seed) + 4242)

    # Real filler sentences from the validation stream (split on periods, keep prose).
    filler: list[str] = []
    for text in fineweb_validation_documents(cfg.data, limit=filler_docs):
        for piece in text.replace("\n", " ").split(". "):
            piece = piece.strip()
            if 20 <= len(piece) <= 200:
                filler.append(piece + ".")
        if len(filler) >= 512:
            break
    if not filler:
        return {"capacity_curve": {}, "breaking_points": {}, "breaking_point": 0, "skipped": True}

    n_keys = min(len(_NEEDLE_KEYS), len(_NEEDLE_VALUES))

    def fact(k: str, v: str) -> str:
        return f" The secret word for {k} is {v}."

    capacity_curve: dict[str, dict[str, float]] = {}
    breaking_points: dict[str, int] = {}

    for mode in depth_modes:
        mode_curve: dict[str, float] = {}
        for num_pairs in num_pairs_list:
            if num_pairs > n_keys:
                mode_curve[str(num_pairs)] = float("nan")
                continue
            correct = 0
            total = 0
            for _ in range(trials):
                perm = torch.randperm(n_keys, generator=rng)[:num_pairs].tolist()
                keys = [_NEEDLE_KEYS[i] for i in perm]
                vals = [_NEEDLE_VALUES[i] for i in perm]
                q = int(torch.randint(0, num_pairs, (1,), generator=rng))

                # Order facts; for "early" the queried fact is first, for "late" last.
                order = list(range(num_pairs))
                if mode == "early":
                    order.remove(q)
                    order = [q] + order
                elif mode == "late":
                    order.remove(q)
                    order = order + [q]

                parts: list[str] = []
                for j in order:
                    parts.append(fact(keys[j], vals[j]))
                    fi = int(torch.randint(0, len(filler), (1,), generator=rng))
                    parts.append(" " + filler[fi])
                context = "".join(parts)

                prompt = f" The secret word for {keys[q]} is"
                prefix_ids = encoder.encode(context + prompt)
                # Honour the eval budget, but NEVER drop the queried fact. The
                # old code unconditionally kept the LAST seq_cap-8 tokens,
                # which for "early" mode deleted exactly the head-planted fact
                # being queried — the trial became unanswerable by construction.
                # "early": keep the head around the fact (the fact is planted
                # first), then re-append the query prompt. "late": keep the
                # tail as before (the queried fact sits just before the query).
                budget = seq_cap - 8
                if len(prefix_ids) > budget:
                    prompt_ids = encoder.encode(prompt)
                    if mode == "early":
                        context_ids = encoder.encode(context)
                        fact_ids = encoder.encode(fact(keys[q], vals[q]))
                        # +2 absorbs BPE boundary merges between the fact and
                        # the filler that follows it.
                        keep = max(len(fact_ids) + 2, budget - len(prompt_ids))
                        prefix_ids = context_ids[:keep] + prompt_ids
                    else:
                        prefix_ids = prefix_ids[-budget:]

                nlls = [_continuation_nll(model, prefix_ids, encoder.encode(" " + v), device) for v in vals]
                if int(torch.tensor(nlls).argmin()) == q:
                    correct += 1
                total += 1

            acc = correct / max(total, 1)
            mode_curve[str(num_pairs)] = acc
        capacity_curve[mode] = mode_curve
        # Chance is 1/num_pairs; require clearing 0.5 as the breaking-point bar.
        breaking_points[mode] = _monotone_breaking_point(mode_curve)

    return {
        "capacity_curve": capacity_curve,
        "breaking_points": breaking_points,
        "breaking_point": breaking_points.get("late", max(breaking_points.values(), default=0)),
        "num_filler": len(filler),
    }


def _sequence_nll(model: torch.nn.Module, ids: list[int], device: torch.device) -> float:
    if len(ids) < 2:
        return float("inf")
    batch = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model(batch, labels=batch)
    loss = out["loss"]
    if loss is None:
        raise RuntimeError("loss was not computed")
    return float(loss.cpu()) * (len(ids) - 1)


# Hand-curated minimal pairs spanning several BLiMP-style grammatical phenomena.
# Each tuple is (acceptable, unacceptable). 80 pairs is enough to make accuracy
# resolution ~0.0125 instead of the 5-pair version's 0.2.
_BLIMP_MINI_PAIRS: list[tuple[str, str]] = [
    # Subject-verb agreement (number)
    ("The dogs are running.", "The dogs is running."),
    ("The child sees the birds.", "The child see the birds."),
    ("These keys open the door.", "These keys opens the door."),
    ("A man walks home.", "A man walk home."),
    ("Several students were absent.", "Several students was absent."),
    ("The neighbor often drives slowly.", "The neighbor often drive slowly."),
    ("My sister bakes excellent bread.", "My sister bake excellent bread."),
    ("Both teachers know the answer.", "Both teachers knows the answer."),
    ("Each cat sleeps on the couch.", "Each cat sleep on the couch."),
    ("Most of the apples were sweet.", "Most of the apples was sweet."),
    # Subject-verb agreement across a relative clause
    ("The book that I read was short.", "The book that I read were short."),
    ("The boys who arrived early are ready.", "The boys who arrived early is ready."),
    ("The actor that they hired performs nightly.", "The actor that they hired perform nightly."),
    ("The friends who left late seem tired.", "The friends who left late seems tired."),
    ("The plant that we bought needs water.", "The plant that we bought need water."),
    # Auxiliary agreement / tense
    ("She has never been late.", "She has never be late."),
    ("They had already eaten.", "They had already ate."),
    ("He has finished his work.", "He has finish his work."),
    ("We have seen that movie.", "We have saw that movie."),
    ("I had taken the train.", "I had took the train."),
    ("She is reading a novel.", "She is read a novel."),
    ("They were walking home.", "They were walk home."),
    ("The boy was playing outside.", "The boy was play outside."),
    # Determiner-noun agreement
    ("She bought a new car.", "She bought an new car."),
    ("I saw an elephant.", "I saw a elephant."),
    ("He gave me an honest answer.", "He gave me a honest answer."),
    ("The teacher graded each paper.", "The teacher graded each papers."),
    ("Every student passed the test.", "Every student passed the tests."),
    ("That dog barks loudly.", "That dogs barks loudly."),
    ("Those books belong to her.", "Those book belong to her."),
    # Anaphor / pronoun agreement
    ("John blamed himself for the error.", "John blamed herself for the error."),
    ("The girls helped themselves.", "The girls helped himself."),
    ("Maria praised herself in the mirror.", "Maria praised himself in the mirror."),
    ("The boy hurt himself badly.", "The boy hurt themselves badly."),
    ("The actors prepared themselves.", "The actors prepared himself."),
    # Pronoun case
    ("She and I went to the store.", "Her and me went to the store."),
    ("He gave the gift to me.", "He gave the gift to I."),
    ("They invited her and me.", "They invited she and I."),
    ("Between you and me, this is hard.", "Between you and I, this is hard."),
    # Wh-movement / island violations (acceptable vs unacceptable extraction)
    ("Who did you say arrived late?", "Who did you say that arrived late?"),
    ("Which book did the editor read?", "Which book did the editor read it?"),
    ("What did the cook prepare?", "What did the cook prepare the meal?"),
    ("Whose dog did you walk yesterday?", "Whose did you walk dog yesterday?"),
    # Polarity / negation
    ("She has not visited Paris.", "She has no visited Paris."),
    ("They do not want any cake.", "They do not want some cake."),
    ("I have never seen anything stranger.", "I have never seen something stranger."),
    ("There isn't anybody here.", "There isn't somebody here."),
    ("He didn't say anything useful.", "He didn't say something useful."),
    # Argument structure (transitivity)
    ("She arranged the flowers.", "She arranged."),
    ("The teacher explained the lesson.", "The teacher explained."),
    ("They built a house.", "They built."),
    ("He devoured the meal.", "He devoured."),
    # Word-order / inversion
    ("Where did you go yesterday?", "Where you did go yesterday?"),
    ("How does this machine work?", "How this machine does work?"),
    ("Never have I seen such beauty.", "Never I have seen such beauty."),
    ("Only then did she understand.", "Only then she did understand."),
    # Mass / count nouns
    ("There is little water left.", "There are little water left."),
    ("She gave me much advice.", "She gave me many advice."),
    ("We have few options remaining.", "We have little options remaining."),
    ("He bought several books.", "He bought several book."),
    # Tense / aspect consistency
    ("Yesterday she walked home.", "Yesterday she walks home."),
    ("Tomorrow we will leave.", "Tomorrow we left."),
    ("Last year they moved here.", "Last year they move here."),
    ("By next year he will have finished.", "By next year he will finished."),
    # Comparatives
    ("She is taller than I am.", "She is more tall than I am."),
    ("This is the best result.", "This is the most best result."),
    ("He runs faster than his brother.", "He runs more faster than his brother."),
    ("That puzzle is harder than this one.", "That puzzle is hard than this one."),
    # Complementiser / subordinate clauses
    ("I think that she is right.", "I think than she is right."),
    ("He said that he would come.", "He said when he would come."),
    ("She knew the answer was correct.", "She knew the answer were correct."),
    # Reflexive vs reciprocal
    ("The siblings hugged each other.", "The siblings hugged themselves."),
    ("The two friends helped each other.", "The two friends helped them."),
    # Subjunctive / counterfactual
    ("If I were you I would go.", "If I am you I would go."),
    ("I wish she were here.", "I wish she was being here."),
    # Coordination ellipsis
    ("She can sing and I can too.", "She can sing and I can also."),
    ("He runs and she does too.", "He runs and she do too."),
]


@torch.no_grad()
def blimp_mini_probe(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, float | int]:
    encoder = None if cfg.data.dataset in {"synthetic", "mqar"} else build_tokenizer(cfg.data.tokenizer)
    correct = 0
    margins: list[float] = []
    skipped = 0
    for good, bad in _BLIMP_MINI_PAIRS:
        if encoder is None:
            good_ids = [b % cfg.model.vocab_size for b in good.encode("utf-8")]
            bad_ids = [b % cfg.model.vocab_size for b in bad.encode("utf-8")]
        else:
            good_ids = encoder.encode(good)
            bad_ids = encoder.encode(bad)
        good_ids = good_ids[: cfg.model.max_seq_len]
        bad_ids = bad_ids[: cfg.model.max_seq_len]
        if len(good_ids) < 2 or len(bad_ids) < 2:
            skipped += 1
            continue
        good_nll = _sequence_nll(model, good_ids, device)
        bad_nll = _sequence_nll(model, bad_ids, device)
        if good_nll < bad_nll:
            correct += 1
        margins.append(bad_nll - good_nll)
    scored = len(_BLIMP_MINI_PAIRS) - skipped
    return {
        "accuracy": correct / max(scored, 1),
        "mean_margin_nats": (sum(margins) / max(len(margins), 1)) if margins else 0.0,
        "examples": scored,
        "skipped": skipped,
    }


# Module attribute names whose forward returns the memory-derived residual added
# to the hidden state. Adding a new memory architecture? Add its read-side module
# name here so the ablation actually zeroes its contribution.
_ABLATABLE_MODULE_NAMES: tuple[str, ...] = (
    "mem_read",  # summary_memory, flow_memory, flow_meanflow, flow_pma, titans_mac, ...
    "engram",  # engram_lite
)


class _AblationState:
    """What `_memory_read_ablation` actually patched while active."""

    def __init__(self) -> None:
        self.patched: list[str] = []

    def __bool__(self) -> bool:
        return bool(self.patched)


@contextlib.contextmanager
def _memory_read_ablation(model: torch.nn.Module) -> Iterator[_AblationState]:
    """Zero out memory-derived residuals during the wrapped block.

    Different memory architectures take different first-argument shapes/dtypes
    (e.g. summary_memory.mem_read takes a float hidden state, engram_lite.engram
    takes a long token-id tensor). We can't construct the right zero tensor from
    the inputs alone, so we run the original forward and then zero the output —
    this is a probe, so the extra compute is fine.

    Yields an `_AblationState` recording every patched read path. Some memory
    variants read memory through paths this probe does NOT know how to patch
    (e.g. phase_memory's bound `PhaseMemoryBlock._read` method, or any
    non-module read): for those nothing matches, the "ablated" forward is
    bit-identical to the normal one, and the caller must treat delta_bpb as
    unmeasured rather than reporting the fabricated 0.0 it would produce.
    """
    state = _AblationState()
    originals: list[tuple[Any, Any]] = []
    try:
        for module in model.modules():
            for attr_name in _ABLATABLE_MODULE_NAMES:
                target = getattr(module, attr_name, None)
                if target is None or not hasattr(target, "forward"):
                    continue
                original = target.forward
                originals.append((target, original))
                state.patched.append(f"{type(module).__name__}.{attr_name}")

                def make_zero_forward(orig):
                    def zero_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                        return torch.zeros_like(orig(*args, **kwargs))

                    return zero_forward

                target.forward = make_zero_forward(original)
        yield state
    finally:
        for target, original in originals:
            target.forward = original


@torch.no_grad()
def memory_ablation_probe(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    doc_limit: int | None = 32,
) -> dict[str, float | int]:
    normal = evaluate_documents(model, cfg, device, doc_limit=doc_limit, bootstrap=False)
    with _memory_read_ablation(model) as ablation:
        if ablation:
            ablated = evaluate_documents(model, cfg, device, doc_limit=doc_limit, bootstrap=False)
        else:
            # Nothing was patched (unpatchable read path, or a variant with no
            # memory at all). The ablated pass would be a bit-identical rerun of
            # the normal pass and delta_bpb a fabricated 0.0 — skip the compute
            # and mark the metric unmeasured (`ablated: false`, NaN deltas) so
            # downstream ranking excludes it instead of tying every such
            # variant at a fake zero.
            ablated = None
    result: dict[str, float | int] = {
        "normal_bpb": float(normal["bpb"]),
        "validation_docs": int(normal["validation_docs"]),
        "ablated": ablated is not None,
    }
    if ablated is not None:
        result["ablated_bpb"] = float(ablated["bpb"])
        result["delta_bpb"] = float(ablated["bpb"]) - float(normal["bpb"])
    else:
        result["ablated_bpb"] = float("nan")
        result["delta_bpb"] = float("nan")
    return result


def attach_average_ranks(rows: list[dict[str, Any]], *, prefix: str = "rank_") -> None:
    """Attach a `composite_avg_rank` headline to each row (lower = better).

    For every metric column named `prefix*`, rank the rows carrying a numeric
    value for it (1 = best, ascending); each row's headline is the mean of its
    ranks. This is the CANONICAL average-rank composite: probe-side tests and
    consumers import it from here, and scripts/aggregate_sweep_results.py
    imports this same function for its sweep tables — it has no private copy,
    and merely rounds the headline to 3 decimals where its output table is
    produced (a presentation-only step, so the value it prints is
    `round(canonical, 3)` by construction).

    Why not a geomean of the rank inputs: the old `geomean_loss_like` took
    `exp(mean(log(max(v, 1e-9))))` over rank_inputs that include NEGATED
    breaking points (values <= 0). Every one clamped to log(1e-9) = -20.72, so
    the three capacity metrics carried zero signal while a strictly-positive
    metric near zero dominated — a perfect blimp score (error 0.0, clamped)
    versus 0.0125 swung the headline ~7.7x. Ranks are scale-free and signed
    only through the value ordering, so negated breaking points rank correctly
    and no metric can swamp the composite.
    """
    rank_cols = sorted({k for row in rows for k in row if k.startswith(prefix)})
    if not rank_cols:
        return
    ranks_by_row: dict[int, list[int]] = {i: [] for i in range(len(rows))}
    for col in rank_cols:
        indexed = [(i, row[col]) for i, row in enumerate(rows) if isinstance(row.get(col), (int, float))]
        indexed.sort(key=lambda item: item[1])
        for rank, (i, _) in enumerate(indexed, start=1):
            ranks_by_row[i].append(rank)
    for i, ranks in ranks_by_row.items():
        if ranks:
            rows[i]["composite_avg_rank"] = sum(ranks) / len(ranks)


@torch.no_grad()
def run_composite_eval(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    *,
    device: torch.device,
    doc_limit: int | None = 64,
) -> dict[str, Any]:
    was_training = model.training
    # The probes below need LOGITS (`model(seq)["logits"]`). A model with the
    # fused-eval CE flag on (fused_linear_ce_eval) returns logits=None from
    # labels-carrying forwards, so force the flag off for the duration and
    # restore it after — mirroring the train()/eval() state handling around
    # this block. Today's logits reads all call without labels (so they take
    # the eager branch regardless) and the label-carrying helpers
    # (_continuation_nll/_sequence_nll) read only the loss; pinning the flag
    # off keeps every probe on the exact eager numerics it was validated on
    # and makes any future labels-carrying logits read correct by
    # construction instead of by accident.
    fused_eval_was_on = getattr(model, "fused_linear_ce_eval", False)
    if fused_eval_was_on:
        model.fused_linear_ce_eval = False
    model.eval()
    try:
        _empty_cuda_cache(device)
        fineweb = evaluate_documents(
            model,
            cfg,
            device,
            doc_limit=doc_limit,
            bootstrap=True,
            bootstrap_samples=1000,
        )
        _empty_cuda_cache(device)
        induction = induction_copy_probe(model, cfg, device)
        _empty_cuda_cache(device)
        assoc = associative_recall_probe(model, cfg, device)
        _empty_cuda_cache(device)
        mqar = mqar_probe(model, cfg, device)
        _empty_cuda_cache(device)
        text_recall = text_recall_probe(model, cfg, device)
        _empty_cuda_cache(device)
        needle = needle_in_text_probe(model, cfg, device)
        _empty_cuda_cache(device)
        memory_ablation = memory_ablation_probe(model, cfg, device, doc_limit=min(doc_limit or 32, 32))
        _empty_cuda_cache(device)
        blimp = blimp_mini_probe(model, cfg, device)
    finally:
        _empty_cuda_cache(device)
        if fused_eval_was_on:
            model.fused_linear_ce_eval = True
        if was_training:
            model.train()

    rank_inputs = {
        "fineweb_bpb": float(fineweb["bpb"]),
        "induction_copy_loss": float(induction["loss"]),
        "assoc_recall_loss": float(assoc["loss"]),
        # Bigger breaking_point = better capacity; negate for lower-is-better ranking.
        "mqar_neg_breaking_point": -float(mqar["breaking_point"]),
        # On-manifold recall: the FineWeb-valid discriminator (see eval-validation plan).
        "text_recall_neg_breaking_point": -float(text_recall["breaking_point"]),
        # Needle-in-real-text: recall where the TASK (not just tokens) is on-manifold.
        "needle_neg_breaking_point": -float(needle["breaking_point"]),
        "blimp_mini_error": 1.0 - float(blimp["accuracy"]),
    }
    if memory_ablation.get("ablated", False):
        # Bigger positive delta means memory helped more, so negate for
        # lower-is-better ranking. ONLY when the ablation actually patched a
        # read path: an unpatched run (unpatchable read path, or a variant
        # with no memory) has a fabricated delta of 0.0 — ranking it would tie
        # every such variant at a fake zero.
        rank_inputs["memory_ablation_neg_delta_bpb"] = -float(memory_ablation["delta_bpb"])
    return {
        "variant": cfg.model.variant,
        "seed": int(cfg.training.seed),
        "config": asdict(cfg),
        "metrics": {
            "fineweb": fineweb,
            "induction_copy": induction,
            "associative_recall": assoc,
            "mqar": mqar,
            "text_recall": text_recall,
            "needle_in_text": needle,
            "memory_ablation": memory_ablation,
            "blimp_mini": blimp,
        },
        "rank_inputs": rank_inputs,
        "lower_is_better": list(rank_inputs),
        # Headline history: this dict used to end with
        #   "geomean_loss_like": exp(mean(log(max(v, 1e-9)) for v in rank_inputs))
        # which was broken by construction — rank_inputs contain NEGATED
        # breaking points (<= 0), all clamping to log(1e-9) = -20.72, so the
        # capacity metrics contributed a constant and any near-zero positive
        # metric dominated (perfect blimp error 0.0 vs 0.0125 = a ~7.7x swing).
        # The headline is now the average-rank composite computed across runs
        # by `attach_average_ranks` above / aggregate_sweep_results.py
        # (`composite_avg_rank`), which is rank-based and scale-free; a
        # single-run scalar headline is not recoverable from one row, so no
        # per-run replacement is emitted here.
    }
