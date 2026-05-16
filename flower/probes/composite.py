from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F

from flower.config import ExperimentConfig
from flower.data import build_tokenizer
from flower.eval import evaluate_documents


def _probe_vocab(cfg: ExperimentConfig) -> tuple[int, int]:
    # Avoid the first few token ids because tokenizers often reserve them for
    # special/control tokens. Synthetic configs can still use the full range.
    if cfg.data.dataset == "synthetic":
        return 0, min(cfg.model.vocab_size, cfg.data.synthetic_vocab_size)
    return min(8, cfg.model.vocab_size - 1), cfg.model.vocab_size


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
    pattern_len = max(4, min(32, cfg.model.max_seq_len // 4))
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.training.seed) + 101)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for _ in range(batches):
        pattern = torch.randint(lo, hi, (batch_size, pattern_len), generator=gen, dtype=torch.long)
        filler = torch.randint(lo, hi, (batch_size, pattern_len), generator=gen, dtype=torch.long)
        seq = torch.cat([pattern, filler, pattern], dim=1).to(device)
        logits = model(seq)["logits"]
        start = pattern_len * 2 - 1
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
    seq_len = min(cfg.model.max_seq_len, max(32, pairs * 2 + 8))
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.training.seed) + 202)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for _ in range(batches):
        keys = torch.randint(lo, hi, (batch_size, pairs), generator=gen, dtype=torch.long)
        vals = torch.randint(lo, hi, (batch_size, pairs), generator=gen, dtype=torch.long)
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
    encoder = None if cfg.data.dataset == "synthetic" else build_tokenizer(cfg.data.tokenizer)
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
    "mem_read",  # summary_memory, flow_memory, flow_meanflow, flow_pma, titans_mac
    "engram",  # engram_lite
)


@contextlib.contextmanager
def _memory_read_ablation(model: torch.nn.Module) -> Iterator[None]:
    """Zero out memory-derived residuals during the wrapped block.

    Different memory architectures take different first-argument shapes/dtypes
    (e.g. summary_memory.mem_read takes a float hidden state, engram_lite.engram
    takes a long token-id tensor). We can't construct the right zero tensor from
    the inputs alone, so we run the original forward and then zero the output —
    this is a probe, so the extra compute is fine.
    """
    originals: list[tuple[Any, Any]] = []
    try:
        for module in model.modules():
            for attr_name in _ABLATABLE_MODULE_NAMES:
                target = getattr(module, attr_name, None)
                if target is None or not hasattr(target, "forward"):
                    continue
                original = target.forward
                originals.append((target, original))

                def make_zero_forward(orig):
                    def zero_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                        return torch.zeros_like(orig(*args, **kwargs))

                    return zero_forward

                target.forward = make_zero_forward(original)
        yield
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
    with _memory_read_ablation(model):
        ablated = evaluate_documents(model, cfg, device, doc_limit=doc_limit, bootstrap=False)
    return {
        "normal_bpb": float(normal["bpb"]),
        "ablated_bpb": float(ablated["bpb"]),
        "delta_bpb": float(ablated["bpb"]) - float(normal["bpb"]),
        "validation_docs": int(normal["validation_docs"]),
    }


@torch.no_grad()
def run_composite_eval(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    *,
    device: torch.device,
    doc_limit: int | None = 64,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        fineweb = evaluate_documents(
            model,
            cfg,
            device,
            doc_limit=doc_limit,
            bootstrap=True,
            bootstrap_samples=1000,
        )
        induction = induction_copy_probe(model, cfg, device)
        assoc = associative_recall_probe(model, cfg, device)
        memory_ablation = memory_ablation_probe(model, cfg, device, doc_limit=min(doc_limit or 32, 32))
        blimp = blimp_mini_probe(model, cfg, device)
    finally:
        if was_training:
            model.train()

    rank_inputs = {
        "fineweb_bpb": float(fineweb["bpb"]),
        "induction_copy_loss": float(induction["loss"]),
        "assoc_recall_loss": float(assoc["loss"]),
        # Bigger positive delta means memory helped more, so negate for lower-is-better ranking.
        "memory_ablation_neg_delta_bpb": -float(memory_ablation["delta_bpb"]),
        "blimp_mini_error": 1.0 - float(blimp["accuracy"]),
    }
    return {
        "variant": cfg.model.variant,
        "seed": int(cfg.training.seed),
        "config": asdict(cfg),
        "metrics": {
            "fineweb": fineweb,
            "induction_copy": induction,
            "associative_recall": assoc,
            "memory_ablation": memory_ablation,
            "blimp_mini": blimp,
        },
        "rank_inputs": rank_inputs,
        "lower_is_better": list(rank_inputs),
        "geomean_loss_like": math.exp(sum(math.log(max(v, 1e-9)) for v in rank_inputs.values()) / len(rank_inputs)),
    }
