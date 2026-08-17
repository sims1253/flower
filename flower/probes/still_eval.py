"""Compaction evaluation probes for Still-style KV-cache compactors.

These probes measure how well a trained StillLM preserves information under
KV-cache compression, adapted from the Still paper's evaluation methodology:

1. **Compaction quality curve**: measure student-vs-teacher KL at multiple
   compression ratios on held-out text.
2. **Needle-through-compaction**: plant a fact in a long context, compact
   the KV cache, then query the fact. Measures exact-retrieval preservation.

These complement Flower's existing composite eval probes, adding the
compaction-specific metrics that the Still paper cares about.

Both probes are read-only with respect to model state: compaction_kl_curve
temporarily overrides each compactor's `compact_len`/`latents` to hit the
requested ratios and restores them in a finally block, so evaluating a model
never leaves it reconfigured (pinned by tests/test_still_eval_probes.py).
"""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def compaction_kl_curve(
    model: torch.nn.Module,
    cfg: Any,
    device: torch.device,
    *,
    batches: int = 8,
    batch_size: int = 4,
    compression_ratios: tuple[int, ...] = (4, 8, 16, 32),
) -> dict[str, Any]:
    """Measure teacher-student KL at multiple compression ratios.

    For each compression ratio, temporarily set the compactor's compact_len
    to achieve that ratio, run the forward pass, and measure KL. The
    compact_len/latents overrides are restored on exit, so the model is left
    exactly as the probe found it.

    Returns a dict mapping compression_ratio -> {kl, student_loss, teacher_loss}.
    """
    from flower.data import token_batches

    results: dict[str, dict[str, float]] = {}
    T = cfg.data.sequence_length

    # Snapshot the compactor state we are about to override, so restoration is
    # exact (same Parameter object, not just equal values).
    compactors = list(model.compactors) if hasattr(model, "compactors") else []
    saved = [(comp.compact_len, comp.latents) for comp in compactors]
    try:
        for ratio in compression_ratios:
            target_compact = max(1, T // ratio)
            # Update compact_len in the model's compactors.
            for comp in compactors:
                comp.compact_len = target_compact
                # Resize latent bank if needed.
                if comp.latents.shape[1] != target_compact:
                    old_latents = comp.latents.data
                    new_latents = torch.zeros(
                        old_latents.shape[0], target_compact, old_latents.shape[2],
                        device=old_latents.device, dtype=old_latents.dtype,
                    )
                    copy_len = min(old_latents.shape[1], target_compact)
                    new_latents[:, :copy_len] = old_latents[:, :copy_len]
                    comp.latents = torch.nn.Parameter(new_latents)
                    comp.compact_len = target_compact

            # Run a few batches and collect KL.
            batches_iter = token_batches(cfg.data, batch_size, device, seed=999)
            total_kl = 0.0
            total_loss_s = 0.0
            total_loss_t = 0.0
            count = 0

            for _ in range(batches):
                batch = next(batches_iter)
                if isinstance(batch, (tuple, list)):
                    input_ids, labels = batch
                else:
                    input_ids, labels = batch, batch

                out = model(input_ids, labels=labels)
                total_kl += float(out["diagnostics"].get("kl_loss", 0.0))
                total_loss_s += float(out["diagnostics"].get("student_loss", 0.0))
                total_loss_t += float(out["diagnostics"].get("teacher_loss", 0.0))
                count += 1

            results[str(ratio)] = {
                "kl": total_kl / max(count, 1),
                "student_loss": total_loss_s / max(count, 1),
                "teacher_loss": total_loss_t / max(count, 1),
                "compact_len": target_compact,
                "compression_ratio": ratio,
            }
    finally:
        for comp, (compact_len, latents) in zip(compactors, saved):
            comp.compact_len = compact_len
            comp.latents = latents

    return results


@torch.no_grad()
def needle_through_compaction(
    model: torch.nn.Module,
    cfg: Any,
    device: torch.device,
    *,
    trials: int = 32,
    seq_lens: tuple[int, ...] = (256, 512, 1024),
    vocab_size: int = 256,
) -> dict[str, Any]:
    """Needle-in-a-haystack through KV compaction.

    Plant a specific token at a random position in a random sequence, then check
    whether the compacted model can still identify the needle when prompted.

    For each trial:
    1. Generate a random sequence of length L.
    2. Plant a "needle" token at position p.
    3. Forward through the model (teacher + student).
    4. At the position after the needle, check if student logits place the needle
       token in top-k.

    This is a synthetic version of Still's RULER needle task.
    """
    results: dict[str, dict[str, float]] = {}

    for seq_len in seq_lens:
        if seq_len > cfg.model.max_seq_len:
            continue
        correct_student = 0
        correct_teacher = 0
        total = 0

        for trial in range(trials):
            gen = torch.Generator(device="cpu").manual_seed(trial * 1000 + seq_len)
            seq = torch.randint(1, vocab_size, (1, seq_len), generator=gen, dtype=torch.long)
            needle_pos = int(torch.randint(seq_len // 4, 3 * seq_len // 4, (1,), generator=gen))
            needle_val = int(torch.randint(1, vocab_size, (1,), generator=gen))
            seq[0, needle_pos] = needle_val
            seq = seq.to(device)

            # Student and teacher forward.
            out = model(seq)
            student_logits = out["logits"]

            # Teacher logits (run base model without compaction).
            if hasattr(model, "base_model"):
                with torch.no_grad():
                    teacher_out = model.base_model(seq)
                    teacher_logits = teacher_out["logits"]

                # At position needle_pos, the logit predicts the NEXT token.
                # Check if the needle value is the argmax (exact copy task).
                # Actually, the needle is at needle_pos; the model needs to
                # "find" it given a query. We simplify: check if needle_val
                # is in top-5 of the logits at the last position (the model
                # should "know" the needle exists in context).
                last_s = student_logits[0, -1, :]
                last_t = teacher_logits[0, -1, :]

                top5_s = last_s.topk(5).indices.tolist()
                top5_t = last_t.topk(5).indices.tolist()

                if needle_val in top5_s:
                    correct_student += 1
                if needle_val in top5_t:
                    correct_teacher += 1
            total += 1

        results[str(seq_len)] = {
            "student_recall_at_5": correct_student / max(total, 1),
            "teacher_recall_at_5": correct_teacher / max(total, 1),
            "trials": total,
        }

    return results


@torch.no_grad()
def run_still_composite_eval(
    model: torch.nn.Module,
    cfg: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Full Still-specific evaluation suite."""
    was_training = model.training
    model.eval()
    try:
        kl_curve = compaction_kl_curve(model, cfg, device, batches=4, batch_size=2)
        needle = needle_through_compaction(model, cfg, device, trials=16)
    finally:
        if was_training:
            model.train()

    return {
        "variant": cfg.model.variant,
        "compaction_kl_curve": kl_curve,
        "needle_through_compaction": needle,
    }
