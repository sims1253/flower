"""MTP auxiliary losses must not leak into the eval loss.

MTP is a TRAINING-ONLY auxiliary objective: the extra heads predict t+2, t+3 to
shape the representation, but the model is judged on the main tied head's t+1
prediction. val_bpb is compared across arms, so anything else in the eval loss
makes those comparisons meaningless.

This is not hypothetical. The first MTP screen (`runs/mtp_screen_450m`, 2026-08-13)
measured val_bpb 1.128 -> 2.036 (1 head) -> 3.069 (2 heads), which reads as
catastrophic divergence. It was entirely a leak: the fused CE path is gated on
`self.training`, so eval ALWAYS falls into the eager branch, and that branch
added `mtp_weight * mtp_loss` unconditionally. The implied auxiliary losses were
1.61x and 1.83x the main loss — monotone in offset, exactly as predicting
further ahead should be, which is what identified the cause.
"""

import torch

from flower.config import ModelConfig
from flower.models import build_model


def _cfg(heads: int, weight: float = 0.5) -> ModelConfig:
    return ModelConfig(
        variant="vanilla_local", vocab_size=64, d_model=64, num_heads=4,
        num_layers=2, ffn_dim=128, max_seq_len=32, local_window=8,
        mtp_extra_heads=heads, mtp_weight=weight,
    )


def _eval_loss(model, ids) -> float:
    model.eval()
    with torch.no_grad():
        return float(model(input_ids=ids, labels=ids)["loss"])


def test_eval_loss_is_identical_with_and_without_mtp_heads():
    """The headline invariant: extra heads must not move the eval loss at all.

    Built from one seed so the shared trunk is bit-identical between the two
    models; only the presence of the auxiliary heads differs. Those heads are
    untied and unused at eval, so the loss must match exactly.
    """
    ids = torch.randint(0, 64, (2, 16))

    torch.manual_seed(0)
    plain = build_model(_cfg(0))
    torch.manual_seed(0)
    with_mtp = build_model(_cfg(2))

    # Copy the shared trunk across so the only difference is the MTP heads.
    src = dict(plain.named_parameters())
    for name, p in with_mtp.named_parameters():
        if name in src:
            p.data.copy_(src[name].data)

    assert with_mtp.mtp_heads is not None and len(with_mtp.mtp_heads) == 2
    torch.testing.assert_close(
        torch.tensor(_eval_loss(with_mtp, ids)),
        torch.tensor(_eval_loss(plain, ids)),
        rtol=1e-6, atol=1e-6,
    )


def test_eval_loss_does_not_scale_with_head_count():
    """The exact signature of the bug that shipped.

    Under the leak, eval loss grew roughly linearly in the number of heads
    (ratios 1.81x and 2.72x at 450M). Anything monotone in head count here means
    the auxiliary objective is being scored.
    """
    ids = torch.randint(0, 64, (2, 16))
    losses = []
    for heads in (0, 1, 2):
        torch.manual_seed(0)
        losses.append(_eval_loss(build_model(_cfg(heads)), ids))

    spread = max(losses) - min(losses)
    assert spread < 1e-5, f"eval loss varies with head count {losses} (spread {spread:.2e})"


def test_eval_loss_is_independent_of_mtp_weight():
    """`mtp_weight` is a training knob; it must be invisible at eval.

    Independent of the head-count test: a leak that summed the auxiliary losses
    without the weight would pass this and fail the one above, and a weight of
    0.0 would pass the one above while still leaving the leak in place.
    """
    ids = torch.randint(0, 64, (2, 16))
    out = []
    for w in (0.25, 0.5, 1.0):
        torch.manual_seed(0)
        out.append(_eval_loss(build_model(_cfg(2, weight=w)), ids))

    assert max(out) - min(out) < 1e-5, f"eval loss depends on mtp_weight: {out}"


def test_training_loss_still_includes_the_auxiliary_objective():
    """The guard must not disable MTP — that would make the feature a no-op.

    The complement of the tests above: in train mode the auxiliary heads MUST
    contribute, and more of them (or a larger weight) must raise the loss.
    """
    ids = torch.randint(0, 64, (2, 16))

    def train_loss(heads: int, weight: float = 0.5) -> float:
        torch.manual_seed(0)
        m = build_model(_cfg(heads, weight))
        m.train()
        return float(m(input_ids=ids, labels=ids)["loss"].detach())

    base, one, two = train_loss(0), train_loss(1), train_loss(2)
    assert one > base + 1e-4, f"1 MTP head did not add to the training loss ({one} vs {base})"
    assert two > one + 1e-4, f"2 MTP heads did not add over 1 ({two} vs {one})"
    assert train_loss(2, 1.0) > train_loss(2, 0.25) + 1e-4, "mtp_weight has no training effect"
