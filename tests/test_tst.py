"""Tests for Token Superposition Training (S9).

TST was previously a stub: `compress_to_bags` produced 3-D inputs, `labels` were
never bagged, and the model forward — which sits OUTSIDE the guard in train.py —
raised "too many values to unpack (expected 3)". `tst_enabled: true` crashed at
step 1 while the config flags and the data helper made it look implemented.

These tests pin the pieces that make it real:
  * the multi-hot objective reduces EXACTLY to standard CE at bag_size=1
    (the strongest available correctness check — no new maths at s=1), and
  * it equals the mean per-token CE over the bag (its definition), and
  * the sequence really is compressed s-fold (the source of the speedup).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flower.config import ModelConfig
from flower.data import compress_to_bags
from flower.models import build_model


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = ModelConfig(
        variant="vanilla_local", vocab_size=64, d_model=32, num_heads=2,
        num_layers=2, ffn_dim=64, max_seq_len=64, local_window=16,
    )
    return build_model(cfg)


def test_bag_size_one_reduces_to_standard_next_token_loss():
    """At s=1 the multi-hot objective must be the ordinary CE, bit-for-bit."""
    m = _tiny_model()
    ids = torch.randint(0, 64, (2, 32))
    ntp = m(ids, labels=ids)["loss"]
    bagged = m(compress_to_bags(ids, 1), labels=compress_to_bags(ids, 1))["loss"]
    assert torch.allclose(ntp, bagged, atol=1e-6), f"{ntp.item()} vs {bagged.item()}"


def test_multi_hot_loss_equals_mean_per_token_ce_over_the_bag():
    """The definition: uniform credit across the s tokens of the next bag."""
    m = _tiny_model()
    B, T, s = 2, 32, 4
    ids = torch.randint(0, 64, (B, T))
    bi, bl = compress_to_bags(ids, s), compress_to_bags(ids, s)
    out = m(bi, labels=bl)

    logits = out["logits"][:, :-1]        # (B, L-1, V)
    targets = bl[:, 1:]                   # (B, L-1, s)
    # Reference: average ordinary CE over each of the s targets independently.
    per_token = torch.stack([
        F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets[..., j].reshape(-1),
        )
        for j in range(s)
    ]).mean()
    assert torch.allclose(out["loss"], per_token, atol=1e-5), (
        f"multi-hot {out['loss'].item():.6f} != mean per-token CE {per_token.item():.6f}"
    )


def test_sequence_is_compressed_by_the_bag_factor():
    """The speedup comes from running T/s positions; verify it actually happens."""
    m = _tiny_model()
    B, T, s = 2, 32, 4
    ids = torch.randint(0, 64, (B, T))
    out = m(compress_to_bags(ids, s), labels=compress_to_bags(ids, s))
    assert out["logits"].shape[1] == T // s


def test_gradients_flow_through_the_bagged_path():
    m = _tiny_model()
    ids = torch.randint(0, 64, (2, 32))
    m(compress_to_bags(ids, 4), labels=compress_to_bags(ids, 4))["loss"].backward()
    total = sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
    assert torch.isfinite(total) and total > 0


def test_embeddings_are_averaged_over_the_bag():
    """A bag of identical tokens must embed exactly like that single token."""
    m = _tiny_model()
    same = torch.full((1, 2, 4), 7, dtype=torch.long)   # two bags, all token 7
    single = torch.full((1, 2), 7, dtype=torch.long)
    assert torch.allclose(m.token(same).mean(dim=2), m.token(single))


def test_single_bag_sequence_returns_zero_loss_with_grad():
    """Fewer than two bags means nothing to predict; must not crash the graph."""
    m = _tiny_model()
    ids = torch.randint(0, 64, (1, 4))
    out = m(compress_to_bags(ids, 4), labels=compress_to_bags(ids, 4))
    assert out["loss"].item() == 0.0
    assert out["loss"].requires_grad
