import torch

from flower.config import ModelConfig
from flower.models import build_model
from flower.models.base import causal_mask


def test_local_causal_mask_excludes_future_and_old_tokens():
    mask = causal_mask(5, torch.device("cpu"), local_window=2)
    expected = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [False, True, True, False, False],
            [False, False, True, True, False],
            [False, False, False, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_future_token_does_not_change_past_logits():
    cfg = ModelConfig(
        variant="vanilla_local",
        vocab_size=64,
        d_model=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=8,
        local_window=4,
    )
    model = build_model(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (1, 8))
    b = a.clone()
    b[:, -1] = (b[:, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_a = model(a)["logits"][:, :-1]
        logits_b = model(b)["logits"][:, :-1]
    assert torch.allclose(logits_a, logits_b, atol=1e-5)
