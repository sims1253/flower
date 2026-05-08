import pytest
import torch

from flower.config import ModelConfig
from flower.models import build_model


@pytest.mark.parametrize(
    "variant",
    ["vanilla_local", "vanilla_full", "linear_memory", "summary_memory", "flow_attention", "flow_memory", "fa_sm", "fa_fm"],
)
def test_variant_forward_shapes(variant: str):
    cfg = ModelConfig(variant=variant, vocab_size=128, d_model=32, num_heads=4, num_layers=1, ffn_dim=64, max_seq_len=16, local_window=4, memory_slots=4, flow_steps=1)
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(tokens, labels=tokens)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"].ndim == 0
