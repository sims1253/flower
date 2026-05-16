import pytest
import torch

from flower.config import ModelConfig
from flower.models import build_model

# fla_gdn uses FLA's Triton kernels which require CUDA. Other variants run on CPU.
CPU_VARIANTS = [
    "vanilla_local",
    "vanilla_full",
    "linear_memory",
    "summary_memory",
    "flow_attention",
    "flow_memory",
    "fa_sm",
    "fa_fm",
    "engram_lite",
    "titans_mac",
    "flow_meanflow",
    "flow_pma",
]
CUDA_ONLY_VARIANTS = ["fla_gdn"]


@pytest.mark.parametrize("variant", CPU_VARIANTS)
def test_variant_forward_shapes_cpu(variant: str):
    cfg = ModelConfig(
        variant=variant,
        vocab_size=128,
        d_model=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=16,
        local_window=4,
        memory_slots=4,
        flow_steps=1,
    )
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(tokens, labels=tokens)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"].ndim == 0


@pytest.mark.parametrize("variant", CUDA_ONLY_VARIANTS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA Triton kernels require CUDA")
def test_variant_forward_shapes_cuda(variant: str):
    cfg = ModelConfig(
        variant=variant,
        vocab_size=128,
        d_model=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=16,
        local_window=4,
        memory_slots=4,
        flow_steps=1,
    )
    device = torch.device("cuda")
    model = build_model(cfg).to(device)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    out = model(tokens, labels=tokens)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"].ndim == 0
