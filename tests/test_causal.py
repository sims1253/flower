import pytest
import torch

from flower.config import ModelConfig
from flower.models import build_model
from flower.models.base import causal_mask
from flower.models.memory import SDPCrossAttention, causal_prefix_attention


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


# ---------------------------------------------------------------------------
# Causal memory writes (ModelConfig.causal_memory).
#
# Every memory variant's legacy write aggregates the ENTIRE window — future
# tokens included — into the memory bank, and the next layer broadcasts that
# bank to every position. logits[t] can therefore depend on input tokens > t,
# which is answer leakage for a next-token objective. The tests below pin the
# fixed behaviour: with causal_memory=True, perturbing the LAST input token
# must leave logits[:, :-1] unchanged, and logits at any prefix must be
# invariant to truncating the sequence after that prefix (the strict
# autoregressive property).
# ---------------------------------------------------------------------------

CAUSAL_MEMORY_VARIANTS = [
    "vanilla_local",  # no memory path: must pass unconditionally
    "linear_memory",
    "summary_memory",
    "phase_memory",
    "partitioned_memory",
    "titans_mac",
    "flow_ot_memory",
    "surprise_memory",
    "frequency_decay_memory",
    "bloom_memory",
]

MEMORY_ONLY_VARIANTS = [v for v in CAUSAL_MEMORY_VARIANTS if v != "vanilla_local"]


def _tiny_cfg(variant: str, causal: bool = True) -> ModelConfig:
    return ModelConfig(
        variant=variant,
        vocab_size=64,
        d_model=64,
        num_heads=4,
        num_layers=3,  # >=3 layers: the leak rides the write->read threading
        ffn_dim=128,
        max_seq_len=48,
        local_window=16,
        memory_slots=8,
        causal_memory=causal,
    )


def _last_token_leak(model: torch.nn.Module, vocab_size: int) -> float:
    torch.manual_seed(1)
    a = torch.randint(0, vocab_size, (2, 48))
    b = a.clone()
    b[:, -1] = (b[:, -1] + 1) % vocab_size
    with torch.no_grad():
        logits_a = model(a)["logits"][:, :-1]
        logits_b = model(b)["logits"][:, :-1]
    return (logits_a - logits_b).abs().max().item()


@pytest.mark.parametrize("variant", CAUSAL_MEMORY_VARIANTS)
def test_causal_memory_future_token_does_not_change_past_logits(variant):
    cfg = _tiny_cfg(variant, causal=True)
    model = build_model(cfg).eval()
    delta = _last_token_leak(model, cfg.vocab_size)
    assert delta == 0.0, f"last-token perturbation moved past logits by {delta}"


@pytest.mark.parametrize("variant", MEMORY_ONLY_VARIANTS)
def test_causal_memory_prefix_truncation_invariance(variant):
    """Strict autoregressive property: running the model on a prefix must give
    exactly the same logits at that prefix as running it on the full sequence."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(variant, causal=True)
    model = build_model(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 48))
    cut = 30
    with torch.no_grad():
        full = model(x)["logits"][:, :cut]
        trunc = model(x[:, :cut])["logits"]
    assert torch.allclose(full, trunc, atol=1e-5), (
        f"max delta {(full - trunc).abs().max().item()}"
    )


@pytest.mark.parametrize("variant", CAUSAL_MEMORY_VARIANTS)
def test_causal_memory_training_step_is_differentiable(variant):
    torch.manual_seed(0)
    cfg = _tiny_cfg(variant, causal=True)
    model = build_model(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 48))
    out = model(x, labels=x)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert with_grad, "no parameter received gradient"


def test_causal_prefix_attention_matches_naive_reference():
    """The cumsum/cummax formulation must equal a per-position softmax loop."""
    torch.manual_seed(0)
    B, H, P, T, hd = 2, 3, 4, 7, 5
    scores = torch.randn(B, H, P, T) * 3.0
    v = torch.randn(B, H, T, hd)
    out = causal_prefix_attention(scores, v)  # (B, H, T, P, hd)
    for t in range(T):
        w = torch.softmax(scores[..., : t + 1], dim=-1)  # (B, H, P, t+1)
        ref = torch.einsum("bhpt,bhtd->bhpd", w, v[:, :, : t + 1])
        assert torch.allclose(out[:, :, t], ref, atol=1e-5), f"mismatch at t={t}"


def test_sdpcross_attention_causal_forward_matches_prefix_attention():
    """causal_forward(latents, x)[..., t, :, :] must equal an unmasked
    forward(latents, x[:, :t+1]) — same parameters, prefix-restricted."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, num_heads=4, max_seq_len=16, local_window=16, memory_slots=8)
    sdp = SDPCrossAttention(cfg).eval()
    x = torch.randn(2, 12, 32)
    latents = torch.randn(2, 5, 32)
    with torch.no_grad():
        causal = sdp.causal_forward(latents, x)  # (B, T, P, D)
        for t in (0, 1, 6, 11):
            ref = sdp.forward(latents, x[:, : t + 1])  # (B, P, D)
            assert torch.allclose(causal[:, t], ref, atol=1e-5), f"mismatch at t={t}"


# ---------------------------------------------------------------------------
# Flow-hybrid audit (READ-ONLY; these variants are NOT fixed by the causal
# memory PR — their write paths live in files owned by other changes). The
# table pins the measured leak status so any future fix flips an expectation
# here and forces this table (and the PR notes) to be updated.
# Measured max |logits[:, :-1] delta| from perturbing only the last input
# token, tiny 3-layer config, seed-controlled (see test): every hybrid leaks.
# ---------------------------------------------------------------------------

FLOW_HYBRID_LEAK_STATUS = {
    "flow_memory": 0.00397,
    "flow_meanflow": 0.01196,
    "flow_pma": 0.00541,
    "fa_sm": 0.00374,
    "fa_fm": 0.00250,
}


@pytest.mark.parametrize("variant", sorted(FLOW_HYBRID_LEAK_STATUS))
def test_flow_hybrid_leak_status_is_as_documented(variant):
    cfg = _tiny_cfg(variant, causal=False)  # current (legacy) code path
    model = build_model(cfg).eval()
    delta = _last_token_leak(model, cfg.vocab_size)
    leaks = delta > 1e-5
    assert leaks, (
        f"{variant} no longer leaks (delta={delta:.2e}) — it was fixed; update "
        f"FLOW_HYBRID_LEAK_STATUS and the causal-memory PR notes"
    )
