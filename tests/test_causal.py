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
    # fa_sm builds SummaryMemoryBlocks (fixed above) plus a last-dim-only
    # EulerFlow memory read, so it is FULLY causal under the flag — measured
    # last-token leak exactly 0.0 (pullfrog review of the audit table moved
    # it here from the unfixed column).
    "fa_sm",
]

MEMORY_ONLY_VARIANTS = [v for v in CAUSAL_MEMORY_VARIANTS if v != "vanilla_local"]


def _tiny_cfg(variant: str, causal: bool = True, **options) -> ModelConfig:
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
        **options,
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


def test_causal_memory_prefix_truncation_rbf_kernel_bias_floor():
    """memory_kernel_bias="rbf" + causal_memory=True: DOCUMENTED ~1.5e-5 floor.

    The rbf grid in MemoryRead._bias_causal normalises each token's query
    position by the CURRENT sequence length (linspace(0, 1, q_len), same
    construction as the legacy read), so truncating the window from T to
    `cut` rescales every query position t/(T-1) -> t/(cut-1) and moves the
    logits slightly. Measured max delta ~1.5e-5 at this seed — just above the
    1e-5 atol that the kernel_bias="none" variants hold in the test above,
    hence this dedicated case with a relaxed, annotated tolerance.

    This is length-CONDITIONAL bias, not token-value leakage: no future token
    values enter the bias or the write (the last-token perturbation test
    measures exactly 0.0 with rbf too), and the legacy read has the same
    T-dependence. Pinned so a future change to the grid (e.g. normalising by
    max_seq_len to make it length-invariant) flips this expectation
    deliberately."""
    torch.manual_seed(0)
    cfg = _tiny_cfg("summary_memory", causal=True, memory_kernel_bias="rbf")
    model = build_model(cfg).eval()
    torch.manual_seed(2)
    x = torch.randint(0, cfg.vocab_size, (2, 48))
    cut = 30
    with torch.no_grad():
        full = model(x)["logits"][:, :cut]
        trunc = model(x[:, :cut])["logits"]
    delta = (full - trunc).abs().max().item()
    assert delta < 5e-5, f"rbf truncation delta {delta} blew past the documented ~1.5e-5 floor"


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
# Flow-hybrid audit. Of the five flow hybrids, only fa_sm is fixed by the
# causal-memory flag (it builds SummaryMemoryBlocks — see CAUSAL_MEMORY_VARIANTS
# above; measured leak exactly 0.0). The other four (flow_memory,
# flow_meanflow, flow_pma, fa_fm) contain NO causal_memory handling: their
# write paths live in files owned by the causal-flow-hybrids follow-up PR
# (#12). Setting causal_memory=True on them would silently train non-causal,
# so build_model refuses loudly (see CAUSAL_MEMORY_UNSUPPORTED_VARIANTS in
# flower/models/__init__.py). This table pins the LEGACY (flag-off) leak that
# justifies the guard: measured max |logits[:, :-1] delta| from perturbing
# only the last input token, tiny 3-layer config, seed-controlled (see test).
# When PR #12 fixes a variant it must remove it here AND drop the guard.
# ---------------------------------------------------------------------------

FLOW_HYBRID_LEAK_STATUS = {
    "flow_memory": 0.00397,
    "flow_meanflow": 0.01196,
    "flow_pma": 0.00541,
    "fa_fm": 0.00250,
}


@pytest.mark.parametrize("variant", sorted(FLOW_HYBRID_LEAK_STATUS))
def test_unsupported_flow_hybrid_fails_loudly_and_legacy_leak_is_as_documented(variant):
    # causal_memory=True must not silently train the legacy write: model
    # construction raises, naming the variant and the follow-up PR.
    with pytest.raises(ValueError, match=variant):
        build_model(_tiny_cfg(variant, causal=True))
    # The reason for the guard is still true: the legacy (flag-off) write
    # aggregates the whole window, so the last-token perturbation leaks into
    # past logits. Seed BEFORE build_model so the pinned table above is
    # reproducible regardless of which tests ran before this one.
    torch.manual_seed(3)
    cfg = _tiny_cfg(variant, causal=False)  # legacy code path
    model = build_model(cfg).eval()
    delta = _last_token_leak(model, cfg.vocab_size)
    leaks = delta > 1e-5
    assert leaks, (
        f"{variant} no longer leaks (delta={delta:.2e}) — it was fixed; update "
        f"FLOW_HYBRID_LEAK_STATUS, drop it from CAUSAL_MEMORY_UNSUPPORTED_VARIANTS, "
        f"and move it to CAUSAL_MEMORY_VARIANTS"
    )


def test_fa_sm_is_supported_but_fa_fm_is_not():
    """fa_sm (SummaryMemoryBlocks) honours the flag; fa_fm must still guard."""
    build_model(_tiny_cfg("fa_sm", causal=True))  # no ValueError
    with pytest.raises(ValueError, match="fa_fm"):
        build_model(_tiny_cfg("fa_fm", causal=True))


# ---------------------------------------------------------------------------
# bf16 regression (pullfrog A1): causal_prefix_attention used to keep the
# scores' dtype through the softmax while floating v, so any causal path
# routing through it crashed pure-bf16 models with
# "expected scalar type Float but found BFloat16". The scores are now floated
# before the masked softmax; these tests pin both the primitive and every
# model-level causal path that routes through it (summary perceiver /
# attention-aggregation, bloom, flow_ot).
# ---------------------------------------------------------------------------


def test_causal_prefix_attention_bf16_mixed_scores_and_values():
    torch.manual_seed(0)
    B, H, P, T, hd = 2, 3, 4, 7, 5
    scores = (torch.randn(B, H, P, T) * 3.0).to(torch.bfloat16)
    v = torch.randn(B, H, T, hd).to(torch.bfloat16)
    out = causal_prefix_attention(scores, v)  # used to raise RuntimeError
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    # fp32-computed reference on the same values: bf16 inputs lose precision,
    # but the fp32 softmax keeps the agreement tight.
    ref = causal_prefix_attention(scores.float(), v.float())
    assert torch.allclose(out.float(), ref, atol=1e-2), (out.float() - ref).abs().max().item()


BF16_CAUSAL_CASES = [
    ("linear_memory", {}),
    ("summary_memory", {}),
    # Both causal_prefix_attention users inside SummaryMemoryBlock:
    ("summary_memory", {"summary_style": "perceiver"}),
    ("summary_memory", {"memory_aggregation": "attention"}),
    ("phase_memory", {}),
    ("partitioned_memory", {}),
    ("titans_mac", {}),
    ("flow_ot_memory", {}),  # causal source attention
    ("surprise_memory", {}),
    ("frequency_decay_memory", {}),
    ("bloom_memory", {}),  # causal summary attention
    ("fa_sm", {}),  # SummaryMemoryBlock write + flow memory read
]


@pytest.mark.parametrize("variant,options", BF16_CAUSAL_CASES)
def test_causal_memory_bf16_forward_and_backward_do_not_crash(variant, options):
    """A pure-bf16 causal model must forward (and backward) cleanly.

    Exercises the fp32-floated prefix softmax in causal_prefix_attention
    (perceiver summary + attention aggregation, bloom, flow_ot), which
    crashed before the fix; the remaining variants pin that no other causal
    write/read path mixes dtypes either."""
    cfg = _tiny_cfg(variant, causal=True, **options)
    model = build_model(cfg).to(torch.bfloat16)
    x = torch.randint(0, cfg.vocab_size, (2, 48))
    out = model(x, labels=x)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
