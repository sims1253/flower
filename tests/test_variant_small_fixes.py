"""Small verified fixes for the model-variant LMs (adversarial-review bundle).

One test group per fix; see the PR body for the per-finding context:

1. EngramLiteLM / FrequencyDecayLM construct through CausalLM.__init__ (param
   counts and state_dict keys unchanged; base-owned flags now exist).
2. The hard eval-length caps were removed where sound (engram n-gram hash is
   position-independent; freq-decay state is allocated per forward).
3. SummaryMemoryBlock._orthogonal_residual docstring now tells the truth
   (per-row Gram projection); behavior pinned as-is.
4. PartitionedMemoryBlock computes ln_mem once per block; logits bit-for-bit
   identical to the old two-evaluation path.
5. Frequency-decay write_mag accumulation/saturation documented and pinned;
   per-block write-magnitude diagnostic emitted.
"""

import math

import pytest
import torch
from torch import nn

from flower.config import ModelConfig
from flower.models import build_model
from flower.models.base import CausalLM
from flower.models.engram_lite import EngramLiteBlock, EngramLiteLM
from flower.models.frequency_decay_memory import FrequencyDecayBlock, FrequencyDecayLM
from flower.models.partitioned_memory import PartitionedMemoryBlock
from flower.models.summary_memory import SummaryMemoryBlock

# Shared tiny config: 3 layers so cross-block state threading is exercised.
_COMMON = dict(
    vocab_size=128,
    d_model=32,
    num_heads=4,
    num_layers=3,
    ffn_dim=64,
    max_seq_len=16,
    local_window=4,
    memory_slots=4,
)


def _cfg(variant: str, **overrides) -> ModelConfig:
    return ModelConfig(variant=variant, **_COMMON, **overrides)


# ---------------------------------------------------------------------------
# Fix 1: CausalLM.__init__ is now the constructor of record for both variants.
# ---------------------------------------------------------------------------


class _HandBuiltEngram(nn.Module):
    """The pre-refactor EngramLiteLM construction (for key/count comparison)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([EngramLiteBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight


class _HandBuiltFrequencyDecay(nn.Module):
    """The pre-refactor FrequencyDecayLM construction (for key/count comparison)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([FrequencyDecayBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight


@pytest.mark.parametrize(
    "variant, ref_cls",
    [
        ("engram_lite", _HandBuiltEngram),
        ("frequency_decay_memory", _HandBuiltFrequencyDecay),
    ],
)
def test_causalm_init_preserves_param_count_and_state_dict_keys(variant, ref_cls):
    """super().__init__(config, blocks) must not change what the model IS:
    same parameter count and byte-for-byte the same state_dict keys as the
    old hand-built construction (checkpoints stay loadable)."""
    cfg = _cfg(variant)
    torch.manual_seed(0)
    model = build_model(cfg)
    torch.manual_seed(0)
    reference = ref_cls(cfg)
    assert isinstance(model, CausalLM)
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    assert list(model.state_dict().keys()) == list(reference.state_dict().keys())


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_causalm_init_flags_exist_on_variant_lms(variant):
    """The base CausalLM flags must at least EXIST on the variant LMs. Before
    the refactor a sweep setting any of them silently did nothing here (the
    variants skipped CausalLM.__init__, so the attributes were never created).
    Attribute presence only — the variant forwards don't consume the flags."""
    cfg = _cfg(
        variant,
        activation_checkpoint="ffn",
        fused_linear_ce=True,
        fp8_lm_head=True,
        bf16_cross_entropy=True,
        mtp_extra_heads=1,
    )
    model = build_model(cfg)
    assert model.activation_checkpoint == "ffn"  # stored verbatim, not bool()-collapsed
    assert model.fused_linear_ce is True
    assert model.fp8_lm_head is True
    assert model.bf16_cross_entropy is True
    assert model.mtp_extra_heads == 1
    assert isinstance(model.mtp_heads, nn.ModuleList) and len(model.mtp_heads) == 1
    assert model.attn_res_sites == [] and model.depth_router is None
    assert model._static_diagnostics is None  # cache attribute owned by the parent
    # The forward still runs with the flags set (it just doesn't use them).
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(tokens, labels=tokens)
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_smoke_train_step(variant):
    """One optimizer step: forward -> finite loss -> backward -> grads -> step."""
    torch.manual_seed(0)
    model = build_model(_cfg(variant))
    tokens = torch.randint(0, model.config.vocab_size, (2, model.config.max_seq_len))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = model(tokens, labels=tokens)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    opt.zero_grad()
    with torch.no_grad():
        loss_after = model(tokens, labels=tokens)["loss"]
    assert torch.isfinite(loss_after)


# ---------------------------------------------------------------------------
# Fix 2: eval beyond max_seq_len is allowed where sound (base CausalLM already
# allows it; RoPE cache extends lazily, local mask is built on-the-fly).
# ---------------------------------------------------------------------------


def test_engram_lite_eval_beyond_max_seq_len():
    """The n-gram hash is position-independent (it hashes only the trailing n
    token ids), so evaluating at T > max_seq_len is sound. Prefix logits of a
    long run must match a short run of the same prefix: local attention +
    trailing-n-gram residual + pointwise FFN are all independent of suffix
    tokens."""
    cfg = _cfg("engram_lite")
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))  # 24 > max_seq_len=16
    with torch.no_grad():
        long_out = model(tokens, labels=tokens)
        short_out = model(tokens[:, :16], labels=tokens[:, :16])
    assert long_out["logits"].shape == (2, 24, cfg.vocab_size)
    assert torch.isfinite(long_out["loss"])
    assert torch.allclose(long_out["logits"][:, :16], short_out["logits"], atol=1e-5)


def test_frequency_decay_eval_beyond_max_seq_len():
    """freq-decay memory/write_mag are allocated per forward from the input's
    own T — nothing is tied to max_seq_len, so long eval is sound."""
    cfg = _cfg("frequency_decay_memory")
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))  # 24 > max_seq_len=16
    with torch.no_grad():
        out = model(tokens, labels=tokens)
    assert out["logits"].shape == (2, 24, cfg.vocab_size)
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("variant", ["summary_memory", "partitioned_memory"])
@pytest.mark.parametrize("causal", [False, True])
def test_summary_and_partitioned_eval_beyond_max_seq_len(variant, causal):
    """summary/partitioned build on CausalLM directly, whose forward has no
    cap (base.py:989 deliberately allows eval beyond max_seq_len); their
    memory states are allocated per forward, so nothing blocks long eval.
    Pins that these variants keep working under Sweep 7 A1 conditions."""
    cfg = _cfg(variant, causal_memory=causal, num_memory_banks=2)
    model = build_model(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        out = model(tokens, labels=tokens)
    assert out["logits"].shape == (2, 24, cfg.vocab_size)
    assert torch.isfinite(out["loss"])


# ---------------------------------------------------------------------------
# Fix 3: _orthogonal_residual docstring now matches the math (per-row Gram
# projection). Behavior pinned exactly as computed.
# ---------------------------------------------------------------------------


def test_orthogonal_residual_is_per_row_projection():
    """Pin the actual behavior: each output row is orthogonal to its OWN
    (same-slot) memory row, but NOT to the span of the other rows — this is a
    per-row Gram projection, not a row-span projection."""
    torch.manual_seed(0)
    B, S, D = 2, 4, 8
    memory = torch.randn(B, S, D)
    update = torch.randn(B, S, D)
    eps = 1e-6
    out = SummaryMemoryBlock._orthogonal_residual(update, memory, eps)
    mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + eps)
    # Orthogonal to the corresponding memory row (that's what is computed).
    own_dot = (out * mem_norm).sum(dim=-1)
    assert torch.allclose(own_dot, torch.zeros_like(own_dot), atol=1e-4)
    # NOT projected against the other rows: cross-row overlap is nonzero in
    # general (guards against someone "fixing" this into a full row-span
    # projection without a flag — that would zero these too).
    cross = torch.einsum("sd,td->st", out[0], mem_norm[0])
    off_diag = cross[~torch.eye(S, dtype=torch.bool)]
    assert off_diag.abs().max() > 1e-3


# ---------------------------------------------------------------------------
# Fix 4: ln_mem computed once per partitioned block; logits bit-for-bit
# identical (same input -> same deterministic LayerNorm output).
# ---------------------------------------------------------------------------


def _partitioned_cfg(causal: bool) -> ModelConfig:
    return _cfg("partitioned_memory", causal_memory=causal, num_memory_banks=2)


@pytest.mark.parametrize("causal", [False, True])
def test_partitioned_ln_mem_evaluated_once_per_block(causal):
    cfg = _partitioned_cfg(causal)
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    counts = []
    for block in model.blocks:
        calls = {"n": 0}
        original = block.ln_mem.forward

        def counting_forward(t, _orig=original, _calls=calls):
            _calls["n"] += 1
            return _orig(t)

        block.ln_mem.forward = counting_forward
        counts.append(calls)
    model(tokens)
    assert all(c["n"] == 1 for c in counts), [c["n"] for c in counts]


@pytest.mark.parametrize("causal", [False, True])
def test_partitioned_ln_mem_dedup_is_bit_for_bit(causal):
    """Recompute the pre-dedup path (ln_mem evaluated twice per block) with
    the model's own weights and require bitwise-identical logits."""
    cfg = _partitioned_cfg(causal)
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    with torch.no_grad():
        out = model(tokens)
        # --- old path: two separate ln_mem evaluations per block ---
        x = model.token(tokens)
        memory = None
        for block in model.blocks:
            if memory is None or memory.shape[-3] != block.num_banks:
                memory = block._initial_memory_causal(x) if causal else block._initial_memory(x)
            x = x + block.local(block.ln1(x))
            bank_w, bank_logw = block._bank_weights(block.ln_mem(x))
            x = x + block.mem_read(block.ln_mem(x), memory, bank_logw)
            x = x + block.ff(block.ln2(x))
            if block.config.memory_update_frequency <= 1 or x.shape[1] % block.config.memory_update_frequency == 0:
                memory = block._update_memory(memory, x, bank_w)
        old_logits = model.head(model.ln(x))
    assert torch.equal(old_logits, out["logits"])


# ---------------------------------------------------------------------------
# Fix 5: freq-decay write_mag accumulation/saturation documented + pinned,
# per-block write-magnitude diagnostic emitted.
# ---------------------------------------------------------------------------


def test_write_mag_accumulates_monotonically_across_blocks():
    """Each block adds its candidate magnitudes on top of the running
    write_mag, so the per-block exit values are non-decreasing over the
    stack (and strictly positive)."""
    cfg = _cfg("frequency_decay_memory")
    torch.manual_seed(0)
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(tokens, labels=tokens)
    mags = [block.last_diag_write_mag for block in model.blocks]
    assert all(m > 0 for m in mags)
    assert all(b >= a for a, b in zip(mags, mags[1:]))
    diag = out["diagnostics"]
    assert diag["write_mag_mean"] == pytest.approx(sum(mags) / len(mags))
    assert diag["write_mag_max"] == pytest.approx(max(mags))
    # The pre-existing final-state diagnostics are still emitted.
    assert diag["frequency_decay_mean_mag"] == pytest.approx(mags[-1])
    assert diag["frequency_decay_max_mag"] >= diag["frequency_decay_mean_mag"]


def test_write_mag_accumulates_across_loops():
    """loop_count > 1 keeps summing into the same accumulator (no reset
    between loops), so the final magnitude strictly grows with loop count.
    Same seed -> identical weights; only the loop count differs."""
    tokens = torch.randint(0, 128, (2, 16))
    final_mags = []
    for loops in (1, 2):
        torch.manual_seed(0)
        model = build_model(_cfg("frequency_decay_memory", loop_count=loops))
        out = model(tokens, labels=tokens)
        final_mags.append(out["diagnostics"]["frequency_decay_mean_mag"])
    assert final_mags[1] > final_mags[0]


def test_write_mag_saturation_zeroes_retention():
    """Once base * (1 + freq_penalty * write_mag) hits the clamp at 1.0,
    retention (1 - effective) is 0 and new_memory == candidate: the slot's
    previous content is fully discarded on that write."""
    cfg = _cfg("frequency_decay_memory", causal_memory=False)
    torch.manual_seed(0)
    block = FrequencyDecayBlock(cfg)
    B, T, S, D = 2, 6, cfg.memory_slots, cfg.d_model
    memory = torch.randn(B, S, D)
    x = torch.randn(B, T, D)
    # candidate, replicated from _update's non-causal path for this memory/x.
    token_summary = block._aggregate(x).expand(-1, S, -1)
    combined = block.token_mlp(token_summary) + block.mem_mlp(memory)
    candidate = block.update(combined) / max(1, cfg.num_layers)

    new_mem_saturated, _ = block._update(memory, torch.full((B, S), 1e9), x)
    assert torch.allclose(new_mem_saturated, candidate, atol=1e-6)

    # Unsaturated (write_mag = 0): retention is base decay < 1, so the old
    # content survives and the result differs from the raw candidate.
    new_mem_fresh, _ = block._update(memory, torch.zeros(B, S), x)
    base = torch.sigmoid(block.decay_logit).view(1, -1)
    assert (base < 1).all()
    assert not torch.allclose(new_mem_fresh, candidate, atol=1e-6)
    assert torch.allclose(new_mem_fresh, (1 - base).unsqueeze(-1) * memory + candidate, atol=1e-5)
