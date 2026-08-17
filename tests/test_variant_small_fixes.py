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
6. (pullfrog follow-up) The variant forwards route the final projection and
   loss through CausalLM._compute_logits/_cross_entropy plus the fused-CE and
   activation-checkpoint paths, so fp8_lm_head / bf16_cross_entropy /
   fused_linear_ce / activation_checkpoint are live instead of silent no-ops —
   while flag-off numerics stay bit-identical to the old inline ops.
"""

import math

import pytest
import torch
import torch.nn.functional as F
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
    return ModelConfig(variant=variant, **{**_COMMON, **overrides})


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
    Attribute presence here; the follow-up fix (pullfrog review) routes the
    forward through the base helpers so the head/CE/checkpoint flags are
    actually CONSUMED — pinned by the tests below."""
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
    # The forward runs with every flag set: checkpointed blocks ("ffn" falls
    # back to full), eager head on CPU (fp8 is CUDA+eval-only; fused CE falls
    # back off-CUDA with a warning), bf16 loss.
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


# ---------------------------------------------------------------------------
# Fix 6 (pullfrog review of PR #11): the variant forwards now route the final
# projection + CE through the CausalLM helpers (and mirror base's fused-CE /
# activation-checkpoint paths), so the base flags are no longer silent no-ops.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_flag_off_forward_is_bit_identical_to_inline_path(variant):
    """Helper routing must not change default numerics: with every flag off,
    _compute_logits reduces to self.head(self.ln(x)) and _cross_entropy to the
    inline shifted F.cross_entropy, so logits and loss are BIT-identical to
    the pre-change inline path recomputed by hand with the same weights."""
    torch.manual_seed(0)
    model = build_model(_cfg(variant)).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, model.config.max_seq_len))
    with torch.no_grad():
        out = model(tokens, labels=tokens)
        # --- the old inline path, recomputed with the model's own weights ---
        x = model.token(tokens)
        if variant == "engram_lite":
            for block in model.blocks:
                x = block(x, tokens)
        else:
            state = None
            for block in model.blocks:
                x, state = block(x, state)
        old_logits = model.head(model.ln(x))
        old_loss = F.cross_entropy(
            old_logits[:, :-1].reshape(-1, old_logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )
    assert torch.equal(old_logits, out["logits"])
    assert torch.equal(old_loss, out["loss"])


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_bf16_cross_entropy_casts_loss_path(variant):
    """bf16_cross_entropy=True must change the variant's loss path exactly like
    base's: the shifted logits are cast to bf16 before CE (loss comes back
    bf16), instead of silently keeping the fp32 inline CE."""
    tokens = torch.randint(0, 128, (2, 16))
    torch.manual_seed(0)
    model_off = build_model(_cfg(variant)).eval()
    torch.manual_seed(0)
    model_on = build_model(_cfg(variant, bf16_cross_entropy=True)).eval()
    with torch.no_grad():
        out_off = model_off(tokens, labels=tokens)
        out_on = model_on(tokens, labels=tokens)
    # The cast happened: fp32 loss off, bf16 loss on.
    assert out_off["loss"].dtype == torch.float32
    assert out_on["loss"].dtype == torch.bfloat16
    # The flag touches only the loss, not the logits (same seed -> same weights).
    assert torch.equal(out_off["logits"], out_on["logits"])
    # The bf16 loss is EXACTLY the manual inline bf16 CE over the flag-off
    # logits (mirrors base's _cross_entropy when the flag is on).
    manual = F.cross_entropy(
        out_off["logits"][:, :-1].to(torch.bfloat16).reshape(-1, out_off["logits"].size(-1)),
        tokens[:, 1:].reshape(-1),
    )
    assert float(out_on["loss"]) == pytest.approx(float(manual), abs=1e-6)
    # Consistent with base's tolerance for bf16 CE (lossy but close).
    assert torch.allclose(out_on["loss"].float(), out_off["loss"].float(), rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 head path needs CUDA")
@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_fp8_lm_head_routes_through_fp8_head_in_eval(variant):
    """fp8_lm_head=True + eval + bf16 + CUDA routes the tied-head matmul
    through CausalLM._fp8_head (via _compute_logits); training falls back to
    the BF16 head (no _scaled_mm backward kernel). Only the ROUTING and
    finiteness are pinned here — the head's numerics are pinned by the S3
    tests in test_training_speedups.py."""
    torch.manual_seed(0)
    cfg = _cfg(variant, fp8_lm_head=True)
    model = build_model(cfg).to("cuda").to(torch.bfloat16)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device="cuda")
    calls = {"n": 0}
    original = model._fp8_head

    def spy(t, _orig=original, _calls=calls):
        _calls["n"] += 1
        return _orig(t)

    model._fp8_head = spy
    model.eval()
    with torch.no_grad():
        out = model(tokens, labels=tokens)
    assert calls["n"] == 1  # routed through the FP8 head exactly once
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["logits"].dtype == torch.bfloat16
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["loss"])

    # Train mode: BF16 head (backward must work); no further FP8-head calls.
    model.train()
    out_train = model(tokens, labels=tokens)
    assert calls["n"] == 1
    out_train["loss"].backward()
    assert torch.isfinite(out_train["loss"])


def _run_variant_training_step(
    variant: str, activation_checkpoint, ids: torch.Tensor, backward: bool
) -> float:
    torch.manual_seed(0)
    cfg = _cfg(variant, dropout=0.1, activation_checkpoint=activation_checkpoint)
    model = build_model(cfg)
    model.train()
    out = model(ids, labels=ids)
    if backward:
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    return float(out["loss"])


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
@pytest.mark.parametrize("mode", [True, "selective"])
def test_activation_checkpoint_loss_identical_dropout_rng(variant, mode):
    """Core checkpointing gate, mirroring the base tests (which are also
    forward-only): with dropout > 0 the loss must be bit-identical with and
    without checkpointing at the same seed — use_reentrant=False saves/restores
    RNG state so the forward matches, and the same contract holds for the
    custom block signatures (input_ids arg for engram, (memory, write_mag)
    tuple state for freq-decay).

    NOTE on "selective": backward through selective checkpointing crashes with
    torch 2.13's "Tensor cached during selective activation checkpoint has
    been mutated" on the SDPA path — for the VANILLA base model identically
    (pre-existing, not introduced here; base's own selective identity test is
    forward-only for this reason, and its only selective+backward test runs
    under flex+autocast). The variant backward-under-selective path is pinned
    on CUDA+flex below, mirroring that test."""
    torch.manual_seed(123)
    ids = torch.randint(0, 128, (2, 16))
    loss_no = _run_variant_training_step(variant, False, ids, backward=False)
    loss_ckpt = _run_variant_training_step(variant, mode, ids, backward=False)
    assert loss_no == loss_ckpt, (
        f"{variant} {mode!r} checkpointing changed the loss with dropout>0: "
        f"{loss_no} vs {loss_ckpt} (delta {abs(loss_no - loss_ckpt):.2e}); "
        f"RNG state not preserved"
    )


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_activation_checkpoint_full_backward_produces_grads(variant):
    """Full checkpointing must carry gradients through the checkpointed custom
    block calls (input_ids arg / tuple state) and backprop finitely."""
    torch.manual_seed(123)
    ids = torch.randint(0, 128, (2, 16))
    loss = _run_variant_training_step(variant, True, ids, backward=True)
    assert math.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="selective backward is exercised under flex on CUDA")
@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_activation_checkpoint_selective_backward_under_flex(variant):
    """Selective checkpointing + backward through the variant blocks, under the
    flex+autocast configuration base's own selective-backward test uses (the
    plain-SDPA selective backward crashes in torch 2.13 for vanilla and these
    variants alike — see the identity test's NOTE). Pins that the recompute
    handles the custom block signatures: gradients flow and stay finite."""
    from flower.models.base import prebuild_attention_masks

    dev = torch.device("cuda")
    torch.manual_seed(0)
    # d_model/num_heads -> head_dim 16: Triton's flex kernel requires >= 16.
    cfg = _cfg(
        variant,
        d_model=64,
        num_heads=4,
        ffn_dim=128,
        num_layers=2,
        dropout=0.1,
        activation_checkpoint="selective",
        flex_attention=True,
    )
    model = build_model(cfg).to(dev).to(torch.bfloat16)
    prebuild_attention_masks(model, cfg.max_seq_len, dev)
    model.train()
    torch.manual_seed(123)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(ids, labels=ids)["loss"]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g.float()).all() for g in grads)


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_activation_checkpoint_ffn_falls_back_to_full(variant, capsys):
    """"ffn" has no clean FFN-only boundary in these custom blocks (the n-gram
    residual / memory update interleave with attention+FFN), so it falls back
    to FULL checkpointing with a one-time printed notice — the documented
    base.py contract for non-vanilla blocks. The fallback is real
    checkpointing, not a silent no-op: bit-identical loss to full mode, and
    the notice prints exactly once."""
    torch.manual_seed(123)
    ids = torch.randint(0, 128, (2, 16))
    loss_full = _run_variant_training_step(variant, True, ids, backward=False)
    loss_ffn = _run_variant_training_step(variant, "ffn", ids, backward=False)
    assert loss_ffn == loss_full

    model = build_model(_cfg(variant, activation_checkpoint="ffn"))
    model.train()
    capsys.readouterr()  # drain
    model(ids, labels=ids)
    first = capsys.readouterr().out
    assert "activation_checkpoint='ffn'" in first
    assert "falling back to full checkpointing" in first
    model(ids, labels=ids)
    assert "falling back to full checkpointing" not in capsys.readouterr().out


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_activation_checkpoint_is_noop_in_eval(variant):
    """Eval mode must run uncheckpointed (no backward -> a recompute would be
    pure waste): same seed, flag on vs off -> bit-identical logits."""
    torch.manual_seed(123)
    ids = torch.randint(0, 128, (2, 16))
    torch.manual_seed(0)
    model_off = build_model(_cfg(variant)).eval()
    torch.manual_seed(0)
    model_on = build_model(_cfg(variant, activation_checkpoint=True)).eval()
    with torch.no_grad():
        logits_off = model_off(ids)["logits"]
        logits_on = model_on(ids)["logits"]
    assert torch.equal(logits_off, logits_on)


@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_fused_linear_ce_cpu_fallback_warns_once(variant):
    """Mirrors base's CPU-fallback test: fused_linear_ce=True off CUDA must
    fall back to eager WITH a one-time warning — not a silent no-op."""
    import warnings

    cfg = _cfg(variant, fused_linear_ce=True)
    model = build_model(cfg)  # CPU model — the fused kernel cannot run here
    model.train()
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(tokens, labels=tokens)  # first call: should warn
        model(tokens, labels=tokens)  # second call: must NOT warn again
    fallback_warnings = [w for w in caught if "fused_linear_ce" in str(w.message)]
    assert len(fallback_warnings) == 1, (
        f"expected exactly one fallback warning, got {len(fallback_warnings)}"
    )
    assert "falling back to the eager" in str(fallback_warnings[0].message)
    assert getattr(model, "_warned_fused_ce_fallback", False) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger fused CE kernel is CUDA/Triton-only")
@pytest.mark.parametrize("variant", ["engram_lite", "frequency_decay_memory"])
def test_fused_linear_ce_matches_eager_and_keeps_tied_grad(variant):
    """On CUDA the variants run the real Liger fused kernel through the same
    _fused_cross_entropy helper as base: the (B*T, vocab) logits tensor is
    never materialized, the loss matches eager to bf16 precision, and the
    tied embedding gradient flows through the by-reference weight."""
    dev = torch.device("cuda")
    torch.manual_seed(0)
    eager = build_model(_cfg(variant)).to(dev).to(torch.bfloat16)
    torch.manual_seed(0)
    fused = build_model(_cfg(variant, fused_linear_ce=True)).to(dev).to(torch.bfloat16)
    fused.load_state_dict(eager.state_dict())
    eager.train()
    fused.train()
    assert fused._ensure_liger_fce() is True  # the real kernel path is live
    tokens = torch.randint(0, eager.config.vocab_size, (2, eager.config.max_seq_len), device=dev)
    out_e = eager(tokens, labels=tokens)
    out_f = fused(tokens, labels=tokens)
    assert out_f["logits"] is None  # fused path never materializes logits
    assert torch.isfinite(out_f["loss"])
    # Same loss to bf16 precision (kernel-selection noise, as in base's test).
    assert torch.allclose(out_f["loss"].float(), out_e["loss"].float(), rtol=5e-2)
    out_f["loss"].backward()
    out_e["loss"].backward()
    # The tie stays load-bearing: the embedding table receives the head's
    # gradient through the fused kernel's by-reference weight.
    assert fused.token.weight.grad is not None
    assert torch.allclose(
        fused.token.weight.grad.float(), eager.token.weight.grad.float(), rtol=0.05, atol=0.05
    )
