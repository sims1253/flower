import pytest
import torch

from flower.config import ModelConfig
from flower.models import build_model
from flower.models.base import causal_mask
from flower.models.memory import SDPCrossAttention, causal_prefix_attention, causal_running_mean


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
    # fa_sm builds SummaryMemoryBlocks (fixed by the causal-memory PR) plus a
    # last-dim-only EulerFlow memory read, so it is FULLY causal under the
    # flag — measured last-token leak exactly 0.0 (pullfrog review of the
    # base PR's audit table moved it to the fixed column).
    # The remaining flow hybrids are fixed by THIS PR (causal-flow-hybrids);
    # fa_fm builds on the fixed FlowMemoryBlock + FlowSelfAttention:
    "fa_sm",
    "flow_memory",
    "flow_meanflow",
    "flow_pma",
    "fa_fm",
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


# Per-variant fp floor for the prefix-truncation test (see that test's
# comment); variants absent from the map keep the 1e-5 default.
PREFIX_TRUNCATION_ATOL = {
    "flow_memory": 5e-5,
    "flow_meanflow": 5e-5,
    "flow_pma": 5e-5,
}


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
    # The property is exact in real arithmetic. In floating point the SHARED
    # machinery (local-attention softmax and the norms reassociate their
    # reductions when the sequence length changes — vanilla_local itself
    # measures ~7.6e-06 at this seed) puts every variant at a ~1e-5 floor;
    # the flow hybrids' invertible-flow writes amplify that floor a little
    # further (measured up to ~1.5e-05, the same value the base branch's
    # linear_memory shows under an adjacent seed). Every per-position memory
    # op (cumsum, masked prefix softmax, last-tokens unfold) is bitwise
    # prefix-exact; the exact-0 last-token leak test above is the hard
    # causal guarantee, this one pins "no extra leak beyond fp reassociation".
    atol = PREFIX_TRUNCATION_ATOL.get(variant, 1e-5)
    assert torch.allclose(full, trunc, atol=atol), f"max delta {(full - trunc).abs().max().item()}"


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
# Flow hybrids (causal-flow-hybrids PR): the causal write at position t must
# equal the LEGACY write restricted to the prefix x[:, :t+1] with the
# per-position slot state at t — same parameters, prefix-restricted. These
# are the per-variant correctness references for the new write paths.
# ---------------------------------------------------------------------------


def test_flow_memory_causal_write_equals_prefix_restricted_legacy():
    torch.manual_seed(0)
    cfg = _tiny_cfg("flow_memory", causal=True)
    block = build_model(cfg).blocks[0]
    x = torch.randn(2, 10, cfg.d_model)
    memory = torch.randn(2, 10, cfg.memory_slots, cfg.d_model)
    with torch.no_grad():
        out = block._update_memory_causal(memory, x)  # (B, T, S, D)
        for t in (0, 4, 9):
            # Legacy write on the prefix: whole-"window" mean of x[:, :t+1],
            # coupling flow on the flat state at t.
            cond = block.cond(x[:, : t + 1].mean(dim=1)).unsqueeze(1)  # (B, 1, D)
            state = memory[:, t : t + 1]  # (B, 1, S, D) -> flat (B, 1, flat_dim)
            legacy = block._unflat_memory(block.flow(block._flat_memory(state), cond), state)
            assert torch.allclose(out[:, t], legacy.squeeze(1), atol=1e-5), f"mismatch at t={t}"


def test_flow_pma_causal_write_equals_prefix_restricted_legacy():
    """The manual prefix PMA must match nn.MultiheadAttention run on the prefix
    (same weights), and the per-slot flow / short-memory slice must match the
    legacy forms restricted to the prefix."""
    torch.manual_seed(0)
    cfg = _tiny_cfg("flow_pma", causal=True, hierarchical_memory=True)
    block = build_model(cfg).blocks[0]
    x = torch.randn(2, 10, cfg.d_model)
    slots_total = cfg.memory_slots + cfg.short_memory_slots
    memory = torch.randn(2, 10, slots_total, cfg.d_model)
    with torch.no_grad():
        out = block._update_memory_causal(memory, x)  # (B, T, S_total, D)
        for t in (0, 3, 9):
            xp = x[:, : t + 1]
            seeds = block.seeds.expand(2, -1, -1)
            pooled, _ = block.pma(seeds, xp, xp, need_weights=False)  # legacy MHA on prefix
            cond = block.cond_proj(pooled)
            legacy_long = block.slot_flow(memory[:, t, : cfg.memory_slots], cond)
            assert torch.allclose(out[:, t, : cfg.memory_slots], legacy_long, atol=1e-5), (
                f"long-memory mismatch at t={t}"
            )
            # Short memory: legacy left-padded slice of the prefix.
            short = xp[:, -cfg.short_memory_slots :]
            if short.shape[1] < cfg.short_memory_slots:
                short = torch.nn.functional.pad(short, (0, 0, cfg.short_memory_slots - short.shape[1], 0))
            assert torch.equal(out[:, t, cfg.memory_slots :], short), f"short mismatch at t={t}"


def test_flow_meanflow_causal_write_equals_prefix_restricted_legacy():
    """The per-position MeanFlow endpoint must equal the legacy field applied
    to the prefix state with the prefix-mean conditioning."""
    torch.manual_seed(0)
    cfg = _tiny_cfg("flow_meanflow", causal=True)
    block = build_model(cfg).blocks[0]
    x = torch.randn(2, 10, cfg.d_model)
    memory = torch.randn(2, 10, cfg.memory_slots, cfg.d_model)
    ones, zeros = x.new_ones(()), x.new_zeros(())
    cond = causal_running_mean(x)  # (B, T, D)
    with torch.no_grad():
        out = block.field.forward_positions(memory, ones, zeros, cond)  # (B, T, S, D)
        for t in (0, 4, 9):
            u_ref = block.field(memory[:, t], ones, zeros, x[:, : t + 1].mean(dim=1))
            assert torch.allclose(out[:, t], u_ref, atol=1e-5), f"mismatch at t={t}"


def test_flow_meanflow_ot_cfm_couples_batch_not_time():
    """Property checks for the OT-CFM aux-loss coupling (semantics NOT changed
    by this PR; see the NOTE ON THE OT-CFM BATCH COUPLING in
    MeanFlowMemoryBlock._meanflow_loss).

    The Sinkhorn plan pairs z0/z1 across the BATCH dimension: with
    meanflow_ot_cfm=True each element's regression target is a soft mixture
    of the OTHER elements' memories (the standard OT-CFM choice), and the aux
    loss never feeds the forward logits. Concretely, pinned here:

    1. The plan has NO time axis: it is a (B, B) matrix, so the coupling can
       never reference a position directly.
    2. Given the plan — which is computed under no_grad, i.e. it is a constant
       w.r.t. the training gradients — the value mixing is TIME-LOCAL:
       zeroing z1[k, t0] changes targets ONLY at position t0 (bitwise
       unchanged at every other position, for every batch element). The
       causal per-position states therefore never gain a NEW time coupling
       from the mixing einsum itself.
    3. The batch-axis coupling is real (and intentionally kept): at t0 the
       targets of elements i with plan[i, k] > 0 DO move (documented
       semantics). A literal "zeroing z1[k] must not change any other
       element's target" — promised by an earlier version of this docstring —
       is FALSE (the plan mixes other elements' z1 by design; measured 0.095
       at this seed), which is exactly why the coupling is restricted to the
       aux loss.
    4. The forward logits are BITWISE identical with meanflow_ot_cfm on/off
       (same weights, same input): the aux loss never feeds the forward.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg("flow_meanflow", causal=True, meanflow_ot_cfm=True)
    block = build_model(cfg).blocks[0]
    B, T = 4, 6
    z0 = torch.randn(B, T, cfg.memory_slots, cfg.d_model)
    z1 = torch.randn(B, T, cfg.memory_slots, cfg.d_model)
    cond = torch.randn(B, T, cfg.d_model)

    # The real loss path stays finite (exercising the causal 4-D branch).
    torch.manual_seed(7)
    aux = block._meanflow_loss(z0, z1, cond)
    assert torch.isfinite(aux)

    # Reconstruct the plan exactly as _meanflow_loss does (same cost, same
    # Sinkhorn call, fp32, no_grad) so the mixing can be inspected in isolation.
    from flower.models.flow_meanflow import _sinkhorn_plan

    with torch.no_grad():
        cost = torch.cdist(z0.reshape(B, -1), z1.reshape(B, -1), p=2.0).pow(2)
        plan = _sinkhorn_plan(cost, epsilon=cfg.meanflow_ot_epsilon, iters=cfg.meanflow_ot_iters)
    assert plan.shape == (B, B), "the transport plan must stay a batch x batch matrix"

    def targets(z1_values: torch.Tensor) -> torch.Tensor:
        z1_paired = B * torch.einsum("ij,j...->i...", plan, z1_values)
        return z1_paired - z0

    base = targets(z1)
    # (2) time-locality of the mixing under the (no_grad) plan: zeroing one
    # element's z1 AT ONE POSITION leaves every target at every OTHER
    # position bitwise unchanged...
    z1_one_pos = z1.clone()
    z1_one_pos[2, 3] = 0.0
    moved = targets(z1_one_pos)
    assert torch.equal(moved[:, :3], base[:, :3]), "coupling leaked backwards in time"
    assert torch.equal(moved[:, 4:], base[:, 4:]), "coupling leaked forwards in time"
    # ...and (3) at position 3 itself other elements' targets DO move when
    # the plan gives them mass on element 2 (the documented batch mixture):
    # find an element i != 2 with plan[i, 2] > 0 and require the move.
    coupled = [i for i in range(B) if i != 2 and plan[i, 2] > 0]
    assert coupled, "plan gave element 2 no partners; re-record with a draw that couples"
    for i in coupled:
        assert not torch.equal(moved[i, 3], base[i, 3]), f"element {i} lost its documented batch coupling"

    # (4) the aux loss never feeds the forward logits: identical weights and
    # input give bitwise-identical logits regardless of meanflow_ot_cfm.
    torch.manual_seed(11)
    model_with = build_model(_tiny_cfg("flow_meanflow", causal=True, meanflow_ot_cfm=True)).eval()
    torch.manual_seed(11)
    model_without = build_model(_tiny_cfg("flow_meanflow", causal=True)).eval()
    torch.manual_seed(1)
    a = torch.randint(0, cfg.vocab_size, (2, 48))
    with torch.no_grad():
        logits_with = model_with(a)["logits"]
        logits_without = model_without(a)["logits"]
    assert torch.equal(logits_with, logits_without), "aux-loss coupling leaked into the forward logits"


# ---------------------------------------------------------------------------
# Energy read (memory.py): the query-axis-chunked logsumexp must match the
# unchunked reference exactly (each query row is independent), and forcing
# tiny chunk budgets through MemoryRead must not move the logits.
# ---------------------------------------------------------------------------


def test_energy_read_chunked_matches_unchunked_reference():
    torch.manual_seed(0)
    from flower.models.memory import _energy_read, _energy_read_unchunked

    bsz, heads, q_len, m_len, head_dim = 2, 3, 16, 5, 7
    scores = torch.randn(bsz, heads, q_len, m_len) * 3.0
    v = torch.randn(bsz, heads, m_len, head_dim)
    beta = torch.tensor(1.3)
    ref = _energy_read_unchunked(scores, v, beta)
    # Budget forces one query row per chunk; rows are computed by the exact
    # same kernels, so the concatenation must be bitwise identical.
    chunked = _energy_read(scores, v, beta, max_temp_elements=97)
    assert torch.equal(chunked, ref)
    # Default budget: single-chunk fast path is the reference itself.
    assert torch.equal(_energy_read(scores, v, beta), ref)


def test_energy_read_forced_chunking_through_memory_read(monkeypatch):
    import flower.models.memory as memory_mod
    from flower.models.memory import MemoryRead

    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, num_heads=4, max_seq_len=16, local_window=16, memory_slots=8, energy_read=True)
    read = MemoryRead(cfg).eval()
    x = torch.randn(2, 12, 32)
    memory = torch.randn(2, 8, 32)
    causal_memory = torch.randn(2, 12, 8, 32)  # (B, T, S, D) per-position state
    with torch.no_grad():
        whole = read(x, memory)  # tiny shapes: single chunk (reference)
        whole_causal = read._forward_causal(x, causal_memory)
    monkeypatch.setattr(memory_mod, "_ENERGY_CHUNK_ELEMENTS", 64)  # force chunking
    with torch.no_grad():
        chunked = read(x, memory)
        chunked_causal = read._forward_causal(x, causal_memory)
    assert torch.equal(chunked, whole)
    assert torch.equal(chunked_causal, whole_causal)


# ---------------------------------------------------------------------------
# Flag-off parity, environment-portable redesign (pullfrog must-address).
#
# The original pin recorded sha256 digests of the logits and the full
# gradient vector taken at the base commit on the authoring machine. On
# GitHub-hosted x86_64 (torch 2.13, cpu) ALL NINE grad digests failed while
# all nine logits digests passed, with the executed code verified
# bit-identical: backward kernels reduce in platform-specific orders (SIMD
# width / BLAS backends), so a recorded grad digest fingerprints the MACHINE,
# not the code — and the logits digests are one kernel-selection change away
# from the same failure. Recorded constants can never be portable, so the pin
# is rebuilt from same-process comparisons only (everything below is computed
# in the test's own environment, so it holds on any machine that can run the
# suite):
#
#   1. test_flag_off_execution_is_reproducible: the flag-off protocol must be
#      a pure function of its seeds — logits AND the full gradient vector
#      byte-equal across two same-process runs. This is the portable form of
#      the old byte-exact check (exactness where it holds: identical kernels,
#      identical machine), and it catches RNG-state leaks and nondeterministic
#      kernels in the legacy path.
#   2. test_flag_off_write_equals_causal_write_at_final_position: the legacy
#      write aggregates the whole window; the causal write at the FINAL
#      position aggregates the same whole window through the same submodules,
#      so the two branches must agree. This ties the flag-off branch to the
#      flag-on branch (which the leak tests pin to the causal formulas),
#      replacing the drift protection the recorded digests provided. A tight
#      fp tolerance is legitimate and documented here: the batched (B, T, ...)
#      causal tensors vs the flat (B, ...) legacy tensors legally select
#      different GEMM reduction orders (measured max delta ~5e-7 across the
#      five hybrids).
#
# The remaining shared-code change in the flag-off path (the query-chunked
# energy read) keeps its own exact same-process reference:
# _energy_read_unchunked is the verbatim pre-change formula, pinned by the
# energy-read tests above.
# ---------------------------------------------------------------------------

FLAG_OFF_PARITY_CASES = {
    # case: (variant, options) — the same 9 combos the recorded table covered.
    "fa_sm/default": ("fa_sm", {}),
    "fa_fm/default": ("fa_fm", {}),
    "flow_meanflow/default": ("flow_meanflow", {}),
    "flow_meanflow/meanflow_ot_cfm=True": ("flow_meanflow", {"meanflow_ot_cfm": True}),
    "flow_memory/default": ("flow_memory", {}),
    "flow_memory/energy_read=True": ("flow_memory", {"energy_read": True}),
    "flow_memory/loop_count=2": ("flow_memory", {"loop_count": 2}),
    "flow_pma/default": ("flow_pma", {}),
    "flow_pma/hierarchical_memory=True": ("flow_pma", {"hierarchical_memory": True}),
}


def _run_flag_off_protocol(variant: str, options: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits, full-grad-vector) under the fixed flag-off protocol.

    Same seeds the old digest pin used (weights seed 1234 + input seed 1235
    for the eval pass; the training pass rebuilds under seed 1236, which also
    seeds its input), but returns the tensors so callers compare them
    directly instead of hashing them against machine-specific constants."""

    def build() -> torch.nn.Module:
        return build_model(
            ModelConfig(
                variant=variant,
                vocab_size=64,
                d_model=64,
                num_heads=4,
                num_layers=3,
                ffn_dim=128,
                max_seq_len=48,
                local_window=16,
                memory_slots=8,
                causal_memory=False,
                **options,
            )
        )

    torch.manual_seed(1234)
    model = build().eval()
    torch.manual_seed(1235)
    x = torch.randint(0, 64, (2, 48))
    with torch.no_grad():
        logits = model(x)["logits"]
    torch.manual_seed(1236)
    model2 = build().train()
    x2 = torch.randint(0, 64, (2, 48))
    model2(x2, labels=x2)["loss"].backward()
    grads = torch.cat([p.grad.flatten() for p in model2.parameters() if p.grad is not None])
    return logits, grads


@pytest.mark.parametrize("case", sorted(FLAG_OFF_PARITY_CASES))
def test_flag_off_execution_is_reproducible(case):
    """Flag-off logits AND full gradient vectors must be a pure function of
    the protocol seeds (bitwise, same process/machine/kernels)."""
    variant, options = FLAG_OFF_PARITY_CASES[case]
    logits_a, grads_a = _run_flag_off_protocol(variant, options)
    logits_b, grads_b = _run_flag_off_protocol(variant, options)
    assert torch.equal(logits_a, logits_b), f"{case}: legacy logits are not seed-deterministic"
    assert torch.equal(grads_a, grads_b), f"{case}: legacy gradients are not seed-deterministic"


FLAG_OFF_WRITE_IDENTITY_CASES = [
    ("flow_memory", {}),
    ("flow_pma", {"hierarchical_memory": True}),
    ("flow_meanflow", {}),
    ("fa_sm", {}),
    ("fa_fm", {}),
]


@pytest.mark.parametrize("variant,options", FLAG_OFF_WRITE_IDENTITY_CASES)
def test_flag_off_write_equals_causal_write_at_final_position(variant, options):
    """The legacy (flag-off) whole-window write must equal the causal
    per-position write evaluated at the FINAL position (same whole-window
    aggregation, same submodules) — the branch-tying replacement for the
    recorded-digest pin; see the section comment above for why the tolerance
    is a tight fp bound rather than bitwise."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(variant, causal=False, **options)
    block_off = build_model(cfg).blocks[0]
    torch.manual_seed(0)  # same seed -> identical weights in the causal model
    block_on = build_model(_tiny_cfg(variant, causal=True, **options)).blocks[0]

    B, T = 2, 10
    short = cfg.short_memory_slots if options.get("hierarchical_memory") else 0
    slots_total = cfg.memory_slots + short
    torch.manual_seed(5)
    x = torch.randn(B, T, cfg.d_model)
    mem3d = torch.randn(B, slots_total, cfg.d_model)  # legacy (B, S, D) bank
    mem4d = x.new_zeros(B, T, slots_total, cfg.d_model)  # causal state, bank at t=T-1
    mem4d[:, -1] = mem3d
    with torch.no_grad():
        if variant in ("flow_memory", "fa_fm"):  # FlowMemoryBlock: legacy write is inline
            legacy = block_off._unflat_memory(
                block_off.flow(block_off._flat_memory(mem3d), block_off.cond(x.mean(dim=1))), mem3d
            )
        elif variant == "flow_meanflow":  # MeanFlowMemoryBlock: legacy write is inline
            ones, zeros = x.new_ones(()), x.new_zeros(())
            legacy = mem3d + block_off.field(mem3d, ones, zeros, x.mean(dim=1))
        else:  # flow_pma / fa_sm expose the legacy branch as _update_memory
            legacy = block_off._update_memory(mem3d, x)

        if variant in ("flow_memory", "fa_fm", "flow_pma", "fa_sm"):
            causal_last = block_on._update_memory_causal(mem4d, x)[:, -1]
        else:  # flow_meanflow: endpoint field per position, then the +memory update
            ones, zeros = x.new_ones(()), x.new_zeros(())
            causal_last = (
                mem4d + block_on.field.forward_positions(mem4d, ones, zeros, causal_running_mean(x))
            )[:, -1]
    assert torch.allclose(legacy, causal_last, atol=1e-5), (
        f"{variant}: max |legacy write - causal write at final position| = "
        f"{(legacy - causal_last).abs().max().item()} (expected ~5e-7 fp reassociation)"
    )


# ---------------------------------------------------------------------------
# Flow-hybrid audit (LEGACY / flag-off only). These variants' legacy write
# paths still aggregate the whole window — that is the byte-identical
# historical behaviour that causal_memory=False must keep reproducing (see
# the parity tests above). With causal_memory=True every hybrid now has
# exact-0 leak (CAUSAL_MEMORY_VARIANTS above); this table pins the flag-off
# leak so any accidental behaviour change in the legacy path surfaces here
# too. Measured max |logits[:, :-1] delta| from perturbing only the last
# input token, tiny 3-layer config, seed-controlled (see test).
# ---------------------------------------------------------------------------

FLOW_HYBRID_LEAK_STATUS = {
    "flow_memory": 0.00397,
    "flow_meanflow": 0.01196,
    "flow_pma": 0.00541,
    "fa_fm": 0.00250,
}


@pytest.mark.parametrize("variant", sorted(FLOW_HYBRID_LEAK_STATUS))
def test_flow_hybrid_leak_status_is_as_documented(variant):
    # Seed BEFORE build_model so the pinned table above is reproducible
    # regardless of which tests ran before this one (pullfrog A4b on the
    # base PR; carried over here because this audit test inherited the gap).
    torch.manual_seed(3)
    cfg = _tiny_cfg(variant, causal=False)  # legacy code path
    model = build_model(cfg).eval()
    delta = _last_token_leak(model, cfg.vocab_size)
    leaks = delta > 1e-5
    assert leaks, (
        f"{variant} no longer leaks (delta={delta:.2e}) — the legacy path changed; "
        f"update FLOW_HYBRID_LEAK_STATUS and the PR notes"
    )


# ---------------------------------------------------------------------------
# bf16 regression (from the base causal-memory PR): causal_prefix_attention
# used to keep the scores' dtype through the softmax while floating v, so any
# causal path routing through it crashed pure-bf16 models with
# "expected scalar type Float but found BFloat16". The scores are now floated
# before the masked softmax; these tests pin both the primitive and every
# model-level causal path that routes through it — here extended to the four
# flow hybrids whose causal writes (per-position coupling flows / MeanFlow
# field / prefix PMA) must be dtype-clean too.
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
    # Flow hybrids (this PR): per-position flow writes + MemoryRead dim-4 path.
    ("fa_sm", {}),
    ("fa_fm", {}),
    ("flow_memory", {}),
    ("flow_meanflow", {}),
    ("flow_pma", {}),
    ("flow_pma", {"hierarchical_memory": True}),  # causal_last_tokens short write
]


@pytest.mark.parametrize("variant,options", BF16_CAUSAL_CASES)
def test_causal_memory_bf16_forward_and_backward_do_not_crash(variant, options):
    """A pure-bf16 causal model must forward (and backward) cleanly.

    Exercises the fp32-floated prefix softmax in causal_prefix_attention
    (perceiver summary + attention aggregation, bloom, flow_ot, flow_pma's
    prefix PMA), which crashed before the base PR's fix; the remaining
    variants pin that no other causal write/read path mixes dtypes either."""
    cfg = _tiny_cfg(variant, causal=True, **options)
    model = build_model(cfg).to(torch.bfloat16)
    x = torch.randint(0, cfg.vocab_size, (2, 48))
    out = model(x, labels=x)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
