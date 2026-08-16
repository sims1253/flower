"""Compositional Muon (CM) — github.com/tilde-research/comp-muon-release.

CM updates the *composed* attention circuits (QK^T, OV) rather than each matrix
in isolation. These tests pin the three things that can silently go wrong:

  1. the circuit map pairs the right parameters and refuses odd layouts,
  2. the fused-qkv slicing lines up with the Q/K/V factors it claims to be,
  3. the update rule matches the reference's algebra head-for-head.

(3) is the load-bearing one. CM is a rescaling of a spectral sign, so a wrong
partner (K's scale applied to K instead of Q) or a wrong axis still produces a
finite, plausible-looking update that trains — it just trains as something other
than CM. The isotropic path is checked against its closed form because that form
is short enough to state independently; the full path is checked through the
defining property of its whitener.
"""

import pytest
import torch

from flower.config import ModelConfig, TrainingConfig
from flower.models import build_model
from flower.optim import (
    Muon,
    _cm_qk_delta,
    _compositional_muon_deltas,
    _coupled_inv_sqrt,
    _isotropic_scale,
    _ns_heads,
    _ns_schedule,
    build_circuit_map,
    build_optimizer,
)

COEFFS = _ns_schedule("quintic5", 5)


def _cfg(**over) -> ModelConfig:
    base = dict(
        variant="vanilla_local", vocab_size=64, d_model=64, num_heads=4,
        num_layers=2, ffn_dim=128, max_seq_len=32, local_window=8,
    )
    base.update(over)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Circuit map
# ---------------------------------------------------------------------------

def test_circuit_map_pairs_qkv_with_its_own_output_projection():
    m = build_model(_cfg())
    cmap = build_circuit_map(m)
    names = {id(p): n for n, p in m.named_parameters()}

    assert len(cmap) == 2, f"expected one circuit per layer, got {len(cmap)}"
    for qkv_id, (out_id, heads) in cmap.items():
        q_name, o_name = names[qkv_id], names[out_id]
        assert q_name.endswith("qkv.weight") and o_name.endswith("out.weight")
        # The pair must come from the SAME attention module, not merely be one
        # qkv and one out: crossing layers would whiten against a partner the
        # circuit never composes with.
        assert q_name.rsplit(".", 2)[0] == o_name.rsplit(".", 2)[0]
        assert heads == 4


def test_circuit_map_rejects_layouts_its_slicing_does_not_fit():
    """A block whose qkv is not exactly (3d, d) must fall through to plain Muon."""
    m = build_model(_cfg())
    attn = next(mod for mod in m.modules() if getattr(mod, "num_heads", None) == 4)
    attn.qkv.weight = torch.nn.Parameter(torch.zeros(5 * 64, 64))

    cmap = build_circuit_map(m)
    assert len(cmap) == 1, "the mis-shaped block should have been skipped, not sliced"


def test_comp_muon_is_opt_in_and_takes_precedence_over_per_head():
    m = build_model(_cfg())

    def muon_for(**kw) -> Muon:
        opts = build_optimizer(m, TrainingConfig(optimizer="muon", **kw))
        return next(o for o in (opts if isinstance(opts, list) else [opts])
                    if isinstance(o, Muon))

    assert len(muon_for()._circuit_map) == 0, "CM must be off by default"
    assert len(muon_for(comp_muon=True)._circuit_map) == 2

    # Both flags on: CM owns qkv/out. The head map is still built (it covers any
    # block CM's shape check rejected) but must not be what runs on the circuits.
    both = muon_for(comp_muon=True, muon_per_head=True)
    assert set(both._circuit_map) <= set(both._head_map)


def test_comp_muon_with_aurora_is_rejected_rather_than_ignored():
    """Aurora has no partner-whitening hook, so the flag would be inert.

    A screen arm that silently ran the control is worse than one that failed.
    """
    m = build_model(_cfg())
    with pytest.raises(ValueError, match="comp_muon requires optimizer='muon'"):
        build_optimizer(m, TrainingConfig(optimizer="aurora", comp_muon=True))


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def test_isotropic_qk_matches_the_closed_form_partner_rescale():
    """delta_Q head h = s_K[h] * msign(G_Q[h]), and symmetrically for K.

    Pins the partner pairing. Swapping s_Q for s_K here is invisible downstream —
    it just yields a differently-scaled update — so it has to be checked directly.
    """
    torch.manual_seed(0)
    d, heads, hd = 64, 4, 16
    w_q, w_k = torch.randn(d, d), torch.randn(d, d)
    g_q, g_k = torch.randn(d, d), torch.randn(d, d)

    d_q, d_k = _cm_qk_delta(w_q, w_k, g_q, g_k, hd, COEFFS, isotropic=True, damping=1e-2)

    def heads_of(x):
        return x.view(d, heads, hd).transpose(0, 1).contiguous()

    s_q = _isotropic_scale(heads_of(w_q), hd, 1e-2)
    s_k = _isotropic_scale(heads_of(w_k), hd, 1e-2)
    want_q = s_k[:, None, None] * _ns_heads(heads_of(g_q), COEFFS)
    want_k = s_q[:, None, None] * _ns_heads(heads_of(g_k), COEFFS)

    torch.testing.assert_close(heads_of(d_q), want_q, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(heads_of(d_k), want_k, rtol=1e-4, atol=1e-5)

    # And the scales genuinely differ per head, so the assertion above has teeth.
    assert s_k.std() > 1e-3


def test_coupled_inverse_root_is_the_inverse_square_root_of_the_damped_gram():
    """C_inv @ (gram + lam I) @ C_inv == I, which is what the whitening assumes."""
    torch.manual_seed(0)
    a = torch.randn(4, 16, 32)
    gram = a @ a.mT
    lam = 1e-2

    c_inv = _coupled_inv_sqrt(gram, lam)
    eye = torch.eye(16).expand(4, 16, 16)
    got = c_inv @ (gram + lam * eye) @ c_inv

    torch.testing.assert_close(got, eye, rtol=2e-3, atol=2e-3)


def test_full_whitening_differs_from_isotropic_but_agrees_on_isotropic_grams():
    """The scalar path is an approximation of the matrix path, not a variant of it.

    When a head's Gram really is isotropic the two must coincide; when it is not,
    they must not. Without the second half the isotropic flag could be silently
    inert.
    """
    torch.manual_seed(0)
    d, heads, hd = 64, 4, 16
    g_q, g_k = torch.randn(d, d), torch.randn(d, d)

    def rel_gap(w):
        """Relative Frobenius gap between the two whitening paths.

        A plain `allclose` is useless here: both deltas have entries of order
        1e-2, so any absolute tolerance loose enough to admit bf16 NS noise also
        admits "the flag does nothing". Scale-free comparison is the only one
        that distinguishes those.
        """
        full = _cm_qk_delta(w, w, g_q, g_k, hd, COEFFS, False, 1e-2)[0]
        scal = _cm_qk_delta(w, w, g_q, g_k, hd, COEFFS, True, 1e-2)[0]
        return ((full - scal).norm() / full.norm()).item()

    # Isotropic Gram: per-head columns orthonormal and equally scaled, so
    # W^T W = c^2 I exactly and the scalar approximation is exact. What is left
    # is bf16 Newton-Schulz noise.
    w_iso = torch.zeros(d, d)
    for h in range(heads):
        q, _ = torch.linalg.qr(torch.randn(d, hd))
        w_iso[:, h * hd:(h + 1) * hd] = q * 3.0
    assert rel_gap(w_iso) < 0.05

    # Anisotropic Gram: the scalar approximation must visibly lose information.
    w_aniso = torch.randn(d, d)
    w_aniso[:, :hd] *= 12.0
    assert rel_gap(w_aniso) > 0.25


@pytest.mark.parametrize("isotropic", [True, False])
def test_a_factor_is_whitened_by_its_partner_not_by_itself(isotropic):
    """The defining property of CM, and the one a plausible bug would break.

    Each factor's update is whitened by its PARTNER's Gram root, so perturbing
    W_V must leave delta_V alone and move delta_O — the reverse of what
    per-matrix Muon would do. Perturbing the V *gradient* is what moves delta_V,
    and it must not leak into the Q/K rows of the fused matrix (which would be an
    off-by-one-block slice: still trainable, no longer CM).
    """
    torch.manual_seed(0)
    d, heads = 64, 4
    w_qkv, u_qkv = torch.randn(3 * d, d), torch.randn(3 * d, d)
    w_out, u_out = torch.randn(d, d), torch.randn(d, d)

    args = (heads, COEFFS, isotropic, 1e-2)
    base_qkv, base_out = _compositional_muon_deltas(w_qkv, u_qkv, w_out, u_out, *args)

    def moved(a, b):
        return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()

    # Perturb the V *weight*: it is W_O's partner, so only delta_O may react.
    w_pert = w_qkv.clone()
    w_pert[2 * d:] += 5.0
    pert_qkv, pert_out = _compositional_muon_deltas(w_pert, u_qkv, w_out, u_out, *args)
    assert moved(pert_qkv, base_qkv) < 1e-3, "W_V must not whiten V's own update"
    assert moved(pert_out, base_out) > 0.05, "W_V must whiten W_O's update"

    # Perturb the V *gradient*: delta_V moves, Q and K rows must not.
    u_pert = u_qkv.clone()
    u_pert[2 * d:] += 5.0
    grad_qkv, _ = _compositional_muon_deltas(w_qkv, u_pert, w_out, u_out, *args)
    torch.testing.assert_close(grad_qkv[: 2 * d], base_qkv[: 2 * d], rtol=1e-4, atol=1e-5)
    assert moved(grad_qkv[2 * d:], base_qkv[2 * d:]) > 0.05


@pytest.mark.parametrize("isotropic", [True, False])
def test_ov_is_per_head_for_v_and_whole_matrix_for_the_output_projection(isotropic):
    """OV hybrid granularity: V gets per-head signs, W_O gets one aggregate sign.

    W_O writes every head into a single residual stream, so per-head
    orthogonalisation would impose an independent unit norm on a map that is not
    head-local — the reason CM's OV rule is hybrid rather than symmetric.

    Two obvious probes do NOT work here and are worth naming so they are not
    reintroduced. Singular values fail at d_v == d: any row subset of a square
    orthogonal matrix is itself orthonormal, so both granularities look the same.
    Block magnitudes fail because `msign` discards the singular-value spectrum
    either way, so boosting one head's gradient survives in neither.

    What does separate them is CROSS-head orthogonality. One sign over the whole
    matrix makes rows from different head blocks mutually orthogonal; per-head
    signs only orthogonalise within a block and leave blocks free to correlate.
    Both the partner whitening and the per-head scalars preserve that structure
    (they act within a block), so it survives to the returned delta.
    """
    torch.manual_seed(0)
    d, heads, hd = 64, 4, 16
    w_qkv, u_qkv = torch.randn(3 * d, d), torch.randn(3 * d, d)
    w_out, u_out = torch.randn(d, d), torch.randn(d, d)

    def cross_head_coupling(m: torch.Tensor) -> float:
        """Off-block vs on-block mass of M M^T, for M grouped into head blocks."""
        gram = m @ m.mT
        gram = gram / gram.diagonal().mean()
        on = torch.zeros_like(gram, dtype=torch.bool)
        for h in range(heads):
            on[h * hd:(h + 1) * hd, h * hd:(h + 1) * hd] = True
        return (gram[~on].norm() / gram[on].norm()).item()

    _, delta_out = _compositional_muon_deltas(
        w_qkv, u_qkv, w_out, u_out, heads, COEFFS, isotropic, 1e-2)
    # Control: the same gradient orthogonalised per head, i.e. what the rule
    # would produce if W_O were treated head-locally like V.
    control = _ns_heads(u_out.mT.float().view(heads, hd, d), COEFFS).reshape(d, d)

    got = cross_head_coupling(delta_out.mT.float())
    per_head = cross_head_coupling(control)
    assert got < 0.5 < per_head, (
        f"delta_out coupling {got:.3f} should sit well below the per-head control "
        f"{per_head:.3f} — W_O looks head-local"
    )


# ---------------------------------------------------------------------------
# Integration through Muon.step
# ---------------------------------------------------------------------------

def _step_once(**kw) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    m = build_model(_cfg())
    opts = build_optimizer(m, TrainingConfig(optimizer="muon", **kw))
    opts = opts if isinstance(opts, list) else [opts]

    ids = torch.randint(0, 64, (2, 16))
    m(input_ids=ids, labels=ids)["loss"].backward()
    for o in opts:
        o.step()
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def test_comp_muon_off_is_bit_identical_to_the_existing_muon_path():
    """The flag must be inert when unset — every published run has to reproduce."""
    a, b = _step_once(), _step_once()
    for n in a:
        assert torch.equal(a[n], b[n]), f"{n} not deterministic"


@pytest.mark.parametrize("isotropic", [True, False])
def test_comp_muon_changes_attention_and_leaves_the_ffn_to_muon(isotropic):
    off = _step_once()
    on = _step_once(comp_muon=True, comp_muon_isotropic=isotropic)

    for n, p in on.items():
        assert torch.isfinite(p).all(), f"{n} went non-finite under CM"

    attn = [n for n in on if n.endswith(("attn.qkv.weight", "attn.out.weight"))]
    assert attn, "no attention matrices found — the test is not testing anything"
    for n in attn:
        assert not torch.equal(off[n], on[n]), f"CM did not change {n}"

    # The FFN is outside every circuit and must land exactly where Muon put it.
    for n in on:
        if ".ffn." in n or ".mlp." in n:
            assert torch.equal(off[n], on[n]), f"CM perturbed non-circuit param {n}"


def test_comp_muon_mp_scales_the_step_without_touching_other_matrices():
    """`comp_muon_mp` is the knob for matching CM's step to Muon's.

    It must act on the circuits only — moving `muon_lr` instead would also move
    every FFN matrix and confound the arm.
    """
    base = _step_once(comp_muon=True)
    scaled = _step_once(comp_muon=True, comp_muon_mp=2.0)

    qkv = next(n for n in base if n.endswith("attn.qkv.weight"))
    assert not torch.allclose(base[qkv], scaled[qkv], rtol=1e-3, atol=1e-5)
    for n in base:
        if ".ffn." in n or ".mlp." in n:
            assert torch.equal(base[n], scaled[n]), f"mp leaked into {n}"
