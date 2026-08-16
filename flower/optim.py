"""Optimizer factory for Flower.

`build_optimizer` returns either:
  - vanilla AdamW over all parameters, or
  - Muon for 2D hidden weights + AdamW for 1D/embedding/head/scalar params.

Muon (Keller Jordan, 2024) = SGD-momentum + Newton-Schulz orthogonalisation of the
update matrix. Validated across the NanoGPT speedrun community at the 30M-200M scale
that Flower targets. Should NOT be applied to:
  - 1D parameters (biases, LayerNorm gains)
  - Embeddings and lm_head (tied here, single Embedding)
  - Anything not a 2D matrix

Reference: https://github.com/KellerJordan/Muon
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from flower.config import TrainingConfig

# Newton-Schulz polynomial coefficients for Muon orthogonalisation.
# A step applies x <- a*x + b*(x@x.T)@x + c*(x@x.T)^2@x.
# quintic: standard speedrun schedule (15 matmuls for 5 steps).
# cubic: cheaper 10-matmul schedule; ~1e-3 val-loss difference vs quintic
#        (arXiv:2606.00371 "How Much Orthogonalization Does Muon Need?").
# stabilize: DeepSeek-V4's final-step schedule that pins singular values at 1
#        (arXiv:2606.19348). Used after 8 quintic steps in hybrid_v4.
_NS_QUINTIC = (3.4445, -4.7750, 2.0315)
_NS_CUBIC = (1.5, -0.5, 0.0)
_NS_STABILIZE = (2.0, -1.5, 0.5)


def _ns_schedule(name: str, steps: int) -> list[tuple[float, float, float]]:
    """Expand a schedule name into a list of (a, b, c) coefficient triples.

    Supported names:
      quintic5   -> steps x quintic coefficients
      cubic5     -> steps x cubic coefficients
      hybrid_v4  -> 8 x quintic + 2 x stabilize (DeepSeek-V4)
    Unknown names raise ValueError (strict, to catch typos in configs).
    """
    if name == "quintic5":
        return [_NS_QUINTIC] * max(1, steps)
    if name == "cubic5":
        return [_NS_CUBIC] * max(1, steps)
    if name == "hybrid_v4":
        return [_NS_QUINTIC] * 8 + [_NS_STABILIZE] * 2
    raise ValueError(f"Unknown muon_ns_schedule: {name!r} (want quintic5|cubic5|hybrid_v4)")


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int, schedule: str = "quintic5") -> torch.Tensor:
    """Newton-Schulz iteration for the matrix polar decomposition (orthogonalisation).

    Approximates U @ V^T where g = U S V^T. Default schedule is the quintic
    (3.4445, -4.7750, 2.0315) used by the speedrun community; see `_ns_schedule`
    for alternatives. Operates in bf16 for speed; the polynomial is numerically
    robust there.

    This is the per-matrix (legacy) path. The batched path
    `_zeropower_newtonschulz5_batched` is mathematically identical — a `bmm` over a
    stack of same-shape matrices reduces slice-for-slice to looping this function
    — but collapses N `mm` launches into one, which is the dominant win when the
    optimizer step is launch-bound (see NEXT_IDEAS.md section 5).
    """
    assert g.ndim == 2
    coeffs = _ns_schedule(schedule, steps)
    x = g.to(torch.bfloat16)
    # Normalise spectral norm to <= 1 to keep the polynomial in its convergence basin.
    x = x / (x.norm() + 1e-7)
    # Operate on the smaller side for compute efficiency (transpose if rows > cols).
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for a, b, c in coeffs:
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


def _zeropower_newtonschulz5_batched(
    stack: torch.Tensor, coeffs: list[tuple[float, float, float]]
) -> torch.Tensor:
    """Batched Newton-Schulz over a stack of same-shape 2D matrices.

    `stack` is (B, m, n); every slice gets the same `coeffs` schedule applied via
    `bmm`, so this is exactly `_zeropower_via_newtonschulz5` run on every slice at
    once — the only numerical difference vs the per-matrix path is bf16 kernel
    selection noise, not a different algorithm. Per-slice Frobenius normalisation
    (`x / x.norm()` for each slice independently) is what makes batching safe: the
    legacy normaliser is per-matrix, so a grouped norm (one scalar per slice) is
    the correct generalisation, not an approximation.

    The transpose-when-rows>cols step is the caller's responsibility — callers
    build `stack` already oriented (smaller side on the left), so every slice in a
    stack shares one orientation and one set of matmul shapes.
    """
    assert stack.ndim == 3
    x = stack.to(torch.bfloat16)
    # Per-slice Frobenius norm: one scalar per matrix, keeping the leading batch dim.
    x = x / (x.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-7)
    for a, b, c in coeffs:
        # x @ x.T over the batch: (B,m,n) @ (B,n,m) -> (B,m,m). Identical reductions
        # to per-matrix `x @ x.T`; bmm just fuses the B dispatches into one launch.
        a_mat = torch.bmm(x, x.transpose(-2, -1))
        b_mat = b * a_mat + c * torch.bmm(a_mat, a_mat)
        x = a * x + torch.bmm(b_mat, x)
    return x


# ---------------------------------------------------------------------------
# Compositional Muon (CM) — github.com/tilde-research/comp-muon-release (Apache 2.0)
# ---------------------------------------------------------------------------
#
# Muon controls the operator norm of each weight update in isolation. The loss
# does not see W_Q and W_K, it sees the QK product W_Q W_K^T (and the OV product
# W_O W_V). CM controls the operator norm of the *composed* update by whitening
# each factor's gradient with its partner's inverse Gram root before the spectral
# sign and scaling by it again afterward:
#
#     Delta_Q = -(eta/2) msign(G_Q C_K^{-1}) C_K^{-1},  C_K = (W_K^T W_K + lam I)^{1/2}
#
# This is the principled version of the per-head arm that won the 1500-step Muon
# screen (`muon_per_head`, -0.027 val_bpb): CM's QK rule is head-local by
# construction, and additionally supplies the partner whitening that plain
# per-head splitting lacks.
#
# WHAT IS VENDORED AND WHAT IS NOT. The reference exposes a large variant space
# (joint vs half-split budget, gauge/connection fixes via a Sylvester solve,
# momentum reprojection onto the horizontal bundle, leg-norm-restore). Only the
# two configurations the CM writeup actually recommends are implemented here:
# `half_split` budget with `connection="none"`, `momentum_reproject=False`,
# `per_mat_renorm=False`, `whitening="both"`, and OV `hybrid=True` (V per-head
# sign, W_O one aggregating full-matrix sign). `comp_muon_isotropic` selects
# between the scalar Gram-root approximation and the full coupled-Newton-Schulz
# inverse root. The unimplemented knobs are ablations, not the method.
#
# DELIBERATE DEVIATION FROM THE REFERENCE: the reference's `msign` runs 8 steps
# of the `polar_express` polynomial. This uses Flower's existing Newton-Schulz
# (`muon_ns_schedule`, default 5x quintic) so that a CM arm differs from the
# `muon_baseline` arm in the CM rule *only* and not also in the orthogonaliser.
# It also reuses `_zeropower_newtonschulz5_batched`, whose per-slice Frobenius
# normalisation is what makes the per-head stacks safe to batch.

# CANS coupled-iteration schedule (arXiv:2506.10935): 9 tuned (a, b) pairs then
# classic (1.5, -0.5) padding. Drives Y -> W^{1/2}, Z -> W^{-1/2} without an eigh.
_CANS_COEFFS: list[tuple[float, float]] = [
    (5.182503604966906, -5.178098480082684),
    (2.586120737395915, -0.6479542005271643),
    (2.567364126726186, -0.6454968804392178),
    (2.520560084348265, -0.6393528082067044),
    (2.410759275435182, -0.6248683598710716),
    (2.1883348130094173, -0.5952022073798908),
    (1.8595760874873613, -0.5504490972723968),
    (1.589020160467417, -0.5126569802066718),
    (1.5051653981684994, -0.5007377068751799),
] + [(1.5, -0.5)] * 16


def _coupled_inv_sqrt(gram: torch.Tensor, damping: float) -> torch.Tensor:
    """`(gram + damping I)^{-1/2}` via the CANS coupled Newton-Schulz iteration.

    `gram` is (H, n, n) — one damped Gram per attention head. Both iterates are
    symmetrised each step to stay on the SPD manifold. fp32 throughout: this is a
    matrix inverse root on 64x64 blocks, where bf16 would cost real accuracy and
    saves nothing (the cost is 25 tiny bmms, not bandwidth).
    """
    n = gram.shape[-1]
    eye = torch.eye(n, device=gram.device, dtype=torch.float32)
    w = gram.float() + damping * eye
    scale = (1 - 1e-3) / w.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    y = w * scale
    z = eye.expand_as(w).contiguous()
    for a, b in _CANS_COEFFS:
        m = z @ y
        y_next = torch.baddbmm(y, y, m, beta=a, alpha=b)
        z_next = torch.baddbmm(z, m, z, beta=a, alpha=b)
        y = 0.5 * (y_next + y_next.mT)
        z = 0.5 * (z_next + z_next.mT)
    return z * torch.sqrt(scale)


def _isotropic_scale(w_h: torch.Tensor, head_dim: int, damping: float) -> torch.Tensor:
    """Isotropic approximation `C^{-1} ~ s I`, one scalar per head.

    `s = (||W_h||_F^2 / head_dim + lam)^{-1/2}`. This is the cheap CM path: it
    replaces the (head_dim, head_dim) inverse Gram root with a scalar, which
    reduces the whole method to a partner-rescaled per-head Muon at essentially
    zero cost over `muon_per_head`.
    """
    sq = w_h.float().pow(2).sum(dim=(-2, -1))
    return (sq / head_dim + damping).clamp_min(1e-12).rsqrt()


def _to_heads_row(w: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """`(d, H*hd) -> (H, d, hd)`: split heads along the column axis."""
    return w.view(w.shape[0], heads, head_dim).transpose(0, 1).contiguous()


def _ns_heads(x: torch.Tensor, coeffs: list[tuple[float, float, float]]) -> torch.Tensor:
    """Batched spectral sign over a per-head stack, orienting for the bmm path.

    `_zeropower_newtonschulz5_batched` wants the smaller side on the left (it
    forms `x @ x.T`), and every slice of a per-head stack shares a shape, so one
    orientation decision covers the batch — same contract as
    `_per_head_orthogonalise`.
    """
    flip = x.size(-2) > x.size(-1)
    if flip:
        x = x.transpose(-2, -1).contiguous()
    out = _zeropower_newtonschulz5_batched(x, coeffs).float()
    return out.transpose(-2, -1).contiguous() if flip else out


def _cm_qk_delta(
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    g_q: torch.Tensor,
    g_k: torch.Tensor,
    head_dim: int,
    coeffs: list[tuple[float, float, float]],
    isotropic: bool,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CM half-split update directions for the QK circuit, math convention.

    Inputs are (d_model, d_q) / (d_model, d_k) — i.e. `nn.Linear.weight.mT` — and
    the returned deltas match. Both legs are orthogonalised per head, which is
    what makes QK head-local: the product W_Q W_K^T only ever couples a head's Q
    with the same head's K.
    """
    d, d_q = w_q.shape
    heads = d_q // head_dim
    w_q_h, w_k_h = _to_heads_row(w_q, heads, head_dim), _to_heads_row(w_k, heads, head_dim)
    g_q_h, g_k_h = _to_heads_row(g_q, heads, head_dim), _to_heads_row(g_k, heads, head_dim)

    if isotropic:
        s_q = _isotropic_scale(w_q_h, head_dim, damping)[:, None, None]
        s_k = _isotropic_scale(w_k_h, head_dim, damping)[:, None, None]
        delta_q_h = s_k * _ns_heads(g_q_h, coeffs)
        delta_k_h = s_q * _ns_heads(g_k_h, coeffs)
    else:
        c_q_inv = _coupled_inv_sqrt(w_q_h.mT @ w_q_h, damping)
        c_k_inv = _coupled_inv_sqrt(w_k_h.mT @ w_k_h, damping)
        delta_q_h = _ns_heads(g_q_h @ c_k_inv, coeffs) @ c_k_inv
        delta_k_h = _ns_heads(g_k_h @ c_q_inv, coeffs) @ c_q_inv

    return (
        delta_q_h.transpose(0, 1).reshape(d, d_q),
        delta_k_h.transpose(0, 1).reshape(d, d_q),
    )


def _cm_ov_delta(
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    g_v: torch.Tensor,
    g_o: torch.Tensor,
    head_dim: int,
    coeffs: list[tuple[float, float, float]],
    isotropic: bool,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CM half-split update directions for the OV circuit, math convention (hybrid).

    `w_v` is (d_model, d_v), `w_o` is (d_v, d_model) — both `nn.Linear.weight.mT`.

    Hybrid granularity, which is the configuration the CM writeup recommends for
    OV: V gets a per-head spectral sign (each head's values are written into its
    own subspace) but W_O gets ONE full-matrix sign, because W_O aggregates all
    heads into a single residual write and splitting it would impose an
    independent unit norm per head on a map that is not head-local.
    """
    d, d_v = w_v.shape
    heads = d_v // head_dim
    w_v_h, g_v_h = _to_heads_row(w_v, heads, head_dim), _to_heads_row(g_v, heads, head_dim)
    w_o_h, g_o_h = w_o.view(heads, head_dim, d), g_o.view(heads, head_dim, d)

    if isotropic:
        s_v = _isotropic_scale(w_v_h, head_dim, damping)[:, None, None]
        s_o = _isotropic_scale(w_o_h, head_dim, damping)[:, None, None]
        delta_v_h = s_o * _ns_heads(g_v_h, coeffs)
        # Pre-whiten per head, then one sign over the reassembled full matrix.
        m_o = _zeropower_via_newtonschulz5(
            (s_v * g_o_h).reshape(d_v, d), len(coeffs), "quintic5"
        ).float().view(heads, head_dim, d)
        delta_o_h = s_v * m_o
    else:
        c_v_inv = _coupled_inv_sqrt(w_v_h.mT @ w_v_h, damping)
        c_o_inv = _coupled_inv_sqrt(w_o_h @ w_o_h.mT, damping)
        delta_v_h = _ns_heads(g_v_h @ c_o_inv, coeffs) @ c_o_inv
        m_o = _zeropower_via_newtonschulz5(
            (c_v_inv @ g_o_h).reshape(d_v, d), len(coeffs), "quintic5"
        ).float().view(heads, head_dim, d)
        delta_o_h = c_v_inv @ m_o

    return delta_v_h.transpose(0, 1).reshape(d, d_v), delta_o_h.reshape(d_v, d)


def _compositional_muon_deltas(
    w_qkv: torch.Tensor,
    u_qkv: torch.Tensor,
    w_out: torch.Tensor,
    u_out: torch.Tensor,
    heads: int,
    coeffs: list[tuple[float, float, float]],
    isotropic: bool,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CM update directions for one attention block's fused qkv + out matrices.

    Flower fuses Q/K/V into a single (3*d_model, d_model) `nn.Linear`, so the
    circuit factors are row slices of one parameter: rows [0:d) are Q, [d:2d) are
    K, [2d:3d) are V. The gradient of a fused matrix is exactly the concatenation
    of the factor gradients, so slicing the momentum direction is equivalent to
    having kept three separate projections.

    Returns `(delta_qkv, delta_out)` in `nn.Linear` (out, in) convention, ready to
    be applied by the caller with its own learning rate.
    """
    d = w_out.shape[0]
    # (out, in) -> math (in, out); fp32 for the Gram roots.
    w_q, w_k, w_v = (w_qkv[i * d : (i + 1) * d].mT.float() for i in range(3))
    u_q, u_k, u_v = (u_qkv[i * d : (i + 1) * d].mT.float() for i in range(3))
    head_dim = d // heads

    d_q, d_k = _cm_qk_delta(w_q, w_k, u_q, u_k, head_dim, coeffs, isotropic, damping)
    d_v, d_o = _cm_ov_delta(
        w_v, w_out.mT.float(), u_v, u_out.mT.float(), head_dim, coeffs, isotropic, damping
    )

    delta_qkv = torch.cat([d_q.mT, d_k.mT, d_v.mT], dim=0).to(w_qkv.dtype)
    return delta_qkv, d_o.mT.to(w_out.dtype)


def _per_head_orthogonalise(
    u: torch.Tensor, spec: tuple[int, int], coeffs: list[tuple[float, float, float]]
) -> torch.Tensor:
    """Orthogonalise `u` one head-block at a time (per-head / Group Muon).

    `spec` is (axis, n_blocks) from `build_head_block_map`. The blocks of a
    single attention matrix all share a shape, so they form a natural batch —
    one `bmm` chain, not `n_blocks` separate launches.

    axis=0 splits ROWS (fused QKV: each head owns `head_dim` output rows);
    axis=1 splits COLUMNS (output projection: each head owns `head_dim` inputs).
    Falls back to whole-matrix NS when the shape does not divide evenly, so a
    variant with an unusual attention layout degrades to standard Muon rather
    than silently orthogonalising mis-aligned blocks.
    """
    axis, n_blocks = spec
    rows, cols = u.shape
    if n_blocks <= 1 or (axis == 0 and rows % n_blocks) or (axis == 1 and cols % n_blocks):
        return _zeropower_via_newtonschulz5(u, len(coeffs), "quintic5")

    if axis == 0:
        blocks = u.view(n_blocks, rows // n_blocks, cols)
    else:
        # (rows, cols) -> per-head column slabs -> (n_blocks, cols/n, rows)
        blocks = u.t().contiguous().view(n_blocks, cols // n_blocks, rows)

    # The batched kernel wants the smaller side on the left, as in the shape-group
    # path; every block shares one orientation so one decision covers the stack.
    flip = blocks.size(1) > blocks.size(2)
    if flip:
        blocks = blocks.transpose(1, 2).contiguous()
    out = _zeropower_newtonschulz5_batched(blocks, coeffs).to(u.dtype)
    if flip:
        out = out.transpose(1, 2)

    if axis == 0:
        return out.reshape(rows, cols)
    return out.reshape(cols, rows).t().contiguous()


class Muon(Optimizer):
    """Muon optimizer for 2D weight matrices.

    Update rule per parameter:
        m_t = mu * m_{t-1} + g_t           (Nesterov momentum)
        g_t' = g_t + mu * m_t              (look-ahead)
        u_t = NewtonSchulz(g_t')           (orthogonalised update direction)
        p_t = p_{t-1} - lr * u_t * scale   (scale = max(1, sqrt(d_out / d_in)))

    The scale factor compensates for the fact that orthogonal matrices have unit
    spectral norm regardless of shape, so wider matrices need a bigger step.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_schedule: str = "quintic5",
        norm_update: bool = False,
        cautious_wd: float = 0.0,
        contra_muon: float = 0.0,
        ns_batched: bool = True,
        head_map: dict[int, tuple[int, int]] | None = None,
        circuit_map: dict[int, tuple[int, int]] | None = None,
        cm_isotropic: bool = False,
        cm_damping: float = 1e-2,
        cm_mp: float = 1.0,
    ) -> None:
        # Not a param-group default: they are keyed by id(param) and shared across
        # groups, and putting a dict in `defaults` would have it deep-copied per
        # group and break the identity lookup.
        self._head_map = head_map or {}
        self._circuit_map = circuit_map or {}
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_schedule=ns_schedule,
            norm_update=norm_update,
            cautious_wd=cautious_wd,
            contra_muon=contra_muon,
            ns_batched=ns_batched,
            cm_isotropic=cm_isotropic,
            cm_damping=cm_damping,
            cm_mp=cm_mp,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            ns_schedule = group["ns_schedule"]
            norm_update = group["norm_update"]
            cautious_wd = group["cautious_wd"]
            contra_muon = group["contra_muon"]
            ns_batched = group["ns_batched"]
            head_map = self._head_map
            circuit_map = self._circuit_map

            # Pass 1: momentum buffer update + build the per-param update-direction
            # buffers. Same as the legacy loop up to the NS call; done per-param
            # because the momentum buffer is persistent optimizer state.
            active: list[tuple] = []  # (p, g, update_dir) per active param
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(
                        f"Muon expects 2D parameters, got shape {tuple(g.shape)}. "
                        "Route 1D/embedding/head params to AdamW instead."
                    )
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                update_dir = g.add(buf, alpha=mu) if nesterov else buf
                active.append((p, g, update_dir))

            orthos: dict[int, torch.Tensor] = {}
            # Pass 1b: Compositional Muon. Consumes attention circuits (fused qkv
            # + out) in PAIRS, so it runs before the shape grouping and removes
            # the params it handled from everything downstream.
            #
            # `cm_lr_mult` records the per-param learning-rate multiplier CM
            # prescribes, which is NOT Muon's. Muon scales by
            # max(1, sqrt(fan_out/fan_in)) because an orthogonal update has unit
            # spectral norm regardless of shape; CM applies no shape scale at all
            # (its whitening already sets the scale) and instead splits a budget
            # of 0.5 across the two legs of each half-split pair.
            cm_lr_mult: dict[int, float] = {}
            if circuit_map:
                coeffs = _ns_schedule(ns_schedule, ns_steps)
                index_of = {id(p): i for i, (p, _g, _u) in enumerate(active)}
                budget = 0.5 * group["cm_mp"]
                for qkv_id, (out_id, heads) in circuit_map.items():
                    i_qkv, i_out = index_of.get(qkv_id), index_of.get(out_id)
                    # Both legs must be live in THIS group. A circuit split across
                    # param groups (or with one leg's grad None) falls through to
                    # standard Muon rather than being half-updated.
                    if i_qkv is None or i_out is None:
                        continue
                    d_qkv, d_out = _compositional_muon_deltas(
                        active[i_qkv][0].data,
                        active[i_qkv][2],
                        active[i_out][0].data,
                        active[i_out][2],
                        heads,
                        coeffs,
                        group["cm_isotropic"],
                        group["cm_damping"],
                    )
                    orthos[i_qkv], orthos[i_out] = d_qkv, d_out
                    cm_lr_mult[i_qkv] = cm_lr_mult[i_out] = budget

            # Pass 2: orthogonalise. The batched path groups same-shape update_dirs
            # into one bmm per NS line (NEXT_IDEAS.md section 5); the per-matrix path
            # is the exact legacy behaviour and the reproducibility fallback.
            if ns_batched and len(active) > 1:
                coeffs = _ns_schedule(ns_schedule, ns_steps)
                # Group active params by oriented shape. Orientation puts the smaller
                # side on the left so every slice in a stack shares one matmul shape;
                # the per-shape transpose flag is recorded to undo it on the way out.
                groups: dict[tuple[int, int, bool], list[int]] = {}
                for i, (p_i, _g, u) in enumerate(active):
                    if i in orthos:
                        continue  # already handled by Compositional Muon
                    # Per-head Muon: attention matrices are orthogonalised one
                    # head-block at a time rather than as one fused matrix, so
                    # every head gets its own unit-spectral-norm update instead
                    # of sharing one across all heads. Handled here rather than
                    # in the shape grouping because the blocks of ONE matrix form
                    # their own natural batch — the split is already a stack.
                    spec = head_map.get(id(p_i)) if head_map else None
                    if spec is not None:
                        orthos[i] = _per_head_orthogonalise(u, spec, coeffs)
                        continue
                    transposed = u.size(0) > u.size(1)
                    key = (max(u.size(0), u.size(1)), min(u.size(0), u.size(1)), transposed)
                    groups.setdefault(key, []).append(i)
                for (_rows, _cols, transposed), idxs in groups.items():
                    if len(idxs) == 1:
                        # A lone shape has nothing to batch; run the exact legacy path
                        # so single-param groups aren't exposed to bf16 stack noise.
                        i = idxs[0]
                        orthos[i] = _zeropower_via_newtonschulz5(
                            active[i][2], ns_steps, ns_schedule
                        )
                        continue
                    # Orient every slice the same way, stack, batch-NS, scatter back.
                    slices = [active[i][2] for i in idxs]
                    if transposed:
                        slices = [s.T.contiguous() for s in slices]
                    stack = torch.stack(slices, dim=0)
                    out = _zeropower_newtonschulz5_batched(stack, coeffs)
                    out = out.to(active[idxs[0]][2].dtype)
                    for j, i in enumerate(idxs):
                        orthos[i] = out[j].T.contiguous() if transposed else out[j]
            else:
                for i, (_p, _g, u) in enumerate(active):
                    if i in orthos:
                        continue  # already handled by Compositional Muon
                    orthos[i] = _zeropower_via_newtonschulz5(u, ns_steps, ns_schedule)

            # Pass 3: NorMuon + spectral scaling + cautious WD + the param update.
            # Unchanged from the legacy loop; `ortho` is whatever Pass 2 produced.
            for i, (p, g, _u) in enumerate(active):
                ortho = orthos[i]
                if i in cm_lr_mult:
                    # Compositional Muon supplies a finished update direction and
                    # its own scale. Contra-Muon and NorMuon are alternative
                    # post-processings of a *Muon* spectral sign — CM's whitening
                    # has already set the singular-value profile these would
                    # rewrite, and composing them is not a rule either method
                    # defines. Cautious WD still applies: it is a decay rule keyed
                    # on the sign of the update, independent of how it was formed.
                    if cautious_wd > 0:
                        p.data -= cautious_wd * lr * (ortho * p.data > 0).to(ortho.dtype) * p.data
                    p.add_(ortho, alpha=-lr * cm_lr_mult[i])
                    continue
                # Contra-Muon (github.com/nilin/contra-muon). Newton-Schulz maps
                # every singular value of the update to 1, discarding the
                # gradient's own spectral profile. Contra-Muon subtracts a
                # normalised copy of the pre-orthogonalisation update, whose
                # singular values are sigma_i/sigma_max in (0, 1]. Large-sigma
                # directions are therefore reduced MORE than small-sigma ones,
                # which relatively boosts the small directions and increases
                # singular-value diversity. Multiple nanoGPT Track-3 records
                # (PR #275, PR #301, the latter stacked with NorMuon).
                #
                # NORMALISATION CHOICE: the reference describes an
                # "operator-normalised" update, i.e. divide by the spectral norm.
                # Computing that exactly needs an SVD or power iteration per
                # matrix per step, which would dominate an optimizer that is only
                # ~2.8% of step time. Frobenius is used instead — it is what
                # `_zeropower_via_newtonschulz5` already uses to pull the matrix
                # into the NS convergence basin, and since ||X||_op <= ||X||_F
                # the subtraction is conservative (never over-corrects). The
                # coefficient absorbs the constant factor.
                #
                # NOTE: docs/research/frontier-speedups-2026-08.md gives the
                # one-liner `ortho - 0.2*(ortho/ortho.norm())*momentum.norm()`.
                # That subtracts a multiple of `ortho` itself, which merely
                # RESCALES it and cannot change the singular-value profile at
                # all, so it cannot do what the prose describes. Implemented
                # from the description instead.
                if contra_muon > 0:
                    u = _u
                    ortho = ortho - contra_muon * (u / (u.norm() + 1e-7))
                # NorMuon (arXiv:2510.05491): equalise the orthogonalised update's
                # PER-ROW scales — the Adam-like part of the method. Rows of a
                # Newton-Schulz output are not equally scaled, especially for
                # rectangular matrices, and NorMuon divides each row by its RMS.
                #
                # THIS WAS PREVIOUSLY `ortho / ortho.norm()` — a single GLOBAL
                # Frobenius normalisation, which is a different operation and a
                # damaging one. An orthogonalised (m, n) matrix has
                # ||ortho||_F ~= sqrt(min(m, n)), so dividing by it shrank every
                # update by ~36x at d_model=1280, i.e. a silent 36x cut to the
                # Muon learning rate. Measured in the 1500-step Muon screen:
                # +0.517 val_bpb against baseline (train loss 4.64 vs 3.16) —
                # catastrophic, and it had been sitting in the codebase as a
                # switched-off "implemented" feature.
                #
                # The rescale on the last line keeps ||update||_F unchanged, so
                # this is a pure REDISTRIBUTION of scale across rows rather than
                # a change to the step size. That matters for the A/B: without
                # it, `norm_update` would silently retune the LR and the screen
                # would measure the LR change, not the method.
                if norm_update:
                    row_rms = ortho.pow(2).mean(dim=1, keepdim=True).sqrt()
                    normed = ortho / (row_rms + 1e-7)
                    ortho = normed * (ortho.norm() / (normed.norm() + 1e-7))
                # Scale step by max(1, sqrt(fan_out/fan_in)) so wide layers still move.
                scale = max(1.0, (g.size(0) / g.size(1)) ** 0.5)
                # Cautious Weight Decay: only decay where the update is already
                # shrinking the weight (ortho . p > 0). `ortho` is the update
                # direction before the -lr*scale scaling.
                if cautious_wd > 0:
                    cautious_mask = (ortho * p.data > 0).to(ortho.dtype)
                    p.data -= cautious_wd * lr * cautious_mask * p.data
                p.add_(ortho, alpha=-lr * scale)
        return loss


class CautiousAdamW(torch.optim.AdamW):
    """AdamW with Cautious Weight Decay (CWD).

    When cautious_wd > 0, replaces decoupled weight decay with cautious decay:
    only decay a weight coordinate where the optimizer update is already
    shrinking it (update . weight > 0). When cautious_wd == 0, behaves exactly
    like torch.optim.AdamW (standard decoupled WD via the weight_decay group).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, cautious_wd=0.0, *, maximize=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        self.cautious_wd = cautious_wd

    @torch.no_grad()
    def step(self, closure=None):
        if self.cautious_wd <= 0:
            return super().step(closure)
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            cwpd = self.cautious_wd
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(g)
                    state["exp_avg_sq"] = torch.zeros_like(g)
                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias_c1 = 1 - beta1 ** int(state["step"])
                bias_c2 = 1 - beta2 ** int(state["step"])
                step_size = lr * (bias_c2 ** 0.5) / bias_c1
                # AdamW update direction (before lr): m_hat / (sqrt(v_hat) + eps)
                update = m / (v.sqrt().add_(eps))
                # CWD mask: decay only where update is shrinking the weight.
                cautious_mask = (update * p.data > 0).to(update.dtype)
                # Standard decoupled WD + cautious WD.
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                p.data -= cwpd * lr * cautious_mask * p.data
                p.data.add_(update, alpha=-step_size)
        return loss


_NS_SIMPLE_QUINTIC = (2.0, -1.5, 0.5)


def _polar_simple_quintic(g: torch.Tensor, steps: int = 12) -> torch.Tensor:
    """Polar factor via the simple-quintic Newton-Schulz schedule.

    Uses (a, b, c) = (2, -1.5, 0.5) — the sigma=1 super-attracting coefficients —
    for `steps` iterations, matching the modded-nanoGPT track-3 schedule that
    Aurora was released against. Distinct from `_zeropower_via_newtonschulz5`,
    which defaults to the aggressive 5-step quintic.
    """
    assert g.ndim == 2
    x = g.to(torch.bfloat16)
    x = x / (x.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    a, b, c = _NS_SIMPLE_QUINTIC
    for _ in range(steps):
        a_mat = x @ x.T
        x = a * x + (b * a_mat + c * (a_mat @ a_mat)) @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Aurora(Optimizer):
    """Aurora (Tilde Research) — Muon with a joint Stiefel/row-oblique projection.

    Muon's polar step maps the update onto the Stiefel manifold, which leaves row
    norms free. On rectangular matrices that lets rows collapse, and the measured
    consequence is neuron death (>25% dead MLP neurons by step 500 in Muon). Aurora
    alternates the polar step with a row-norm rebalancing toward the isotropic
    target n/m, so the update is (approximately) both orthogonal and leverage-uniform.

    For square matrices the rebalancing is skipped and the update reduces exactly
    to Muon, so the difference is confined to rectangular weights — which for this
    codebase means the FFN up/gate/down (768x3072) and qkv (768x2304).

    Source: github.com/tilde-research/aurora-release (MIT), vendored rather than
    pip-installed because the project ships no package.

    Note: Aurora stays on the per-matrix `_polar_simple_quintic` path even when
    `muon_ns_batched` is set. The row-oblique rebalancing loop recomputes a
    per-row-scaled `d * g32` and re-runs the polar step each iteration, so the NS
    input changes between rebalancing passes and the per-pass scaling is per-matrix
    — there is no stable same-shape stack to batch across. The win is also smaller
    here (Aurora is not the sweep default), so the cuBLAS path is retained.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        pp_iterations: int = 2,
        pp_beta: float = 0.5,
        ns_steps: int = 12,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        eps = 1e-7
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(
                        f"Aurora expects 2D parameters, got shape {tuple(g.shape)}. "
                        "Route 1D/embedding/head params to AdamW instead."
                    )
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mu)
                update = g.lerp(buf, mu) if group["nesterov"] else buf.clone()

                m, n = update.size(-2), update.size(-1)
                if m == n:
                    update = _polar_simple_quintic(update, group["ns_steps"])
                else:
                    transposed = m < n
                    if transposed:
                        update = update.mT
                        m, n = n, m
                    g32 = update.to(torch.float32)
                    target_row_sq = n / m
                    d = 1.0 / g32.norm(dim=-1, keepdim=True).clamp_(min=eps)
                    u = _polar_simple_quintic(d * g32, group["ns_steps"])
                    for _ in range(group["pp_iterations"] - 1):
                        row_sq = u.float().pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                        d = d * (target_row_sq / row_sq).pow(group["pp_beta"])
                        u = _polar_simple_quintic(d * g32, group["ns_steps"])
                    update = (u.mT if transposed else u).to(g.dtype)

                # Same spectral scaling as Muon: orthogonal matrices have unit
                # spectral norm regardless of shape, so wide layers need a bigger step.
                update = update * max(1.0, (g.size(-2) / g.size(-1)) ** 0.5)
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(update, alpha=-lr)
        return loss


def _no_decay_param_ids(model: nn.Module) -> set[int]:
    """Params that should not receive weight decay: embeddings and 1D tensors."""
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            ids.update(id(p) for p in module.parameters(recurse=False))
    ids.update(id(p) for p in model.parameters() if p.ndim < 2)
    return ids


def _is_memory_param(name: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(p in name for p in patterns)


def _classify_params(
    model: nn.Module, memory_patterns: tuple[str, ...] | list[str]
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Return (muon_backbone, muon_memory, adamw_backbone, adamw_memory).

    Routing rules:
      - Embeddings → adamw (always; never Muon).
      - 2D matrices → muon if Muon path requested, else adamw.
      - 1D / non-matrix params → adamw.
      - Within each, `memory` if its qualified name matches any memory pattern.
    Memory routing only matters when `memory_lr > 0`; callers split into LR groups
    based on the returned lists.
    """
    embedding_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                embedding_param_ids.add(id(p))

    muon_backbone: list[nn.Parameter] = []
    muon_memory: list[nn.Parameter] = []
    adamw_backbone: list[nn.Parameter] = []
    adamw_memory: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_memory = _is_memory_param(name, memory_patterns)
        if id(param) in embedding_param_ids:
            (adamw_memory if is_memory else adamw_backbone).append(param)
            continue
        if param.ndim == 2:
            (muon_memory if is_memory else muon_backbone).append(param)
        else:
            (adamw_memory if is_memory else adamw_backbone).append(param)
    return muon_backbone, muon_memory, adamw_backbone, adamw_memory


def build_head_block_map(model: nn.Module) -> dict[int, tuple[int, int]]:
    """Map id(attention weight) -> (split_axis, n_blocks) for per-head Muon.

    Per-head / Group Muon (arXiv:2605.08933; Kimi K3 §2.5) orthogonalises each
    attention head's block separately instead of the whole fused matrix. Muon
    maps every singular value to 1 *per matrix*, so with a fused QKV the update
    scale is shared across all heads and heads with larger gradients dominate;
    splitting equalises the update scale across heads. Kimi K3 uses this at 2.8T
    params for stability; nanoGPT PR #253 measured ~10 fewer steps.

    Which axis:
      * `qkv` is (3*d_model, d_model) — output features are
        [q heads | k heads | v heads], each head owning `head_dim` ROWS, so it
        splits along axis 0 into 3*num_heads blocks.
      * `out` is (d_model, d_model) — it consumes the concatenated heads, so
        each head owns `head_dim` COLUMNS: axis 1, num_heads blocks.

    Built by walking modules rather than tagging Parameters at construction
    time, because FP8 conversion swaps `nn.Linear` for `Float8Linear` and any
    attribute stashed on the original module would be lost. This runs after that
    swap (train.py: build_model -> maybe_convert_fp8 -> build_optimizer).
    """
    mapping: dict[int, tuple[int, int]] = {}
    for module in model.modules():
        heads = getattr(module, "num_heads", None)
        if not isinstance(heads, int) or heads <= 1:
            continue
        qkv = getattr(module, "qkv", None)
        out = getattr(module, "out", None)
        w = getattr(qkv, "weight", None)
        if w is not None and w.ndim == 2 and w.size(0) % (3 * heads) == 0:
            mapping[id(w)] = (0, 3 * heads)
        w = getattr(out, "weight", None)
        if w is not None and w.ndim == 2 and w.size(1) % heads == 0:
            mapping[id(w)] = (1, heads)
    return mapping


def build_circuit_map(model: nn.Module) -> dict[int, tuple[int, int]]:
    """Map id(qkv weight) -> (id(out weight), num_heads) for Compositional Muon.

    CM updates the QK and OV *products*, so it needs both factors of a circuit at
    once. Flower fuses Q/K/V into one matrix and keeps `out` separate, so one
    attention block contributes exactly one entry keyed on the qkv weight.

    Built by walking modules for the same reason as `build_head_block_map`: FP8
    conversion replaces `nn.Linear` with `Float8Linear`, so anything stashed on
    the original module at construction time would be lost. Both maps are built
    after that swap (train.py: build_model -> maybe_convert_fp8 -> build_optimizer).

    A block is only mapped when the shapes are the ones CM's slicing assumes —
    qkv exactly (3*d, d) and out exactly (d, d) with d divisible by num_heads.
    Anything else (a memory variant with a different attention layout) is left
    out of the map and falls through to standard Muon rather than being sliced on
    a guess.
    """
    mapping: dict[int, tuple[int, int]] = {}
    for module in model.modules():
        heads = getattr(module, "num_heads", None)
        if not isinstance(heads, int) or heads <= 1:
            continue
        w_qkv = getattr(getattr(module, "qkv", None), "weight", None)
        w_out = getattr(getattr(module, "out", None), "weight", None)
        if w_qkv is None or w_out is None or w_qkv.ndim != 2 or w_out.ndim != 2:
            continue
        d = w_out.shape[0]
        if w_out.shape != (d, d) or w_qkv.shape != (3 * d, d) or d % heads:
            continue
        mapping[id(w_qkv)] = (id(w_out), heads)
    return mapping


def _split_params_for_muon(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Backward-compatible wrapper: returns (muon, adamw) ignoring the memory split."""
    mb, mm, ab, am = _classify_params(model, ())
    return mb + mm, ab + am


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> Optimizer | list[Optimizer]:
    """Construct the optimizer(s) for `model` from `cfg`.

    Returns a single Optimizer (AdamW) or a list of Optimizers (Muon + AdamW) for
    the Muon path. The trainer calls `.step()` and `.zero_grad()` on each one.

    When `cfg.memory_lr > 0`, memory-bank parameters get a separate group with a
    distinct LR (I8 dual-LR — backbone slow, memory fast).
    """
    name = cfg.optimizer.lower()
    cautious_wd = float(getattr(cfg, "cautious_wd", 0.0))
    adamw_cls = CautiousAdamW if cautious_wd > 0 else AdamW
    memory_patterns = tuple(cfg.memory_param_patterns or ())
    use_dual_lr = cfg.memory_lr > 0 and bool(memory_patterns)
    muon_bb, muon_mem, adamw_bb, adamw_mem = _classify_params(model, memory_patterns)
    # Per-head Muon (opt-in). Empty map = every matrix orthogonalised whole,
    # i.e. exactly the existing behaviour.
    head_map = build_head_block_map(model) if getattr(cfg, "muon_per_head", False) else {}
    # Compositional Muon (opt-in). CM and per-head Muon target exactly the same
    # parameters (fused qkv + out), so CM wins on those and per-head is left to
    # cover any attention block CM's shape check rejected.
    comp_muon = bool(getattr(cfg, "comp_muon", False))
    if comp_muon and name != "muon":
        # Aurora replaces the polar step itself and has no partner-whitening
        # hook, so it would accept the flag and ignore it. Fail loudly: a screen
        # arm that silently ran the control is worse than one that did not run.
        raise ValueError(
            f"comp_muon requires optimizer='muon', got {cfg.optimizer!r}. "
            "Compositional Muon is implemented on the Muon step only."
        )
    circuit_map = build_circuit_map(model) if comp_muon else {}

    opts: list[Optimizer] = []

    exclude_wd = bool(getattr(cfg, "weight_decay_exclude_embeddings", False))
    no_decay_ids = _no_decay_param_ids(model) if exclude_wd else set()

    def _split_by_decay(params: list[nn.Parameter], lr: float) -> list[dict]:
        """Split a param list into decayed / non-decayed AdamW groups.

        Embeddings and 1D params (norm gains, biases) are excluded from weight
        decay when requested: decaying an embedding shrinks rare-token rows in
        proportion to how rarely they are updated, and decaying a norm gain
        works against the normalisation it parameterises.
        """
        if not exclude_wd:
            return [{"params": params, "lr": lr, "weight_decay": cfg.weight_decay}] if params else []
        decayed = [p for p in params if id(p) not in no_decay_ids]
        plain = [p for p in params if id(p) in no_decay_ids]
        groups = []
        if decayed:
            groups.append({"params": decayed, "lr": lr, "weight_decay": cfg.weight_decay})
        if plain:
            groups.append({"params": plain, "lr": lr, "weight_decay": 0.0})
        return groups

    if name == "adamw":
        main = adamw_bb + muon_bb
        memory = adamw_mem + muon_mem
        if use_dual_lr and memory:
            adamw_groups = _split_by_decay(main, cfg.lr) + _split_by_decay(memory, cfg.memory_lr)
        else:
            adamw_groups = _split_by_decay(main + memory, cfg.lr)
        opts.append(adamw_cls(adamw_groups, cautious_wd=cautious_wd) if cautious_wd > 0 else adamw_cls(adamw_groups))
        return opts if use_dual_lr else opts[0]

    if name in {"muon", "aurora"}:
        # Muon/Aurora: route 2D backbone (and optionally 2D memory) to matrix-optimizer instances.
        # AdamW handles 1D/embedding params and the Muon optimizer itself doesn't
        # support param groups with different LRs cleanly, so split into two Muon
        # instances when dual-LR is requested.
        def _matrix_opt(params: list[nn.Parameter], lr: float) -> Optimizer:
            if name == "aurora":
                return Aurora(
                    params,
                    lr=lr,
                    momentum=cfg.muon_momentum,
                    weight_decay=cfg.aurora_weight_decay,
                    pp_iterations=cfg.aurora_pp_iterations,
                    pp_beta=cfg.aurora_pp_beta,
                )
            return Muon(
                params,
                lr=lr,
                momentum=cfg.muon_momentum,
                ns_steps=cfg.muon_ns_steps,
                ns_schedule=cfg.muon_ns_schedule,
                norm_update=getattr(cfg, "norm_update", False),
                cautious_wd=getattr(cfg, "cautious_wd", 0.0),
                ns_batched=getattr(cfg, "muon_ns_batched", True),
                contra_muon=getattr(cfg, "contra_muon", 0.0),
                head_map=head_map,
                nesterov=getattr(cfg, "muon_nesterov", True),
                circuit_map=circuit_map,
                cm_isotropic=bool(getattr(cfg, "comp_muon_isotropic", False)),
                cm_damping=float(getattr(cfg, "comp_muon_damping", 1e-2)),
                cm_mp=float(getattr(cfg, "comp_muon_mp", 1.0)),
            )

        if muon_bb:
            opts.append(_matrix_opt(muon_bb, cfg.muon_lr))
        if muon_mem:
            opts.append(_matrix_opt(muon_mem, cfg.memory_lr if use_dual_lr else cfg.muon_lr))
        # AdamW for everything non-2D (embeddings + biases + LN + scalars).
        adamw_groups = []
        if adamw_bb:
            adamw_groups.extend(_split_by_decay(adamw_bb, cfg.lr))
        if adamw_mem:
            adamw_groups.extend(_split_by_decay(adamw_mem, cfg.memory_lr if use_dual_lr else cfg.lr))
        if adamw_groups:
            opts.append(adamw_cls(adamw_groups, cautious_wd=cautious_wd) if cautious_wd > 0 else adamw_cls(adamw_groups))
        if not opts:
            raise ValueError("No trainable parameters found")
        return opts
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
