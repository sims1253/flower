"""Symplectic Hamiltonian flow for Q/K vectors.

A SympNet-style construction (Jin et al. 2020, "SympNets: Intrinsic structure-
preserving symplectic networks for identifying Hamiltonian systems"). The flow
is symplectic *by architecture* rather than by numerical integrator: it is built
from "gradient shears" whose Jacobians are exact Hessians of scalar potentials.
Composing such shears gives a symplectic map regardless of step size, so volume
preservation and bounded energy drift fall out of the architecture rather than
having to be regularised in.

Construction. Split each head-dim vector into position q and momentum p halves.
A gradient shear of the form

    p_kick(q) = W^T tanh(W q + b)

is the exact gradient of the scalar potential

    U(q) = 1^T log cosh(W q + b)

so its Jacobian (= Hessian of U) is symmetric, which makes the map (q, p) ->
(q, p + dt * p_kick(q)) symplectic for any dt. A position shear is defined
analogously from the momentum side. Alternating the two in a leapfrog pattern

    p <- p + (dt/2) * p_kick(q)
    q <- q + dt * q_kick(p)
    p <- p + (dt/2) * p_kick(q)

gives a symmetric, second-order, structure-preserving step.

Diagnostic. Even though the architecture is exactly symplectic, the *explicit*
Hamiltonian H = U(q) + K(p) we name above is only one Hamiltonian; the
composition with per-step time biases conserves a "shadow" one. Tracking
||H_final - H_init||^2 measures how far the explicit H is from being the
shadow H -- a useful indicator of how smooth the learned potentials are and
how well-conditioned the flow is. It's written into a buffer for free
logging; never enters the loss.

Why not just regularise an Euler flow toward energy conservation? Two reasons.
(1) Architectural constraints are tighter than soft losses -- volume
preservation is exact at every step instead of approximate at the end. (2) The
gradient shear with `tanh` activation has a closed-form scalar potential, which
gives us the diagnostic H without a separate energy net.

Options exposed:
  - hamiltonian_mass="diagonal" learns a per-coordinate kinetic mass, scaling
    p before and after the K-shear. Equivalent to a Nutpie-style diagonal
    preconditioner; lets each head-dim channel set its own timescale.
  - WalnutsHamiltonianFlow does per-macro-step adaptive dt gated by explicit
    energy error (batch-mean). Coarse where the potential is flat, fine where
    it is curved. Decisions are made on detached tensors so they never enter
    the gradient.

Cost. Per shear: 2 * half_dim * hidden_dim params. With defaults (half_dim=32,
hidden_dim=64) that's ~4k params per shear, and 2 shears per layer of the flow.
Negligible compared to attention QKV projections.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _log_cosh(z: torch.Tensor) -> torch.Tensor:
    """Numerically stable log cosh."""
    abs_z = z.abs()
    return abs_z + F.softplus(-2.0 * abs_z) - math.log(2.0)


class GradientShear(nn.Module):
    """One symplectic gradient shear.

    Output = W^T tanh(W x + b), i.e. exact gradient of `_log_cosh(W x + b).sum(-1)`.
    Per-step bias is applied externally so a single shear can be reused across
    leapfrog iterations with a time-varying potential while keeping parameter
    count low.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim, bias=True)

    def kick(self, x: torch.Tensor, extra_bias: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x)
        if extra_bias is not None:
            h = h + extra_bias
        a = torch.tanh(h)
        return F.linear(a, self.proj.weight.t())

    def potential(self, x: torch.Tensor, extra_bias: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x)
        if extra_bias is not None:
            h = h + extra_bias
        return _log_cosh(h).sum(dim=-1)


class HamiltonianFlow(nn.Module):
    """SympNet-style symplectic flow that maps (..., dim) -> (..., dim).

    The vector is split into (q, p) halves of size dim // 2. `steps` leapfrog
    iterations are applied with per-step learnable biases to give the potentials
    a depth-varying signature without per-step weight matrices.
    """

    def __init__(
        self,
        dim: int,
        steps: int = 3,
        hidden_dim: int | None = None,
        mass: str = "isotropic",
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"HamiltonianFlow requires even dim, got {dim}")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if mass not in {"isotropic", "diagonal"}:
            raise ValueError("mass must be 'isotropic' or 'diagonal'")
        self.dim = dim
        self.half = dim // 2
        self.steps = steps
        hidden = hidden_dim or (2 * self.half)
        self.v_shear = GradientShear(self.half, hidden)
        self.k_shear = GradientShear(self.half, hidden)
        self.time_bias_v = nn.Parameter(torch.zeros(steps, hidden))
        self.time_bias_k = nn.Parameter(torch.zeros(steps, hidden))
        if mass == "diagonal":
            self.log_mass = nn.Parameter(torch.zeros(self.half))
        else:
            self.register_parameter("log_mass", None)
        # Diagnostic: explicit H drift across the flow. Detached, no grad.
        self.register_buffer("last_diag_hamiltonian_energy_drift", torch.zeros(()), persistent=False)

    def _mass_scale(self) -> torch.Tensor | None:
        if self.log_mass is None:
            return None
        return torch.exp(-0.5 * self.log_mass)

    def _v_kick(self, q: torch.Tensor, step_idx: int) -> torch.Tensor:
        return self.v_shear.kick(q, extra_bias=self.time_bias_v[step_idx])

    def _k_kick(self, p: torch.Tensor, step_idx: int) -> torch.Tensor:
        scale = self._mass_scale()
        if scale is not None:
            p = p * scale
        kick = self.k_shear.kick(p, extra_bias=self.time_bias_k[step_idx])
        if scale is not None:
            kick = kick * scale
        return kick

    def _hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Average potential across step biases. Gives a single, time-invariant
        # H proxy whose drift across the flow is a sane diagnostic.
        p_eff = p
        scale = self._mass_scale()
        if scale is not None:
            p_eff = p_eff * scale
        u = sum(self.v_shear.potential(q, self.time_bias_v[i]) for i in range(self.steps)) / self.steps
        k = sum(self.k_shear.potential(p_eff, self.time_bias_k[i]) for i in range(self.steps)) / self.steps
        return u + k

    def _leapfrog(self, q: torch.Tensor, p: torch.Tensor, dt: float, step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = p + 0.5 * dt * self._v_kick(q, step_idx)
        q = q + dt * self._k_kick(p, step_idx)
        p = p + 0.5 * dt * self._v_kick(q, step_idx)
        return q, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, p = x.chunk(2, dim=-1)
        with torch.no_grad():
            h_init = self._hamiltonian(q, p)
        dt = 1.0 / self.steps
        for i in range(self.steps):
            q, p = self._leapfrog(q, p, dt, i)
        with torch.no_grad():
            h_final = self._hamiltonian(q, p)
            self.last_diag_hamiltonian_energy_drift.copy_((h_final - h_init).pow(2).mean())
        return torch.cat([q, p], dim=-1)


class WalnutsHamiltonianFlow(HamiltonianFlow):
    """Hamiltonian flow with per-macro-step adaptive dt (WALNUTS-style).

    For each macro step, we try progressively halved sub-step sizes (1, 1/2,
    1/4, ...) until the *relative* batch-mean explicit energy error over the
    macro is below `energy_threshold`, or we exhaust `max_subdivisions`
    halvings. Relative (drift / |H|) rather than absolute so the threshold
    stays meaningful regardless of the scale of the learned potential -- an
    absolute threshold saturates at max subdivisions whenever the model
    decides to use large W magnitudes.

    All adaptation decisions are made on detached tensors and don't enter the
    gradient. Trial work is discarded when a finer subdivision is needed --
    correctness-first; we can refine to incremental refinement later if this
    proves promising. Worst-case extra forward cost is 2^(max_subdivisions+1)-1
    leapfrog evaluations per macro step.
    """

    def __init__(
        self,
        dim: int,
        steps: int = 3,
        hidden_dim: int | None = None,
        mass: str = "isotropic",
        energy_threshold: float = 0.05,
        max_subdivisions: int = 2,
    ) -> None:
        super().__init__(dim, steps, hidden_dim, mass)
        self.energy_threshold = float(energy_threshold)
        self.max_subdivisions = int(max_subdivisions)
        # Mean and max substep counts from the last forward, for logging.
        self.register_buffer("last_diag_hamiltonian_substeps_mean", torch.zeros(()), persistent=False)
        self.register_buffer("last_diag_hamiltonian_substeps_max", torch.zeros(()), persistent=False)

    def _trial_macro(self, q: torch.Tensor, p: torch.Tensor, step_idx: int, n_sub: int) -> tuple[torch.Tensor, torch.Tensor]:
        macro_dt = 1.0 / self.steps
        sub_dt = macro_dt / n_sub
        for _ in range(n_sub):
            q, p = self._leapfrog(q, p, sub_dt, step_idx)
        return q, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, p = x.chunk(2, dim=-1)
        with torch.no_grad():
            h_init_total = self._hamiltonian(q, p)
        substep_log: list[int] = []
        for i in range(self.steps):
            chosen_n = 1
            best_q, best_p = q, p
            for trial in range(self.max_subdivisions + 1):
                n = 1 << trial  # 1, 2, 4, ...
                # Probe on detached state, then commit a non-detached run with the
                # winning n so gradients flow through the chosen trajectory.
                with torch.no_grad():
                    qt, pt = self._trial_macro(q.detach(), p.detach(), i, n)
                    h_pre = self._hamiltonian(q.detach(), p.detach())
                    h_post = self._hamiltonian(qt, pt)
                    drift = (h_post - h_pre).abs().mean().item()
                    ref = h_pre.abs().mean().item()
                    rel_drift = drift / (ref + 1e-6)
                if rel_drift < self.energy_threshold or trial == self.max_subdivisions:
                    chosen_n = n
                    best_q, best_p = self._trial_macro(q, p, i, n)
                    break
            q, p = best_q, best_p
            substep_log.append(chosen_n)
        with torch.no_grad():
            steps_tensor = torch.tensor(substep_log, dtype=torch.float32, device=self.last_diag_hamiltonian_substeps_mean.device)
            self.last_diag_hamiltonian_substeps_mean.copy_(steps_tensor.mean())
            self.last_diag_hamiltonian_substeps_max.copy_(steps_tensor.max())
            h_final_total = self._hamiltonian(q, p)
            self.last_diag_hamiltonian_energy_drift.copy_((h_final_total - h_init_total).pow(2).mean())
        return torch.cat([q, p], dim=-1)
