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


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration for the matrix polar decomposition (orthogonalisation).

    Approximates U @ V^T where g = U S V^T. Quintic schedule with the standard
    (3.4445, -4.7750, 2.0315) coefficients used by the speedrun community.
    Operates in bf16 for speed; the polynomial is numerically robust there.
    """
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.to(torch.bfloat16)
    # Normalise spectral norm to <= 1 to keep the polynomial in its convergence basin.
    x = x / (x.norm() + 1e-7)
    # Operate on the smaller side for compute efficiency (transpose if rows > cols).
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


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
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
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
                ortho = _zeropower_via_newtonschulz5(update_dir, ns_steps)
                # Scale step by max(1, sqrt(fan_out/fan_in)) so wide layers still move.
                scale = max(1.0, (g.size(0) / g.size(1)) ** 0.5)
                p.add_(ortho, alpha=-lr * scale)
        return loss


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
    memory_patterns = tuple(cfg.memory_param_patterns or ())
    use_dual_lr = cfg.memory_lr > 0 and bool(memory_patterns)
    muon_bb, muon_mem, adamw_bb, adamw_mem = _classify_params(model, memory_patterns)

    opts: list[Optimizer] = []

    if name == "adamw":
        adamw_groups: list[dict] = []
        if adamw_bb or muon_bb:
            adamw_groups.append({"params": adamw_bb + muon_bb, "lr": cfg.lr})
        if use_dual_lr and (adamw_mem or muon_mem):
            adamw_groups.append({"params": adamw_mem + muon_mem, "lr": cfg.memory_lr})
        else:
            # No dual-LR: fold memory params into the main group at cfg.lr.
            if adamw_groups:
                adamw_groups[0]["params"].extend(adamw_mem + muon_mem)
            else:
                adamw_groups.append({"params": adamw_mem + muon_mem, "lr": cfg.lr})
        opts.append(AdamW(adamw_groups))
        return opts if use_dual_lr else opts[0]

    if name == "muon":
        # Muon: route 2D backbone (and optionally 2D memory) to Muon optimizer instances.
        # AdamW handles 1D/embedding params and the Muon optimizer itself doesn't
        # support param groups with different LRs cleanly, so split into two Muon
        # instances when dual-LR is requested.
        if muon_bb:
            opts.append(Muon(muon_bb, lr=cfg.muon_lr, momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps))
        if muon_mem:
            mem_muon_lr = cfg.memory_lr if use_dual_lr else cfg.muon_lr
            opts.append(Muon(muon_mem, lr=mem_muon_lr, momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps))
        # AdamW for everything non-2D (embeddings + biases + LN + scalars).
        adamw_groups = []
        if adamw_bb:
            adamw_groups.append({"params": adamw_bb, "lr": cfg.lr})
        if adamw_mem:
            adamw_groups.append({"params": adamw_mem, "lr": cfg.memory_lr if use_dual_lr else cfg.lr})
        if adamw_groups:
            opts.append(AdamW(adamw_groups))
        if not opts:
            raise ValueError("No trainable parameters found")
        return opts
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
