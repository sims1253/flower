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
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_schedule=ns_schedule,
            norm_update=norm_update,
            cautious_wd=cautious_wd,
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
                ortho = _zeropower_via_newtonschulz5(update_dir, ns_steps, ns_schedule)
                # NorMuon (arXiv:2510.05491): normalise the orthogonalised update to
                # unit Frobenius norm before the LR/aspect-ratio scaling step.
                if norm_update:
                    ortho = ortho / (ortho.norm() + 1e-7)
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
