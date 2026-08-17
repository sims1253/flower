"""StillLM: Frozen base model + per-layer Still compactors.

Wraps an existing Flower CausalLM (the "base model") with StillCompactor modules
that learn to compress each layer's KV cache. The base model weights are frozen;
only the compactor parameters are trainable.

Training mode:
- Forward pass 1 (teacher): base model with full KV cache -> teacher logits.
- Forward pass 2 (student): base model with compacted KV cache -> student logits.
- Loss: KL(teacher || student) on answer tokens, optionally + CE on answer tokens.

Inference mode:
- Run base model prefill to get full KV cache.
- Apply compactors to get compact cache.
- Continue generation/query against the compact cache.

The implementation extracts per-layer K and V from the attention module during
a forward pass using hooks, then replaces them with the compacted versions for
the student forward pass.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM
from flower.models.still import (
    StillCompactor,
    StillCompactorFlow,
    StillCompactorOT,
    StillCompactorSpectral,
)
from flower.models.still_flow2 import StillCompactorFlowOT, StillCompactorMeanFlow
from flower.models.still_flow3 import StillCompactorFlowKV


def _standard_attention(attn, q, k, v, seq_len: int, device, x_shape) -> torch.Tensor:
    """Causal (+optional window) attention used by the Still standard branches.

    Mirrors CausalSelfAttention's own forward but operates on already-projected
    q/k/v (Still extracts them to cache the teacher KV). Honours
    `attn.use_flex` (S1): when set, use FlexAttention with the module's cached
    block mask; otherwise fall back to SDPA + a materialized causal mask.

    The compact-path branches (mixed compact-prefix + causal-suffix KV with a
    per-layer variable compact_len) intentionally stay on SDPA: that mask
    structure cannot be expressed as a single BlockMask. Still is a research
    variant, not the seq=32K throughput path, so this is the right boundary.
    """
    if getattr(attn, "use_flex", False):
        from flower.models.base import _load_flex_attention

        flex_attention, _ = _load_flex_attention()
        block_mask = attn._get_block_mask(seq_len, device)
        out = flex_attention(q, k, v, block_mask=block_mask)
    else:
        from flower.models.base import causal_mask

        keep = causal_mask(seq_len, device, attn.local_window).view(1, 1, seq_len, seq_len)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    out = out.transpose(1, 2).contiguous().view(x_shape)
    return attn.out(out)


def _suffix_causal_mask(span: int, device: torch.device, local_window: int | None) -> torch.Tensor:
    """Causal mask over the suffix region, honouring the layer's local window.

    The suffix keys are ordinary (uncompacted) tokens, so they must obey exactly
    the same windowing the teacher applies to them. Omitting the window is
    harmless only while `span <= local_window` — which held for the legacy
    geometry (span = compact_len = 64 < window 256) and hid the issue. Once the
    evaluated span is widened past the window, an unwindowed mask would hand the
    student longer local reach than the teacher and confound the comparison with
    the very thing being measured.

    `local_window=None` means the layer is full-attention, so no windowing.
    """
    idx = torch.arange(span, device=device)
    rel = idx.unsqueeze(1) - idx.unsqueeze(0)
    mask = rel >= 0
    if local_window is not None:
        mask &= rel < local_window
    return mask


def _pyramid_budget(base_len: int, num_layers: int) -> list[int]:
    """PyramidKV-style decreasing budget across layers.

    Lower layers (closer to input) get more cache because attention scatters there.
    Upper layers get less because attention concentrates. The budget decreases
    linearly from base_len * 1.5 at layer 0 to base_len * 0.5 at the last layer,
    with the average matching base_len.
    """
    if num_layers <= 1:
        return [base_len]
    result = []
    for i in range(num_layers):
        ratio = 1.5 - i * 1.0 / max(num_layers - 1, 1)
        result.append(max(int(base_len * ratio), 4))
    return result


class StillLM(nn.Module):
    """Frozen base model + per-layer Still KV-cache compactors.

    The base model's attention modules must expose their K and V tensors during
    forward so the compactor can read them. We use forward hooks on the qkv
    projection inside CausalSelfAttention to intercept K and V.
    """

    def __init__(
        self,
        base_model: CausalLM,
        config: ModelConfig,
        compact_len: int = 32,
        num_blocks: int = 2,
        d_latent: int | None = None,
        use_ot_read: bool = False,
        use_energy_read: bool = False,
        use_freq_decay: bool = False,
        use_spectral: bool = False,
        layer_adaptive: bool = False,
        attn_match_weight: float = 0.0,
        ot_reg_weight: float = 0.0,
        flow_steps: int = 0,
        meanflow_steps: int = 0,
        meanflow_loss_weight: float = 0.0,
        flow_ot: bool = False,
        flow_kv: bool = False,
        ot_epsilon: float = 0.1,
        ot_iters: int = 10,
        key_velocity_hidden: int | None = None,
        val_velocity_hidden: int | None = None,
        kl_topk: int = 200,
        kl_weight: float = 1.0,
        ce_weight: float = 0.0,
        compact_from_step: int = 0,
        kl_temperature: float = 1.0,
        base_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.kl_topk = kl_topk
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.compact_from_step = compact_from_step
        self.kl_temperature = kl_temperature
        self._base_warmup_steps = base_warmup_steps
        self._current_step = 0

        if ot_reg_weight > 0:
            # StillCompactorOTReg (still.py) was byte-identical to the plain
            # compactor: its forward read neither ot_reg_weight nor produced a
            # penalty, and no training-loss consumer ever existed, so every
            # run made with this flag was a plain-compactor run regardless.
            # The dead class is deleted; fail loudly instead of training a
            # mislabeled arm.
            raise ValueError(
                "still_ot_reg_weight > 0 selects the removed 'OT-regularized' "
                "compactor arm, which never had any effect: its forward was "
                "identical to the plain StillCompactor and no OT penalty was "
                "ever computed or consumed. No existing results are affected "
                "(they were plain-compactor runs). Use still_use_ot_read "
                "(StillCompactorOT, which does replace the read with a Sinkhorn "
                "plan) or leave this at 0."
            )

        # The single-pass warmup phase is keyed on compact_from_step (see
        # forward), and the docstring promises the base is FROZEN during it.
        # Base unfreezing keys on base_warmup_steps, so
        # base_warmup_steps > compact_from_step would leave the base trainable
        # inside the dual-pass compactor phase (the student pass backprops
        # straight through the base blocks), contradicting the frozen-teacher
        # design and every KL comparison made against it. Refuse it up front.
        if base_warmup_steps > compact_from_step:
            raise ValueError(
                f"still_base_warmup_steps ({base_warmup_steps}) exceeds "
                f"still_compact_from_step ({compact_from_step}): the base would "
                "still be unfrozen when the dual-pass compactor phase begins "
                "(the phase boundary keys on compact_from_step, the freeze on "
                "base_warmup_steps), so the 'frozen base' teacher/student "
                "comparison would train the base through the student pass. "
                "Warm the base entirely inside the single-pass phase "
                "(base_warmup_steps <= compact_from_step)."
            )

        # asdict() over the config was rebuilt on every forward; it is constant.
        self._config_dict: dict[str, Any] | None = None
        self.suffix_len = getattr(config, "still_suffix_len", None)
        self.loss_positions = getattr(config, "still_loss_positions", "all")
        if self.loss_positions not in {"all", "suffix"}:
            raise ValueError(
                f"still_loss_positions must be 'all' or 'suffix', got {self.loss_positions!r}"
            )
        self.layer_adaptive = layer_adaptive
        self.attn_match_weight = attn_match_weight
        self.flow_steps = flow_steps
        self.meanflow_steps = meanflow_steps
        # Weight of the MeanFlow self-consistency loss (0.0 = legacy: the
        # compactor computes it but StillLM discards it, reproducing every
        # prior still_meanflow run).
        self.meanflow_loss_weight = meanflow_loss_weight
        self.flow_ot = flow_ot
        self.flow_kv = flow_kv

        # Freeze base model (will be unfrozen during warmup if base_warmup_steps > 0).
        base_trainable = base_warmup_steps > 0
        for param in self.base_model.parameters():
            param.requires_grad_(base_trainable)

        # Determine KV cache geometry from the base model's attention layers.
        num_layers = config.num_layers
        num_heads = config.num_heads
        head_dim = config.d_model // config.num_heads
        dl = d_latent or 2 * head_dim

        # Resolve per-layer d_latent schedule (Sweep 11: tapered compactor budgets).
        # Default = uniform (current behavior). Identity init only valid at d_latent==2*head_dim.
        schedule_cfg = getattr(config, "still_d_latent_schedule", None)
        if schedule_cfg is not None:
            if len(schedule_cfg) != num_layers:
                raise ValueError(f"still_d_latent_schedule len {len(schedule_cfg)} != num_layers {num_layers}")
            layer_d_latents = list(schedule_cfg)
        else:
            layer_d_latents = [dl] * num_layers
        self.layer_d_latents = layer_d_latents

        # Create one compactor per layer.
        if meanflow_steps > 0:
            compactor_cls = StillCompactorMeanFlow
        elif flow_kv:
            compactor_cls = StillCompactorFlowKV
        elif flow_ot:
            compactor_cls = StillCompactorFlowOT
        elif use_spectral:
            compactor_cls = StillCompactorSpectral
        elif flow_steps > 0:
            compactor_cls = StillCompactorFlow
        elif use_ot_read:
            compactor_cls = StillCompactorOT
        else:
            compactor_cls = StillCompactor

        # Compute per-layer compact lengths (PyramidKV pattern if layer_adaptive).
        if layer_adaptive and num_layers > 1:
            layer_compact_lens = _pyramid_budget(compact_len, num_layers)
        else:
            layer_compact_lens = [compact_len] * num_layers

        self.layer_compact_lens = layer_compact_lens
        # The compactors un-rotate (and re-rotate) the BASE model's RoPE with
        # `base_rope_base`; defaulting it to 10000 while a config sweeps
        # `rope_base` silently corrupted every compact key (wrong inverse
        # rotation). Thread the base config's actual base through.
        base_rope_base = float(getattr(config, "rope_base", 10000.0))
        self.compactors = nn.ModuleList()
        for i in range(num_layers):
            kwargs = dict(
                num_kv_heads=num_heads,
                head_dim=head_dim,
                compact_len=layer_compact_lens[i],
                num_blocks=num_blocks,
                d_latent=layer_d_latents[i],
                base_rope_base=base_rope_base,
                # Identity init is only valid when d_latent == 2 * head_dim (the
                # eye() slice in StillCompactor._init_identity requires it).
                identity_init=(layer_d_latents[i] == 2 * head_dim),
                use_energy_read=use_energy_read,
                freq_decay=use_freq_decay,
            )
            if flow_steps > 0 or flow_ot:
                kwargs["flow_steps"] = flow_steps if flow_steps > 0 else 5
                # Shrink the velocity nets for param-matched comparisons
                # (StillCompactorFlow only; other flow classes ignore it).
                velocity_hidden = getattr(config, "still_velocity_hidden", None)
                if velocity_hidden is not None and compactor_cls is StillCompactorFlow:
                    kwargs["velocity_hidden"] = velocity_hidden
            if meanflow_steps > 0:
                kwargs["meanflow_steps"] = meanflow_steps
            if flow_ot:
                kwargs["ot_epsilon"] = ot_epsilon
                kwargs["ot_iters"] = ot_iters
            if flow_kv:
                kwargs["flow_steps"] = flow_steps if flow_steps > 0 else 5
                if key_velocity_hidden is not None:
                    kwargs["key_velocity_hidden"] = key_velocity_hidden
                if val_velocity_hidden is not None:
                    kwargs["val_velocity_hidden"] = val_velocity_hidden
            self.compactors.append(compactor_cls(**kwargs))

        self._kv_cache: list[dict[str, torch.Tensor]] = [{} for _ in range(num_layers)]
        self._compact_mode = False
        # Parameter counts are constant per build; sum(p.numel()) walked every
        # parameter on every forward (pattern: CausalLM._static_diagnostics).
        self._static_diagnostics: dict[str, Any] = {
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "base_parameter_count": sum(p.numel() for p in self.base_model.parameters()),
            "compactor_parameter_count": sum(p.numel() for p in self.compactors.parameters()),
        }

    def set_step(self, step: int) -> None:
        self._current_step = step
        # If we have a base_warmup_steps, toggle base model training accordingly.
        if hasattr(self, "_base_warmup_steps") and self._base_warmup_steps > 0:
            training_base = step < self._base_warmup_steps
            for param in self.base_model.parameters():
                param.requires_grad_(training_base)

    @property
    def compact_active(self) -> bool:
        return self._current_step >= self.compact_from_step

    def suffix_start(self, seq_len: int) -> int:
        """First position whose logits can differ from the teacher's.

        Everything before this runs plain local attention through the same
        frozen base in both passes and is bit-identical by construction.

        With `still_layer_adaptive` the per-layer compact budgets differ, so the
        split points differ too; the EARLIEST one governs, because a difference
        introduced at any layer propagates forward from there. Hence max() over
        the per-layer budgets.
        """
        span = self.suffix_len if self.suffix_len is not None else max(self.layer_compact_lens)
        return min(max(seq_len - span, 1), seq_len - 1)

    def _suffix_mask(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
        mask[:, self.suffix_start(seq_len) :] = True
        return mask

    def _extract_kv_and_forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_compact: bool = False,
        teacher_kv: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
        """Forward pass through the base model, optionally with compacted KV cache.

        For the teacher pass: standard forward, extract per-layer K, V.
        For the student pass: compact K, V, then forward with compact cache.

        teacher_kv: per-layer dicts with detached teacher 'keys'/'values'. When
        given alongside use_compact=True, the attention-pattern match loss
        (Fast KV Compaction) is computed inline on this pass's own suffix
        queries and compact keys, so the arm needs exactly one teacher pass
        and one student pass (it used to pay a third full base forward that
        re-derived q/k/v and re-ran every compactor with grad).

        Returns (model_output, list_of_kv_dicts) where each kv dict has
        'keys', 'values' per layer. model_output may additionally carry
        'meanflow_losses' (per-compactor consistency losses, student pass) and
        'attn_match_loss' (scalar, when teacher_kv is given).
        """
        # We implement a custom forward that intercepts K, V at each layer.
        # This requires modifying the attention computation, which we do
        # by temporarily patching the attention forward.

        T = input_ids.shape[1]
        device = input_ids.device

        # Collect KV from each layer during the forward pass.
        kv_layers: list[dict[str, torch.Tensor]] = []
        # Per-compactor MeanFlow consistency losses (student pass only).
        meanflow_losses: list[torch.Tensor] = []
        # Attention-match accumulation (student pass with teacher_kv given).
        am_total = torch.zeros((), device=device)
        am_layers = 0

        # We'll replace the attention forward in each block to extract K, V.
        # For the compact (student) pass, we'll inject the compacted cache.

        x = self.base_model.token(input_ids)

        # Determine positions for RoPE.
        positions = torch.arange(T, device=device, dtype=torch.float32)

        for layer_idx, block in enumerate(self.base_model.blocks):
            # Check if block has the standard structure.
            attn = getattr(block, "attn", None) or getattr(block, "local", None)
            if attn is None:
                # Fallback: just run the block normally.
                x, _ = block(x, None) if isinstance(x, torch.Tensor) else block(x)
                kv_layers.append({})
                continue

            # Extract K, V from attention layer.
            ln_out = block.ln1(x)
            # Delegate the whole q/k/v -> QK-norm -> RoPE pipeline to the
            # attention module. Composing attn.qkv + attn.rope by hand here used
            # to work, but it silently skips any step the module adds (QK-norm),
            # which would make the Still passes disagree with the base model's
            # own forward on a base pretrained with it.
            q_rot, k_rot, v_h = attn.qkv_heads(ln_out)
            B_inner, num_heads, T_inner, head_dim = q_rot.shape
            D = num_heads * head_dim

            if use_compact and self.compact_active and layer_idx < len(self.compactors):
                from flower.models.base import causal_mask as _causal_mask_fn

                compactor = self.compactors[layer_idx]
                t_compact = compactor.compact_len
                # Causal prefix/suffix split: compact the prefix KV [0:ctx_end),
                # then suffix queries [ctx_end:T) attend to the compact cache plus
                # causal self-attention.  Prefix queries use standard causal
                # attention.  This mirrors deployment (prefill -> compact ->
                # attend) and prevents the student from peeking at future tokens
                # through the compact cache.
                ctx_end = self.suffix_start(T_inner)
                result = compactor(
                    k_rot[:, :, :ctx_end, :],
                    v_h[:, :, :ctx_end, :],
                    positions=positions[:ctx_end],
                    return_compact_cache=True,
                )
                compact_k = result["compact_keys"]  # (B, H, t_compact, d)
                compact_v = result["compact_values"]
                # The MeanFlow compactor emits a scalar self-consistency loss
                # during training; StillLM.forward weights and sums it (it used
                # to be computed and silently discarded).
                mf = result.get("meanflow_loss")
                if mf is not None:
                    meanflow_losses.append(mf)

                # Suffix queries attend to [compact ; suffix_kv] with a causal mask:
                # all compact entries visible (strictly past), suffix lower-triangular.
                S = T_inner - ctx_end
                suffix_q = q_rot[:, :, ctx_end:, :]

                # Attention-pattern match (Fast KV Compaction, arXiv:2602.16284),
                # computed inline on this pass's own suffix queries / compact
                # keys: MSE between the student's suffix-vs-compact-prefix
                # scores and the teacher's suffix-vs-full-prefix scores,
                # pooled into t_compact bins when the budgets differ. The
                # teacher keys are detached (captured under no_grad).
                if (
                    teacher_kv is not None
                    and layer_idx < len(teacher_kv)
                    and teacher_kv[layer_idx]
                ):
                    t_keys_prefix = teacher_kv[layer_idx]["keys"][:, :, :ctx_end, :]
                    t_scores_prefix = (
                        suffix_q.float() @ t_keys_prefix.float().transpose(-2, -1)
                    ) / math.sqrt(head_dim)
                    s_scores = (
                        suffix_q.float() @ compact_k.float().transpose(-2, -1)
                    ) / math.sqrt(head_dim)
                    if t_compact < ctx_end:
                        bin_size = ctx_end / t_compact
                        pooled_t = torch.zeros(B_inner, num_heads, S, t_compact, device=device)
                        for bi in range(t_compact):
                            start = int(bi * bin_size)
                            end = int((bi + 1) * bin_size)
                            if end <= start:
                                end = start + 1
                            pooled_t[:, :, :, bi] = t_scores_prefix[:, :, :, start:end].mean(dim=-1)
                        t_match = pooled_t
                    else:
                        t_match = t_scores_prefix
                    am_total = am_total + F.mse_loss(s_scores, t_match.detach())
                    am_layers += 1

                all_k = torch.cat([compact_k, k_rot[:, :, ctx_end:, :]], dim=2)
                all_v = torch.cat([compact_v, v_h[:, :, ctx_end:, :]], dim=2)
                compact_visible = torch.ones(1, 1, S, t_compact, device=device, dtype=torch.bool)
                suffix_causal = _suffix_causal_mask(S, device, attn.local_window)
                suffix_mask = torch.cat(
                    [compact_visible, suffix_causal.unsqueeze(0).unsqueeze(0)], dim=3
                )
                suffix_out = F.scaled_dot_product_attention(suffix_q, all_k, all_v, attn_mask=suffix_mask)

                # Prefix queries use standard causal attention (identical to teacher).
                prefix_keep = _causal_mask_fn(ctx_end, device, attn.local_window).view(1, 1, ctx_end, ctx_end)
                prefix_out = F.scaled_dot_product_attention(
                    q_rot[:, :, :ctx_end, :], k_rot[:, :, :ctx_end, :], v_h[:, :, :ctx_end, :], attn_mask=prefix_keep
                )

                out = torch.cat([prefix_out, suffix_out], dim=2)
                out = out.transpose(1, 2).contiguous().view(B_inner, T_inner, D)
                attn_output = attn.out(out)
            else:
                # Standard attention path (honours attn.use_flex, see _standard_attention).
                seq_len = x.shape[1]
                attn_output = _standard_attention(attn, q_rot, k_rot, v_h, seq_len, device, x.shape)

            # Residual + FFN.
            x = x + attn_output
            x = x + block.ff(block.ln2(x))

            # Store the KV for this layer (teacher pass only needs this).
            if not use_compact:
                kv_layers.append({"keys": k_rot.detach(), "values": v_h.detach()})
            else:
                kv_layers.append({})

        logits = self.base_model.head(self.base_model.ln(x))

        loss = None
        if labels is not None:
            # Honour the base model's bf16_cross_entropy flag (S4) so the Still
            # teacher/student CE matches the base model's own loss precision.
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            if getattr(self.base_model, "bf16_cross_entropy", False):
                shift_logits = shift_logits.to(torch.bfloat16)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

        diagnostics: dict[str, Any] = dict(self._static_diagnostics)

        out: dict[str, Any] = {"logits": logits, "loss": loss, "diagnostics": diagnostics}
        # Grad-carrying auxiliary losses ride at the top level (NOT inside
        # `diagnostics`, whose values train.py logs): forward() pops them and
        # folds them into the total loss.
        if use_compact and meanflow_losses:
            out["meanflow_losses"] = meanflow_losses
        if use_compact and teacher_kv is not None and am_layers > 0:
            out["attn_match_loss"] = am_total / am_layers

        return out, kv_layers

    def _topk_kl_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor | None = None,
        answer_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward KL on top-k teacher tokens, gold answer forced into support.

        teacher_logits: (B, T, V)
        student_logits: (B, T, V)
        answer_mask: (B, T) boolean, True at answer positions. If None, use all positions.

        Both teacher and student distributions are renormalized over the same top-k
        support set, so the KL is well-defined as a distribution over the restricted support.
        """
        B, T, V = teacher_logits.shape
        tau = self.kl_temperature

        # Derive an answer mask from labels (-100 = ignore) when not given.
        if answer_mask is None and labels is not None:
            answer_mask = labels != -100

        with torch.no_grad():
            # Top-k teacher tokens per position (using temperature-softened distribution for selection).
            topk_vals, topk_indices = (teacher_logits / tau).float().topk(self.kl_topk, dim=-1)
            # Force gold answer token into the support ONLY where it is absent.
            # The old unconditional splice (drop slot k, append gold) duplicated
            # gold whenever it already sat in the top-(k-1), which double-
            # counted it in the renormalized support (its teacher probability
            # was gathered twice, inflating it to ~2p/(1+p)) and dropped the
            # true rank-k token from the supervision. Clamp -100 (ignore) to 0
            # so negative-index wrapping does not corrupt the gather; those
            # positions are zeroed out by answer_mask below.
            if labels is not None:
                gold = labels.clamp_min(0).unsqueeze(-1)
                gold_in_support = (topk_indices == gold).any(dim=-1, keepdim=True)
                spliced = torch.cat([topk_indices[:, :, :-1], gold], dim=-1)
                topk_indices = torch.where(gold_in_support, topk_indices, spliced)

            # Teacher distribution restricted to top-k support, renormalized.
            teacher_probs = torch.gather(
                (teacher_logits / tau).float().softmax(dim=-1), -1, topk_indices
            )
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Student distribution restricted to the SAME top-k support, renormalized.
        student_probs = torch.gather(
            (student_logits / tau).float().softmax(dim=-1), -1, topk_indices
        )
        student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        student_log_probs = torch.log(student_probs.clamp_min(1e-8))

        # KL(teacher || student) = sum teacher * (log teacher - log student).
        kl = (teacher_probs * (torch.log(teacher_probs.clamp_min(1e-8)) - student_log_probs)).sum(dim=-1)

        if answer_mask is not None:
            kl = kl * answer_mask.float()
            count = answer_mask.float().sum().clamp_min(1.0)
        else:
            count = float(B * T)

        return kl.sum() / count

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        answer_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass: teacher (full cache) + student (compact cache) + KL loss.

        During base warmup (step < compact_from_step): single forward pass, CE
        loss only, base frozen (validated at construction: base_warmup_steps
        must not exceed compact_from_step, or the dual-pass phase would start
        with a still-trainable base).
        After warmup: dual teacher/student pass, KL distillation + optional CE,
        + optional attention-match (still_attn_match_weight > 0) and MeanFlow
        self-consistency (still_meanflow_loss_weight > 0) terms.
        """
        # Phase 1: base model warmup (single pass, CE only).
        if not self.compact_active:
            out = self.base_model(input_ids, labels=labels)
            diagnostics = out["diagnostics"]
            diagnostics["kl_loss"] = 0.0
            # Diagnostics are kept as detached 0-d tensors, never floats: a
            # float() here forces a host sync on every micro-step (8x per
            # optimizer step at accum=8) for a value only read at log time, and
            # it graph-breaks the compiled region. train.py converts at logging.
            diagnostics["student_loss"] = out["loss"].detach() if out["loss"] is not None else 0.0
            diagnostics["teacher_loss"] = 0.0
            diagnostics["compact_active"] = False
            diagnostics["phase"] = "base_warmup"
            return out

        # Phase 2: compactor training (dual pass, KL distillation).
        # Teacher pass (no grad). When the attention-match arm is active, the
        # teacher pass runs through _extract_kv_and_forward so the per-layer KV
        # is captured in the SAME pass; before, that cost a second full base
        # forward purely to extract KV. Otherwise the base model's own
        # optimized forward is used (cheaper: per-layer KV dies with each layer
        # instead of being retained for the whole student pass).
        need_teacher_kv = self.attn_match_weight > 0
        with torch.no_grad():
            if need_teacher_kv:
                teacher_out, teacher_kv = self._extract_kv_and_forward(
                    input_ids, labels=labels, use_compact=False
                )
            else:
                teacher_kv = None
                # fused_linear_ce (S14-5b) is train-gated, and under it the
                # base forward materializes NO logits (base.py returns
                # logits=None so the (B*T, vocab) tensor is never built) — but
                # the teacher's logits are exactly what the KL distillation
                # below consumes, and _topk_kl_loss crashed on
                # teacher_logits.shape. Force the eager head for the teacher
                # call only: this pass is no-grad (the fused path saves nothing
                # here), the flag is restored immediately after, and the
                # student loss path never goes through base_model.forward.
                fused_prev = getattr(self.base_model, "fused_linear_ce", False)
                self.base_model.fused_linear_ce = False
                try:
                    teacher_out = self.base_model(input_ids, labels=labels)
                finally:
                    self.base_model.fused_linear_ce = fused_prev
            teacher_logits = teacher_out["logits"]

        # Student pass (with compact cache). When teacher KV was captured, the
        # attention-pattern match loss is computed inline on this pass's own
        # suffix queries / compact keys (previously a THIRD full base forward
        # that re-derived q/k/v and re-ran every compactor with grad — ~2x the
        # arm's intended cost).
        student_out, _ = self._extract_kv_and_forward(
            input_ids, labels=labels, use_compact=True, teacher_kv=teacher_kv
        )
        student_logits = student_out["logits"]

        # Attention-matching loss (optional), computed inside the student pass;
        # the result tensor retains its grad graph and is added to the loss below.
        attn_match_val = 0.0
        am_loss = student_out.pop("attn_match_loss", None)
        if am_loss is not None:
            attn_match_val = am_loss.detach()

        # Restrict the loss to positions that can actually differ from the
        # teacher. Prefix positions run identical local attention through the
        # same frozen base in both passes, so their KL is exactly zero and their
        # CE is a constant shared by every arm and every seed. Averaging over
        # them divides the KL by seq_len while only the suffix contributes —
        # at 1024/64 that is a silent 16x reduction of still_kl_weight.
        suffix_mask = None
        if self.loss_positions == "suffix":
            suffix_mask = self._suffix_mask(input_ids.shape[0], input_ids.shape[1], input_ids.device)
            if answer_mask is not None:
                suffix_mask = suffix_mask & answer_mask

        # Build total loss.
        loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        kl_val = 0.0
        if self.kl_weight > 0:
            kl = self._topk_kl_loss(
                teacher_logits,
                student_logits,
                labels,
                suffix_mask if suffix_mask is not None else answer_mask,
            )
            loss = loss + self.kl_weight * kl
            kl_val = kl.detach()

        if am_loss is not None:
            loss = loss + self.attn_match_weight * am_loss

        # MeanFlow self-consistency loss (only exists for the still_meanflow
        # arm). still_meanflow_steps always paid for the extra no-grad Euler
        # rollout + one grad net call per compactor per layer, but the loss
        # was previously discarded here; it enters the objective only when
        # still_meanflow_loss_weight > 0 (0.0 reproduces every prior run).
        meanflow_losses = student_out.pop("meanflow_losses", None)
        mf_loss = torch.stack(meanflow_losses).mean() if meanflow_losses else None
        mf_val = 0.0
        if mf_loss is not None:
            if self.meanflow_loss_weight > 0:
                loss = loss + self.meanflow_loss_weight * mf_loss
            mf_val = mf_loss.detach()

        # Optional CE loss (direct supervision).
        if self.ce_weight > 0 and labels is not None:
            if suffix_mask is None:
                ce = F.cross_entropy(
                    student_logits[:, :-1].reshape(-1, student_logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                )
            else:
                # logits at position i predict the token at i+1, so the mask for
                # the *prediction targets* is suffix_mask shifted left by one.
                per_pos = F.cross_entropy(
                    student_logits[:, :-1].reshape(-1, student_logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    reduction="none",
                ).view(student_logits.shape[0], -1)
                target_mask = suffix_mask[:, 1:].to(per_pos.dtype)
                ce = (per_pos * target_mask).sum() / target_mask.sum().clamp_min(1.0)
            loss = loss + self.ce_weight * ce

        diagnostics = student_out["diagnostics"]
        if self.training:
            diagnostics["kl_loss"] = kl_val
            diagnostics["attn_match_loss"] = attn_match_val
            if mf_loss is not None:
                diagnostics["meanflow_loss"] = mf_val
        diagnostics["student_loss"] = student_out["loss"].detach() if student_out["loss"] is not None else 0.0
        diagnostics["teacher_loss"] = teacher_out["loss"].detach() if teacher_out["loss"] is not None else 0.0
        if self._config_dict is None:
            self._config_dict = asdict(self.config)
        diagnostics["config"] = self._config_dict
        diagnostics["compact_active"] = True
        diagnostics["phase"] = "compactor_training"
        # Make the geometry visible: what fraction of positions the loss is
        # actually computed on. If this is small, the effective sample size for
        # comparing arms is correspondingly smaller than the token count.
        seq_len = input_ids.shape[1]
        diagnostics["loss_position_frac"] = (seq_len - self.suffix_start(seq_len)) / seq_len

        return {"logits": student_logits, "loss": loss, "diagnostics": diagnostics}

    def compact_only(
        self,
        input_ids: torch.Tensor,
    ) -> dict[str, Any]:
        """Run the base model with compacted KV cache only (inference mode).

        This is the deployment path: full prefill -> compact -> decode.
        """
        return self._extract_kv_and_forward(input_ids, use_compact=True)[0]


def build_still_model(config: ModelConfig) -> StillLM:
    """Build a StillLM from a config.

    Config fields consumed (in model config):
    - All standard fields for the base model (variant, d_model, num_heads, etc.)
    - still_compact_len: number of compact KV entries (default 32)
    - still_num_blocks: compactor blocks (default 2)
    - still_d_latent: latent dimension (default 2 * head_dim)
    - still_use_ot_read: use OT-coupled cross-attention (default False)
    - still_use_energy_read: use energy-based read (default False)
    - still_use_freq_decay: use frequency decay on compact entries (default False)
    - still_kl_topk: top-k for KL support (default 200)
    - still_kl_weight: KL loss weight (default 1.0)
    - still_ce_weight: CE loss weight (default 0.0)
    - still_compact_from_step: step to start compaction (default 0)
    - still_kl_temperature: temperature for KL distillation (default 1.0)
    - still_meanflow_loss_weight: weight of the MeanFlow self-consistency loss
      (default 0.0 = legacy discard; > 0 finally trains the objective that
      still_meanflow_steps always computed but previously never consumed)
    - still_ot_reg_weight: BROKEN/REMOVED — > 0 raises (the arm never had any
      effect; see ModelConfig.still_ot_reg_weight)
    - still_pretrained_base: path to a pretrained base model checkpoint (optional)
    """
    import os

    from flower.models import build_model

    # StillLM re-implements the base block loop in order to intercept the KV
    # cache, and that re-implementation has no depth-routing path. Silently
    # dropping AttnRes would make the Still student disagree with its own base,
    # so refuse the combination rather than produce a wrong teacher.
    if getattr(config, "attn_res", "none") != "none":
        raise ValueError(
            "attn_res is not supported under the still_* variants: StillLM "
            "re-implements the block loop for KV interception and would skip the "
            "depth router. Run AttnRes as a separate vanilla-base probe."
        )

    # Build the base model (always vanilla_local for Still experiments).
    base_config = ModelConfig(**{**config.__dict__})
    base_config.variant = "vanilla_local"
    base_model = build_model(base_config)

    # Optionally load a pretrained base model checkpoint.
    pretrained_path = getattr(config, "still_pretrained_base", None)
    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"[still] still_pretrained_base set to {pretrained_path!r} but the "
                "file does not exist. Phase-1 runs require the phase-0 base "
                "checkpoint to be present (on vast: upload it separately — "
                "make_repo_archive excludes runs/ and *.pt). Refusing to "
                "silently train against a random-init base."
            )
        payload = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        # Strip "base_model." prefix if present (from a previous StillLM checkpoint).
        cleaned = {k.replace("base_model.", ""): v for k, v in state.items()}
        # S14 Opportunity 2 Part A: if the base is a bloom_memory checkpoint from
        # before the hash_weights refactor, remap its legacy hashes.{i}.weight
        # keys. No-op for non-bloom / new-format state_dicts.
        from flower.models.bloom_memory import remap_legacy_bloom_state_dict
        from flower.models.memory import remap_legacy_mha_state_dict

        cleaned = remap_legacy_bloom_state_dict(cleaned)
        # S14 Opportunity: summary_memory / bloom_memory replaced their
        # nn.MultiheadAttention perceiver with a compile-clean SDPCrossAttention.
        # Remap legacy MHA in_proj_*/out_proj.* keys to the new q/k/v/out_proj
        # layout. `bias` from config (nn.MultiheadAttention always had bias;
        # SDPCrossAttention respects use_bias). No-op for new-format state_dicts.
        cleaned = remap_legacy_mha_state_dict(cleaned, bias=getattr(config, "use_bias", True))
        missing, unexpected = base_model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[still] base model missing keys: {len(missing)}")
        if unexpected:
            print(f"[still] base model unexpected keys: {len(unexpected)}")
        print(f"[still] loaded pretrained base from {pretrained_path}")

    compact_len = getattr(config, "still_compact_len", 32)
    num_blocks = getattr(config, "still_num_blocks", 2)
    d_latent = getattr(config, "still_d_latent", None)
    use_ot = getattr(config, "still_use_ot_read", False)
    use_energy = getattr(config, "still_use_energy_read", False)
    use_freq = getattr(config, "still_use_freq_decay", False)
    use_spectral = getattr(config, "still_use_spectral", False)
    layer_adaptive = getattr(config, "still_layer_adaptive", False)
    attn_match_weight = getattr(config, "still_attn_match_weight", 0.0)
    ot_reg_weight = getattr(config, "still_ot_reg_weight", 0.0)
    flow_steps = getattr(config, "still_flow_steps", 0)
    meanflow_steps = getattr(config, "still_meanflow_steps", 0)
    meanflow_loss_weight = getattr(config, "still_meanflow_loss_weight", 0.0)
    flow_ot = config.variant in ("still_flow_ot",)
    flow_kv = config.variant in ("still_flow_kv",)
    ot_epsilon = getattr(config, "still_ot_epsilon", 0.1)
    ot_iters = getattr(config, "still_ot_iters", 10)
    key_vel_hidden = getattr(config, "still_key_velocity_hidden", None)
    val_vel_hidden = getattr(config, "still_val_velocity_hidden", None)
    kl_topk = getattr(config, "still_kl_topk", 200)
    kl_weight = getattr(config, "still_kl_weight", 1.0)
    ce_weight = getattr(config, "still_ce_weight", 0.0)
    compact_from = getattr(config, "still_compact_from_step", 0)
    kl_temp = getattr(config, "still_kl_temperature", 1.0)
    base_warmup = getattr(config, "still_base_warmup_steps", 0)

    return StillLM(
        base_model=base_model,
        config=config,
        compact_len=compact_len,
        num_blocks=num_blocks,
        d_latent=d_latent,
        use_ot_read=use_ot,
        use_energy_read=use_energy,
        use_freq_decay=use_freq,
        use_spectral=use_spectral,
        layer_adaptive=layer_adaptive,
        attn_match_weight=attn_match_weight,
        ot_reg_weight=ot_reg_weight,
        flow_steps=flow_steps,
        meanflow_steps=meanflow_steps,
        meanflow_loss_weight=meanflow_loss_weight,
        flow_ot=flow_ot,
        flow_kv=flow_kv,
        ot_epsilon=ot_epsilon,
        ot_iters=ot_iters,
        key_velocity_hidden=key_vel_hidden,
        val_velocity_hidden=val_vel_hidden,
        kl_topk=kl_topk,
        kl_weight=kl_weight,
        ce_weight=ce_weight,
        compact_from_step=compact_from,
        kl_temperature=kl_temp,
        base_warmup_steps=base_warmup,
    )
