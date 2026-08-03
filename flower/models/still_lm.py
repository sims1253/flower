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
    StillCompactorOTReg,
    StillCompactorSpectral,
)
from flower.models.still_flow2 import StillCompactorFlowOT, StillCompactorMeanFlow
from flower.models.still_flow3 import StillCompactorFlowKV


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
        self.ot_reg_weight = ot_reg_weight
        self.flow_steps = flow_steps
        self.meanflow_steps = meanflow_steps
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
        elif ot_reg_weight > 0:
            compactor_cls = StillCompactorOTReg
        else:
            compactor_cls = StillCompactor

        # Compute per-layer compact lengths (PyramidKV pattern if layer_adaptive).
        if layer_adaptive and num_layers > 1:
            layer_compact_lens = _pyramid_budget(compact_len, num_layers)
        else:
            layer_compact_lens = [compact_len] * num_layers

        self.layer_compact_lens = layer_compact_lens
        self.compactors = nn.ModuleList()
        for i in range(num_layers):
            kwargs = dict(
                num_kv_heads=num_heads,
                head_dim=head_dim,
                compact_len=layer_compact_lens[i],
                num_blocks=num_blocks,
                d_latent=layer_d_latents[i],
                # Identity init is only valid when d_latent == 2*head_dim (the
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
    ) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
        """Forward pass through the base model, optionally with compacted KV cache.

        For the teacher pass: standard forward, extract per-layer K, V.
        For the student pass: compact K, V, then forward with compact cache.

        Returns (model_output, list_of_kv_dicts) where each kv dict has
        'keys', 'values', 'positions' per layer.
        """
        # We implement a custom forward that intercepts K, V at each layer.
        # This requires modifying the attention computation, which we do
        # by temporarily patching the attention forward.

        T = input_ids.shape[1]
        device = input_ids.device

        # Collect KV from each layer during the forward pass.
        kv_layers: list[dict[str, torch.Tensor]] = []

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

                # Suffix queries attend to [compact ; suffix_kv] with a causal mask:
                # all compact entries visible (strictly past), suffix lower-triangular.
                S = T_inner - ctx_end
                suffix_q = q_rot[:, :, ctx_end:, :]
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
                # Standard attention path.
                from flower.models.base import causal_mask

                seq_len = x.shape[1]
                keep = causal_mask(seq_len, device, attn.local_window).view(1, 1, seq_len, seq_len)
                out = F.scaled_dot_product_attention(q_rot, k_rot, v_h, attn_mask=keep)
                out = out.transpose(1, 2).contiguous().view(x.shape)
                attn_output = attn.out(out)

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
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

        diagnostics: dict[str, Any] = {
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "base_parameter_count": sum(p.numel() for p in self.base_model.parameters()),
            "compactor_parameter_count": sum(p.numel() for p in self.compactors.parameters()),
        }

        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}, kv_layers

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
            # Force gold answer token into the support.  Clamp -100 (ignore) to 0
            # so negative-index wrapping does not corrupt the gather; those
            # positions are zeroed out by answer_mask below.
            if labels is not None:
                gold = labels.clamp_min(0).unsqueeze(-1)
                topk_indices = torch.cat([topk_indices[:, :, :-1], gold], dim=-1)

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

    def _attention_match_loss(
        self,
        teacher_kv: list[dict[str, torch.Tensor]],
        student_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-pattern matching loss (Fast KV Compaction, arXiv:2602.16284).

        Instead of matching output logits (KL), directly match the attention patterns
        between teacher (full KV) and student (compact KV) at each layer. This is
        cheaper and more direct: we preserve the attention structure itself.

        teacher_kv: list of per-layer dicts with 'keys', 'values' from the teacher pass.
        Returns: MSE between teacher and student attention score matrices.
        """
        T = student_input_ids.shape[1]
        device = student_input_ids.device
        x = self.base_model.token(student_input_ids)
        positions = torch.arange(T, device=device, dtype=torch.float32)

        total_loss = torch.tensor(0.0, device=device)
        n_layers = 0

        for layer_idx, block in enumerate(self.base_model.blocks):
            attn = getattr(block, "attn", None) or getattr(block, "local", None)
            if attn is None:
                x = x + block.ff(block.ln2(x))
                continue

            ln_out = block.ln1(x)
            # Same rationale as _extract_kv_and_forward: go through the module's
            # own projection pipeline rather than re-deriving it here.
            q_rot, k_rot, v_h = attn.qkv_heads(ln_out)
            B_inner, num_heads, T_inner, head_dim = q_rot.shape
            D = num_heads * head_dim

            # Causal prefix/suffix split (same as _extract_kv_and_forward).
            compactor = self.compactors[layer_idx]
            t_compact = compactor.compact_len
            ctx_end = self.suffix_start(T_inner)
            S = T_inner - ctx_end
            suffix_q = q_rot[:, :, ctx_end:, :]

            # Teacher attention: suffix queries vs prefix teacher keys (no grad).
            t_keys_prefix = teacher_kv[layer_idx]["keys"][:, :, :ctx_end, :].detach()
            t_scores_prefix = (
                suffix_q.float() @ t_keys_prefix.float().transpose(-2, -1)
            ) / math.sqrt(head_dim)

            # Student attention pattern with compact prefix KV.
            if self.compact_active and layer_idx < len(self.compactors):
                result = compactor(
                    k_rot[:, :, :ctx_end, :], v_h[:, :, :ctx_end, :],
                    positions=positions[:ctx_end], return_compact_cache=True,
                )
                compact_k = result["compact_keys"]
                compact_v = result["compact_values"]
                s_scores = (
                    suffix_q.float() @ compact_k.float().transpose(-2, -1)
                ) / math.sqrt(head_dim)

                # Match teacher (ctx_end keys) vs student (t_compact keys) by binning.
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

                attn_loss = F.mse_loss(s_scores, t_match.detach())
                total_loss = total_loss + attn_loss
                n_layers += 1

                # Forward with prefix/suffix split (causal).
                all_k = torch.cat([compact_k, k_rot[:, :, ctx_end:, :]], dim=2)
                all_v = torch.cat([compact_v, v_h[:, :, ctx_end:, :]], dim=2)
                compact_visible = torch.ones(1, 1, S, t_compact, device=device, dtype=torch.bool)
                suffix_causal = _suffix_causal_mask(S, device, attn.local_window)
                suffix_mask = torch.cat(
                    [compact_visible, suffix_causal.unsqueeze(0).unsqueeze(0)], dim=3
                )
                suffix_out = F.scaled_dot_product_attention(suffix_q, all_k, all_v, attn_mask=suffix_mask)

                from flower.models.base import causal_mask
                prefix_keep = causal_mask(ctx_end, device, attn.local_window).view(1, 1, ctx_end, ctx_end)
                prefix_out = F.scaled_dot_product_attention(
                    q_rot[:, :, :ctx_end, :], k_rot[:, :, :ctx_end, :], v_h[:, :, :ctx_end, :], attn_mask=prefix_keep
                )

                out = torch.cat([prefix_out, suffix_out], dim=2)
                out = out.transpose(1, 2).contiguous().view(B_inner, T_inner, D)
                attn_output = attn.out(out)
            else:
                from flower.models.base import causal_mask
                seq_len = x.shape[1]
                keep = causal_mask(seq_len, device, attn.local_window).view(1, 1, seq_len, seq_len)
                out = F.scaled_dot_product_attention(q_rot, k_rot, v_h, attn_mask=keep)
                out = out.transpose(1, 2).contiguous().view(x.shape)
                attn_output = attn.out(out)

            x = x + attn_output
            x = x + block.ff(block.ln2(x))

        if n_layers > 0:
            return total_loss / n_layers
        return total_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        answer_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass: teacher (full cache) + student (compact cache) + KL loss.

        During base warmup (step < base_warmup_steps): single forward pass, CE loss only.
        After warmup: dual teacher/student pass, KL distillation + optional CE.
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
        # Teacher pass: use the base model's optimized forward (no grad).
        with torch.no_grad():
            teacher_out = self.base_model(input_ids, labels=labels)
            teacher_logits = teacher_out["logits"]

        # Attention-matching loss (optional). Computed once — the result tensor
        # retains its grad graph and is added to the loss below.
        attn_match_val = 0.0
        am_loss = None
        if self.attn_match_weight > 0:
            _, teacher_kv = self._extract_kv_and_forward(input_ids, use_compact=False)
            am_loss = self._attention_match_loss(teacher_kv, input_ids)
            attn_match_val = am_loss.detach()

        # Student pass (with compact cache).
        student_out, _ = self._extract_kv_and_forward(
            input_ids, labels=labels, use_compact=True
        )
        student_logits = student_out["logits"]

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
