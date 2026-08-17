from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    variant: str = "vanilla_local"
    vocab_size: int = 50257
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 1024
    max_seq_len: int = 512
    local_window: int | None = 64
    # Sweep 4: RoPE base frequency. Keep 10000 for the Phase 1 bake-off; larger
    # values can support post-training long-context extension via YaRN/PI.
    rope_base: float = 10000.0
    memory_slots: int = 8
    flow_steps: int = 3
    flow_shared: bool = True
    flow_mode: str = "euler"  # euler, direct, shortcut
    flow_step_size: float = 1.0
    memory_aggregation: str = "mean"  # sum, mean, max, attention, orthogonal
    summary_style: str = "deepsets"  # deepsets, perceiver
    hierarchical_memory: bool = False
    short_memory_slots: int = 4
    memory_kernel_bias: str = "none"  # none, positional, rbf
    memory_update_frequency: int = 1
    deep_shallow_flow: bool = False
    noise_std: float = 0.0
    dropout: float = 0.0
    # Sweep 2 additions
    loop_count: int = 1  # A2: loop blocks K times with shared memory state across loops
    local_attn_kernel_bias: str = "none"  # none, rbf — additive RBF bias on local self-attention
    local_attn_rbf_scale: float = 4.0  # initial scale for local-attention RBF (positions are 0..1 normalised)
    phase_memory_dim: int = 64  # A5: per-slot complex matrix is (phase_memory_dim x phase_memory_dim)
    orthogonal_eps: float = 1e-6  # A1: numerical floor for orthogonal projection
    num_memory_banks: int = 1  # I4: when >1, partition memory into N independent banks with a learned router
    bank_router_temperature: float = 1.0  # I4: softmax temperature for bank routing
    # Sweep 4: hashed n-gram residual memory. ngrams are hashed into a learned
    # table and added to the hidden state inside each block.
    engram_table_size: int = 8192
    engram_ngram_min: int = 2
    engram_ngram_max: int = 3
    engram_scale: float = 0.1
    # Sweep 4 (flow_meanflow): MeanFlow average-velocity parameterisation +
    # optional OT-CFM batch coupling. Loss weight scales the auxiliary regression
    # loss vs. the main cross-entropy. OT-CFM is off by default because it adds
    # O(B^2) coupling work; enable it explicitly for the OT-CFM ablation.
    meanflow_loss_weight: float = 0.1
    meanflow_ot_cfm: bool = False
    meanflow_ot_epsilon: float = 0.05
    meanflow_ot_iters: int = 20
    # Sweep 4 (flow_pma): number of coupling layers in the per-slot transport flow.
    flow_pma_layers: int = 2
    # Sweep 5 (flow_ot_memory): OT-coupled flow-matched memory writes.
    # `ot_source_points` is the number of perceiver-summary points routed to slots
    # via a Sinkhorn coupling. Smaller = cheaper / coarser write; larger = finer.
    ot_source_points: int = 16
    ot_epsilon: float = 0.1
    ot_iters: int = 10
    # Sweep 5 (bloom_memory): K learned hash projections; each routes summary
    # points to slots via a soft top-k (temperature-controlled softmax).
    # Smaller temperature = sharper hash (more Bloom-like, more aliasing risk).
    bloom_num_hashes: int = 4
    bloom_temperature: float = 0.5
    bloom_summary_points: int = 16
    # Sweep 5 (frequency_decay_memory): per-slot inverse spaced repetition.
    # Slots that receive heavy writes across layers get decayed harder so the
    # memory module preferentially preserves rarely-written-but-distinctive
    # content. `freq_penalty` scales the magnitude->decay coupling; the base
    # decay is learned per-slot.
    freq_penalty: float = 1.0
    freq_decay_init: float = -2.0  # sigmoid(-2) ~ 0.12 base retention loss
    # Sweep 5 (surprise_memory): cheap LM-loss-style surprise gating without
    # 2nd-order autograd. Each block has a small judge net that emits a per-token
    # surprise scalar; high-surprise tokens dominate the write summary AND the
    # global write gate is modulated by their mean. Judge is trained end-to-end
    # via the LM CE through the gate.
    surprise_judge_dim: int = 64
    surprise_scale: float = 1.0
    # Sweep 6 (hamiltonian_attention): SympNet-style symplectic Q/K flow.
    # Q/K vectors are split into (q, p) halves; the flow is a stack of
    # gradient shears whose Jacobians are exact Hessians, making it symplectic
    # by construction (volume-preserving, bounded shadow-H drift).
    # `hamiltonian_mass` = "diagonal" learns a per-coord kinetic mass (Nutpie
    # diagonal preconditioner analogue). `hamiltonian_walnuts` enables
    # per-macro-step adaptive dt gated by explicit energy error.
    hamiltonian_mass: str = "isotropic"  # "isotropic" or "diagonal"
    hamiltonian_walnuts: bool = False
    hamiltonian_energy_threshold: float = 0.05
    hamiltonian_max_subdivisions: int = 2
    # Sweep 7 (B2): energy-based memory read. Replaces the convex softmax weighted
    # average in MemoryRead with a log-sum-exp read controlled by a learnable
    # inverse temperature β. β=1 at init (near mean-pool); grows → hard-max.
    energy_read: bool = False
    energy_beta_init: float = 1.0
    # Still: amortized KV-cache compaction (arXiv:2606.07878)
    still_compact_len: int = 32
    still_num_blocks: int = 2
    still_d_latent: int | None = None
    still_use_ot_read: bool = False
    still_use_energy_read: bool = False
    still_use_freq_decay: bool = False
    still_use_spectral: bool = False
    still_layer_adaptive: bool = False
    still_attn_match_weight: float = 0.0
    still_ot_reg_weight: float = 0.0
    still_flow_steps: int = 0
    still_meanflow_steps: int = 0
    still_ot_epsilon: float = 0.1
    still_ot_iters: int = 10
    still_key_velocity_hidden: int | None = None
    still_val_velocity_hidden: int | None = None
    still_velocity_hidden: int | None = None
    still_d_latent_schedule: list[int] | None = None  # per-layer d_latent; None = uniform (current)
    still_kl_topk: int = 200
    still_kl_weight: float = 1.0
    still_ce_weight: float = 0.0
    still_compact_from_step: int = 0
    still_kl_temperature: float = 1.0
    still_pretrained_base: str | None = None
    still_base_warmup_steps: int = 0
    # ------------------------------------------------------------------
    # Still loss geometry.
    #
    # The student compacts the prefix [0:ctx_end) and only queries in
    # [ctx_end:T) ever see the compact cache. Prefix positions run plain local
    # attention through the same frozen base in both passes, so they are
    # bit-identical to the teacher and carry no signal at all.
    #
    # `still_suffix_len` sets that split directly. None reproduces the legacy
    # coupling ctx_end = T - compact_len, which ties the number of *evaluated*
    # positions to the *compaction budget* — at seq 1024 / compact_len 64 that
    # is 64 positions, 6.25% of the sequence.
    #
    # `still_loss_positions` controls the loss reduction:
    #   all    -> legacy. KL is summed over all T positions and divided by T,
    #             even though only the suffix contributes. At 1024/64 that
    #             scales the KL term down by 16x, so still_kl_weight: 1.0 is
    #             effectively 0.0625 against the CE term. CE likewise averages
    #             over a prefix that is a frozen-base constant across all arms.
    #   suffix -> KL and CE are both restricted to the positions that can
    #             actually differ, so the configured weights mean what they say.
    # ------------------------------------------------------------------
    still_suffix_len: int | None = None
    still_loss_positions: str = "all"  # all | suffix
    # ------------------------------------------------------------------
    # Sweep 13 (next-gen base): architecture modernisation.
    #
    # Every field below defaults to the legacy GPT-2-style behaviour so that
    # pre-sweep-13 configs reproduce exactly. Opt in explicitly per config.
    # ------------------------------------------------------------------
    norm_type: str = "layernorm"  # layernorm | rmsnorm
    ffn_activation: str = "gelu"  # gelu | swiglu
    # SwiGLU uses three d x h projections vs GELU's two, so a literal `ffn_dim`
    # swap would silently add ~50% FFN params. With ffn_param_match the hidden
    # width is scaled to 2/3 * ffn_dim (rounded to a multiple of 64) to keep the
    # parameter count comparable across the activation A/B.
    ffn_param_match: bool = True
    # Per-layer FFN width (TLM-style layer-wise parameter allocation). None =
    # uniform `ffn_dim` everywhere. Length must equal num_layers. Keep the sum
    # equal to num_layers * ffn_dim for a budget-preserving comparison — the
    # published result is about the direction of allocation, not the amount.
    ffn_dim_schedule: list[int] | None = None
    # Per-layer attention window (hybrid sliding-window attention). None = every
    # layer uses `local_window`, which is the current behaviour. In a schedule,
    # `null` means that layer attends over the FULL context.
    #
    # `vanilla_local` currently sets local_window=256 on all 14 layers at
    # seq 1024, so no layer ever sees the whole sequence — information can only
    # cross the context by hopping up through depth. Production hybrids
    # interleave a few full-attention layers among the windowed ones (MiMo-v2.5
    # 6:1 at window 128, Laguna-XS 3:1 at window 512) to get near-linear cost
    # while keeping genuine global reach. Length must equal num_layers.
    attn_window_schedule: list[int | None] | None = None
    # RMSNorm on Q and K before RoPE. Muon's full-rank updates inflate the
    # spectral norms of W_Q/W_K and QK^T multiplies them, which is the
    # MaxLogit-explosion failure mode (see concepts/qk-stability-under-muon).
    qk_norm: bool = False
    use_bias: bool = True  # biases on attention/FFN linears
    # Weight initialisation.
    #   torch  -> PyTorch defaults (legacy). nn.Embedding is N(0, 1), and it is
    #             tied to the LM head, so initial logits have std ~sqrt(d_model)
    #             and the initial loss is enormous (569 vs ln(4096)=8.3 at the
    #             102M config). Training burns its first phase just shrinking
    #             the embedding norm.
    #   scaled -> GPT-2 style: N(0, init_std) for Linear/Embedding, zero biases,
    #             and residual output projections scaled by 1/sqrt(2*num_layers)
    #             so the residual stream does not grow with depth.
    init_scheme: str = "torch"  # torch | scaled
    init_std: float = 0.02
    # ------------------------------------------------------------------
    # Depth-axis routing (AttnRes family). Orthogonal to KV compaction: Still
    # compacts along the sequence axis, this retrieves along the depth axis.
    #   delta_block -> Delta Block AttnRes (arXiv:2605.18855): route over
    #     block-level deltas (h_{b+1} - h_b) with additive routing. Plain
    #     cumulative-state AttnRes degrades at scale; the delta form does not.
    #   attn_res_key sliced -> S-LR-ATTNRES (arXiv:2607.09694): the routing key
    #     is the last `attn_res_rank` dims of the source value, so it costs no
    #     extra projection and no extra activation memory.
    # ------------------------------------------------------------------
    attn_res: str = "none"  # none | delta_block
    attn_res_blocks: int = 8
    attn_res_key: str = "full"  # full | sliced
    attn_res_rank: int = 64
    # ------------------------------------------------------------------
    # docs/training-speedups.md. Each knob defaults to the legacy
    # behaviour so published runs reproduce; opt in per config.
    # ------------------------------------------------------------------
    # S1: FlexAttention. Compiles the causal/local mask into the attention
    # kernel without materializing the T x T mask — required for seq=32K.
    # On CPU / when off, the SDPA path is used (flex needs CUDA to be fast
    # but does run, unfused, for tests).
    flex_attention: bool = False
    # S2: attention-window warmup. Linearly ramps every attention module's
    # local_window from attn_warmup_start to local_window over
    # attn_warmup_steps training steps. 0 disables (use local_window always).
    attn_warmup_start: int = 256
    attn_warmup_steps: int = 0
    # S2/S14-bugfix: quantise the warmup window to coarse steps. Each distinct
    # window value forces FlexAttention's create_block_mask to recompile (the
    # mask_mod closure captures the window), so a per-step ramp over N warmup
    # steps causes N recompiles and exhausts torch.compile's recompile limit
    # (default 8), after which flex falls back to the eager dense path and
    # throughput collapses. ``attn_warmup_quantize`` is the number of *steps*
    # between window changes: the ramp is divided into
    # ceil(warmup_steps / quantize) segments, each held at a constant window.
    # e.g. quantize=32 over a 256-step warmup = 8 segments = 8 recompiles
    # (under the limit). 0 = the legacy per-step ramp (a distinct window every
    # step). The final window is always local_window regardless of quantize.
    # The ramp's training-dynamics benefit (start narrow, widen) is preserved
    # — it jumps in `quantize`-step plateaus instead of 1-position steps.
    attn_warmup_quantize: int = 0
    # S3: FP8 matmul for the lm_head projection (Blackwell/Hopper, bf16 only).
    fp8_lm_head: bool = False
    # S4: compute cross-entropy in BF16 instead of FP32.
    bf16_cross_entropy: bool = False
    # S14-5b: Liger FusedLinearCrossEntropy. Fuses the tied lm_head matmul + CE
    # so the (B*T, vocab) logits tensor is never materialized during training —
    # the binding memory constraint at seq=8192/vocab=16K and the blocker for
    # seq=32K on the 32GB 5090. Training-time only; eval keeps the eager logits
    # path so FP8-head / logprob consumers are untouched. CUDA-only (Triton
    # kernel); the forward falls back to the eager path automatically when the
    # flag is on but CUDA is unavailable. False reproduces old runs exactly.
    fused_linear_ce: bool = False
    # S8: Multi-Token Prediction (final-runs only; changes the loss surface).
    # Predicts N extra future tokens with untied heads; aux losses weighted by
    # mtp_weight. 0 disables (standard next-token loss).
    mtp_extra_heads: int = 0
    mtp_weight: float = 0.5
    # FP8-attention quality probe (NOT a speedup — it is slightly slower than
    # bf16). Rounds Q/K/V to e4m3 in the attention forward with a
    # straight-through gradient, so the quality cost of FP8-precision attention
    # can be measured in an ordinary training run without first writing an FP8
    # attention backward. Simulates Q/K/V only, not the softmax probabilities,
    # so it is a LOWER BOUND on a real FP8 kernel's error. See
    # flower/models/base.py::_fake_quant_e4m3 and
    # flower/kernels/fp8_swa_attention.py.
    fp8_attention_sim: bool = False
    # S10: Smooth-SwiGLU. Per-channel scale on the `up` projection that makes
    # the SwiGLU multiply numerically equivalent but with a narrower dynamic
    # range, stabilising FP8/FP4 training. No-op for GELU FFNs.
    smooth_swiglu: bool = False
    # S14-checkpoint: activation checkpointing on the transformer blocks.
    # Wraps each block's forward in torch.utils.checkpoint during training so
    # activations are not retained for backward (recomputed from the block
    # inputs instead), cutting activation memory from O(num_layers) saved
    # tensors to O(1). Trades ~one extra forward per backward (~25-33% more
    # compute) for a large activation-memory reduction — the enabler for
    # seq=32K on the 32GB 5090 (Section 13). use_reentrant=False (the modern
    # recommended path) preserves the differentiable memory tensor that flows
    # between blocks and saves/restores RNG state so dropout is identical
    # between the two passes. No-op in eval mode and when False (old runs
    # reproduce exactly). AttnRes (depth_router) is incompatible: it reads
    # inter-block deltas, so checkpointing is skipped when depth_router is set.
    #
    # Values:
    #   False        -> off (default; old runs reproduce).
    #   True         -> full checkpointing: recompute EVERY activation in the
    #                   block during backward. Maximum memory saving, maximum
    #                   recompute cost (~25-33% throughput loss).
    #   "selective"  -> selective checkpointing: recompute only the large
    #                   activations (default byte-threshold policy targets the
    #                   FFN intermediate, the dominant tensor), keeping cheap
    #                   ones (residual stream, norms) materialized so they are
    #                   not recomputed. Recovers much of the throughput cost
    #                   while retaining most of the memory saving. Uses
    #                   torch.utils.checkpoint.create_selective_checkpoint_contexts
    #                   (torch 2.13) with a context_fn + use_reentrant=False;
    #                   the same RNG-state-save/restore and flex-compile
    #                   workarounds as full checkpointing apply.
    #   "ffn"        -> checkpoint ONLY the feed-forward sub-layer, keeping
    #                   attention (Q/K/V/O) materialized. The best throughput/
    #                   memory tradeoff at long context: FlashAttention/Flex
    #                   already recomputes the O(T^2) score matmul in backward
    #                   from saved Q,K,V, so the only extra cost full
    #                   checkpointing adds on the attention side is the cheap
    #                   qkv/out projections — but it also forces dropping Q,K,V
    #                   which are large at long T. Checkpointing only the FFN
    #                   (whose intermediate is the biggest single activation AND
    #                   whose matmuls are cheap to recompute) saves ~37% peak
    #                   memory for only ~10% throughput cost, vs full's ~50%
    #                   saving for ~100% cost. Measured at d768/L14/seq8192/b4:
    #                   none 15.3GB/185ms, full 3.3GB/373ms, ffn 9.7GB/203ms.
    #                   vanilla_local blocks only (memory-variant forwards are
    #                   left to full/selective; "ffn" falls back to "full" for
    #                   them with a warning).
    activation_checkpoint: bool | str = False
    # S12.2: orthogonal weight initialisation for 2D matrices (pairs with Muon).
    orthogonal_init: bool = False
    # S13: per-component precision routing (scaffolding). The actual FP4/FP8
    # matmul casting requires NVIDIA's transformer_engine and is NOT
    # implemented here — these fields are forward-looking config only and must
    # stay bf16 (memory_precision is bf16-locked by design: Flower's memory
    # write/read path needs high dynamic range).
    ffn_precision: str = "bf16"  # bf16 | fp8 | fp4
    attn_precision: str = "bf16"  # bf16 | fp8
    memory_precision: str = "bf16"  # bf16 only
    head_precision: str = "bf16"  # bf16 | fp8
    bf16_guard_blocks: int = 0
    # S14 Opportunity 3: analytical Titans surprise gradient (research
    # contribution — see NEXT_IDEAS.md §7). Replaces the per-step
    # torch.autograd.grad inner loop in TitansMACBlock._surprise_update with a
    # closed-form gradient of the associative-retrieval MSE w.r.t. memory slots.
    # Eliminates building/destroying an inner autograd graph every forward step
    # while keeping the outer CE gradient intact (every op is a standard
    # differentiable PyTorch op). The closed form is exact — it matches
    # torch.autograd.grad at fp32 ~1e-9 (gate: 1e-4), so the Titans write rule
    # is numerically unchanged and the alpha_logit/write_scale semantics are
    # preserved. False reproduces the legacy autograd path bit-for-bit (old
    # runs and checkpoints reproduce).
    titans_analytical_surprise: bool = False
    # Titans inner-loss reduction (train/eval consistency fix).
    #
    # THE BUG: the inner associative-retrieval MSE is reduced with a mean over
    # the batch AND feature dims (factor 2/(B*D) in the analytical surprise
    # path; the legacy autograd path's F.mse_loss default is the same B*D
    # mean). Titans memory writes are alpha * write_scale * surprise, so every
    # write scales with 1/B: measured at init, mean |memory| after block 0 is
    # ~7.8e-4 at B=1 vs ~4.7e-5 at B=16 — memory writes at batch 1 are ~16x
    # larger than anything a batch-16 training run ever produced. Training
    # runs at training.batch_size, but flower/eval.py's document-level paths
    # (evaluate_documents, sliding_window_document_loss) score one document at
    # a time (B=1), so titans doc-level bpb is computed with batch_size-x
    # larger memory writes than training, and the two eval paths disagree with
    # each other.
    #
    # False (default) keeps the legacy B-dependent reduction and reproduces
    # every published run bit-for-bit.
    #
    # True switches BOTH surprise paths (analytical and autograd) to a
    # per-sample reduction: sum over batch, mean over D only (factor 2/D).
    # Memory dynamics become batch-size-invariant and the B=1 document eval
    # sees the same write magnitudes as B=N training.
    #
    # Interaction with causal_memory: the causal write path is per-position
    # and already per-row (each (batch, position) row's surprise is the
    # gradient of its OWN MSE with factor 2/D — the per-position fix from the
    # causal-memory branch), so titans_per_sample_loss is a NO-OP when
    # causal_memory=True; it only affects the non-causal window-aggregated
    # write. Runs with this flag on are not comparable to runs with it off.
    titans_per_sample_loss: bool = False
    # ------------------------------------------------------------------
    # Causal memory writes (correctness fix).
    #
    # THE BUG: every memory variant's write path aggregates the ENTIRE window
    # — including tokens AFTER position t — into the memory bank (perceiver /
    # max / mean / softmax summaries over all of x), and the next layer's
    # mem_read broadcasts that bank to EVERY position. So logits[t] can depend
    # on input tokens > t: empirically, perturbing only the LAST input token
    # changes logits at positions 0..T-2 by up to 0.29 (linear_memory), 0.03
    # (frequency_decay), 0.015 (bloom), 0.005 (summary), ... while vanilla_local
    # is exactly 0. Because logits[t] predicts labels[t+1], a memory model can
    # read its own answer out of memory, which taints every memory-vs-vanilla
    # sweep comparison.
    #
    # False (default) keeps the legacy leaky write and reproduces every
    # existing run bit-for-bit — bake-off results published so far were
    # obtained with this behavior.
    #
    # True makes the memory visible at position t a function of tokens <= t
    # only (token t itself is allowed): each block computes a per-position
    # write from the layer input and accumulates it causally (running
    # mean/sum via cumsum, running max via cummax, routed/softmax writes via
    # masked cumulative sums), and the read at t consumes the per-position
    # memory state at t. No new parameters — checkpoints stay loadable and
    # param counts are unchanged. Runs trained with the flag off are NOT
    # comparable to runs with it on; rerun the bake-off with
    # causal_memory: true before drawing memory-mechanism conclusions.
    # ------------------------------------------------------------------
    causal_memory: bool = False

    def __post_init__(self) -> None:
        # S13: validate the precision-routing fields. Keep the actual FP4/FP8
        # matmul casting (which requires NVIDIA's transformer_engine) out of
        # scope; these fields are forward-looking config scaffolding only.
        if self.ffn_precision not in {"bf16", "fp8", "fp4"}:
            raise ValueError(f"ffn_precision must be bf16|fp8|fp4, got {self.ffn_precision!r}")
        if self.attn_precision not in {"bf16", "fp8"}:
            raise ValueError(f"attn_precision must be bf16|fp8, got {self.attn_precision!r}")
        if self.memory_precision != "bf16":
            raise ValueError(
                f"memory_precision must stay 'bf16' (Flower's memory write/read path "
                f"needs high dynamic range; low precision collapses routing decisions), "
                f"got {self.memory_precision!r}"
            )
        if self.head_precision not in {"bf16", "fp8"}:
            raise ValueError(f"head_precision must be bf16|fp8, got {self.head_precision!r}")
        # S14-checkpoint: activation_checkpoint accepts False / True / "selective".
        # bool|str validated here so a typo (e.g. "selectiv") fails loudly at
        # config load rather than silently behaving like "off" in forward.
        if self.activation_checkpoint not in {False, True, "selective", "ffn"}:
            raise ValueError(
                f"activation_checkpoint must be False, True, 'selective', or 'ffn', "
                f"got {self.activation_checkpoint!r}"
            )


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    tokenizer: str = "gpt2"
    split: str = "train"
    sequence_length: int = 512
    streaming: bool = True
    text_field: str = "text"
    synthetic_vocab_size: int = 256
    validation_split: str | None = None
    validation_seed: int = 4321
    # Sweep 7 (A1): evaluate at a longer context than training to put memory under
    # pressure. None = use sequence_length. May exceed model.max_seq_len (the RoPE
    # cache is extended lazily in that case).
    eval_seq_len: int | None = None
    # Mean UTF-8 bytes per token for `tokenizer`, measured on held-out text by
    # scripts/analyze_tokenizer.py. Setting it makes training-time validation
    # emit `val_bpb` alongside `val_perplexity`.
    #
    # This matters because **perplexity is not comparable across tokenizers**: a
    # coarser tokenizer packs more text into each token and is "predicting more"
    # per step, so its per-token loss is inflated for reasons that have nothing
    # to do with model quality. Any comparison of a 4k-vocab result against a
    # 16k-vocab result on ppl is meaningless. Bits-per-byte normalises by text
    # and is the correct cross-tokenizer metric. (flower/eval.py already
    # computes true per-document BPB for final numbers; this is the cheap
    # in-loop approximation.)
    bytes_per_token: float | None = None
    # DataLoader worker count for the *training* stream. The baseline profile
    # (docs/profiling/baseline_profile.md) reaches 52.2k tok/s on synthetic
    # tokens while the real 450M runs logged 41.9-44.8k — a ~17% gap that is
    # entirely tokenization throughput.
    #
    # NOT order-preserving: each worker holds its own streaming iterator and
    # shards documents by `islice(worker_id, None, num_workers)`, so the worker
    # count determines which documents land in which batch. Changing it
    # perturbs the training data order the same way a different data seed
    # would — the token *set* is unchanged, the interleaving is not. Runs at
    # different worker counts are therefore seed-comparable, not bit-comparable;
    # the 450M seed band (0.0004 BPB) is the yardstick for that. 2 is the
    # legacy default and reproduces the published runs.
    num_workers: int = 2
    prefetch_factor: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    steps: int = 30_000
    lr: float = 3e-4
    warmup_steps: int = 1000
    # constant | linear_warmup | wsd | cosine
    #
    # `linear_warmup` warms up and then holds flat forever — every sweep before
    # 13 trained with no decay phase at all. `wsd` (warmup-stable-decay, the
    # Delphi recipe) holds flat and then decays linearly to `lr_final_frac` over
    # the last `lr_decay_frac` of training; `cosine` decays from the end of
    # warmup. A decay tail is normally worth more val loss than any of the
    # architectural arms it is being used to compare.
    lr_schedule: str = "linear_warmup"
    lr_decay_frac: float = 0.2  # fraction of total steps spent decaying (wsd)
    lr_final_frac: float = 0.0  # final LR as a fraction of base LR
    grad_clip: float = 1.0
    eval_interval: int = 500
    validation_interval: int = 0
    validation_steps: int = 0
    checkpoint_interval: int = 5000
    save_checkpoints: bool = True  # set false to skip torch.save during sweeps and save disk
    output_dir: str = "runs"
    metrics_json: str | None = None
    log_backend: str = "tensorboard"
    device: str = "auto"
    seed: int = 0
    seeds: tuple[int, ...] = (0, 1, 2)
    composite_eval: bool = False
    composite_eval_interval: int = 0
    composite_eval_json: str | None = None
    find_unused_parameters: bool | None = None
    # ------------------------------------------------------------------
    # Sweep 13: precision and compilation.
    #
    # Everything before sweep 13 trained in fp32 eager — no autocast, no TF32,
    # no compile. Defaults stay fp32/off so old runs reproduce.
    #   fp32 -> unchanged (fp32 matmuls, highest precision)
    #   tf32 -> fp32 params, TF32 matmul kernels
    #   bf16 -> TF32 matmuls + bf16 autocast for the forward/backward
    # bf16 needs no GradScaler (unlike fp16); master weights stay fp32.
    # ------------------------------------------------------------------
    # Scalar-logging cadence, in steps. 0 = legacy behaviour (derive it from
    # `eval_interval`), which in the production configs means logging every 1000
    # steps — only 11 points in a 10k-step run, far too coarse to see a loss
    # spike or a gradient-norm excursion. Set this to 25-50 for any run whose
    # stability you intend to reason about.
    log_interval: int = 0
    precision: str = "fp32"  # fp32 | tf32 | bf16
    # ------------------------------------------------------------------
    # FP8 linear-layer training (torchao). Targets the cutlass GEMM bucket that
    # the baseline profile measures at 62.1% of kernel time. Requires
    # precision: bf16 (FP8 layers consume bf16 activations) and CUDA sm_89+.
    #
    # Measured at the 450M block shape on sm_120: tensorwise 1.39x, rowwise
    # 1.01x. Rowwise is safer numerically but buys nothing on this GPU, so
    # tensorwise is the default and the guardrails carry the risk — see
    # flower/precision.py. Pair with model.smooth_swiglu (S10).
    #
    # Off by default: FP8 changes numerics, so every published run reproduces.
    # ------------------------------------------------------------------
    fp8_linear: bool = False
    fp8_recipe: str = "tensorwise"  # tensorwise | rowwise
    # Number of transformer blocks at each end kept in bf16. The first block
    # sees raw embeddings and the last feeds the LM head; both carry the widest
    # activation ranges. Nemotron-H uses 4 at 28 layers; 1 at 20 layers is the
    # proportionate floor. 0 converts every block.
    fp8_keep_bf16_blocks: int = 1
    # cuBLASLt fast accumulation for the two *backward* FP8 GEMMs (dgrad,
    # wgrad). torchao's tensorwise recipe already fast-accums the forward GEMM;
    # the backward GEMMs default to fp32 accumulation, together ~2/3 of the
    # _scaled_mm bucket (28.1% of CUDA time in the fp8_stack profile). May be
    # a no-op on sm_120 — bench_arms decides. Changes numerics (lower-precision
    # K accumulation), so quality-screen before adopting.
    fp8_use_fast_accum: bool = False
    compile_model: bool = False
    compile_mode: str = "default"  # default | reduce-overhead | max-autotune
    # Sweep 2: optimizer choice. "adamw" = legacy default. "muon" = Muon for 2D weights + AdamW for 1D/embeddings/heads.
    # "aurora" = Aurora (Tilde Research): Muon with a joint Stiefel/row-oblique
    # projection that fixes neuron death on rectangular matrices (the FFN shapes).
    optimizer: str = "adamw"
    # Explicit weight decay for the AdamW group. 0.01 reproduces the torch
    # default that every previous sweep picked up implicitly.
    weight_decay: float = 0.01
    # Exclude embeddings and 1D params (norm gains, biases) from weight decay.
    # Decaying an embedding pulls rare-token rows toward zero in proportion to
    # how rarely they are updated, and decaying a norm gain fights the norm.
    # Default False reproduces the previous behaviour, where the implicit torch
    # AdamW default applied 0.01 to everything.
    weight_decay_exclude_embeddings: bool = False
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # Sweep 13: Newton-Schulz orthogonalisation schedule for Muon.
    #   quintic5 = 5x quintic (3.4445,-4.7750,2.0315), 15 matmuls (default, speedrun std)
    #   cubic5   = 5x cubic (1.5,-0.5,0), 10 matmuls (-33% orth compute, ~1e-3 val loss per arXiv:2606.00371)
    #   hybrid_v4 = 8x quintic + 2x stabilize (2,-1.5,0.5) pinning singular values to 1 (DeepSeek-V4)
    muon_ns_schedule: str = "quintic5"
    # Nesterov momentum on the Muon group. True reproduces every existing run
    # (it was hardcoded True and not configurable, so it could not be tested).
    # NVIDIA's large-scale optimizer study (arXiv:2607.20548, up to 100M-token
    # batches) reports Nesterov did not help for either Muon or SOAP and was
    # skipped for both — so this is worth an A/B here rather than an assumption.
    # Turning it off also removes one add per parameter per step.
    muon_nesterov: bool = True
    # Contra-Muon (github.com/nilin/contra-muon). Subtracts `contra_muon` times
    # the Frobenius-normalised pre-orthogonalisation update from the Newton-Schulz
    # output, which damps large singular directions more than small ones and so
    # increases singular-value diversity. Multiple nanoGPT Track-3 records.
    # 0.0 = off (reproduces every existing run). The reference uses
    # CONTRA_MUON=0.4 meaning an effective 0.2 coefficient; start there.
    contra_muon: float = 0.0
    # Per-head / Group Muon (arXiv:2605.08933; Kimi K3 §2.5). Orthogonalise each
    # attention head's block separately instead of the fused QKV / output matrix,
    # so every head gets its own unit-spectral-norm update rather than sharing
    # one across all heads. Kimi K3 runs this at 2.8T params for stability;
    # nanoGPT PR #253 measured ~10 fewer steps. False reproduces existing runs.
    muon_per_head: bool = False
    # Compositional Muon (github.com/tilde-research/comp-muon-release). Muon
    # controls the operator norm of each matrix in isolation, but the loss sees
    # the composed circuits W_Q W_K^T and W_O W_V. CM whitens each factor's
    # gradient by its partner's inverse Gram root before and after the spectral
    # sign, so the *product's* update is what gets norm-controlled.
    #
    # This is the principled generalisation of `muon_per_head`, which was the
    # best arm in the 1500-step Muon screen (-0.027 val_bpb): CM's QK rule is
    # head-local by construction and adds the partner whitening plain per-head
    # splitting lacks. CM takes precedence over `muon_per_head` on the attention
    # matrices — they are the same parameters, and stacking them is undefined.
    comp_muon: bool = False
    # Whitening regime. True = isotropic (replace each head's inverse Gram root
    # with one scalar), which degenerates to a partner-rescaled per-head Muon at
    # near-zero cost over `muon_per_head` and is the honest ablation against it.
    # False = the full coupled-Newton-Schulz matrix inverse root: 25 extra bmms
    # on (head_dim, head_dim) blocks per circuit, the method as published.
    comp_muon_isotropic: bool = False
    # Gram damping `lam` in C = (W^T W + lam I)^{1/2}. The released default.
    comp_muon_damping: float = 1e-2
    # CM learning-rate multiplier on top of `muon_lr`. CM applies no spectral
    # shape-scale, so its effective step differs from Muon's at the same LR; this
    # is the knob for matching them without moving `muon_lr` (which would also
    # move every non-attention matrix).
    comp_muon_mp: float = 1.0
    # S14 Opportunity 1/5a (NEXT_IDEAS.md section 5): batch the Newton-Schulz
    # iteration over same-shape params — one `bmm` per NS line per shape group
    # instead of one `mm` per param. The optimizer step is launch-bound (GPU idle
    # ~40% of step wall-clock waiting on per-param Python dispatch), so fusing
    # launches is the win, not fusing the per-matrix matmuls. Batched NS is
    # mathematically identical to the per-matrix path (a `bmm` over a stack
    # reduces slice-for-slice to looping the legacy `mm`), so it only differs by
    # bf16 kernel-selection noise. False reproduces the exact legacy dispatch and
    # is the fallback for reproducing runs made before this flag existed.
    muon_ns_batched: bool = True
    # Aurora (optimizer: "aurora"). `aurora_pp_iterations` is the number of
    # row-oblique rebalancing passes around the polar step; 2 is the released
    # default. Square matrices reduce exactly to Muon, so this only changes
    # rectangular weights (FFN up/gate/down, qkv).
    aurora_pp_iterations: int = 2
    aurora_pp_beta: float = 0.5
    aurora_weight_decay: float = 0.0
    # I8: optional faster LR for memory-bank parameters (Carmack's plasticity argument
    # — backbone learns slowly, memory adapts quickly). 0 disables; otherwise overrides
    # `lr` for params whose qualified name matches `memory_param_patterns`.
    memory_lr: float = 0.0
    memory_param_patterns: tuple[str, ...] = (
        "mem_read",
        "mem_mlp",
        "update",
        "perceiver",
        "agg_query",
        "short_project",
        "slot_bias",
        "rbf_scale",
        "proj_key",
        "proj_val",
        "proj_query",
        "proj_back",
        "decay",
        "cond",
        "flow",
    )
    # ------------------------------------------------------------------
    # docs/training-speedups.md — optimizer / training-schedule knobs.
    # All default to the legacy behaviour.
    # ------------------------------------------------------------------
    # S5 (NorMuon, arXiv:2510.05491): normalise the Muon update to unit
    # Frobenius norm before the LR/aspect-ratio scaling step.
    norm_update: bool = False
    # S6 (Cautious Weight Decay): only decay a weight where the optimizer
    # update is already shrinking it (update * weight > 0). Replaces standard
    # weight_decay for the Muon group when > 0; for the AdamW group, replaces
    # decoupled weight decay. 0.0 = disabled.
    cautious_wd: float = 0.0
    # S9 (Token Superposition Training, arXiv:2605.06546). Phase 1 compresses
    # `tst_bag_size` consecutive tokens into bags and trains with multi-hot
    # cross-entropy; phase 2 reverts to standard next-token prediction. The
    # final model is architecturally identical to a baseline. Disabled by
    # default.
    tst_enabled: bool = False
    tst_bag_size: int = 4
    tst_phase_ratio: float = 0.3
    # S12.4 (EMA weight averaging for evaluation): maintain an EMA copy of
    # the weights (decay) and use it for validation/final eval. 0.0 disables.
    ema_decay: float = 0.0
    # Batch size for validation/eval forwards. None (default) uses the training
    # batch_size. Set lower than batch_size when the eval forward spikes memory
    # higher than training (e.g. the 450M long-context run trains at batch 2 but
    # the no-grad eval logits tensor OOMs at batch 2, so eval at batch 1).
    eval_batch_size: int | None = None
    # VRAM allocator cap fraction (train.py configure_vram_limit). 0.85 default
    # leaves headroom against WSL2 silent shared-memory spill. Raise for large
    # models whose validation-pass memory spikes above the training steady-state
    # (e.g. 0.95 for the 450M long-context runs). 0.0 = no cap.
    vram_fraction: float = 0.85


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dataclass(cls: type, values: dict[str, Any]) -> Any:
    base = cls()
    for key, value in values.items():
        if not hasattr(base, key):
            raise ValueError(f"Unknown config field {cls.__name__}.{key}")
        setattr(base, key, value)
    return base


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Config file must contain a mapping")
            raw.update(loaded)
    overrides = overrides or {}
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)

    return ExperimentConfig(
        model=_merge_dataclass(ModelConfig, raw.get("model", {})),
        data=_merge_dataclass(DataConfig, raw.get("data", {})),
        training=_merge_dataclass(TrainingConfig, raw.get("training", {})),
    )
