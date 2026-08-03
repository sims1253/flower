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
    precision: str = "fp32"  # fp32 | tf32 | bf16
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
