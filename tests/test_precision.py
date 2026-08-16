"""Tests for the FP8 linear-conversion guardrails (flower/precision.py).

These are pure-CPU tests of the *filter* logic — which layers get converted.
They deliberately do not exercise the torchao conversion itself, which needs a
CUDA sm_89+ device; the filter is where the guardrails live and where a silent
mistake (converting the tied LM head, converting an unaligned layer) would cost
a multi-hour run.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from flower.precision import _block_index, fp8_module_filter


def test_block_index_parses_block_fqns():
    assert _block_index("blocks.0.ff.gate") == 0
    assert _block_index("blocks.19.attn.qkv") == 19


def test_block_index_returns_none_outside_blocks():
    # The tied LM head and MTP heads hang off the model root, not off `blocks`.
    assert _block_index("head") is None
    assert _block_index("mtp_heads.0") is None
    assert _block_index("token") is None


def test_filter_converts_interior_block_linears():
    keep = fp8_module_filter(num_layers=20, keep_bf16_blocks=1)
    aligned = nn.Linear(1280, 3392, bias=False)
    assert keep(aligned, "blocks.5.ff.gate") is True
    assert keep(aligned, "blocks.18.attn.qkv") is True


def test_filter_keeps_first_and_last_blocks_in_bf16():
    keep = fp8_module_filter(num_layers=20, keep_bf16_blocks=1)
    aligned = nn.Linear(1280, 3392, bias=False)
    assert keep(aligned, "blocks.0.ff.gate") is False
    assert keep(aligned, "blocks.19.ff.gate") is False


def test_filter_honours_wider_bf16_margin():
    keep = fp8_module_filter(num_layers=20, keep_bf16_blocks=4)
    aligned = nn.Linear(1280, 3392, bias=False)
    assert keep(aligned, "blocks.3.ff.gate") is False
    assert keep(aligned, "blocks.16.ff.gate") is False
    assert keep(aligned, "blocks.4.ff.gate") is True


def test_filter_never_converts_the_tied_lm_head():
    """The head shares its weight with the embedding table (CausalLM.__init__).

    Quantizing it would perturb the embedding through the tie, so it must be
    excluded regardless of alignment or block margin.
    """
    keep = fp8_module_filter(num_layers=20, keep_bf16_blocks=0)
    head = nn.Linear(1280, 16384, bias=False)
    assert keep(head, "head") is False


def test_filter_rejects_unaligned_dims():
    """FP8 GEMM needs both dims divisible by 16."""
    keep = fp8_module_filter(num_layers=20, keep_bf16_blocks=0)
    assert keep(nn.Linear(1280, 2730, bias=False), "blocks.5.ff.gate") is False
    assert keep(nn.Linear(2730, 1280, bias=False), "blocks.5.ff.down") is False
    assert keep(nn.Linear(1280, 2752, bias=False), "blocks.5.ff.gate") is True


def test_filter_ignores_non_linear_modules():
    keep = fp8_module_filter(num_layers=20, keep_bf16_blocks=0)
    assert keep(nn.RMSNorm(1280), "blocks.5.n1") is False
    assert keep(nn.Embedding(16384, 1280), "token") is False


def test_keep_zero_converts_every_aligned_block_linear():
    keep = fp8_module_filter(num_layers=20, keep_bf16_blocks=0)
    aligned = nn.Linear(1280, 3392, bias=False)
    assert keep(aligned, "blocks.0.ff.gate") is True
    assert keep(aligned, "blocks.19.ff.gate") is True


def test_fp8_config_fast_accum_off_by_default():
    """torchao's recipes fast-accum only the forward GEMM; document that fact.

    If a torchao upgrade changes this (e.g. fast-accums all three GEMMs), this
    test fails loudly and the fp8_fast_accum bench arm's meaning changes —
    re-read the recipe defaults before trusting the arm.
    """
    pytest.importorskip("torchao")
    from flower.precision import _fp8_config

    cfg = _fp8_config("tensorwise")
    assert cfg.gemm_config_output.use_fast_accum is True
    assert cfg.gemm_config_grad_input.use_fast_accum is False
    assert cfg.gemm_config_grad_weight.use_fast_accum is False


def test_fp8_config_fast_accum_flips_only_the_backward_gemms():
    pytest.importorskip("torchao")
    from flower.precision import _fp8_config

    cfg = _fp8_config("tensorwise", use_fast_accum=True)
    assert cfg.gemm_config_output.use_fast_accum is True
    assert cfg.gemm_config_grad_input.use_fast_accum is True
    assert cfg.gemm_config_grad_weight.use_fast_accum is True
    # The cast configs (the actual quantization recipe) are untouched.
    off = _fp8_config("tensorwise")
    assert cfg.cast_config_input == off.cast_config_input
    assert cfg.cast_config_weight == off.cast_config_weight
    assert cfg.cast_config_grad_output == off.cast_config_grad_output


def test_fp8_config_rejects_unknown_recipe():
    pytest.importorskip("torchao")
    from flower.precision import _fp8_config

    with pytest.raises(RuntimeError, match="fp8_recipe"):
        _fp8_config("int8")
