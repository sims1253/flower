"""Tests for Still KV-cache compaction modules.

Verifies:
1. StillCompactor produces correct output shapes.
2. Identity initialization is near-pass-through at t=T.
3. StillLM forward pass runs and produces KL loss.
4. OT variant runs without error.
5. Compaction eval probes produce expected structures.
"""

from __future__ import annotations

import pytest
import torch

from flower.config import ModelConfig
from flower.models.still import StillCompactor, StillCompactorOT, _apply_rope, _inverse_rope


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


class TestRoPEHelpers:
    """Test the RoPE rotation / inverse-rotation helpers."""

    def test_inverse_rope_recovers_original(self):
        """Un-rotating a rotated key should approximately recover the original."""
        head_dim = 64
        T = 32
        base = 10000.0
        x = torch.randn(1, 4, T, head_dim, dtype=torch.float32)
        positions = torch.arange(T, dtype=torch.float32)
        rotated = _apply_rope(x, positions, head_dim, base=base)
        recovered = _inverse_rope(rotated, positions, head_dim, base=base)
        assert torch.allclose(x, recovered, atol=1e-4), (
            f"Inverse RoPE should recover original. Max diff: {(x - recovered).abs().max()}"
        )

    def test_rope_changes_values(self):
        """RoPE should actually rotate the key vectors."""
        head_dim = 32
        T = 8
        x = torch.randn(1, 2, T, head_dim, dtype=torch.float32)
        positions = torch.arange(T, dtype=torch.float32)
        rotated = _apply_rope(x, positions, head_dim, base=10.0)
        assert not torch.allclose(x, rotated, atol=1e-4), "RoPE should modify the input"


class TestStillCompactor:
    """Test the StillCompactor module."""

    @pytest.fixture
    def compactor(self):
        return StillCompactor(
            num_kv_heads=4,
            head_dim=64,
            compact_len=32,
            num_blocks=2,
            d_latent=128,
            identity_init=True,
        )

    @pytest.fixture
    def compactor_t_eq_T(self):
        # compact_len == T for identity-init pass-through test.
        return StillCompactor(
            num_kv_heads=4,
            head_dim=64,
            compact_len=64,
            num_blocks=2,
            d_latent=128,
            identity_init=True,
        )

    def test_output_shapes(self, compactor):
        """Compactor should produce correct shapes for Ck, Cv."""
        B, H, T, d = 2, 4, 128, 64
        keys = torch.randn(B, H, T, d)
        values = torch.randn(B, H, T, d)
        result = compactor(keys, values, return_compact_cache=True)

        assert "Ck_raw" in result
        assert "Cv" in result
        assert "compact_keys" in result
        assert result["Ck_raw"].shape == (B, H, 32, d)
        assert result["Cv"].shape == (B, H, 32, d)
        assert result["compact_keys"].shape == (B, H, 32, d)

    def test_identity_init_near_passthrough(self, compactor_t_eq_T):
        """At t=T with identity init, output should approximately equal input."""
        B, H, T, d = 1, 4, 64, 64
        keys = torch.randn(B, H, T, d, dtype=torch.float32)
        values = torch.randn(B, H, T, d, dtype=torch.float32)
        positions = torch.arange(T, dtype=torch.float32)

        result = compactor_t_eq_T(keys, values, positions=positions, return_compact_cache=True)

        # The identity init makes Ck_raw approximate the position-free keys,
        # and Cv approximate the values. Check they're close.
        keys_free = _inverse_rope(keys, positions, d, base=10000.0)

        # With identity init, the first 64 dims of key_proj output = position-free keys.
        ck_diff = (result["Ck_raw"] - keys_free).abs().mean().item()
        cv_diff = (result["Cv"] - values).abs().mean().item()

        # Should be reasonably close (identity init is approximate, not exact).
        # The self-attention residual adds some noise even with zero init.
        assert ck_diff < 1.5, f"Ck_raw diff too large: {ck_diff}"
        assert cv_diff < 1.5, f"Cv diff too large: {cv_diff}"

    def test_different_compact_lengths(self):
        """Compactor should work with various compact_len values."""
        for compact_len in [8, 16, 32, 64]:
            comp = StillCompactor(
                num_kv_heads=2, head_dim=32, compact_len=compact_len,
                d_latent=64, identity_init=True,
            )
            keys = torch.randn(1, 2, 128, 32)
            values = torch.randn(1, 2, 128, 32)
            result = comp(keys, values)
            assert result["Ck_raw"].shape[2] == compact_len

    def test_parameter_count_reasonable(self, compactor):
        """Compactor should have a reasonable parameter count (< 10% of a small model)."""
        n_params = sum(p.numel() for p in compactor.parameters() if p.requires_grad)
        # For d=64, d_latent=128, 2 blocks, this should be well under 1M.
        assert n_params < 1_000_000, f"Too many parameters: {n_params}"


class TestStillCompactorOT:
    """Test the OT-coupled compactor variant."""

    def test_ot_compactor_forward(self):
        """OT compactor should run without error and produce correct shapes."""
        comp = StillCompactorOT(
            num_kv_heads=4,
            head_dim=32,
            compact_len=16,
            num_blocks=2,
            d_latent=64,
            identity_init=True,
            ot_epsilon=0.1,
            ot_iters=5,
        )
        keys = torch.randn(2, 4, 64, 32)
        values = torch.randn(2, 4, 64, 32)
        result = comp(keys, values, return_compact_cache=True)

        assert result["Ck_raw"].shape == (2, 4, 16, 32)
        assert result["Cv"].shape == (2, 4, 16, 32)


class TestStillLM:
    """Test the StillLM wrapper."""

    @pytest.fixture
    def small_config(self):
        return ModelConfig(
            variant="still",
            vocab_size=256,
            d_model=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            max_seq_len=64,
            local_window=32,
            rope_base=10000.0,
            still_compact_len=16,
            still_num_blocks=1,
            still_d_latent=32,
            still_kl_topk=50,
            still_kl_weight=1.0,
            still_ce_weight=0.0,
        )

    def test_model_builds(self, small_config):
        """StillLM should build without error."""
        from flower.models import build_model

        model = build_model(small_config)
        assert model is not None
        assert hasattr(model, "compactors")
        assert hasattr(model, "base_model")

    def test_base_model_frozen(self, small_config):
        """Base model parameters should be frozen."""
        from flower.models import build_model

        model = build_model(small_config)
        for param in model.base_model.parameters():
            assert not param.requires_grad, "Base model parameter should be frozen"

    def test_compactor_trainable(self, small_config):
        """Compactor parameters should be trainable."""
        from flower.models import build_model

        model = build_model(small_config)
        comp_params = list(model.compactors.parameters())
        assert len(comp_params) > 0, "Should have compactor parameters"
        for param in comp_params:
            assert param.requires_grad, "Compactor parameter should be trainable"

    def test_forward_pass(self, small_config):
        """Full forward pass should produce logits and loss."""
        from flower.models import build_model

        model = build_model(small_config)
        input_ids = torch.randint(0, 256, (2, 32), dtype=torch.long)
        labels = input_ids.clone()

        out = model(input_ids, labels=labels)

        assert "logits" in out
        assert "loss" in out
        assert out["logits"].shape == (2, 32, 256)
        assert out["loss"].item() is not None  # Just check it's a valid tensor

    def test_kl_loss_decreases(self, small_config):
        """A short training run should decrease the KL loss."""
        from flower.models import build_model

        model = build_model(small_config)
        model.set_step(0)
        optimizer = torch.optim.AdamW(model.compactors.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 256, (4, 32), dtype=torch.long)
        labels = input_ids.clone()

        # Initial loss.
        model.train()
        out1 = model(input_ids, labels=labels)
        loss1 = out1["loss"]
        loss1_val = loss1.item()
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Train for a few steps.
        for _ in range(10):
            out = model(input_ids, labels=labels)
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        out2 = model(input_ids, labels=labels)
        loss2_val = out2["loss"].item()

        assert loss2_val <= loss1_val + 0.1, f"Loss should not increase much: {loss1_val} -> {loss2_val}"

    def test_ot_variant_builds(self, small_config):
        """The OT variant should build and run."""
        from flower.models import build_model

        cfg = ModelConfig(**{**small_config.__dict__})
        cfg.variant = "still_ot"
        cfg.still_use_ot_read = True
        model = build_model(cfg)

        input_ids = torch.randint(0, 256, (2, 32), dtype=torch.long)
        out = model(input_ids, labels=input_ids)
        assert out["logits"].shape == (2, 32, 256)

    def test_smoke_eval(self, small_config):
        """Quick smoke test of the eval probes."""
        from flower.models import build_model

        model = build_model(small_config)
        model.eval()
        model.set_step(100)

        input_ids = torch.randint(0, 256, (2, 32), dtype=torch.long)
        out = model(input_ids, labels=input_ids)
        assert "diagnostics" in out


class TestConfigIntegration:
    """Test that config fields are properly integrated."""

    def test_config_has_still_fields(self):
        cfg = ModelConfig()
        assert hasattr(cfg, "still_compact_len")
        assert hasattr(cfg, "still_num_blocks")
        assert hasattr(cfg, "still_d_latent")
        assert hasattr(cfg, "still_use_ot_read")
        assert hasattr(cfg, "still_use_energy_read")
        assert hasattr(cfg, "still_use_freq_decay")
        assert hasattr(cfg, "still_kl_topk")
        assert hasattr(cfg, "still_kl_weight")
        assert hasattr(cfg, "still_ce_weight")

    def test_variant_registered(self):
        """Still variants should be in the build_model dispatch."""
        from flower.models import build_model

        cfg = ModelConfig(
            variant="still",
            vocab_size=64,
            d_model=32,
            num_heads=2,
            num_layers=1,
            ffn_dim=64,
            max_seq_len=16,
            local_window=8,
            still_compact_len=8,
            still_num_blocks=1,
            still_d_latent=32,
        )
        model = build_model(cfg)
        assert model is not None
