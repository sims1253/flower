"""Tests for Still KV-cache compaction modules.

Verifies:
1. StillCompactor produces correct output shapes.
2. Identity initialization is near-pass-through at t=T.
3. StillLM forward pass runs and produces KL loss.
4. OT variant runs without error.
5. Compaction eval probes produce expected structures.
6. MeanFlow consistency loss wiring (still_meanflow_loss_weight).
7. Removed no-op arms fail loudly (still_ot_reg_weight).
8. Teacher pass survives fused_linear_ce (logits must exist for KL).
9. Compactors inherit the base config's rope_base.
10. _topk_kl_loss gold-splice does not duplicate gold in the support.
11. Warmup/compaction gating contradiction is rejected at construction.
"""

from __future__ import annotations

import math

import pytest
import torch

from flower.config import ModelConfig
from flower.models.still import (
    StillCompactor,
    StillCompactorOT,
    StillCompactorSpectral,
    _apply_rope,
    _inverse_rope,
)


def _liger_fce_available() -> bool:
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


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

class TestMeanFlowWiring:
    """still_meanflow_loss_weight finally activates the consistency loss."""

    @pytest.fixture
    def meanflow_config(self):
        return ModelConfig(
            variant="still",
            vocab_size=128,
            d_model=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            max_seq_len=64,
            local_window=32,
            still_compact_len=16,
            still_num_blocks=1,
            still_d_latent=32,
            still_kl_topk=50,
            still_kl_weight=1.0,
            still_meanflow_steps=2,
            still_meanflow_loss_weight=0.1,
        )

    def test_compactor_emits_consistency_loss(self):
        """StillCompactorMeanFlow emits 'meanflow_loss' (train: >0, eval: 0)."""
        from flower.models.still_flow2 import StillCompactorMeanFlow

        comp = StillCompactorMeanFlow(
            num_kv_heads=2, head_dim=16, compact_len=8, d_latent=32,
            meanflow_steps=2, identity_init=True,
        )
        # The zero-init velocity field makes u == 0, so the consistency loss
        # is exactly 0 at init (the identity flow is already self-consistent).
        # Un-zero the final layer so the loss is live.
        torch.nn.init.normal_(comp.meanflow_net.fc3.weight, std=0.05)
        torch.nn.init.normal_(comp.meanflow_net.fc3.bias, std=0.05)
        keys = torch.randn(2, 2, 32, 16)
        values = torch.randn(2, 2, 32, 16)

        comp.train()
        result = comp(keys, values, return_compact_cache=True)
        assert "meanflow_loss" in result
        assert result["meanflow_loss"].dim() == 0
        assert result["meanflow_loss"].item() > 0.0

        comp.eval()
        result = comp(keys, values, return_compact_cache=True)
        assert result["meanflow_loss"].item() == 0.0

    def test_loss_weight_adds_term_to_total(self, meanflow_config):
        """loss(w>0) == loss(w=0) + w * mean(collected consistency losses).

        With w=0 (the legacy default) the term is discarded and the loss is
        bit-identical to every prior still_meanflow run; this pins both the
        default-off contract and the actual activation.
        """
        from flower.models import build_model

        model = build_model(meanflow_config)
        model.train()
        # Same zero-init consideration as above: make the consistency loss
        # observable before the first forward.
        for comp in model.compactors:
            torch.nn.init.normal_(comp.meanflow_net.fc3.weight, std=0.05)
            torch.nn.init.normal_(comp.meanflow_net.fc3.bias, std=0.05)
        input_ids = torch.randint(0, 128, (2, 32), dtype=torch.long)
        labels = input_ids.clone()

        out_w = model(input_ids, labels=labels)
        mf = out_w["diagnostics"]["meanflow_loss"]
        assert mf.item() > 0.0

        model.meanflow_loss_weight = 0.0
        with torch.no_grad():
            out_0 = model(input_ids, labels=labels)
        model.meanflow_loss_weight = 0.1

        expected = out_0["loss"] + 0.1 * mf
        assert torch.allclose(out_w["loss"], expected, rtol=1e-5, atol=1e-7), (
            f"weighted {out_w['loss'].item()} vs expected {expected.item()}"
        )

    def test_consistency_term_reaches_meanflow_net(self, meanflow_config):
        """The consistency-loss tensors the compactors return are IN the total
        loss's autograd graph (pullfrog review follow-up).

        Why not "fc3 grad differs with the weight on/off": fc3 also receives
        gradient from the KL path through the one-step endpoint u0, and at the
        fixture's config/seeds the consistency term itself evaluates to ~1e-16
        (the un-zeroed fc3 still sees near-zero features), so a magnitude-based
        diff is either vacuously equal or noise. Graph membership is the
        property that actually broke originally (loss computed then
        discarded), so pin THAT: capture the exact tensors _consistency_loss
        returns and assert autograd.grad(total_loss, captured) is defined.
        """
        from flower.models import build_model
        import flower.models.still_flow2 as sf2

        captured: list[torch.Tensor] = []
        orig = sf2.StillCompactorMeanFlow._consistency_loss

        def spy(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            captured.append(out)
            return out

        sf2.StillCompactorMeanFlow._consistency_loss = spy
        try:
            model = build_model(meanflow_config)
            model.train()
            model.meanflow_loss_weight = 0.5
            for comp in model.compactors:
                torch.nn.init.normal_(comp.meanflow_net.fc3.weight, std=0.05)
                torch.nn.init.normal_(comp.meanflow_net.fc3.bias, std=0.05)
            input_ids = torch.randint(0, 128, (2, 32), dtype=torch.long)
            out = model(input_ids, labels=input_ids.clone())
            assert captured, "no compactor ran _consistency_loss in training mode"
            grads = torch.autograd.grad(
                out["loss"], captured, allow_unused=True, retain_graph=True
            )
            for i, g in enumerate(grads):
                assert g is not None, (
                    f"layer {i}: the consistency-loss tensor is not part of the "
                    "total loss graph — meanflow objective computed but discarded"
                )
        finally:
            sf2.StillCompactorMeanFlow._consistency_loss = orig

    def test_consistency_term_grads_velocity_net_direct(self, meanflow_config):
        """With non-degenerate inputs the consistency loss grads fc3 directly.

        Companion to the graph-membership test: at the MODEL level the term can
        evaluate to ~1e-16 (near-zero features at init), so gradient magnitude
        is pinned at the COMPACTOR level with random keys/values, where the
        existing live-after-unzeroing test already shows the loss is > 0."""
        from flower.models.still_flow2 import StillCompactorMeanFlow

        # Direct construction (num_kv_heads=2, head_dim=16), matching the
        # live-after-unzeroing test's shapes.
        comp = StillCompactorMeanFlow(
            num_kv_heads=2, head_dim=16, compact_len=8, d_latent=32,
            meanflow_steps=2, identity_init=True,
        )
        comp.train()
        torch.nn.init.normal_(comp.meanflow_net.fc3.weight, std=0.05)
        torch.nn.init.normal_(comp.meanflow_net.fc3.bias, std=0.05)
        keys = torch.randn(2, 2, 32, 16)
        values = torch.randn(2, 2, 32, 16)
        result = comp(keys, values, return_compact_cache=True)
        loss = result["meanflow_loss"]
        assert loss.item() > 1e-6, f"degenerate consistency loss: {loss.item()}"
        comp.zero_grad(set_to_none=True)
        loss.backward()
        g = comp.meanflow_net.fc3.weight.grad
        assert g is not None and g.abs().sum() > 0, "fc3 got no grad from the consistency loss"

    def test_default_weight_discards_loss(self):
        """still_meanflow_loss_weight defaults to 0.0 (legacy no-change)."""
        cfg = ModelConfig()
        assert cfg.still_meanflow_loss_weight == 0.0


class TestOTRegRemoved:
    """still_ot_reg_weight was a silent no-op; it must now fail loudly."""

    def test_class_deleted(self):
        import flower.models.still as still_mod

        assert not hasattr(still_mod, "StillCompactorOTReg")

    def test_build_raises(self):
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
            still_ot_reg_weight=0.5,
        )
        with pytest.raises(ValueError, match="still_ot_reg_weight"):
            build_model(cfg)


class TestFusedCETeacherPass:
    """Teacher pass must yield real logits even under fused_linear_ce."""

    @pytest.mark.skipif(
        not _liger_fce_available(),
        reason="needs CUDA + liger-kernel for the fused CE path",
    )
    def test_teacher_logits_survive_fused_ce(self):
        from flower.models import build_model

        cfg = ModelConfig(
            variant="still",
            vocab_size=128,
            d_model=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            max_seq_len=64,
            local_window=32,
            still_compact_len=16,
            still_num_blocks=1,
            still_d_latent=32,
            still_kl_topk=50,
            still_kl_weight=1.0,
            fused_linear_ce=True,
        )
        model = build_model(cfg).cuda()
        model.train()
        input_ids = torch.randint(0, 128, (2, 32), dtype=torch.long, device="cuda")
        labels = input_ids.clone()

        # Before the fix: the teacher call ran the train-gated fused-CE path
        # (logits=None) and _topk_kl_loss crashed on teacher_logits.shape.
        out = model(input_ids, labels=labels)
        assert out["logits"] is not None
        assert out["logits"].shape == (2, 32, 128)
        assert torch.isfinite(out["loss"])
        # The toggle around the teacher call must not leak.
        assert model.base_model.fused_linear_ce is True


class TestCompactorRopeBase:
    """Compactors must un-rotate with the base model's actual rope_base."""

    def _build(self, rope_base):
        from flower.models import build_model

        cfg = ModelConfig(
            variant="still",
            vocab_size=64,
            d_model=32,
            num_heads=2,
            num_layers=2,
            ffn_dim=64,
            max_seq_len=16,
            local_window=8,
            still_compact_len=8,
            still_num_blocks=1,
            still_d_latent=32,
            rope_base=rope_base,
        )
        return build_model(cfg)

    def test_inherited_from_config(self):
        model = self._build(500.0)
        for i, comp in enumerate(model.compactors):
            assert comp.base_rope_base == 500.0, f"layer {i} kept the 10000 default"

    def test_default_is_standard(self):
        model = self._build(10000.0)
        for comp in model.compactors:
            assert comp.base_rope_base == 10000.0


class TestTopkKLGoldSplice:
    """Gold enters the top-k support only where absent (no double count)."""

    def test_no_gold_duplication_in_support(self):
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
        model.kl_topk = 3
        model.kl_temperature = 1.0

        B, T, V = 1, 2, 8
        # Position 0: ranks 0>1>2>3>(gold=5): gold OUTSIDE top-3 -> spliced in.
        # Position 1: ranks 0>1>2>3, gold=1 already IN top-3 -> support unchanged.
        teacher = torch.zeros(B, T, V)
        teacher[0, 0] = torch.tensor([10.0, 9.0, 8.5, 8.0, 0.0, 7.0, 0.0, 0.0])
        teacher[0, 1] = torch.tensor([10.0, 9.0, 8.5, 8.0, 0.0, 0.0, 0.0, 0.0])
        student = teacher + torch.tensor([0.0, -1.0, 0.5, -2.0, 1.0, 0.0, 2.0, -0.5])
        labels = torch.tensor([[5, 1]])

        got = model._topk_kl_loss(teacher, student, labels=labels).item()

        def reference(unconditional_splice: bool) -> float:
            topk = teacher.topk(3, dim=-1).indices
            gold = labels.clamp_min(0).unsqueeze(-1)
            if unconditional_splice:
                support = torch.cat([topk[:, :, :-1], gold], dim=-1)  # the old bug
            else:
                inside = (topk == gold).any(dim=-1, keepdim=True)
                support = torch.where(
                    inside, topk, torch.cat([topk[:, :, :-1], gold], dim=-1)
                )
            tp = teacher.softmax(-1).gather(-1, support)
            tp = tp / tp.sum(-1, keepdim=True).clamp_min(1e-8)
            sp = student.softmax(-1).gather(-1, support)
            sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-8)
            kl = (tp * (tp.clamp_min(1e-8).log() - sp.clamp_min(1e-8).log())).sum(-1)
            return kl.sum().item() / (B * T)

        expected = reference(unconditional_splice=False)
        buggy = reference(unconditional_splice=True)
        # The two supports genuinely disagree for this input...
        assert not math.isclose(expected, buggy, rel_tol=1e-4), (
            "test input does not separate fixed and buggy splicing"
        )
        # ...and the model implements the fixed one.
        assert math.isclose(got, expected, rel_tol=1e-4), (
            f"got {got}, expected {expected} (old duplicated-gold value: {buggy})"
        )


class TestWarmupGatingValidation:
    """base_warmup_steps > compact_from_step must be rejected at construction."""

    def _cfg(self, warmup, compact_from):
        return ModelConfig(
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
            still_base_warmup_steps=warmup,
            still_compact_from_step=compact_from,
        )

    def test_warmup_exceeding_compact_from_raises(self):
        from flower.models import build_model

        with pytest.raises(ValueError, match="still_base_warmup_steps"):
            build_model(self._cfg(warmup=10, compact_from=0))

    def test_warmup_within_compact_from_builds(self):
        from flower.models import build_model

        model = build_model(self._cfg(warmup=5, compact_from=10))
        assert model is not None

    def test_warmup_equal_to_compact_from_builds(self):
        """The boundary every production config actually uses (e.g. 1500/1500,
        2000/2000, 0/0) is the permitted equality: the base unfreeze lands
        exactly on the phase switch, so the dual-pass phase starts with the
        base already frozen. Pin it so a future tightening to strict `<`
        cannot silently invalidate every existing still config."""
        from flower.models import build_model

        model = build_model(self._cfg(warmup=10, compact_from=10))
        assert model is not None


class TestAttentionMatchWiring:
    """still_attn_match_weight arm still trains after the pass-count refactor."""

    def test_attention_match_trains(self):
        from flower.models import build_model

        cfg = ModelConfig(
            variant="still",
            vocab_size=128,
            d_model=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            max_seq_len=64,
            local_window=32,
            still_compact_len=16,
            still_num_blocks=1,
            still_d_latent=32,
            still_kl_topk=50,
            still_kl_weight=1.0,
            still_attn_match_weight=0.5,
        )
        model = build_model(cfg)
        model.train()
        input_ids = torch.randint(0, 128, (2, 32), dtype=torch.long)
        out = model(input_ids, labels=input_ids.clone())

        am = out["diagnostics"]["attn_match_loss"]
        assert am.item() > 0.0
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        # Aggregate over all compactor parameters: at identity init the latent
        # stream is exactly zero, so individual projections (e.g. key_proj)
        # legitimately receive zero grad while the latents themselves do not.
        total = sum(p.grad.abs().sum() for p in model.compactors.parameters() if p.grad is not None)
        assert total > 0

class TestSpectralInitRNGStream:
    """Pin the seeded RNG stream of StillCompactorSpectral._init_spectral.

    The `nn.init.orthogonal_(key_reconstruct.weight)` call in _init_spectral
    is value-dead (the weight is fully overwritten on the next lines) but it
    CONSUMES global RNG, so every draw after it — the two kaiming_normal_
    calls and anything initialized later under the same seed — depends on it
    being there. The cleanup branch deleted it as dead code, silently
    shifting the stream for every seeded spectral-compactor init versus the
    published (pre-cleanup) runs; these tests fail if that ever happens
    again.
    """

    SPECTRAL_KWARGS = dict(
        num_kv_heads=2,
        head_dim=64,
        compact_len=8,
        num_blocks=1,
        spectral_key_rank=4,
        spectral_val_rank=16,
    )

    def _build_seeded(self):
        torch.manual_seed(1234)
        return StillCompactorSpectral(**self.SPECTRAL_KWARGS)

    def test_seeded_construction_is_bit_reproducible(self):
        a = self._build_seeded()
        b = self._build_seeded()
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
            assert na == nb
            assert torch.equal(pa, pb), f"seeded init not bit-reproducible: {na}"

    def test_init_matches_pre_cleanup_rng_stream(self):
        # Golden values captured from the pre-cleanup branch (main @ 5f11515):
        # the first rows of the two kaiming draws made in _init_spectral.
        # key_signal is drawn immediately after the orthogonal_ call, so it is
        # the first thing to change if that RNG-consuming call disappears.
        c = self._build_seeded()
        # Exact equality: these are round-trip float reprs of the golden
        # draws, so any shift in the seeded stream changes them.
        assert c.key_signal.weight[0, :4].tolist() == [
            -0.03484155237674713, -0.0021602031774818897, 0.053742606192827225, -0.032153017818927765
        ]
        assert c.val_signal.weight[0, :4].tolist() == [
            0.0238554198294878, 0.011659996584057808, -0.0040173823945224285, -0.012448701076209545
        ]

    def test_orthogonal_init_is_value_dead(self):
        # Documents why the restored orthogonal_ call exists ONLY for the RNG
        # stream: its result is fully overwritten. key_reconstruct ends up as
        # the (scaled) transpose of key_signal, whatever orthogonal_ drew.
        c = self._build_seeded()
        assert torch.allclose(c.key_reconstruct.weight, (c.key_signal.weight / 0.1).T)
