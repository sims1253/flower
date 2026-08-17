"""Fused CE at eval (fused_linear_ce_eval): correctness contracts.

`fused_linear_ce` (S14-5b) fuses the lm_head matmul + CE during training so
the (B*T, vocab) logits tensor is never materialized. At validation the eager
head still materialized it — the eval-time VRAM spike that forces
`eval_batch_size: 1` and `vram_fraction: 0.95` in the production long-context
configs (measured on the 5090 at seq 2048 / vocab 16384 / batch 2 / bf16: the
eval forward peaked 648 MiB above steady state eager vs 58 MiB fused).

`fused_linear_ce_eval` extends the fused path to LOSS-ONLY eval forwards. The
contracts pinned here:

  1. Default off reproduces everything (eval keeps the eager logits path).
  2. The flag EXTENDS `fused_linear_ce`, it does not replace it: without the
     training flag, training forwards stay eager.
  3. Flag-on eval forwards return loss with logits=None — so evaluate() (which
     reads only the loss) works unchanged, and logits consumers must keep the
     flag off.
  4. The fused eval loss matches the eager eval loss: bit-close in fp32;
     within bf16 kernel-selection noise under autocast (measured ~1.5e-3
     relative on the 5090 — Liger accumulates the CE in fp32 over chunked
     matmuls while the eager path runs one big bf16 cuBLAS GEMM then a bf16
     CE; the honest threshold below is ~3x the measured noise). Parity is
     pinned on the production paths too, not just synthetic all-labels data:
     MQAR validation batches (labels -100 except at answer positions — both
     CEs default ignore_index=-100) and the mixed-precision regime train.py
     actually runs (fp32 master weights + bf16 autocast forward).
  5. MTP auxiliary losses stay OUT of the fused eval loss (they are a
     training-only objective; eval reports the t+1 loss alone — the same leak
     tests/test_mtp_eval_loss.py pinned on the eager path would reintroduce
     itself here without the `self.training` guard inside the fused branch).
  6. run_composite_eval needs logits, so it force-disables the flag around its
     forwards and restores it after.

CUDA gates mirror tests/test_training_speedups.py: the Liger kernel is
CUDA/Triton-only, so parity/logits-None contracts are CUDA-gated and the
default/fallback contracts run everywhere.
"""

from __future__ import annotations

import math

import pytest
import torch

from flower.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from flower.data import validation_token_batches
from flower.models import build_model
from flower.probes.composite import run_composite_eval
from flower.train import evaluate


def tiny(**overrides) -> ModelConfig:
    base = dict(
        variant="vanilla_local",
        vocab_size=256,
        d_model=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=192,
        max_seq_len=64,
        local_window=16,
        memory_slots=4,
        fused_linear_ce=True,
        fused_linear_ce_eval=True,
        # Small tied-logit magnitudes (loss ~ ln(vocab) rather than ~60 under
        # the legacy torch init), so the parity tolerances below are measured
        # at representative loss scales.
        init_scheme="scaled",
    )
    base.update(overrides)
    return ModelConfig(**base)


def _ids(cfg: ModelConfig, device: torch.device, batch: int = 2, seq: int = 32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1234)
    return torch.randint(0, cfg.vocab_size, (batch, seq), generator=gen).to(device)


# ---------------------------------------------------------------------------
# Defaults + off-CUDA fallback (run everywhere).
# ---------------------------------------------------------------------------


def test_fused_linear_ce_eval_defaults_off():
    assert ModelConfig().fused_linear_ce_eval is False


def test_fused_eval_cpu_flag_on_keeps_eager_logits():
    # Off-CUDA the fused path cannot run (Triton kernel), so the eval flag
    # inherits the training flag's contract: fall back to eager, materialize
    # logits, warn once. A CPU validation run with the flag on must therefore
    # behave exactly like a run with it off.
    import warnings

    cfg = tiny()
    model = build_model(cfg)  # CPU
    model.eval()
    ids = _ids(cfg, torch.device("cpu"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            out = model(ids, labels=ids)
    assert out["logits"] is not None
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    fallback = [w for w in caught if "fused_linear_ce" in str(w.message)]
    assert len(fallback) == 1  # one-time, like the training flag


# ---------------------------------------------------------------------------
# CUDA contracts (Liger kernel is CUDA/Triton-only).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger fused CE kernel is CUDA/Triton-only")
class TestFusedEvalCUDA:
    dev = torch.device("cuda")

    def _model(self, **overrides) -> torch.nn.Module:
        torch.manual_seed(0)
        model = build_model(tiny(**overrides)).to(self.dev)
        model.eval()
        return model

    def test_flag_on_returns_loss_with_logits_none(self):
        # Contract 3: the whole point of the flag is that the logits tensor is
        # never materialized at eval — loss must still be computed.
        cfg = tiny()
        model = self._model()
        ids = _ids(cfg, self.dev)
        with torch.no_grad():
            out = model(ids, labels=ids)
        assert out["logits"] is None
        assert torch.isfinite(out["loss"])

    def test_flag_off_keeps_logits_path_intact(self):
        # Contract 1: with the eval flag off (default), fused_linear_ce=True
        # alone must not change eval — logits are materialized as before.
        cfg = tiny(fused_linear_ce_eval=False)
        model = self._model(fused_linear_ce_eval=False)
        assert model.fused_linear_ce_eval is False
        ids = _ids(cfg, self.dev)
        with torch.no_grad():
            out = model(ids, labels=ids)
        assert out["logits"] is not None
        assert out["logits"].shape == (2, 32, cfg.vocab_size)
        assert torch.isfinite(out["loss"])

    def test_eval_flag_alone_does_not_fuse_training(self):
        # Contract 2: fused_linear_ce_eval extends the fused path to eval; it
        # must not silently enable it during training when fused_linear_ce is
        # off (that would change a run's training numerics under a flag whose
        # name says "eval").
        model = self._model(fused_linear_ce=False, fused_linear_ce_eval=True)
        model.train()
        ids = _ids(tiny(), self.dev)
        out = model(ids, labels=ids)
        assert out["logits"] is not None, "eval-only flag must not fuse training forwards"

    def test_fused_eval_loss_matches_eager_fp32(self):
        # Contract 4, fp32: both paths accumulate the CE reduction in fp32 and
        # the only difference is the matmul blocking — measured bit-identical
        # on the 5090 at this size, so a 1e-6 floor is honest.
        cfg = tiny()
        model = self._model()
        ids = _ids(cfg, self.dev)
        with torch.no_grad():
            model.fused_linear_ce_eval = False
            eager = model(ids, labels=ids)["loss"]
            model.fused_linear_ce_eval = True
            fused = model(ids, labels=ids)["loss"]
        torch.testing.assert_close(fused.float(), eager.float(), rtol=0.0, atol=1e-6)

    def test_fused_eval_loss_matches_eager_bf16(self):
        # Contract 4, bf16 (the production precision): the eager path runs one
        # big bf16 cuBLAS head GEMM then a bf16 CE; Liger runs chunked matmuls
        # with an fp32 CE accumulation. Measured relative difference on the
        # 5090: ~1.5e-3 at loss ~5.6 (vocab 256) and ~1.1e-3 at loss ~9.8
        # (vocab 16384). 5e-3 gives ~3x headroom over the measured noise while
        # still catching a real regression (a wrong label shift or offset
        # moves the loss by O(1), three orders of magnitude past it).
        cfg = tiny(vocab_size=16384, d_model=256, max_seq_len=128)
        model = self._model(vocab_size=16384, d_model=256, max_seq_len=128).to(torch.bfloat16)
        ids = _ids(cfg, self.dev, batch=2, seq=64)
        with torch.no_grad():
            model.fused_linear_ce_eval = False
            eager = model(ids, labels=ids)["loss"]
            model.fused_linear_ce_eval = True
            fused = model(ids, labels=ids)["loss"]
        torch.testing.assert_close(fused.float(), eager.float(), rtol=5e-3, atol=5e-3)

    def test_fused_eval_loss_matches_eager_on_mqar_ignore_labels(self):
        # Contract 4 on the ACTUAL in-loop evaluate() data for mqar configs:
        # validation_token_batches over dataset="mqar" yields (ids, labels)
        # pairs whose labels are -100 everywhere except at answer positions
        # (87.5% of this batch is ignored). Both CEs default
        # ignore_index=-100 — F.cross_entropy on the eager path, Liger's
        # LigerFusedLinearCrossEntropyLoss on the fused path — so the fused
        # eval must skip the ignored positions and average the same surviving
        # tokens. Measured bit-identical on the 5090 in fp32 at this size, so
        # the 1e-6 floor matches the fp32 parity test above.
        cfg = tiny()
        model = self._model()
        data = DataConfig(
            dataset="mqar",
            tokenizer="byte",
            sequence_length=32,
            synthetic_vocab_size=cfg.vocab_size,
            eval_seq_len=32,
        )
        ids, labels = next(iter(validation_token_batches(data, 2, self.dev)))
        assert (labels == -100).any(), "batch must actually exercise the ignore index"
        with torch.no_grad():
            model.fused_linear_ce_eval = False
            eager = model(ids, labels=labels)["loss"]
            model.fused_linear_ce_eval = True
            fused = model(ids, labels=labels)["loss"]
        assert torch.isfinite(eager) and torch.isfinite(fused)
        torch.testing.assert_close(fused.float(), eager.float(), rtol=0.0, atol=1e-6)

    def test_fused_eval_loss_matches_eager_mixed_precision_autocast(self):
        # Contract 4 at the precision train.py actually runs: with
        # training.precision=bf16 the model keeps fp32 master weights and
        # evaluate() wraps the forward in autocast_ctx(bf16) — NOT the
        # .to(bfloat16) whole-model regime of the bf16 test above. Under
        # autocast both paths run the head matmul in bf16 and the eager CE
        # upcasts to fp32, so the fused/eager gap is smaller than all-bf16:
        # measured ~2e-4 relative at loss ~9.7 (vocab 16384) on the 5090.
        # 1e-3 gives ~5x headroom while still catching a real regression
        # (a wrong label shift or offset moves the loss by O(1)).
        cfg = tiny(vocab_size=16384, d_model=256, max_seq_len=128)
        model = self._model(vocab_size=16384, d_model=256, max_seq_len=128)
        assert next(model.parameters()).dtype == torch.float32  # master weights stay fp32
        ids = _ids(cfg, self.dev, batch=2, seq=64)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.fused_linear_ce_eval = False
            eager = model(ids, labels=ids)["loss"]
            model.fused_linear_ce_eval = True
            fused = model(ids, labels=ids)["loss"]
        torch.testing.assert_close(fused.float(), eager.float(), rtol=1e-3, atol=1e-3)

    def test_fused_eval_excludes_mtp_aux_losses(self):
        # Contract 5: MTP heads are a training-only auxiliary objective. The
        # fused branch now serves eval, so its aux-head loop keeps the
        # `self.training` guard — otherwise fused_linear_ce_eval would
        # reintroduce the eval-loss leak test_mtp_eval_loss pins (val_bpb
        # 1.128 -> 2.036 -> 3.069 in the first MTP screen).
        cfg = tiny(mtp_extra_heads=2, mtp_weight=0.9)
        model = self._model(mtp_extra_heads=2, mtp_weight=0.9)
        assert model.mtp_heads is not None
        ids = _ids(cfg, self.dev)
        with torch.no_grad():
            model.fused_linear_ce_eval = True
            fused = model(ids, labels=ids)["loss"]
            model.fused_linear_ce_eval = False
            eager = model(ids, labels=ids)["loss"]
        torch.testing.assert_close(fused.float(), eager.float(), rtol=0.0, atol=1e-6)

    def test_evaluate_runs_with_flag_on(self):
        # Contract 3, integration: evaluate() reads only out["loss"], so a
        # flag-on validation (logits=None) must produce the same val_loss as
        # the flag-off validation on the same stream — via the same
        # flower.train.evaluate() the training loop and final metrics use.
        cfg = tiny()
        model = self._model()
        data = DataConfig(
            dataset="synthetic",
            tokenizer="byte",
            sequence_length=32,
            synthetic_vocab_size=cfg.vocab_size,
            eval_seq_len=32,
        )
        device = self.dev
        with torch.no_grad():
            model.fused_linear_ce_eval = False
            m_off = evaluate(model, validation_token_batches(data, 2, device), 4, device)
            model.fused_linear_ce_eval = True
            m_on = evaluate(model, validation_token_batches(data, 2, device), 4, device)
        assert math.isfinite(m_on["val_loss"]) and m_on["val_tokens"] > 0
        torch.testing.assert_close(
            torch.tensor(m_on["val_loss"]), torch.tensor(m_off["val_loss"]), rtol=0.0, atol=1e-6
        )

    def test_evaluate_on_compiled_model_with_flag_on(self):
        # The non-EMA validation path evaluates the torch.compile'd model.
        # The fused branch's only compile interaction is the mandatory
        # @torch._dynamo.disable break at _call_liger_fce — the SAME single
        # break the compiled training forward already takes (pinned by
        # test_fused_ce_compile.py), not a new one: eval with the flag on
        # enters the branch training uses, instead of tracing a second
        # (eager-logits) graph as flag-off eval does. This test pins that the
        # compiled flag-on evaluate() runs and matches the eager flag-off
        # validation.
        cfg = tiny()
        model = self._model()
        data = DataConfig(
            dataset="synthetic",
            tokenizer="byte",
            sequence_length=32,
            synthetic_vocab_size=cfg.vocab_size,
            eval_seq_len=32,
        )
        device = self.dev
        with torch.no_grad():
            model.fused_linear_ce_eval = False
            expected = evaluate(model, validation_token_batches(data, 2, device), 4, device)
        model.fused_linear_ce_eval = True
        compiled = torch.compile(model, dynamic=False)
        got = evaluate(compiled, validation_token_batches(data, 2, device), 4, device)
        torch.testing.assert_close(
            torch.tensor(got["val_loss"]), torch.tensor(expected["val_loss"]), rtol=0.0, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Composite eval (logits consumer) — runs on CUDA when available, CPU otherwise.
# ---------------------------------------------------------------------------


def test_composite_eval_forces_fused_eval_off_and_restores():
    # Contract 6: run_composite_eval's probes read `model(seq)["logits"]`; a
    # labels-carrying forward under the flag returns logits=None. The composite
    # must (a) still return logits-derived metrics with the flag on — forced
    # off around its forwards — and (b) restore the caller's flag afterwards,
    # mirroring its model.train()/eval() save-restore.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ExperimentConfig(
        model=tiny(),
        data=DataConfig(
            dataset="synthetic",
            tokenizer="byte",
            sequence_length=32,
            synthetic_vocab_size=256,
            # 64 (not 32) so at least the smallest MQAR level (16 pairs ->
            # 32 kv + 8 qa + short delay tokens) fits the probe budget and the
            # capacity curve carries a measured, finite point.
            eval_seq_len=64,
        ),
        training=TrainingConfig(seed=0, batch_size=4),
    )
    torch.manual_seed(0)
    model = build_model(cfg.model).to(device)
    model.eval()
    assert model.fused_linear_ce_eval is True  # from the config

    result = run_composite_eval(model, cfg, device=device, doc_limit=1)

    # (a) logits-derived metrics came back real, not None-driven garbage: the
    # induction probe slices `logits[:, start:-1, :]` (a TypeError if logits
    # were None) and its loss/accuracy are finite; the associative-recall and
    # smallest MQAR levels likewise.
    induction = result["metrics"]["induction_copy"]
    assert math.isfinite(induction["loss"]) and induction["tokens"] > 0
    assert 0.0 <= induction["accuracy"] <= 1.0
    assoc = result["metrics"]["associative_recall"]
    assert math.isfinite(assoc["loss"]) and assoc["examples"] > 0
    mqar_short = result["metrics"]["mqar"]["capacity_curve"]["short"]
    assert any(math.isfinite(v) for v in mqar_short.values())
    # (b) the caller's flag state is restored.
    assert model.fused_linear_ce_eval is True
