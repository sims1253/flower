#!/usr/bin/env bash
# Overnight queue: wait for the bake-off sweep to finish, then launch the
# 400M @ seq=32K memory comparison (vanilla vs bloom) — the new-capability
# confirmation at scale. Runs detached; logs everything.
set -u
cd /home/m0hawk/Documents/flower
export PYTHONPATH=.
export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] queue watcher started; waiting for bake-off sweep (PID 331721) to exit..."

# Wait for the bake-off sweep process to finish (poll every 60s)
while kill -0 331721 2>/dev/null; do
    sleep 60
done
echo "[$(date)] bake-off sweep finished. Cooling down 60s..."
sleep 60

# Launch the 400M @ seq=32K comparison: vanilla (already have a baseline, but
# re-run at the same seed/config for a clean comparison) + bloom_memory arm.
# Build a small sweep config for the two arms at 400M/32K.
cat > /tmp/sweep_400m_32k_bakeoff.yaml <<'YAML'
sweep:
  name: sweep13_400m_longctx32k_bakeoff

  defaults:
    model:
      variant: vanilla_local
      vocab_size: 16384
      d_model: 1024
      num_heads: 16
      num_layers: 24
      ffn_dim: 4096
      max_seq_len: 32768
      local_window: 2048
      rope_base: 10000.0
      dropout: 0.0
      norm_type: rmsnorm
      ffn_activation: swiglu
      ffn_param_match: true
      qk_norm: true
      use_bias: false
      init_scheme: scaled
      init_std: 0.02
      flex_attention: true
      activation_checkpoint: "ffn"
      bf16_cross_entropy: true
      memory_slots: 8

    data:
      dataset: fineweb_edu
      tokenizer: "custom:tokenizers/fineweb_16k.json"
      sequence_length: 32768
      eval_seq_len: 32768
      bytes_per_token: 4.279

    training:
      batch_size: 1
      gradient_accumulation_steps: 8
      steps: 2000
      lr: 0.002
      warmup_steps: 400
      lr_schedule: wsd
      lr_decay_frac: 0.2
      lr_final_frac: 0.0
      grad_clip: 1.0
      eval_interval: 500
      checkpoint_interval: 1000
      save_checkpoints: true
      device: auto
      seed: 0
      log_backend: tensorboard
      composite_eval: false
      optimizer: muon
      muon_lr: 0.0018
      muon_momentum: 0.95
      weight_decay: 0.01
      weight_decay_exclude_embeddings: true
      ema_decay: 0.997
      validation_steps: 20
      validation_interval: 500
      precision: bf16
      compile_model: true
      compile_mode: default

  variants:
    - name: vanilla_local
      model: { variant: vanilla_local }
    - name: bloom_memory
      model:
        variant: bloom_memory
        bloom_num_hashes: 4
        bloom_summary_points: 16
        bloom_temperature: 0.5
YAML

echo "[$(date)] launching 400M @ seq=32K bake-off (vanilla + bloom, 1 seed each)..."
uv run python -m flower.sweep \
    --config /tmp/sweep_400m_32k_bakeoff.yaml \
    --output-dir runs/sweep13_400m_longctx32k_bakeoff \
    --device cuda >> runs_400m_32k_bakeoff.log 2>&1
echo "[$(date)] 400M bake-off finished."
