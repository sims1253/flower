#!/usr/bin/env bash
# Queue the tokenizer screen (production 16K BPE vs regex-free "superword"
# BPE) behind whatever is currently using the GPU.
#
# Generic GPU-free waiter (unlike queue_perhead_confirm.sh, which waits on a
# specific predecessor artifact): polls for any flower.sweep / flower.train
# process and falls through after two consecutive idle checks, so a dead
# predecessor releases the GPU instead of idling it. Skips its own work if
# already complete, so re-running is harmless.
#
# COLLISION WARNING: if another queue_* waiter is already pending (check
# `pgrep -af queue_`), whoever wakes first takes the GPU. Chain behind it
# (wait on its summary.json like queue_perhead_confirm.sh does) rather than
# racing it.
#
# USAGE
#   nohup bash scripts/queue_tokenizer_screen.sh > runs_queue_tokenizer.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
# Cap inductor's parallel compile pool: torch spawns one compile worker per CPU
# (24 here) that each hold ~0.5GB RSS (~1GB true) for the whole run. 8 keeps
# most of the parallel warmup at a quarter of the pool's RAM footprint.
export TORCHINDUCTOR_COMPILE_THREADS=8

OUT=runs/tokenizer_screen_450m
idle=0

if [ -f "$OUT/summary.json" ]; then
  echo "[tokenizer] already complete ($OUT/summary.json) — nothing to do."
  exit 0
fi

echo "[tokenizer] waiting for the GPU (any flower.sweep / flower.train)"
while pgrep -f "flower.sweep|flower.train" > /dev/null; do
  sleep 120
done
# Fall-through path: also proceed if the GPU looked free twice in a row.
while [ "$idle" -lt 2 ]; do
  if pgrep -f "flower.sweep|flower.train" > /dev/null; then
    idle=0
  else
    idle=$((idle + 1))
    [ "$idle" -ge 2 ] && break
  fi
  sleep 120
done

sleep 30

echo "[tokenizer] === tokenizer screen: bpe vs superword BPE (~4h) === $(date -Is)"
uv run python -m flower.sweep \
  --config configs/tokenizer_screen_450m.yaml \
  --output-dir "$OUT"
echo "[tokenizer] exit=$? at $(date -Is)"

echo "[tokenizer] read the result:"
echo "  doc     docs/profiling/tokenizer_candidates_results.md (verdicts + caveats)"
echo "  arms    $OUT/tok_bpe_control.metrics.json  vs  $OUT/tok_noregex.metrics.json"
echo "  metric  val_bpb at matched steps (BPB is cross-tokenizer comparable;"
echo "          perplexity is NOT). Criterion: tok_noregex <= control."
echo "  if pass: +13.3% bytes/token at zero param cost -> consider the SuperBPE"
echo "           two-stage variant as the refined follow-up."
