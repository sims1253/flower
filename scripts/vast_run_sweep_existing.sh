#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vast_common.sh"
load_vast_defaults

instance_id=""
dry_run="false"
config="configs/research_sweep_remote.yaml"
output_dir="runs/vast/research_sweep_remote"
steps=""
device="auto"
limit=""
variants=""
smoke="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-id) instance_id="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --variants) variants="$2"; shift 2 ;;
    --smoke) smoke="true"; shift ;;
    --dry-run) dry_run="true"; shift ;;
    -h|--help) echo "Usage: $0 --instance-id ID [--config PATH] [--output-dir DIR] [--steps N] [--device DEVICE] [--limit N] [--variants a,b] [--smoke] [--dry-run]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$instance_id" ]] || { echo "ERROR: --instance-id is required" >&2; exit 2; }

cmd=(uv run python -m flower.sweep --config "$config" --device "$device" --output-dir "$output_dir")
[[ -z "$steps" ]] || cmd+=(--steps "$steps")
[[ -z "$limit" ]] || cmd+=(--limit "$limit")
[[ -z "$variants" ]] || cmd+=(--variants "$variants")
[[ "$smoke" != "true" ]] || cmd+=(--smoke)
printf -v remote_cmd '%q ' "${cmd[@]}"
remote_cmd="${remote_cmd% }"

upload_args=("$REPO_ROOT/scripts/vast_run_upload.sh" --instance-id "$instance_id" --cmd "$remote_cmd")
[[ "$dry_run" != "true" ]] || upload_args+=(--dry-run)
"${upload_args[@]}"
