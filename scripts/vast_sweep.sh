#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vast_common.sh"
load_vast_defaults

dry_run="false"
yes="false"
max_price="$VAST_MAX_PRICE"
query="$VAST_QUERY"
limit="1"
cmd_template='uv run python -m flower.train --config configs/research_sweep_remote.yaml --variant {variant} --device auto --steps 30000'
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-price) max_price="$2"; shift 2 ;;
    --query) query="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --cmd-template) cmd_template="$2"; shift 2 ;;
    --dry-run) dry_run="true"; shift ;;
    --yes) yes="true"; shift ;;
    -h|--help) echo "Usage: $0 [--max-price 0.20] [--limit 1] [--cmd-template CMD_WITH_{variant}] [--dry-run] --yes"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
validate_price "$max_price"
[[ "$yes" == "true" ]] || require_yes
if [[ "$dry_run" == "true" ]]; then
  echo "DRY RUN: would search offers: $query dph<=$max_price"
  python3 - <<'PY'
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path('configs/research_sweep_remote.yaml').read_text())
for v in cfg['sweep']['variants']:
    print(v['name'])
PY
  exit 0
fi
ensure_vast_api_key
ensure_vast_cli
mapfile -t offers < <(vast search offers "$query dph<=$max_price" --raw | python3 - "$limit" <<'PY'
import json, sys
n=int(sys.argv[1]); data=json.load(sys.stdin)
for row in sorted(data, key=lambda r: float(r.get('dph_total') or r.get('dph_base') or 999))[:n]:
    print(row.get('id') or row.get('ask_contract_id'))
PY
)
[[ ${#offers[@]} -gt 0 ]] || { echo "ERROR: no offers found" >&2; exit 1; }
python3 - <<'PY' > /tmp/flower-vast-variants.txt
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path('configs/research_sweep_remote.yaml').read_text())
for v in cfg['sweep']['variants']:
    print(v['name'])
PY
for variant in $(cat /tmp/flower-vast-variants.txt); do
  offer="${offers[0]}"
  create_out="$("$REPO_ROOT/scripts/vast_create.sh" --offer-id "$offer" --max-price "$max_price" --yes)"
  echo "$create_out"
  instance_id="$(printf '%s\n' "$create_out" | grep -Eo '[0-9]+' | tail -1)"
  [[ -n "$instance_id" ]] || { echo "ERROR: could not determine instance id" >&2; exit 1; }
  cmd="${cmd_template//\{variant\}/$variant} --metrics-json runs/vast/${variant}.json"
  "$REPO_ROOT/scripts/vast_run_upload.sh" --instance-id "$instance_id" --cmd "$cmd"
done
