#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vast_common.sh"
load_vast_defaults

action="stop"
instance_id=""
yes="false"
dry_run="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-id) instance_id="$2"; shift 2 ;;
    --action) action="$2"; shift 2 ;;
    --yes) yes="true"; shift ;;
    --dry-run) dry_run="true"; shift ;;
    -h|--help) echo "Usage: $0 --instance-id ID [--action stop|destroy] [--dry-run] --yes"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$instance_id" ]] || { echo "ERROR: --instance-id is required" >&2; exit 2; }
[[ "$action" == "stop" || "$action" == "destroy" ]] || { echo "ERROR: --action must be stop or destroy" >&2; exit 2; }
[[ "$yes" == "true" ]] || require_yes
if [[ "$dry_run" != "true" ]]; then ensure_vast_api_key; fi
cmd=(uvx --from vastai python - "$action" "$instance_id")
if ! command -v uvx >/dev/null 2>&1; then
  cmd=(python3 - "$action" "$instance_id")
fi
print_or_run "$dry_run" "${cmd[@]}" <<'PY'
import json
import os
import sys
from vastai import VastAI

action, instance_id = sys.argv[1], int(sys.argv[2])
client = VastAI(api_key=os.environ["VAST_API_KEY"])
if action == "stop":
    result = client.stop_instance(instance_id)
elif action == "destroy":
    result = client.destroy_instance(instance_id)
else:
    raise SystemExit(f"unsupported action: {action}")
print(json.dumps(result, sort_keys=True))
PY
