#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vast_common.sh"
load_vast_defaults

instance_id=""
dest="$REPO_ROOT/runs/vast_pull"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-id) instance_id="$2"; shift 2 ;;
    --dest) dest="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 --instance-id ID [--dest PATH]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$instance_id" ]] || { echo "ERROR: --instance-id is required" >&2; exit 2; }
ensure_vast_api_key
mkdir -p "$dest"
ssh_url="$(python3 "$REPO_ROOT/scripts/vast_instance_info.py" ssh-url "$instance_id")"
read -r user host port < <(ssh_args_from_url "$ssh_url")
build_ssh_opts "$port"
scp -r "${scp_opts[@]}" "$user@$host:$VAST_REMOTE_DIR/$VAST_RUN_DIR/" "$dest/"
