#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vast_common.sh"
load_vast_defaults

limit="20"
query="$VAST_QUERY"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) limit="$2"; shift 2 ;;
    --query) query="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--limit N] [--query VAST_QUERY]"; usage_common; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ensure_vast_api_key
ensure_vast_cli

# Current Vast CLI versions expose JSON reliably through the Python SDK. Some
# CLI `search offers --raw` builds return Python objects to the dispatcher or
# segfault after printing table output, so keep SDK JSON as the primary path and
# retain CLI text parsing as a fallback for older installs.
validate_price "$VAST_MAX_PRICE"
if command -v uvx >/dev/null 2>&1; then
  uvx --from vastai python - "$VAST_API_KEY" "$query" "$limit" <<'PY' | python3 "$REPO_ROOT/scripts/vast_parse_offers.py" "$limit" "$VAST_MAX_PRICE"
import json, sys
from vastai import VastAI
api_key, query, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = VastAI(api_key=api_key).search_offers(query=query, limit=limit)
print(json.dumps(rows))
PY
else
  vast search offers "$query" --limit "$limit" --no-color | python3 "$REPO_ROOT/scripts/vast_parse_offers.py" "$limit" "$VAST_MAX_PRICE"
fi
