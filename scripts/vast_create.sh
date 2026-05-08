#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vast_common.sh"
load_vast_defaults

dry_run="false"
yes="false"
offer_id=""
max_price="$VAST_MAX_PRICE"
offer_type="on-demand"
bid_price=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --offer-id) offer_id="$2"; shift 2 ;;
    --max-price) max_price="$2"; shift 2 ;;
    --offer-type) offer_type="$2"; shift 2 ;;
    --bid-price) bid_price="$2"; shift 2 ;;
    --dry-run) dry_run="true"; shift ;;
    --yes) yes="true"; shift ;;
    -h|--help) echo "Usage: $0 --offer-id ID [--max-price 0.20] [--dry-run] --yes"; usage_common; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$offer_id" ]] || { echo "ERROR: --offer-id is required" >&2; exit 2; }
validate_price "$max_price"
case "$offer_type" in on-demand|reserved|bid|interruptible) ;; *) echo "ERROR: invalid --offer-type: $offer_type" >&2; exit 2 ;; esac
if [[ -n "$bid_price" ]]; then validate_price "$bid_price"; fi
[[ "$yes" == "true" ]] || require_yes
if [[ "$dry_run" != "true" ]]; then ensure_vast_api_key; fi
extra_args=(--offer-type "$offer_type")
if [[ -n "$bid_price" ]]; then extra_args+=(--bid-price "$bid_price"); fi
cmd=(uvx --from vastai python "$REPO_ROOT/scripts/vast_create_instance.py" --offer-id "$offer_id" --image "$VAST_IMAGE" --disk "$VAST_DISK_GB" --max-price "$max_price" "${extra_args[@]}" --python-fallback "$VAST_PYTHON_FALLBACK" --yes)
if ! command -v uvx >/dev/null 2>&1; then
  cmd=(python3 "$REPO_ROOT/scripts/vast_create_instance.py" --offer-id "$offer_id" --image "$VAST_IMAGE" --disk "$VAST_DISK_GB" --max-price "$max_price" "${extra_args[@]}" --python-fallback "$VAST_PYTHON_FALLBACK" --yes)
fi
print_or_run "$dry_run" "${cmd[@]}"
