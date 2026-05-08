#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

HARD_MAX_PRICE = 0.20


def positive_price(value: str) -> float:
    price = float(value)
    if price > HARD_MAX_PRICE:
        raise argparse.ArgumentTypeError(f"max price {price:.4f} exceeds hard safety default {HARD_MAX_PRICE:.2f}/hr")
    return price


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create one Vast.ai instance using the Python SDK.")
    parser.add_argument("--offer-id", required=True, type=int)
    parser.add_argument("--image", required=True)
    parser.add_argument("--disk", required=True, type=float)
    parser.add_argument("--max-price", required=True, type=positive_price)
    parser.add_argument("--offer-type", choices=("on-demand", "reserved", "bid", "interruptible"), default="on-demand")
    parser.add_argument("--bid-price", type=positive_price)
    parser.add_argument("--python-fallback", default="3.12")
    parser.add_argument("--yes", action="store_true", help="Required confirmation for spending money")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.yes:
        print("ERROR: --yes is required to create an instance", file=sys.stderr)
        return 2

    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("ERROR: VAST_API_KEY must be set in the environment", file=sys.stderr)
        return 2

    from vastai import VastAI

    price = None
    if args.offer_type in ("bid", "interruptible"):
        price = args.bid_price if args.bid_price is not None else args.max_price

    client = VastAI(api_key=api_key)
    result: dict[str, Any] = client.create_instance(
        args.offer_id,
        image=args.image,
        disk=args.disk,
        price=price,
        env={"VAST_PYTHON_FALLBACK": args.python_fallback},
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
