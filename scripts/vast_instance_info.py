#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Vast.ai instance connection/status information using the Python SDK.")
    parser.add_argument("action", choices=["ssh-url", "status"])
    parser.add_argument("instance_id", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("ERROR: VAST_API_KEY must be set in the environment", file=sys.stderr)
        return 2

    from vastai import VastAI

    client = VastAI(api_key=api_key)
    if args.action == "ssh-url":
        print(client.ssh_url(args.instance_id))
    else:
        result: dict[str, Any] = client.show_instance(args.instance_id)
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
