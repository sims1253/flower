#!/usr/bin/env python3
from __future__ import annotations

import argparse


def select_image(cuda_max_good: float | None) -> str:
    """Select a PyTorch image whose CUDA runtime does not exceed the host driver."""
    if cuda_max_good is None:
        return "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel"
    if cuda_max_good >= 12.8:
        return "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"
    if cuda_max_good >= 12.4:
        return "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
    return "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel"


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a CUDA-compatible PyTorch image for Vast benchmarks.")
    parser.add_argument("cuda_max_good", nargs="?", type=float)
    args = parser.parse_args()
    print(select_image(args.cuda_max_good))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
