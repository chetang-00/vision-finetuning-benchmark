#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from benchmarks.inference import run_benchmark
from foodvision.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FoodVision inference")
    parser.add_argument("--config", default="configs/quickstart.yaml")
    parser.add_argument("--batch-sizes", default="1,8,16,32")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    sizes = [int(value) for value in args.batch_sizes.split(",")]
    report = run_benchmark(config, sizes, args.iterations, args.compile)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
