#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from foodvision.config import load_config
from foodvision.data import build_dataloaders
from foodvision.evaluation import evaluate_checkpoint
from foodvision.runtime import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FoodVision checkpoint")
    parser.add_argument("--config", default="configs/quickstart.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    config = load_config(args.config)
    runtime = select_device(args.device)
    data = build_dataloaders(config.data, config.project.seed)
    result = evaluate_checkpoint(
        config, data, runtime, args.checkpoint or config.inference.checkpoint
    )
    summary = {key: value for key, value in result.items() if key != "per_class"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
