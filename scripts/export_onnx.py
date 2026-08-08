#!/usr/bin/env python3
from __future__ import annotations

import argparse

from foodvision.config import load_config
from foodvision.export import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a FoodVision checkpoint to ONNX")
    parser.add_argument("--config", default="configs/quickstart.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    path = export_onnx(config, args.checkpoint)
    print(f"Exported ONNX model to {path}")


if __name__ == "__main__":
    main()
