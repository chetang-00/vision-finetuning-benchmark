#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

MODELS = ["resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"]
STRATEGIES = ["frozen", "partial", "full"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or run the model/fine-tuning matrix")
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument(
        "--run", action="store_true", help="Run experiments after generating configs"
    )
    args = parser.parse_args()
    source = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    generated = Path("configs/generated")
    generated.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in MODELS:
        for strategy in STRATEGIES:
            config = yaml.safe_load(yaml.safe_dump(source))
            config["model"]["name"] = model
            config["model"]["fine_tune"] = strategy
            config["project"]["output_dir"] = f"artifacts/{model}-{strategy}"
            path = generated / f"{model}-{strategy}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            paths.append(path)
    print(f"Generated {len(paths)} experiment configurations in {generated}")
    if args.run:
        for path in paths:
            subprocess.run([sys.executable, "scripts/train.py", "--config", str(path)], check=True)


if __name__ == "__main__":
    main()
