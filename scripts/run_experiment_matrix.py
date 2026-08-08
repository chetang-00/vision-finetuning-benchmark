#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from foodvision.experiments import generate_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or run the model/fine-tuning matrix")
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="configs/generated")
    parser.add_argument(
        "--run", action="store_true", help="Run experiments after generating configs"
    )
    args = parser.parse_args()
    generated = Path(args.output_dir)
    paths = generate_configs(Path(args.base), generated)
    print(f"Generated {len(paths)} experiment configurations in {generated}")
    if args.run:
        for path in paths:
            subprocess.run([sys.executable, "scripts/train.py", "--config", str(path)], check=True)


if __name__ == "__main__":
    main()
