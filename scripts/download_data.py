#!/usr/bin/env python3
from __future__ import annotations

import argparse

from foodvision.config import load_config
from foodvision.data import build_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate Food-101")
    parser.add_argument("--config", default="configs/quickstart.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    bundle = build_dataloaders(config.data, config.project.seed)
    print(
        f"Food-101 ready: train={len(bundle.train.dataset):,}, "
        f"validation={len(bundle.validation.dataset):,}, test={len(bundle.test.dataset):,}"
    )


if __name__ == "__main__":
    main()
