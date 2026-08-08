#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.datasets import Food101

from foodvision.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Food-101 class distribution")
    parser.add_argument("--config", default="configs/quickstart.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train = Food101(config.data.root, split="train", download=config.data.download)
    test = Food101(config.data.root, split="test", download=config.data.download)
    train_counts = Counter(train._labels)
    test_counts = Counter(test._labels)
    report = {
        "classes": len(train.classes),
        "train_images": len(train),
        "test_images": len(test),
        "distribution": {
            name: {"train": train_counts[index], "test": test_counts[index]}
            for index, name in enumerate(train.classes)
        },
    }
    output_dir = Path(config.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(18, 6))
    axis.bar(
        range(len(train.classes)),
        [train_counts[index] for index in range(len(train.classes))],
    )
    axis.set_title("Food-101 training images per class")
    axis.set_xlabel("Class index")
    axis.set_ylabel("Images")
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(figure)
    summary = {key: value for key, value in report.items() if key != "distribution"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
