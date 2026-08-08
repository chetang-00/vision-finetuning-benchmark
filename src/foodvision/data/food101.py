from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import Food101

from foodvision.config import DataConfig
from foodvision.data.transforms import build_transforms


@dataclass
class DataBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    classes: list[str]


def _fractional_subset(dataset: Any, fraction: float, seed: int) -> Dataset:
    if fraction >= 1:
        return cast(Dataset, dataset)
    count = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def build_dataloaders(config: DataConfig, seed: int = 42) -> DataBundle:
    train_transform, evaluation_transform = build_transforms(
        config.image_size, config.augmentation
    )
    root = Path(config.root)

    full_train_aug = Food101(root, split="train", transform=train_transform, download=config.download)
    full_train_eval = Food101(
        root, split="train", transform=evaluation_transform, download=config.download
    )
    test_dataset = Food101(
        root, split="test", transform=evaluation_transform, download=config.download
    )

    generator = torch.Generator().manual_seed(seed)
    val_count = max(1, int(len(full_train_aug) * config.val_fraction))
    train_count = len(full_train_aug) - val_count
    all_indices = torch.randperm(len(full_train_aug), generator=generator).tolist()
    validation_indices = all_indices[:val_count]
    train_indices = all_indices[val_count : val_count + train_count]
    train_dataset: Dataset = Subset(full_train_aug, train_indices)
    validation_dataset: Dataset = Subset(full_train_eval, validation_indices)

    train_dataset = _fractional_subset(train_dataset, config.subset_fraction, seed)
    validation_dataset = _fractional_subset(validation_dataset, config.subset_fraction, seed + 1)
    test_dataset = _fractional_subset(test_dataset, config.subset_fraction, seed + 2)

    classes = list(full_train_aug.classes)
    root.mkdir(parents=True, exist_ok=True)
    (root / "food101_classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    return DataBundle(
        train=DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        ),
        validation=DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        ),
        test=DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        ),
        classes=classes,
    )
