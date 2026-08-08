from __future__ import annotations

from typing import Tuple

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, augmentation: str) -> Tuple[transforms.Compose, transforms.Compose]:
    if augmentation not in {"none", "standard", "strong"}:
        raise ValueError("augmentation must be none, standard, or strong")

    train_steps = [transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0))]
    if augmentation in {"standard", "strong"}:
        train_steps.append(transforms.RandomHorizontalFlip())
    if augmentation == "strong":
        train_steps.extend(
            [
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomGrayscale(p=0.05),
            ]
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.15 if augmentation == "strong" else 0.0),
        ]
    )

    evaluation = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_steps), evaluation


def build_inference_transform(image_size: int) -> transforms.Compose:
    return build_transforms(image_size, "none")[1]

