from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import timm
import torch.nn as nn

from foodvision.config import ModelConfig

MODEL_ALIASES: dict[str, str] = {
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "vit_b_16": "vit_base_patch16_224",
    "vit_tiny": "vit_tiny_patch16_224",
}


@dataclass
class ModelBundle:
    model: nn.Module
    architecture: str
    trainable_parameters: int
    total_parameters: int


def _is_head(name: str) -> bool:
    return any(token in name for token in ("classifier", "fc", "head"))


def _configure_fine_tuning(model: nn.Module, strategy: str, unfreeze_blocks: int) -> None:
    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if _is_head(name):
            parameter.requires_grad = True

    if strategy == "partial":
        blocks = getattr(model, "blocks", None) or getattr(model, "stages", None)
        if blocks is not None:
            modules = list(blocks)[-max(1, unfreeze_blocks) :]
        else:
            modules = []
            for layer_name in ("layer4", "layer3", "layer2", "layer1"):
                if hasattr(model, layer_name) and len(modules) < max(1, unfreeze_blocks):
                    modules.append(getattr(model, layer_name))
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True


def build_model(config: ModelConfig) -> ModelBundle:
    if config.name not in MODEL_ALIASES:
        raise ValueError(f"Unknown model '{config.name}'. Choose from {sorted(MODEL_ALIASES)}")
    model = timm.create_model(
        MODEL_ALIASES[config.name],
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        drop_rate=config.dropout,
    )
    if config.gradient_checkpointing:
        setter = getattr(model, "set_grad_checkpointing", None)
        if setter is None:
            raise ValueError(f"{config.name} does not expose gradient checkpointing through timm")
        setter(enable=True)

    _configure_fine_tuning(model, config.fine_tune, config.unfreeze_blocks)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ModelBundle(model, config.name, trainable, total)


def parameter_groups(
    model: nn.Module, head_lr: float, backbone_lr: float
) -> list[dict[str, Iterable[nn.Parameter] | float]]:
    head: list[nn.Parameter] = []
    backbone: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head if _is_head(name) else backbone).append(parameter)
    groups: list[dict[str, Iterable[nn.Parameter] | float]] = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    if head:
        groups.append({"params": head, "lr": head_lr})
    return groups
