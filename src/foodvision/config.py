from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str = "foodvision-production-platform"
    seed: int = 42
    output_dir: str = "artifacts"


@dataclass
class DataConfig:
    root: str = "data"
    dataset: str = "food101"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    val_fraction: float = 0.1
    download: bool = True
    subset_fraction: float = 1.0
    augmentation: str = "strong"


@dataclass
class ModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 101
    fine_tune: str = "partial"
    unfreeze_blocks: int = 1
    dropout: float = 0.2
    gradient_checkpointing: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    backbone_learning_rate: float = 3e-5
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    precision: str = "auto"
    warmup_epochs: int = 2
    early_stopping_patience: int = 5
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    resume_from: str | None = None


@dataclass
class TrackingConfig:
    mlflow: bool = True
    mlflow_uri: str = "file:./mlruns"
    experiment_name: str = "foodvision-food101"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "foodvision-production-platform"


@dataclass
class InferenceConfig:
    backend: str = "pytorch"
    checkpoint: str = "artifacts/best.pt"
    onnx_model: str = "artifacts/model.onnx"
    top_k: int = 5
    warmup_iterations: int = 5


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_name = raw.pop("_base_", None)
    if base_name:
        base_path = (config_path.parent / base_name).resolve()
        with base_path.open("r", encoding="utf-8") as handle:
            base_raw = yaml.safe_load(handle) or {}
        base_raw.pop("_base_", None)
        raw = _deep_merge(base_raw, raw)

    config = AppConfig(
        project=ProjectConfig(**raw.get("project", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.data.dataset.lower() != "food101":
        raise ValueError("This release supports the Food-101 dataset only")
    if not 0 < config.data.subset_fraction <= 1:
        raise ValueError("data.subset_fraction must be in (0, 1]")
    if config.model.fine_tune not in {"frozen", "partial", "full"}:
        raise ValueError("model.fine_tune must be frozen, partial, or full")
    if config.training.accumulation_steps < 1:
        raise ValueError("training.accumulation_steps must be >= 1")
