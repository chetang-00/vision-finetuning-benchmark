from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from foodvision.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    ProjectConfig,
    TrackingConfig,
    TrainingConfig,
)
from foodvision.data.food101 import DataBundle
from foodvision.models.factory import ModelBundle
from foodvision.runtime import RuntimeInfo
from foodvision.training import Trainer


def test_training_engine_smoke(tmp_path: Path) -> None:
    images = torch.randn(6, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    data = DataBundle(loader, loader, loader, ["a", "b", "c"])
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    bundle = ModelBundle(
        model=model,
        architecture="tiny-test-model",
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )
    config = AppConfig(
        project=ProjectConfig(output_dir=str(tmp_path)),
        data=DataConfig(image_size=8, batch_size=2),
        model=ModelConfig(
            name="resnet50",
            pretrained=False,
            num_classes=3,
            fine_tune="full",
        ),
        training=TrainingConfig(
            epochs=1,
            learning_rate=1e-3,
            backbone_learning_rate=1e-3,
            warmup_epochs=0,
            early_stopping_patience=1,
            mixup_alpha=0.0,
            cutmix_alpha=0.0,
        ),
        tracking=TrackingConfig(mlflow=False, tensorboard=False, wandb=False),
    )
    runtime = RuntimeInfo(torch.device("cpu"), "CPU", "cpu")
    result = Trainer(config, bundle, data, runtime).fit()
    assert 0 <= result["best_validation_top1"] <= 1
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "history.json").exists()
