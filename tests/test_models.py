from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

from foodvision.config import ModelConfig
from foodvision.models import build_model


@pytest.mark.parametrize("strategy", ["frozen", "partial", "full"])
def test_fine_tuning_strategies(strategy: str) -> None:
    config = replace(
        ModelConfig(),
        name="resnet50",
        pretrained=False,
        num_classes=101,
        fine_tune=strategy,
    )
    bundle = build_model(config)
    assert bundle.trainable_parameters > 0
    assert bundle.total_parameters >= bundle.trainable_parameters
    if strategy == "frozen":
        assert bundle.trainable_parameters < bundle.total_parameters
    if strategy == "full":
        assert bundle.trainable_parameters == bundle.total_parameters


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(replace(ModelConfig(), name="not-a-model", pretrained=False))
