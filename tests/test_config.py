from pathlib import Path

import pytest

from foodvision.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_quickstart_config() -> None:
    config = load_config(ROOT / "configs/quickstart.yaml")
    assert config.model.name == "resnet50"
    assert config.model.num_classes == 101
    assert config.data.subset_fraction == 0.1


def test_base_configuration_merge() -> None:
    config = load_config(ROOT / "configs/vit_full.yaml")
    assert config.model.name == "vit_b_16"
    assert config.model.fine_tune == "full"
    assert config.data.dataset == "food101"


def test_rejects_invalid_subset(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  subset_fraction: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="subset_fraction"):
        load_config(path)
