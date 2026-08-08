from pathlib import Path

import pytest

from foodvision.config import load_config
from foodvision.experiments import MODELS, STRATEGIES, generate_configs

ROOT = Path(__file__).resolve().parents[1]


def test_load_quickstart_config() -> None:
    config = load_config(ROOT / "configs/quickstart.yaml")
    assert config.model.name == "resnet50"
    assert config.model.num_classes == 101
    assert config.data.subset_fraction == 0.1
    assert config.project.output_dir == "artifacts/quickstart/resnet50-frozen"
    assert config.inference.checkpoint == "artifacts/quickstart/resnet50-frozen/best.pt"


def test_base_configuration_merge() -> None:
    config = load_config(ROOT / "configs/vit_full.yaml")
    assert config.model.name == "vit_b_16"
    assert config.model.fine_tune == "full"
    assert config.data.dataset == "food101"
    assert config.project.output_dir == "artifacts/vit_b_16/full"
    assert config.inference.checkpoint == "artifacts/vit_b_16/full/best.pt"


def test_base_resnet_paths_are_isolated() -> None:
    config = load_config(ROOT / "configs/base.yaml")
    assert config.model.name == "resnet50"
    assert config.model.fine_tune == "partial"
    assert config.project.output_dir == "artifacts/resnet50/partial"
    assert config.inference.checkpoint == "artifacts/resnet50/partial/best.pt"


def test_generated_matrix_has_unique_consistent_paths(tmp_path: Path) -> None:
    paths = generate_configs(ROOT / "configs/base.yaml", tmp_path / "generated")
    assert len(paths) == len(MODELS) * len(STRATEGIES) == 12
    output_dirs = set()
    checkpoint_paths = set()
    onnx_paths = set()
    for path in paths:
        config = load_config(path)
        expected_dir = f"artifacts/{config.model.name}/{config.model.fine_tune}"
        assert config.project.output_dir == expected_dir
        assert config.inference.checkpoint == f"{expected_dir}/best.pt"
        assert config.inference.onnx_model == f"{expected_dir}/model.onnx"
        output_dirs.add(config.project.output_dir)
        checkpoint_paths.add(config.inference.checkpoint)
        onnx_paths.add(config.inference.onnx_model)
    assert len(output_dirs) == 12
    assert len(checkpoint_paths) == 12
    assert len(onnx_paths) == 12


def test_rejects_invalid_subset(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  subset_fraction: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="subset_fraction"):
        load_config(path)
