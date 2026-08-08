from __future__ import annotations

from pathlib import Path

import yaml

MODELS = ["resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"]
STRATEGIES = ["frozen", "partial", "full"]


def generate_configs(base_path: Path, generated: Path) -> list[Path]:
    """Generate one isolated configuration per model and fine-tuning strategy."""
    source = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    generated.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in MODELS:
        for strategy in STRATEGIES:
            config = yaml.safe_load(yaml.safe_dump(source))
            experiment_dir = f"artifacts/{model}/{strategy}"
            config["project"]["name"] = f"foodvision-{model}-{strategy}"
            config["project"]["output_dir"] = experiment_dir
            config["model"]["name"] = model
            config["model"]["fine_tune"] = strategy
            config["inference"]["checkpoint"] = f"{experiment_dir}/best.pt"
            config["inference"]["onnx_model"] = f"{experiment_dir}/model.onnx"
            model_dir = generated / model
            model_dir.mkdir(parents=True, exist_ok=True)
            path = model_dir / f"{strategy}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            paths.append(path)
    return paths
