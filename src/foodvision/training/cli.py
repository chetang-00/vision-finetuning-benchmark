from __future__ import annotations

import argparse

from foodvision.config import load_config
from foodvision.data import build_dataloaders
from foodvision.models import build_model
from foodvision.runtime import seed_everything, select_device
from foodvision.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FoodVision models")
    parser.add_argument("--config", default="configs/quickstart.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.project.seed)
    runtime = select_device(args.device)
    print(f"Using {runtime.accelerator}: {runtime.name}")
    data = build_dataloaders(config.data, config.project.seed)
    model = build_model(config.model)
    print(
        f"Model {model.architecture}: {model.trainable_parameters:,} trainable / "
        f"{model.total_parameters:,} total parameters"
    )
    result = Trainer(config, model, data, runtime).fit()
    print(result)


if __name__ == "__main__":
    main()
