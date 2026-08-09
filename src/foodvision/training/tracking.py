from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foodvision.config import AppConfig


class ExperimentTracker:
    """Fan-out tracker with optional MLflow, TensorBoard, and W&B integrations."""

    def __init__(self, config: AppConfig, run_name: str) -> None:
        self.config = config
        self.run_name = run_name
        self.mlflow: Any | None = None
        self.writer: Any | None = None
        self.wandb: Any | None = None
        self.output_dir = Path(config.project.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        flat_config = _flatten(self.config.to_dict())
        if self.config.tracking.mlflow:
            try:
                import mlflow
            except ImportError as exc:
                message = "Install tracking extras: python -m pip install '.[tracking]'"
                raise RuntimeError(message) from exc
            mlflow.set_tracking_uri(self.config.tracking.mlflow_uri)
            mlflow.set_experiment(self.config.tracking.experiment_name)
            mlflow.start_run(run_name=self.run_name)
            mlflow.log_params(flat_config)
            self.mlflow = mlflow

        if self.config.tracking.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError("TensorBoard is enabled but not installed") from exc
            self.writer = SummaryWriter(self.output_dir / "tensorboard" / self.run_name)

        if self.config.tracking.wandb:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("W&B is enabled but not installed") from exc
            wandb.init(
                project=self.config.tracking.wandb_project,
                name=self.run_name,
                config=flat_config,
            )
            self.wandb = wandb

        (self.output_dir / "resolved_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
        if self.wandb:
            self.wandb.log({**metrics, "epoch": step})

    def log_artifact(self, path: Path) -> None:
        if self.mlflow and path.exists():
            self.mlflow.log_artifact(str(path))

    def close(self) -> None:
        if self.writer:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()
        if self.mlflow:
            self.mlflow.end_run()


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        elif value is not None:
            result[name] = value
    return result
