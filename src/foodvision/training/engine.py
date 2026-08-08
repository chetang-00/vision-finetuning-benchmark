from __future__ import annotations

import json
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm

from foodvision.config import AppConfig
from foodvision.data.food101 import DataBundle
from foodvision.models.factory import ModelBundle, parameter_groups
from foodvision.runtime import RuntimeInfo
from foodvision.training.augment import mix_batch
from foodvision.training.metrics import RunningMetrics, soft_cross_entropy
from foodvision.training.scheduler import warmup_cosine_scheduler
from foodvision.training.tracking import ExperimentTracker


class Trainer:
    def __init__(
        self,
        config: AppConfig,
        model_bundle: ModelBundle,
        data: DataBundle,
        runtime: RuntimeInfo,
    ) -> None:
        self.config = config
        self.model_bundle = model_bundle
        self.model = model_bundle.model.to(runtime.device)
        self.data = data
        self.runtime = runtime
        self.output_dir = Path(config.project.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        groups = parameter_groups(
            self.model,
            config.training.learning_rate,
            config.training.backbone_learning_rate,
        )
        self.optimizer = AdamW(groups, weight_decay=config.training.weight_decay)
        updates_per_epoch = max(1, len(data.train) // config.training.accumulation_steps)
        self.scheduler = warmup_cosine_scheduler(
            self.optimizer,
            updates_per_epoch * config.training.epochs,
            updates_per_epoch * config.training.warmup_epochs,
        )
        self.scaler = self._build_scaler()
        run_name = f"{config.model.name}-{config.model.fine_tune}-{int(time.time())}"
        self.tracker = ExperimentTracker(config, run_name)
        self.start_epoch = 0
        self.best_top1 = 0.0
        self.bad_epochs = 0
        if config.training.resume_from:
            self._resume(Path(config.training.resume_from))

    def _build_scaler(self) -> Any | None:
        precision = self.config.training.precision
        use_fp16 = precision in {"fp16", "16"} or (
            precision == "auto" and self.runtime.device.type == "cuda"
        )
        if self.runtime.device.type == "cuda" and use_fp16:
            return torch.amp.GradScaler("cuda")
        return None

    def _autocast(self) -> Any:
        precision = self.config.training.precision
        device_type = self.runtime.device.type
        if device_type == "cuda" and precision in {"auto", "fp16", "16"}:
            return torch.autocast("cuda", dtype=torch.float16)
        if device_type == "cuda" and precision in {"bf16", "bfloat16"}:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if device_type == "mps" and precision in {"fp16", "16"}:
            return torch.autocast("mps", dtype=torch.float16)
        if device_type == "cpu" and precision in {"bf16", "bfloat16"}:
            return torch.autocast("cpu", dtype=torch.bfloat16)
        return nullcontext()

    def fit(self) -> dict[str, float]:
        self.tracker.start()
        history = []
        try:
            for epoch in range(self.start_epoch, self.config.training.epochs):
                train_metrics = self._train_epoch(epoch)
                validation_metrics = self._evaluate(self.data.validation, "validation")
                metrics = {
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"validation/{key}": value for key, value in validation_metrics.items()},
                    "learning_rate": self.optimizer.param_groups[-1]["lr"],
                }
                history.append({"epoch": epoch + 1, **metrics})
                self.tracker.log_metrics(metrics, epoch + 1)
                self._save_checkpoint("last.pt", epoch, validation_metrics["top1"])

                if validation_metrics["top1"] > self.best_top1:
                    self.best_top1 = validation_metrics["top1"]
                    self.bad_epochs = 0
                    self._save_checkpoint("best.pt", epoch, self.best_top1)
                else:
                    self.bad_epochs += 1

                print(
                    f"epoch={epoch + 1} train_top1={train_metrics['top1']:.4f} "
                    f"val_top1={validation_metrics['top1']:.4f} "
                    f"val_top5={validation_metrics['top5']:.4f}"
                )
                if self.bad_epochs >= self.config.training.early_stopping_patience:
                    print("Early stopping triggered")
                    break

            history_path = self.output_dir / "history.json"
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            self.tracker.log_artifact(history_path)
            return {"best_validation_top1": self.best_top1}
        finally:
            self.tracker.close()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        metrics = RunningMetrics()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = self.config.training.accumulation_steps
        progress = tqdm(self.data.train, desc=f"train {epoch + 1}", leave=False)

        for batch_index, (images, labels) in enumerate(progress):
            images = images.to(self.runtime.device, non_blocking=True)
            labels = labels.to(self.runtime.device, non_blocking=True)
            mixed_images, mixed_targets = mix_batch(
                images,
                labels,
                self.config.model.num_classes,
                self.config.training.mixup_alpha,
                self.config.training.cutmix_alpha,
            )
            with self._autocast():
                logits = self.model(mixed_images)
                loss = soft_cross_entropy(
                    logits, mixed_targets, self.config.training.label_smoothing
                )
                scaled_loss = loss / accumulation

            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_update = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(self.data.train)
            )
            if should_update:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            metrics.update(loss.item(), logits.detach(), labels)
            progress.set_postfix(loss=f"{metrics.compute()['loss']:.3f}")
        return metrics.compute()

    @torch.inference_mode()
    def _evaluate(self, loader: Iterable[Any], split: str) -> dict[str, float]:
        self.model.eval()
        metrics = RunningMetrics()
        for images, labels in tqdm(loader, desc=split, leave=False):
            images = images.to(self.runtime.device, non_blocking=True)
            labels = labels.to(self.runtime.device, non_blocking=True)
            with self._autocast():
                logits = self.model(images)
                loss = soft_cross_entropy(logits, labels, self.config.training.label_smoothing)
            metrics.update(loss.item(), logits, labels)
        return metrics.compute()

    def _save_checkpoint(self, filename: str, epoch: int, score: float) -> None:
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "score": score,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "config": self.config.to_dict(),
                "classes": self.data.classes,
                "architecture": self.model_bundle.architecture,
            },
            path,
        )
        if filename == "best.pt":
            self.tracker.log_artifact(path)

    def _resume(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.runtime.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_top1 = float(checkpoint.get("score", 0.0))
