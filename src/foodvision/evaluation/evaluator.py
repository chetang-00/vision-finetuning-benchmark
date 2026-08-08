from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from foodvision.config import AppConfig
from foodvision.data.food101 import DataBundle
from foodvision.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from foodvision.models.factory import build_model
from foodvision.runtime import RuntimeInfo


@torch.inference_mode()
def evaluate_checkpoint(
    config: AppConfig,
    data: DataBundle,
    runtime: RuntimeInfo,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=runtime.device, weights_only=False)
    bundle = build_model(replace(config.model, pretrained=False))
    model = bundle.model.to(runtime.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions: list[int] = []
    targets: list[int] = []
    confidences: list[float] = []
    failures: list[tuple[torch.Tensor, int, int, float]] = []
    top5_correct = 0
    for images, labels in tqdm(data.test, desc="test"):
        logits = model(images.to(runtime.device))
        probabilities = logits.softmax(dim=1)
        top5 = probabilities.topk(min(5, probabilities.size(1)), dim=1)
        predictions.extend(top5.indices[:, 0].cpu().tolist())
        confidences.extend(top5.values[:, 0].cpu().tolist())
        targets.extend(labels.tolist())
        for position, predicted in enumerate(top5.indices[:, 0].cpu().tolist()):
            actual = int(labels[position])
            if predicted != actual and len(failures) < 25:
                failures.append(
                    (
                        images[position].cpu(),
                        actual,
                        int(predicted),
                        float(top5.values[position, 0].cpu()),
                    )
                )
        top5_correct += int(top5.indices.cpu().eq(labels.view(-1, 1)).any(dim=1).sum())

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(data.classes))),
        target_names=data.classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(data.classes))))
    expected_calibration_error = _ece(np.asarray(confidences), np.equal(predictions, targets))
    result = {
        "top1": float(np.mean(np.equal(predictions, targets))),
        "top5": top5_correct / max(1, len(targets)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "expected_calibration_error": expected_calibration_error,
        "samples": len(targets),
        "per_class": {name: report[name] for name in data.classes},
    }
    output_dir = Path(config.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(output_dir / "confusion_matrix.npy", matrix)
    _plot_confusion(matrix, data.classes, output_dir / "confusion_matrix.png")
    _plot_failures(failures, data.classes, output_dir / "failure_gallery.png")
    return result


def _ece(confidences: np.ndarray, correctness: np.ndarray, bins: int = 15) -> float:
    score = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            score += float(mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean()))
    return score


def _plot_confusion(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title("Food-101 confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    ticks = np.arange(len(classes))
    axis.set_xticks(ticks[::5], [classes[i] for i in ticks[::5]], rotation=90, fontsize=6)
    axis.set_yticks(ticks[::5], [classes[i] for i in ticks[::5]], fontsize=6)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_failures(
    failures: list[tuple[torch.Tensor, int, int, float]], classes: list[str], path: Path
) -> None:
    if not failures:
        return
    figure, axes = plt.subplots(5, 5, figsize=(15, 15))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (image, actual, predicted, confidence) in zip(axes.flat, failures, strict=False):
        rendered = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axis.imshow(rendered)
        axis.set_title(
            f"actual: {classes[actual]}\npred: {classes[predicted]} ({confidence:.1%})",
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle("Representative high-confidence and random test failures")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
