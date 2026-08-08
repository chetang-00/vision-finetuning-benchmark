from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RunningMetrics:
    loss_sum: float = 0.0
    top1_correct: int = 0
    top5_correct: int = 0
    samples: int = 0

    def update(self, loss: float, logits: torch.Tensor, targets: torch.Tensor) -> None:
        hard_targets = targets.argmax(dim=1) if targets.ndim == 2 else targets
        batch = hard_targets.size(0)
        max_k = min(5, logits.size(1))
        predictions = logits.topk(max_k, dim=1).indices
        self.loss_sum += float(loss) * batch
        self.top1_correct += int((predictions[:, 0] == hard_targets).sum().item())
        self.top5_correct += int(predictions.eq(hard_targets.view(-1, 1)).any(dim=1).sum().item())
        self.samples += batch

    def compute(self) -> dict[str, float]:
        denominator = max(1, self.samples)
        return {
            "loss": self.loss_sum / denominator,
            "top1": self.top1_correct / denominator,
            "top5": self.top5_correct / denominator,
        }


def soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    if targets.ndim == 1:
        return torch.nn.functional.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    if label_smoothing:
        classes = targets.size(1)
        targets = targets * (1 - label_smoothing) + label_smoothing / classes
    return -(targets * torch.nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
