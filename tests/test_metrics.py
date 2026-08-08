import torch

from foodvision.training.metrics import RunningMetrics, soft_cross_entropy


def test_running_metrics() -> None:
    logits = torch.tensor([[5.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    targets = torch.tensor([0, 1])
    metrics = RunningMetrics()
    metrics.update(0.25, logits, targets)
    result = metrics.compute()
    assert result["top1"] == 1.0
    assert result["top5"] == 1.0
    assert result["loss"] == 0.25


def test_soft_cross_entropy_accepts_mixed_targets() -> None:
    logits = torch.tensor([[2.0, 0.5]])
    targets = torch.tensor([[0.7, 0.3]])
    assert soft_cross_entropy(logits, targets).item() > 0
