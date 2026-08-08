from __future__ import annotations

import torch


def mix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, targets

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or torch.rand(1).item() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(images.size(0), device=images.device)

    if use_cutmix:
        height, width = images.shape[-2:]
        ratio = (1.0 - lam) ** 0.5
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        center_y = int(torch.randint(height, (1,)).item())
        center_x = int(torch.randint(width, (1,)).item())
        y1, y2 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
        x1, x2 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
    else:
        mixed = lam * images + (1.0 - lam) * images[permutation]
    return mixed, lam * targets + (1.0 - lam) * targets[permutation]
