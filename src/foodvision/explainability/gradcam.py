from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Grad-CAM for CNN-like models; selects the last convolution when omitted."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or self._last_convolution(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)

    @staticmethod
    def _last_convolution(model: nn.Module) -> nn.Module:
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not convolutions:
            raise ValueError(
                "Grad-CAM requires a convolutional model; use attention rollout for ViT"
            )
        return convolutions[-1]

    def _forward_hook(self, _module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
        self.activations = output
        output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient.detach()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        index = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        logits[:, index].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations.detach()).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
