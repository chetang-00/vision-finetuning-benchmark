from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from foodvision.config import AppConfig
from foodvision.models import build_model


def export_onnx(config: AppConfig, checkpoint_path: str | Path | None = None) -> Path:
    checkpoint_path = Path(checkpoint_path or config.inference.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = build_model(replace(config.model, pretrained=False))
    bundle.model.load_state_dict(checkpoint["model_state"])
    bundle.model.eval()
    output = Path(config.inference.onnx_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, config.data.image_size, config.data.image_size)
    torch.onnx.export(
        bundle.model,
        (example,),
        output,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    return output
