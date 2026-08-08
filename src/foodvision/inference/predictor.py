from __future__ import annotations

import io
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from foodvision.config import AppConfig
from foodvision.data.transforms import build_inference_transform
from foodvision.models import build_model
from foodvision.runtime import RuntimeInfo, select_device, synchronize


@dataclass(frozen=True)
class Prediction:
    label: str
    class_index: int
    probability: float


class Predictor:
    def __init__(
        self,
        config: AppConfig,
        runtime: RuntimeInfo | None = None,
        compile_model: bool = False,
    ) -> None:
        self.config = config
        self.runtime = runtime or select_device()
        self.transform = build_inference_transform(config.data.image_size)
        self.backend = config.inference.backend.lower()
        self.model: Any = None
        self.session: Any = None
        self.input_name = "images"
        self.classes: list[str] = []
        self.checkpoint_path = Path(config.inference.checkpoint)

        if self.backend == "pytorch":
            self._load_pytorch(compile_model)
        elif self.backend == "onnx":
            self._load_onnx()
        else:
            raise ValueError("inference.backend must be pytorch or onnx")

    def _load_pytorch(self, compile_model: bool) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. Run training first."
            )
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.runtime.device, weights_only=False
        )
        model_config = replace(self.config.model, pretrained=False)
        bundle = build_model(model_config)
        model: Any = bundle.model
        model.load_state_dict(checkpoint["model_state"])
        model.eval().to(self.runtime.device)
        if compile_model and hasattr(torch, "compile") and self.runtime.device.type != "mps":
            model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.classes = list(checkpoint.get("classes", []))
        if not self.classes:
            self.classes = [str(index) for index in range(self.config.model.num_classes)]

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install ONNX extras: pip install -e '.[onnx]'") from exc
        path = Path(self.config.inference.onnx_model)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}. Export the model first.")
        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        providers = [provider for provider in providers if provider in available]
        try:
            self.session = ort.InferenceSession(str(path), providers=providers)
        except Exception:
            if "CPUExecutionProvider" not in available or providers == ["CPUExecutionProvider"]:
                raise
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self.classes = list(checkpoint.get("classes", []))
        if not self.classes:
            self.classes = [str(index) for index in range(self.config.model.num_classes)]

    def prepare(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image.convert("RGB"))
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Inference transform did not return a tensor")
        return tensor

    def predict_bytes(self, payload: bytes, top_k: int | None = None) -> dict[str, Any]:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        return self.predict_images([image], top_k=top_k)[0]

    def predict_images(
        self, images: Sequence[Image.Image], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        if not images:
            return []
        batch = torch.stack([self.prepare(image) for image in images])
        start = time.perf_counter()
        if self.backend == "pytorch":
            with torch.inference_mode():
                logits = self.model(batch.to(self.runtime.device))
                probabilities = logits.softmax(dim=1).cpu().numpy()
            synchronize(self.runtime.device)
        else:
            logits = self.session.run(None, {self.input_name: batch.numpy()})[0]
            shifted = logits - logits.max(axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        k = min(top_k or self.config.inference.top_k, len(self.classes))
        results = []
        for row in probabilities:
            indices = np.argsort(row)[::-1][:k]
            predictions = [
                Prediction(self.classes[int(index)], int(index), float(row[index])).__dict__
                for index in indices
            ]
            results.append(
                {
                    "predictions": predictions,
                    "latency_ms": elapsed_ms / len(images),
                    "batch_latency_ms": elapsed_ms,
                    "backend": self.backend,
                    "device": self._execution_device(),
                    "model": self.config.model.name,
                }
            )
        return results

    def _execution_device(self) -> str:
        if self.backend == "pytorch":
            return self.runtime.accelerator
        return str(self.session.get_providers()[0])
