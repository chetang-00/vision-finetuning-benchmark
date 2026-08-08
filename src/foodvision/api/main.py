from __future__ import annotations

import base64
import io
import os
import threading
import uuid
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any, cast

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from foodvision.api.schemas import HealthResponse, MetricsResponse, PredictionResponse
from foodvision.config import load_config
from foodvision.explainability import GradCAM
from foodvision.inference import Predictor
from foodvision.runtime import select_device

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


class ServiceMetrics:
    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0
        self.images = 0
        self.total_latency_ms = 0.0
        self.lock = threading.Lock()

    def record(self, images: int, latency_ms: float) -> None:
        with self.lock:
            self.requests += 1
            self.images += images
            self.total_latency_ms += latency_ms

    def error(self) -> None:
        with self.lock:
            self.errors += 1

    def snapshot(self) -> dict[str, float | int]:
        with self.lock:
            return {
                "requests": self.requests,
                "errors": self.errors,
                "images": self.images,
                "average_latency_ms": self.total_latency_ms / max(1, self.requests),
            }


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.getenv("FOODVISION_CONFIG", "configs/quickstart.yaml")
    app.state.metrics = ServiceMetrics()
    app.state.predictor = None
    app.state.startup_error = None
    try:
        config = load_config(config_path)
        checkpoint = os.getenv("FOODVISION_CHECKPOINT")
        if checkpoint:
            config.inference.checkpoint = checkpoint
        onnx_model = os.getenv("FOODVISION_ONNX_MODEL")
        if onnx_model:
            config.inference.onnx_model = onnx_model
        runtime = select_device(os.getenv("FOODVISION_DEVICE", "auto"))
        app.state.predictor = Predictor(config, runtime=runtime)
    except Exception as exc:
        app.state.startup_error = str(exc)
    yield


app = FastAPI(
    title="FoodVision Production API",
    version="0.1.0",
    description="Food-101 classification with PyTorch and ONNX Runtime backends.",
    lifespan=lifespan,
)


def _predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not ready: {request.app.state.startup_error}",
        )
    return cast(Predictor, predictor)


async def _read_image(upload: UploadFile) -> bytes:
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    payload = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid or corrupted image") from exc
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return files("foodvision").joinpath("web/index.html").read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = request.app.state.predictor
    if predictor is None:
        return HealthResponse(
            status="degraded",
            model_ready=False,
            detail=request.app.state.startup_error,
        )
    return HealthResponse(
        status="healthy",
        model_ready=True,
        backend=predictor.backend,
        device=predictor.runtime.name,
    )


@app.get("/models")
def models(request: Request) -> dict[str, Any]:
    predictor = _predictor(request)
    return {
        "active": predictor.config.model.name,
        "backend": predictor.backend,
        "classes": len(predictor.classes),
        "fine_tune_strategy": predictor.config.model.fine_tune,
    }


@app.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> dict[str, float | int]:
    service_metrics = cast(ServiceMetrics, request.app.state.metrics)
    return service_metrics.snapshot()


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    try:
        payload = await _read_image(image)
        result = _predictor(request).predict_bytes(payload, top_k=top_k)
        request.app.state.metrics.record(1, result["batch_latency_ms"])
        return {"request_id": str(uuid.uuid4()), **result}
    except HTTPException:
        request.app.state.metrics.error()
        raise
    except Exception as exc:
        request.app.state.metrics.error()
        raise HTTPException(status_code=500, detail="Inference failed") from exc


@app.post("/predict/batch")
async def predict_batch(
    request: Request,
    images: list[UploadFile] = File(...),
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    if len(images) > 32:
        raise HTTPException(status_code=413, detail="Batch size is limited to 32 images")
    try:
        payloads = [await _read_image(image) for image in images]
        pil_images = [Image.open(io.BytesIO(payload)).convert("RGB") for payload in payloads]
        results = _predictor(request).predict_images(pil_images, top_k=top_k)
        latency = results[0]["batch_latency_ms"] if results else 0.0
        request.app.state.metrics.record(len(results), latency)
        return {"request_id": str(uuid.uuid4()), "results": results}
    except HTTPException:
        request.app.state.metrics.error()
        raise
    except Exception as exc:
        request.app.state.metrics.error()
        raise HTTPException(status_code=500, detail="Batch inference failed") from exc


@app.post("/explain")
async def explain(request: Request, image: UploadFile = File(...)) -> dict[str, Any]:
    predictor = _predictor(request)
    if predictor.backend != "pytorch":
        raise HTTPException(status_code=400, detail="Grad-CAM requires the PyTorch backend")
    payload = await _read_image(image)
    original = Image.open(io.BytesIO(payload)).convert("RGB")
    tensor = predictor.prepare(original).unsqueeze(0).to(predictor.runtime.device)
    tensor.requires_grad_(True)
    try:
        cam = GradCAM(predictor.model)
        heatmap = cam(tensor)
        cam.close()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    heat = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
    heat[..., 0] = (255 * heatmap).astype(np.uint8)
    heat[..., 1] = (160 * np.maximum(0, heatmap - 0.35)).astype(np.uint8)
    overlay_base = original.resize((heatmap.shape[1], heatmap.shape[0]))
    overlay = Image.blend(overlay_base, Image.fromarray(heat), alpha=0.42)
    output = io.BytesIO()
    overlay.save(output, format="PNG")
    prediction = predictor.predict_images([original], top_k=1)[0]
    return {
        "prediction": prediction["predictions"][0],
        "overlay": f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('ascii')}",
    }


def run() -> None:
    import uvicorn

    uvicorn.run("foodvision.api.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
