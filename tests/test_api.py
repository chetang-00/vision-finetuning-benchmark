import io
from dataclasses import replace
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from foodvision.api.main import app
from foodvision.config import ModelConfig
from foodvision.models import build_model


def test_root_and_degraded_health_without_checkpoint(monkeypatch) -> None:
    monkeypatch.setenv("FOODVISION_CONFIG", "configs/quickstart.yaml")
    monkeypatch.setenv("FOODVISION_CHECKPOINT", "artifacts/does-not-exist.pt")
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is False
        assert client.get("/models").status_code == 503


def test_prediction_endpoint_with_local_checkpoint(monkeypatch, tmp_path: Path) -> None:
    model_config = replace(ModelConfig(), pretrained=False)
    bundle = build_model(model_config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": bundle.model.state_dict(),
            "classes": [f"class_{index}" for index in range(101)],
        },
        checkpoint,
    )
    image = Image.new("RGB", (256, 256), color=(210, 90, 35))
    payload = io.BytesIO()
    image.save(payload, format="JPEG")

    monkeypatch.setenv("FOODVISION_CONFIG", "configs/quickstart.yaml")
    monkeypatch.setenv("FOODVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("FOODVISION_DEVICE", "cpu")
    with TestClient(app) as client:
        response = client.post(
            "/predict?top_k=3",
            files={"image": ("food.jpg", payload.getvalue(), "image/jpeg")},
        )
        assert response.status_code == 200
        result = response.json()
        assert len(result["predictions"]) == 3
        assert result["backend"] == "pytorch"
        assert result["device"] == "cpu"
