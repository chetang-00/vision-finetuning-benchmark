from __future__ import annotations

from pydantic import BaseModel, Field


class ClassPrediction(BaseModel):
    label: str
    class_index: int
    probability: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    request_id: str
    predictions: list[ClassPrediction]
    latency_ms: float
    batch_latency_ms: float
    backend: str
    device: str
    model: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    backend: str | None = None
    device: str | None = None
    detail: str | None = None


class MetricsResponse(BaseModel):
    requests: int
    errors: int
    images: int
    average_latency_ms: float
