"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=8000, examples=["The room was tiny but the location was perfect."])


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)


class PredictResponse(BaseModel):
    label: str
    confidence: float | None = Field(None, description="max softmax probability; None if the model cannot produce it")
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_path: str
    device: str
