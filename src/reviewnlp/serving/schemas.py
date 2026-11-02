"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# One review, bounded at both ends: the tokenizer truncates to 256 tokens anyway, and an
# unbounded string is a memory/CPU bill the caller chooses for the server.
ReviewText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=8000)]

# A batch is capped per item *and* per request. `texts: list[str]` alone let one request
# carry 256 multi-megabyte strings straight into the tokenizer.
MAX_BATCH_ITEMS = 64


class PredictRequest(BaseModel):
    text: ReviewText = Field(..., examples=["The room was tiny but the location was perfect."])


class BatchPredictRequest(BaseModel):
    texts: list[ReviewText] = Field(..., min_length=1, max_length=MAX_BATCH_ITEMS)


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
