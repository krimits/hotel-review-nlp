"""FastAPI router for ABSA analysis endpoints."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends

from reviewnlp.absa.extract import overall_from_aspects
from reviewnlp.serving.absa_model_manager import AbsaModelManager
from reviewnlp.serving.absa_schemas import (
    AbsaAspect,
    AbsaRequest,
    AbsaResponse,
    BatchAbsaRequest,
    BatchAbsaResponse,
)

router = APIRouter(prefix="/absa", tags=["ABSA"])

_model_manager = AbsaModelManager()


def get_absa_model() -> AbsaModelManager:
    """Dependency to get the ABSA model manager singleton."""
    return _model_manager


def _to_response(record: dict, hotel_id: str, review_id: str | None, elapsed_ms: float | None):
    """Turn one generated record into the wire shape."""
    aspects = record["aspects"]
    return AbsaResponse(
        hotel_id=hotel_id,
        review_id=review_id,
        aspects=[
            AbsaAspect(aspect=item["aspect"], sentiment=item["sentiment"], quote=item["quote"])
            for item in aspects
        ],
        overall_sentiment=overall_from_aspects(aspects),
        json_valid=record["json_valid"],
        salvaged=record["salvaged"],
        entries_dropped=record["entries_dropped"],
        processing_time_ms=elapsed_ms,
    )


@router.post("", response_model=AbsaResponse)
def analyze_review(
    request: AbsaRequest,
    model: Annotated[AbsaModelManager, Depends(get_absa_model)],
) -> AbsaResponse:
    """Analyze a single hotel review and extract aspects with sentiments.

    This endpoint accepts a hotel review and returns structured aspect-based
    sentiment analysis including aspect category, sentiment polarity, and
    supporting quotes from the review text.
    """
    started_at = time.perf_counter()
    record = model.analyze(request.text)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

    return _to_response(record, request.hotel_id, request.review_id, elapsed_ms)


@router.post("/batch", response_model=BatchAbsaResponse)
def analyze_reviews_batch(
    request: BatchAbsaRequest,
    model: Annotated[AbsaModelManager, Depends(get_absa_model)],
) -> BatchAbsaResponse:
    """Analyze multiple hotel reviews in one padded forward pass per batch.

    The reviews go through the model together rather than one at a time, which
    is what makes this cheaper than the same number of single requests.

    Because they are generated together, no per-review time is measured: each
    result carries a null processing_time_ms and the real figure for the work
    is total_processing_time_ms.
    """
    started_at = time.perf_counter()
    records = model.analyze_batch([review.text for review in request.reviews])
    total_time_ms = round((time.perf_counter() - started_at) * 1000, 2)

    return BatchAbsaResponse(
        hotel_id=request.hotel_id,
        results=[
            _to_response(record, review.hotel_id, review.review_id, None)
            for record, review in zip(records, request.reviews, strict=True)
        ],
        total_processing_time_ms=total_time_ms,
    )
