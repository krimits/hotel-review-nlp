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

    result = model.analyze(request.text)
    aspects = result["aspects"]

    response_aspects = [
        AbsaAspect(
            aspect=item["aspect"],
            sentiment=item["sentiment"],
            quote=item["quote"],
        )
        for item in aspects
    ]

    return AbsaResponse(
        hotel_id=request.hotel_id,
        review_id=request.review_id,
        aspects=response_aspects,
        overall_sentiment=overall_from_aspects(aspects),
        json_valid=result["json_valid"],
        salvaged=result["salvaged"],
        entries_dropped=result["entries_dropped"],
        processing_time_ms=round(
            (time.perf_counter() - started_at) * 1000,
            2,
        ),
    )


@router.post("/batch", response_model=BatchAbsaResponse)
def analyze_reviews_batch(
    request: BatchAbsaRequest,
    model: Annotated[AbsaModelManager, Depends(get_absa_model)],
) -> BatchAbsaResponse:
    """Analyze multiple hotel reviews in a batch.

    This endpoint processes multiple reviews for a hotel in a single request,
    which is more efficient than making individual requests.
    """
    started_at = time.perf_counter()

    results = []
    for review_req in request.reviews:
        result = model.analyze(review_req.text)
        aspects = result["aspects"]

        response_aspects = [
            AbsaAspect(
                aspect=item["aspect"],
                sentiment=item["sentiment"],
                quote=item["quote"],
            )
            for item in aspects
        ]

        results.append(
            AbsaResponse(
                hotel_id=review_req.hotel_id,
                review_id=review_req.review_id,
                aspects=response_aspects,
                overall_sentiment=overall_from_aspects(aspects),
                json_valid=result["json_valid"],
                salvaged=result["salvaged"],
                entries_dropped=result["entries_dropped"],
                processing_time_ms=0.0,  # Will be updated below
            )
        )

    total_time_ms = round((time.perf_counter() - started_at) * 1000, 2)

    return BatchAbsaResponse(
        hotel_id=request.hotel_id,
        results=results,
        total_processing_time_ms=total_time_ms,
    )
