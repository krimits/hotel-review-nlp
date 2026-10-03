"""POST /triage: sentiment, then complaint topics, then suggested actions for one review.

Experimental. The hotel's API key is checked before any model runs and before anything leaves the server, and
the Jev and Qwen stages are off unless they are switched on in the environment (docs/TRIAGE.md).
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from reviewnlp.analytics.store import AspectStore, get_aspect_store
from reviewnlp.serving.analytics_router import NO_STORE_DETAIL
from reviewnlp.serving.model_wrapper import ModelWrapper
from reviewnlp.serving.security import authorize_hotel
from reviewnlp.triage.jev_client import JevClient, JevConfig
from reviewnlp.triage.pipeline import TriagePipeline, hotel_context
from reviewnlp.triage.qwen_generator import QwenActionGenerator
from reviewnlp.triage.schemas import TriageRequest, TriageResponse, TriageSummaryResponse

log = logging.getLogger("reviewnlp.serving")
router = APIRouter(tags=["Triage"])

_pipeline: TriagePipeline | None = None
_pipeline_lock = threading.Lock()


def get_sentiment_wrapper() -> ModelWrapper:
    """The app's sentiment model. Looked up at call time: the app imports this module, not the other way round."""
    # import_module, not `from reviewnlp.serving import app`: the package re-exports the FastAPI object under that name.
    return importlib.import_module("reviewnlp.serving.app").wrapper


def get_triage_pipeline(wrapper: Annotated[ModelWrapper, Depends(get_sentiment_wrapper)]) -> TriagePipeline:
    """One pipeline for the process, built from the environment on first use. Nothing is called while building it."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None or _pipeline.wrapper is not wrapper:
            _pipeline = TriagePipeline(wrapper, JevClient(JevConfig.from_env()), QwenActionGenerator.from_env())
        return _pipeline


@router.post("/triage", response_model=TriageResponse)
def triage_review(
    request: TriageRequest,
    pipeline: Annotated[TriagePipeline, Depends(get_triage_pipeline)],
    store: Annotated[AspectStore | None, Depends(get_aspect_store)],
    x_api_key: str | None = Header(default=None),
) -> TriageResponse:
    """Sentiment from the app's model, complaint topics from Jev and suggested actions from Qwen.

    Only the sentiment stage is required. If Jev or Qwen is off, the response says so; if one fails, the response
    is partial and names the kind of failure. Every response says `validation_status: unvalidated`: no stage has
    been measured on whole or mixed reviews. With a database configured and a `review_id` given, the result is
    stored without the review text, one per hotel, source and review id: running the review again replaces it, so
    it is counted once. `/hotels/{hotel_id}/triage/summary` reads the stored results back.
    """
    authorize_hotel(request.hotel_id, x_api_key)
    # Stored under the review's own id only. A generated id would make every run of the same review a new review
    # in the counts, so without an id the result is returned and not kept.
    stored = store is not None and request.review_id is not None
    context = hotel_context(store, request.hotel_id) if store and pipeline.generator is not None else None
    try:
        result = pipeline.run(request.text, hotel_context=context)
    except Exception as exc:  # noqa: BLE001
        log.exception("triage inference error")
        raise HTTPException(status_code=500, detail="inference failed") from exc
    response = TriageResponse(
        hotel_id=request.hotel_id, review_id=request.review_id, stored=stored,
        not_stored_reason=None if stored else "no_database" if store is None else "no_review_id",
        status=result.status, sentiment=result.sentiment, complaints=result.complaints, routing=result.routing,
        actions=result.actions, timings=result.timings)
    if stored:
        store.save_triage(hotel_id=request.hotel_id, review_id=request.review_id, source=request.source or "api",
                          text=request.text, language=request.language, review_date=request.review_date,
                          result=response.model_dump(mode="json"))
    return response


@router.get("/hotels/{hotel_id}/triage/summary", response_model=TriageSummaryResponse,
            responses={501: {"description": "No aspect store is configured"}})
def triage_summary(
    hotel_id: str,
    store: Annotated[AspectStore | None, Depends(get_aspect_store)],
    days: int = Query(default=30, ge=7, le=365),
    x_api_key: str | None = Header(default=None),
) -> TriageSummaryResponse:
    """What triage found in this hotel's stored reviews: sentiment, complaint topics, actions, reviews to check."""
    authorize_hotel(hotel_id, x_api_key)
    if store is None:
        raise HTTPException(status_code=501, detail=NO_STORE_DETAIL)
    return TriageSummaryResponse(**store.triage_summary(hotel_id, days, limit_actions=20, limit_review=20))
