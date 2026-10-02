"""POST /triage: sentiment, then complaint topics, then suggested actions for one review.

Experimental. The hotel's API key is checked before any model runs and before anything leaves the server, and
the Jev and Qwen stages are off unless they are switched on in the environment (docs/TRIAGE.md).
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from reviewnlp.serving.model_wrapper import ModelWrapper
from reviewnlp.serving.security import authorize_hotel
from reviewnlp.triage.jev_client import JevClient, JevConfig
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.qwen_generator import QwenActionGenerator
from reviewnlp.triage.schemas import TriageRequest, TriageResponse

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
    x_api_key: str | None = Header(default=None),
) -> TriageResponse:
    """Sentiment from the app's model, complaint topics from Jev and suggested actions from Qwen.

    Only the sentiment stage is required. If Jev or Qwen is off, the response says so; if one fails, the response
    is partial and names the kind of failure. Every response says `validation_status: unvalidated`: no stage has
    been measured on whole or mixed reviews.
    """
    authorize_hotel(request.hotel_id, x_api_key)
    review_id = request.review_id
    try:
        result = pipeline.run(request.text)
    except Exception as exc:  # noqa: BLE001
        log.exception("triage inference error")
        raise HTTPException(status_code=500, detail="inference failed") from exc
    return TriageResponse(
        hotel_id=request.hotel_id, review_id=review_id, stored=False, status=result.status,
        sentiment=result.sentiment, complaints=result.complaints, routing=result.routing,
        actions=result.actions, timings=result.timings)
