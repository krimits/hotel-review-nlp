"""FastAPI inference service.

Endpoints:
    GET  /health          liveness + model info (for k8s probes)
    POST /predict         single review
    POST /predict/batch   up to 256 reviews

Configuration (env): MODEL_TYPE, MODEL_PATH - see model_wrapper.py.

Local run:
    MODEL_TYPE=stub uvicorn reviewnlp.serving.app:app --port 8000
    MODEL_TYPE=encoder MODEL_PATH=runs/distilbert uvicorn reviewnlp.serving.app:app
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse

from reviewnlp.analytics.store import get_aspect_store
from reviewnlp.serving.absa_router import router as absa_router
from reviewnlp.serving.analytics_router import router as analytics_router
from reviewnlp.serving.greek_router import router as greek_router
from reviewnlp.serving.model_wrapper import ModelWrapper, timed_predict
from reviewnlp.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from reviewnlp.serving.security import require_api_key, validate_production_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reviewnlp.serving")

app = FastAPI(
    title="Hotel Review Sentiment API",
    description="Serves the benchmarked sentiment models (NB/LR-SGD, BiLSTM, DistilBERT, Qwen-QLoRA) and ABSA analysis.",
    version="0.1.0",
)
wrapper = ModelWrapper()

# Include ABSA and analytics routers
app.include_router(absa_router)
app.include_router(analytics_router)
app.include_router(greek_router)


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    """Local hotel-operations view; its data requests enforce hotel API keys."""
    return FileResponse(Path(__file__).with_name("dashboard.html"))


@app.on_event("startup")
def warm_up() -> None:
    """Load the model once at boot; fail fast if the checkpoint is broken."""
    if os.getenv("REVIEWNLP_ENV") == "production":
        if not os.getenv("REVIEWNLP_DB_PATH") or not os.getenv("REVIEWNLP_API_KEYS_JSON"):
            raise RuntimeError("production requires REVIEWNLP_DB_PATH and REVIEWNLP_API_KEYS_JSON")
        if wrapper.model_type == "stub":
            raise RuntimeError("production cannot serve the stub classifier")
        validate_production_config()
        get_aspect_store()
    try:
        wrapper.load()
        log.info("model loaded: %s", wrapper.info)
    except Exception as exc:  # noqa: BLE001
        log.exception("model load failed: %s", exc)
        if wrapper.model_type != "stub":
            raise


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", **wrapper.info)


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = timed_predict(wrapper, req.text)
    except Exception as exc:  # noqa: BLE001
        log.exception("inference error")
        raise HTTPException(status_code=500, detail="inference failed") from exc
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse, dependencies=[Depends(require_api_key)])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Batch inference: one forward pass over the whole request."""
    import time

    t0 = time.perf_counter()
    try:
        pairs = wrapper.predict_batch(req.texts)
    except Exception as exc:  # noqa: BLE001
        log.exception("batch inference error")
        raise HTTPException(status_code=500, detail="inference failed") from exc
    wall_ms = (time.perf_counter() - t0) * 1000
    per_item = wall_ms / len(req.texts)
    return BatchPredictResponse(
        predictions=[
            PredictResponse(label=label, confidence=conf, latency_ms=round(per_item, 2))
            for label, conf in pairs
        ]
    )
