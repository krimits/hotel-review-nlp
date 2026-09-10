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

from fastapi import FastAPI, HTTPException

from reviewnlp.serving.model_wrapper import ModelWrapper, timed_predict
from reviewnlp.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reviewnlp.serving")

app = FastAPI(
    title="Hotel Review Sentiment API",
    description="Serves the benchmarked sentiment models (NB/LR-SGD, BiLSTM, DistilBERT, Qwen-QLoRA).",
    version="0.1.0",
)
wrapper = ModelWrapper()


@app.on_event("startup")
def warm_up() -> None:
    """Load the model once at boot; fail fast if the checkpoint is broken."""
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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = timed_predict(wrapper, req.text)
    except Exception as exc:  # noqa: BLE001
        log.exception("inference error")
        raise HTTPException(status_code=500, detail="inference failed") from exc
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
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
