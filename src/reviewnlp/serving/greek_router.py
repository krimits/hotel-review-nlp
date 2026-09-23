"""Greek sentiment is available only when an explicit checkpoint is configured."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from reviewnlp.greek.serve import GreekModelManager, get_greek_model
from reviewnlp.serving.security import require_api_key

router = APIRouter(prefix="/greek", tags=["Greek sentiment"], dependencies=[Depends(require_api_key)])


class GreekRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=8000)


class GreekBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)


class GreekPrediction(BaseModel):
    label: str
    confidence: float
    training_domain: str = "greek_tweets"
    hotel_domain_validated: bool = False


def _predict(model: GreekModelManager, texts: list[str]) -> list[GreekPrediction]:
    try:
        return [GreekPrediction(**item) for item in model.predict_batch(texts)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict", response_model=GreekPrediction)
def predict_greek(
    request: GreekRequest, model: Annotated[GreekModelManager, Depends(get_greek_model)],
) -> GreekPrediction:
    return _predict(model, [request.text])[0]


@router.post("/predict/batch", response_model=list[GreekPrediction])
def predict_greek_batch(
    request: GreekBatchRequest, model: Annotated[GreekModelManager, Depends(get_greek_model)],
) -> list[GreekPrediction]:
    return _predict(model, request.texts)
