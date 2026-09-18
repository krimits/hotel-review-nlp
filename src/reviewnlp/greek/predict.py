"""Inference utilities for the Greek sentiment classifier."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reviewnlp.greek.data import LABEL_NAMES


def probs_to_prediction(probs: np.ndarray) -> tuple[str, float]:
    """Pure decision rule over ordered class probabilities — testable offline."""
    probs = np.asarray(probs, dtype=float)
    idx = int(probs.argmax())
    return LABEL_NAMES[idx], float(probs[idx])


def predict_texts(
    texts: list[str],
    model_dir: str,
    max_length: int = 160,
    batch_size: int = 32,
    device: str | None = None,
) -> list[str]:
    if not texts:
        return []
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = [str(text) for text in texts[start : start + batch_size]]
        inputs = tokenizer(
            chunk, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions.extend(probs_to_prediction(row)[0] for row in probs)
    return predictions