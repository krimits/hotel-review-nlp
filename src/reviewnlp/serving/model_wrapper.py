"""Model loading + inference wrapper for the API.

Selects a family via env vars (12-factor style):

    MODEL_TYPE = stub | classical | encoder | qwen_qlora
    MODEL_PATH = path to the checkpoint / joblib pipeline / adapter

``stub`` lets the API run (and be tested) with zero model files - used by the
CI test-suite and for smoke-testing the container before weights are mounted.
"""

from __future__ import annotations

import os
import threading
import time

import torch

from reviewnlp.llm.predict import predict_classical, predict_qwen_qlora

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelWrapper:
    """Lazy single-model holder; thread-safe for Uvicorn workers."""

    def __init__(self, model_type: str | None = None, model_path: str | None = None):
        self.model_type = model_type or os.environ.get("MODEL_TYPE", "stub")
        self.model_path = model_path or os.environ.get("MODEL_PATH", "")
        self._lock = threading.Lock()
        self._loaded = False
        self._obj = None

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if self.model_type == "stub":
                self._obj = None
            elif self.model_type == "classical":
                import joblib

                self._obj = joblib.load(self.model_path)
            elif self.model_type in {"encoder", "qwen_qlora"}:
                # heavy models load here once; predict_* functions re-load per
                # call in offline use, so we keep a dedicated fast path below
                self._obj = _load_heavy(self.model_type, self.model_path)
            else:
                raise ValueError(f"unknown MODEL_TYPE: {self.model_type}")
            self._loaded = True

    def predict(self, text: str) -> tuple[str, float | None]:
        """Single-text predict -> (label, confidence or None)."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float | None]]:
        self.load()
        if self.model_type == "stub":
            # deterministic pseudo-classifier: hash-based, good for API tests
            out = []
            for t in texts:
                h = abs(hash(t)) % 1000 / 1000.0
                out.append(("positive" if h > 0.5 else "negative", max(h, 1 - h)))
            return out

        with self._lock:
            if self.model_type == "classical":
                labels = predict_classical(self._obj, texts)
                probs = self._obj.predict_proba(texts).max(axis=1)
                return [(str(lbl), float(p)) for lbl, p in zip(labels, probs, strict=False)]
            if self.model_type == "encoder":
                return _predict_encoder_fast(self._obj, texts)
            if self.model_type == "qwen_qlora":
                labels = predict_qwen_qlora(self.model_path, texts)
                return [(str(lbl), None) for lbl in labels]
        raise RuntimeError("unreachable")

    @property
    def info(self) -> dict:
        return {"model_type": self.model_type, "model_path": self.model_path, "device": DEVICE}


def _load_heavy(model_type: str, path: str):
    """Preload heavy artifacts at startup (avoids first-request latency spike)."""
    if model_type == "encoder":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
        return {"tokenizer": tokenizer, "model": model}
    return None  # qwen_qlora loads inside predict via peft


def _predict_encoder_fast(bundle: dict, texts: list[str]) -> list[tuple[str, float | None]]:
    tokenizer, model = bundle["tokenizer"], bundle["model"]
    enc = tokenizer([str(t) for t in texts], truncation=True, max_length=256, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    id2label = model.config.id2label
    return [(id2label[int(i)], float(c)) for i, c in zip(idx.cpu(), conf.cpu(), strict=False)]


def timed_predict(wrapper: ModelWrapper, text: str) -> dict:
    """Predict + wall-clock latency in ms (used by the API layer)."""
    t0 = time.perf_counter()
    label, confidence = wrapper.predict(text)
    return {
        "label": label,
        "confidence": confidence,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
