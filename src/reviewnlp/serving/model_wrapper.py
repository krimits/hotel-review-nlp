"""Model loading + inference wrapper for the API.

Selects a family via env vars (12-factor style):

    MODEL_TYPE = stub | classical | encoder | qwen_qlora
    MODEL_PATH = path to the checkpoint / joblib pipeline / adapter

``stub`` lets the API run (and be tested) with zero model files - used by the
CI test-suite and for smoke-testing the container before weights are mounted.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Any

import torch

from reviewnlp.llm.predict import predict_classical, predict_qwen_qlora

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _stub_score(text: str) -> float:
    """Deterministic pseudo-probability for the zero-dependency stub model.

    `hash()` is salted per process, so the stub documented as deterministic returned
    different labels in different workers. SHA-256 of the text does not.
    """
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 1000 / 1000.0


def _max_concurrency() -> int:
    """How many forward passes may run at once (REVIEWNLP_MAX_CONCURRENCY).

    A single blanket lock serialised every request behind the slowest one. A bounded
    semaphore keeps the bound (memory, CPU oversubscription) without making the queue
    strict FIFO against all callers. CPU-bound torch usually wants 1; small inputs on a
    many-core box can use more.
    """
    configured = os.environ.get("REVIEWNLP_MAX_CONCURRENCY", "").strip()
    if configured:
        try:
            return max(1, int(configured))
        except ValueError as error:
            raise ValueError(f"REVIEWNLP_MAX_CONCURRENCY must be an integer, got {configured!r}") from error
    return max(1, min(4, os.cpu_count() or 1))


QUANTIZE_MODES = ("", "dynamic_int8")


def quantize_mode() -> str:
    """`MODEL_QUANTIZE`: empty for FP32, `dynamic_int8` for the measured quantization.

    Only the mode the decision note measured is accepted. A typo must fail at load with the
    allowed values named, not quietly serve FP32 while the deployment believes it is INT8.
    """
    mode = os.environ.get("MODEL_QUANTIZE", "").strip().lower()
    if mode in {"0", "false", "off", "none"}:
        mode = ""
    if mode not in QUANTIZE_MODES:
        allowed = ", ".join(repr(value) for value in QUANTIZE_MODES if value)
        raise ValueError(f"MODEL_QUANTIZE must be empty or one of {allowed}, got {mode!r}")
    return mode


def quantize_encoder(model):
    """Dynamic INT8 on the linear layers — exactly what the INT8 decision note measured.

    Weights are quantized in place at load time, so the deployment needs no second artifact:
    the published FP32 checkpoint plus this call is the served INT8 model.
    """
    from torch.ao.quantization import quantize_dynamic

    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


class ModelWrapper:
    """Lazy single-model holder; thread-safe for Uvicorn workers."""

    def __init__(self, model_type: str | None = None, model_path: str | None = None):
        self.model_type = model_type or os.environ.get("MODEL_TYPE", "stub")
        self.model_path = model_path or os.environ.get("MODEL_PATH", "")
        self.quantize = quantize_mode()
        self._lock = threading.Lock()          # guards load(), nothing else
        self._semaphore = threading.BoundedSemaphore(_max_concurrency())
        self._loaded = False
        self._obj: Any = None

    def _require_model_path(self) -> None:
        """Every family but `stub` needs weights; say so before loading them.

        Without this the empty default reaches the loader and surfaces as
        `HFValidationError: Repo id must use alphanumeric chars ... : ''`,
        which names neither MODEL_PATH nor the container flag that sets it.
        """
        if not self.model_path:
            raise ValueError(
                f"MODEL_TYPE={self.model_type} needs MODEL_PATH pointing at the "
                f"checkpoint, but it is empty. Mount the weights and set it, e.g. "
                f"`docker run -v $(pwd)/models:/models -e MODEL_TYPE={self.model_type} "
                f"-e MODEL_PATH=/models/distilbert ...`, or use MODEL_TYPE=stub to "
                f"run the API without weights."
            )

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if self.model_type == "stub":
                self._obj = None
            elif self.model_type == "classical":
                self._require_model_path()
                import joblib

                self._obj = joblib.load(self.model_path)
            elif self.model_type in {"encoder", "qwen_qlora"}:
                self._require_model_path()
                # heavy models load here once; predict_* functions re-load per
                # call in offline use, so we keep a dedicated fast path below
                self._obj = _load_heavy(self.model_type, self.model_path, quantize=self.quantize == "dynamic_int8")
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
            out: list[tuple[str, float | None]] = []
            for t in texts:
                h = _stub_score(t)
                out.append(("positive" if h > 0.5 else "negative", max(h, 1 - h)))
            return out

        with self._semaphore:
            if self.model_type == "classical":
                labels = predict_classical(self._obj, texts)
                probs = self._obj.predict_proba(texts).max(axis=1)
                return [(str(lbl), float(p)) for lbl, p in zip(labels, probs, strict=False)]
            if self.model_type == "encoder":
                return _predict_encoder_fast(self._obj, texts)
            if self.model_type == "qwen_qlora":
                labels = predict_qwen_qlora(self.model_path, texts, bundle=self._obj)
                return [(str(lbl), None) for lbl in labels]
        raise RuntimeError("unreachable")

    def distribution_batch(self, texts: list[str]) -> list[dict[str, float] | None]:
        """Label -> probability for each text, or None where the model cannot give a distribution.

        `predict_batch` returns only the chosen label and its probability. Triage routes on how sure the model is,
        so it needs the whole distribution. The label that wins here is the one with the largest probability.
        """
        self.load()
        if self.model_type == "stub":
            # Same pseudo-classifier as predict_batch; "negative" comes first so a tie falls on it, as there.
            out: list[dict[str, float] | None] = []
            for t in texts:
                h = _stub_score(t)
                out.append({"negative": 1 - h, "positive": h})
            return out
        with self._semaphore:
            if self.model_type == "classical":
                classes = [str(c) for c in self._obj.classes_]
                return [dict(zip(classes, (float(p) for p in row), strict=True))
                        for row in self._obj.predict_proba(texts)]
            if self.model_type == "encoder":
                return _distribution_encoder(self._obj, texts)
        return [None] * len(texts)  # qwen_qlora generates a word and has no probabilities

    @property
    def info(self) -> dict:
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": DEVICE,
            "quantized": self.quantize == "dynamic_int8",
        }


def _load_heavy(model_type: str, path: str, quantize: bool = False):
    """Preload heavy artifacts at startup (avoids first-request latency spike)."""
    if model_type == "encoder":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
        if quantize:
            model = quantize_encoder(model)
        return {"tokenizer": tokenizer, "model": model}
    from reviewnlp.llm.predict import load_qwen_qlora

    return load_qwen_qlora(path)


def _predict_encoder_fast(bundle: dict, texts: list[str]) -> list[tuple[str, float | None]]:
    tokenizer, model = bundle["tokenizer"], bundle["model"]
    enc = tokenizer([str(t) for t in texts], truncation=True, max_length=256, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    id2label = model.config.id2label
    return [(id2label[int(i)], float(c)) for i, c in zip(idx.cpu(), conf.cpu(), strict=False)]


def _distribution_encoder(bundle: dict, texts: list[str]) -> list[dict[str, float] | None]:
    tokenizer, model = bundle["tokenizer"], bundle["model"]
    enc = tokenizer([str(t) for t in texts], truncation=True, max_length=256, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
    probs = torch.softmax(logits, dim=-1).cpu()
    id2label = model.config.id2label
    return [{id2label[i]: float(p) for i, p in enumerate(row)} for row in probs]


def timed_predict(wrapper: ModelWrapper, text: str) -> dict:
    """Predict + wall-clock latency in ms (used by the API layer)."""
    t0 = time.perf_counter()
    label, confidence = wrapper.predict(text)
    return {
        "label": label,
        "confidence": confidence,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
