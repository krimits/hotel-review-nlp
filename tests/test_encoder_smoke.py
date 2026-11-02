"""A tiny random-weight encoder through the non-stub serving path.

CI used to exercise only `MODEL_TYPE=stub`, so the encoder branch of `ModelWrapper` — the
one the deployed image uses — was covered by no test at all. This builds a DistilBERT of
the smallest possible shape locally (no download) and drives it end to end through the API.
"""

from __future__ import annotations

import sys

import torch
from fastapi.testclient import TestClient
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from reviewnlp.serving.app import app
from reviewnlp.serving.model_wrapper import (
    ModelWrapper,
    _distribution_encoder,
    _predict_encoder_fast,
)

LABELS = {0: "negative", 1: "positive"}


class _FakeTokenizer:
    """Maps every text to four in-vocabulary ids; the point is the forward pass, not tokenising."""

    def __call__(self, texts, truncation=True, max_length=256, padding=True, return_tensors="pt"):  # noqa: ARG002
        rows = [[1, 2, 3, 4] for _ in texts]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 4), dtype=torch.long),
        }


def _tiny_bundle() -> dict:
    config = DistilBertConfig(
        vocab_size=100,
        max_position_embeddings=64,
        n_layers=1,
        n_heads=2,
        dim=32,
        hidden_dim=64,
        num_labels=2,
        id2label=LABELS,
        label2id={name: idx for idx, name in LABELS.items()},
    )
    model = DistilBertForSequenceClassification(config).eval()
    return {"tokenizer": _FakeTokenizer(), "model": model}


def test_encoder_fast_path_returns_a_label_and_its_probability():
    label, confidence = _predict_encoder_fast(_tiny_bundle(), ["The room was spotless."])[0]
    assert label in set(LABELS.values())
    assert 0.5 <= confidence <= 1.0


def test_encoder_distribution_is_a_normalised_two_way_distribution():
    (distribution,) = _distribution_encoder(_tiny_bundle(), ["Breakfast was cold."])
    assert set(distribution) == set(LABELS.values())
    assert abs(sum(distribution.values()) - 1.0) < 1e-5


def test_api_serves_a_loaded_encoder_without_the_stub(monkeypatch):
    # `reviewnlp.serving.app` is shadowed in the package namespace by the FastAPI object,
    # so `import ... as module` would hand back the app, not the module holding `wrapper`.
    app_module = sys.modules["reviewnlp.serving.app"]

    wrapper = ModelWrapper(model_type="encoder", model_path="unused-in-this-test")
    wrapper._obj = _tiny_bundle()   # what load() would have produced from a checkpoint
    wrapper._loaded = True
    monkeypatch.setattr(app_module, "wrapper", wrapper)

    response = TestClient(app).post("/predict", json={"text": "Wonderful staff, tiny room."})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in set(LABELS.values())
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0


def test_api_batch_path_uses_the_encoder(monkeypatch):
    app_module = sys.modules["reviewnlp.serving.app"]

    wrapper = ModelWrapper(model_type="encoder", model_path="unused-in-this-test")
    wrapper._obj = _tiny_bundle()
    wrapper._loaded = True
    monkeypatch.setattr(app_module, "wrapper", wrapper)

    texts = ["Lovely view.", "Noise all night."]
    response = TestClient(app).post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
