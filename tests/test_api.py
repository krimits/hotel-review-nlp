"""API contract tests using the zero-dependency stub model (CI-safe)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reviewnlp.serving.app import app
from reviewnlp.serving.model_wrapper import ModelWrapper


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_TYPE", "stub")
    # re-import via fresh wrapper so env vars apply
    import reviewnlp.serving.app as app_module

    app_module.wrapper = ModelWrapper("stub", "")
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "stub"


def test_predict_returns_label_confidence_latency(client):
    r = client.post("/predict", json={"text": "The hotel was wonderful and clean."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"positive", "negative"}
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0


def test_predict_rejects_short_text(client):
    r = client.post("/predict", json={"text": "ok"})
    assert r.status_code == 422  # pydantic min_length=3


def test_batch_predict_roundtrip(client):
    texts = ["Great stay, amazing staff!", "Filthy room, rude service."]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert {p["label"] for p in preds} <= {"positive", "negative"}


def test_batch_rejects_empty(client):
    r = client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422


@pytest.mark.parametrize("model_type", ["classical", "encoder", "qwen_qlora"])
def test_missing_model_path_is_reported_before_loading(model_type):
    """The image ships no weights and MODEL_PATH defaults to empty. Without
    this guard the empty string reached the loader and surfaced as
    `HFValidationError: Repo id must use alphanumeric chars ... : ''`, which
    named neither MODEL_PATH nor the docker flag that sets it."""
    wrapper = ModelWrapper(model_type=model_type, model_path="")
    with pytest.raises(ValueError, match="needs MODEL_PATH") as excinfo:
        wrapper.load()
    message = str(excinfo.value)
    assert model_type in message
    assert "MODEL_TYPE=stub" in message  # the way out is in the message


def test_stub_needs_no_model_path():
    """The documented smoke test must keep working with no weights present."""
    wrapper = ModelWrapper(model_type="stub", model_path="")
    wrapper.load()
    label, confidence = wrapper.predict("Perfect stay, spotless room.")
    assert label in {"positive", "negative"}
    assert 0.5 <= confidence <= 1.0
