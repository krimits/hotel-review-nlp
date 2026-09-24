from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from reviewnlp.greek import evaluate as evaluation
from reviewnlp.greek.serve import get_greek_model
from reviewnlp.serving.app import app


def test_greek_route_requires_a_checkpoint(monkeypatch):
    monkeypatch.delenv("GREEK_MODEL_PATH", raising=False)
    with TestClient(app) as client:
        result = client.post("/greek/predict", json={"text": "Καθαρό δωμάτιο"})
    assert result.status_code == 503
    assert "GREEK_MODEL_PATH" in result.json()["detail"]


def test_greek_batch_keeps_order_and_declares_training_domain():
    class Fake:
        def predict_batch(self, texts):
            return [{"label": "positive" if i == 0 else "negative", "confidence": 0.8}
                    for i, _ in enumerate(texts)]

    app.dependency_overrides[get_greek_model] = lambda: Fake()
    try:
        with TestClient(app) as client:
            response = client.post("/greek/predict/batch", json={"texts": [
                "Ευγενικό προσωπικό", "Βρώμικο δωμάτιο"]})
        assert response.status_code == 200
        assert [x["label"] for x in response.json()] == ["positive", "negative"]
        assert all(x["hotel_domain_validated"] is False for x in response.json())
    finally:
        app.dependency_overrides.clear()


def test_greek_hotel_evaluation_rejects_unreviewed_or_duplicate_data(tmp_path, monkeypatch):
    path = tmp_path / "hotel.csv"
    pd.DataFrame({"text": ["Ωραίο δωμάτιο", "Αθλιο προσωπικό"],
                  "label": ["positive", "negative"]}).to_csv(path, index=False)
    monkeypatch.setattr(evaluation, "predict_logits", lambda *_args, **_kwargs:
                        np.array([[0.1, 0.9], [0.9, 0.1]]))
    report = evaluation.evaluate_hotel_csv("model", path, tmp_path / "output")
    assert report["test"]["macro_f1"] == 1.0
    assert report["rows"] == 2
    pd.DataFrame({"text": ["Ωραίο δωμάτιο", "ωραιο δωματιο"],
                  "label": ["positive", "negative"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.evaluate_hotel_csv("model", path, tmp_path / "output")
