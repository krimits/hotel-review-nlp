"""Serving a dynamically quantized encoder still behaves like the model the API serves.

The INT8 decision note measured the quantized model on the full test set; this checks the
serving path can be switched to it (``MODEL_QUANTIZE=dynamic_int8``) with the same wiring
the deployed container uses, and that a typo in that variable fails loudly instead of
quietly serving FP32.
"""

from __future__ import annotations

import sys

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from reviewnlp.serving.app import app
from reviewnlp.serving.model_wrapper import ModelWrapper, quantize_encoder, quantize_mode

LABELS = {0: "negative", 1: "positive"}


class _FakeTokenizer:
    def __call__(self, texts, truncation=True, max_length=256, padding=True, return_tensors="pt"):  # noqa: ARG002
        rows = [[1, 2, 3, 4] for _ in texts]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 4), dtype=torch.long),
        }


def _tiny_model() -> DistilBertForSequenceClassification:
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
    return DistilBertForSequenceClassification(config).eval()


def _inputs(count: int = 1) -> dict:
    return {
        "input_ids": torch.ones((count, 4), dtype=torch.long),
        "attention_mask": torch.ones((count, 4), dtype=torch.long),
    }


def test_quantize_encoder_quantizes_the_linear_layers():
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear

    quantized = quantize_encoder(_tiny_model())
    dynamic = [module for module in quantized.modules() if isinstance(module, DynamicQuantizedLinear)]
    assert dynamic, "expected the linear layers to become dynamic-quantized linears, found none"


def test_quantized_forward_is_finite_and_deterministic():
    quantized = quantize_encoder(_tiny_model())
    with torch.no_grad():
        first = quantized(**_inputs(3)).logits
        second = quantized(**_inputs(3)).logits
    assert first.shape == (3, 2)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


@pytest.mark.parametrize("value", ["", "off", "none", "0", "false"])
def test_quantization_can_be_switched_off(monkeypatch, value):
    monkeypatch.setenv("MODEL_QUANTIZE", value)
    assert quantize_mode() == ""


def test_quantization_on_is_dynamic_int8(monkeypatch):
    monkeypatch.setenv("MODEL_QUANTIZE", "dynamic_int8")
    assert quantize_mode() == "dynamic_int8"


def test_an_unknown_mode_is_refused_with_the_allowed_values(monkeypatch):
    monkeypatch.setenv("MODEL_QUANTIZE", "int4")
    with pytest.raises(ValueError, match="dynamic_int8"):
        quantize_mode()


def test_health_reports_a_quantized_model_and_predict_still_answers(monkeypatch):
    monkeypatch.setenv("MODEL_QUANTIZE", "dynamic_int8")
    wrapper = ModelWrapper(model_type="encoder", model_path="unused-in-this-test")
    wrapper._obj = {"tokenizer": _FakeTokenizer(), "model": quantize_encoder(_tiny_model())}
    wrapper._loaded = True
    monkeypatch.setattr(sys.modules["reviewnlp.serving.app"], "wrapper", wrapper)

    client = TestClient(app)
    health = client.get("/health").json()
    assert health["quantized"] is True

    response = client.post("/predict", json={"text": "The room was spotless."})
    assert response.status_code == 200
    assert response.json()["label"] in set(LABELS.values())
