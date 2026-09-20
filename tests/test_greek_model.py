"""Offline tests for the Greek model layer — no downloads, tiny local BERT."""

from __future__ import annotations

import numpy as np
from transformers import BertConfig, BertForSequenceClassification

from reviewnlp.greek.model import ID2LABEL, LABEL2ID
from reviewnlp.greek.predict import probs_to_prediction


def test_label_mapping_is_consistent():
    assert LABEL2ID == {name: idx for idx, name in ID2LABEL.items()}
    assert ID2LABEL == {0: "negative", 1: "positive"}


def test_probs_to_prediction_decision_rule():
    assert probs_to_prediction(np.array([0.9, 0.1])) == ("negative", 0.9)
    assert probs_to_prediction(np.array([0.2, 0.8]))[0] == "positive"
    label, confidence = probs_to_prediction(np.array([0.5, 0.5]))
    assert label in ("negative", "positive")
    assert confidence == 0.5


def test_id2label_roundtrips_through_a_checkpoint_config():
    config = BertConfig(
        vocab_size=100, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=1, intermediate_size=32,
        num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model = BertForSequenceClassification(config)
    assert model.config.id2label == ID2LABEL
    assert model.config.id2label[model.config.num_labels - 1] == "positive"
