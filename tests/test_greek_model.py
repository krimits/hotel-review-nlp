"""Offline tests for the Greek model layer — no downloads, tiny local BERT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from transformers import BertConfig, BertForSequenceClassification

from reviewnlp.greek.config import ModelConfig, TrainingConfig
from reviewnlp.greek.model import ID2LABEL, LABEL2ID, build_model
from reviewnlp.greek.predict import probs_to_prediction
from reviewnlp.greek.train import build_training_arguments


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


def test_build_model_refuses_a_class_count_it_has_no_names_for():
    """num_labels is honoured, not assumed — but this module cannot invent a
    name for a third class, so the mismatch is refused rather than mislabelled."""
    with pytest.raises(ValueError, match="num_labels=3"):
        build_model(ModelConfig(name="nlpaueb/bert-base-greek-uncased-v1", num_labels=3))


def test_training_arguments_carry_every_configured_value():
    """The `training:` block is the contract; this asserts it reaches HF."""
    config = TrainingConfig(
        num_epochs=3, batch_size=8, eval_batch_size=64, learning_rate=3e-5,
        warmup_ratio=0.2, weight_decay=0.05, fp16=False, seed=7,
        logging_steps=13, save_total_limit=1, metric_for_best_model="macro_f1",
    )
    args = build_training_arguments(config, Path("runs/unit-test"))

    assert args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 8
    assert args.per_device_eval_batch_size == 64
    assert args.learning_rate == pytest.approx(3e-5)
    assert args.warmup_ratio == pytest.approx(0.2)
    assert args.weight_decay == pytest.approx(0.05)
    assert args.seed == 7 and args.data_seed == 7
    assert args.logging_steps == 13
    assert args.save_total_limit == 1
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "macro_f1"
    assert args.greater_is_better is True
