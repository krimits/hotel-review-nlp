"""Model construction for the Greek sentiment classifier."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reviewnlp.greek.config import ModelConfig

ID2LABEL = {0: "negative", 1: "positive"}
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}


def build_tokenizer(config: ModelConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(config.name)


def build_model(config: ModelConfig) -> AutoModelForSequenceClassification:
    """Fresh classification head on the Greek BERT encoder.

    Dataset labels are 0=negative, 1=positive (asserted at load time in
    data.py); id2label is written into the checkpoint config so any later
    consumer (serving, Space, benchmark) resolves class names correctly.

    `num_labels` is honoured rather than assumed, but the label names are a
    fixed binary pair — a third class would need names this module cannot
    invent, so the mismatch is refused instead of silently mislabelled.
    """
    if config.num_labels != len(ID2LABEL):
        raise ValueError(
            f"num_labels={config.num_labels} but this module defines names for "
            f"{len(ID2LABEL)} classes ({sorted(LABEL2ID)}). Add the names to "
            f"ID2LABEL before training a {config.num_labels}-class model."
        )
    return AutoModelForSequenceClassification.from_pretrained(
        config.name,
        num_labels=config.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
