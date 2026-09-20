"""Model construction for the Greek sentiment classifier."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer

ID2LABEL = {0: "negative", 1: "positive"}
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}


def build_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str) -> AutoModelForSequenceClassification:
    """Fresh classification head on the Greek BERT encoder.

    Dataset labels are 0=negative, 1=positive (asserted at load time in
    data.py); id2label is written into the checkpoint config so any later
    consumer (serving, Space, benchmark) resolves class names correctly.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
