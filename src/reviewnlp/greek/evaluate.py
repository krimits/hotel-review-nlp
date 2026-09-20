"""Standalone evaluation of a saved Greek checkpoint.

scripts/evaluate_greek.py imported `evaluate_checkpoint` from this module,
which was an empty file — every invocation died on ImportError. Evaluating
from a checkpoint (rather than only at the end of training) is what makes a
published model checkable by someone who did not run the training, so it is
implemented here against the same config that produced the splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reviewnlp.evaluation.metrics import binary_metrics
from reviewnlp.greek.config import GreekConfig
from reviewnlp.greek.data import LABEL_NAMES, dataset_manifest, load_greek_splits

LABEL_VALUES = tuple(range(len(LABEL_NAMES)))


def predict_logits(
    model_dir: str,
    texts: list[str],
    max_length: int,
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Raw logits for `texts`, in order. Torch imported lazily."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = [str(text) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            chunks.append(model(**encoded).logits.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def score_split(frame: pd.DataFrame, logits: np.ndarray) -> dict:
    """Shared metric schema, so Greek rows drop into the unified benchmark."""
    return binary_metrics(
        frame["label"].to_numpy(),
        logits.argmax(axis=-1),
        label_names=LABEL_NAMES,
        label_values=LABEL_VALUES,
    )


def evaluate_checkpoint(
    model_dir: str,
    config_path: str,
    batch_size: int = 64,
    splits: tuple[str, ...] = ("validation", "test"),
) -> dict:
    """Rebuild the configured splits and score `model_dir` on them.

    Writes metrics.json plus per-split logits into `evaluation.output_dir`,
    matching the layout reviewnlp.evaluation.benchmark already reads.
    """
    config = GreekConfig.from_yaml(config_path)
    frames = load_greek_splits(config.data)

    output_dir = Path(config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = {}
    for split in splits:
        frame = frames[split]
        logits = predict_logits(
            model_dir, frame["text"].tolist(), config.data.max_length, batch_size
        )
        np.save(output_dir / f"{split}_logits.npy", logits)
        np.save(output_dir / f"{split}_labels.npy", frame["label"].to_numpy(dtype=np.int64))
        scores[split] = score_split(frame, logits)

    report = {
        "schema_version": 2,
        "model_dir": str(model_dir),
        "base_model": config.model.name,
        "max_length": config.data.max_length,
        "data_manifest": dataset_manifest(frames, config.data),
        **scores,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
