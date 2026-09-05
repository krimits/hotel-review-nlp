"""Classification metrics - thin, explicit wrappers with zero surprises.

We compute from raw counts (sklearn) but return a plain dict so results are
JSON-serializable for the benchmark artifact and README tables.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

LABELS = ("negative", "positive")


def binary_metrics(y_true, y_pred, label_names=LABELS, label_values=None) -> dict:
    """Accuracy, macro/micro/weighted F1, per-class P/R/F1, confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("empty evaluation set")

    values = list(label_names if label_values is None else label_values)
    names = list(label_names)
    if len(values) != len(names):
        raise ValueError("label_values and label_names must have the same length")

    acc = float((y_true == y_pred).mean())
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=values, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=values, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, labels=values, average="weighted", zero_division=0
    )[2]
    cm = confusion_matrix(y_true, y_pred, labels=values)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {
            name: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, name in enumerate(names)
        },
        "confusion_matrix": cm.tolist(),  # rows = true, cols = predicted
    }


def positive_probabilities(logits) -> np.ndarray:
    """Convert two-class logits to positive-class probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"expected logits with shape (n, 2), got {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, 1] / exp.sum(axis=1)


def metrics_at_threshold(y_true, positive_scores, threshold: float) -> dict:
    """Score numeric 0/1 labels at a fixed positive-class threshold."""
    y_true = np.asarray(y_true)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    if len(y_true) != len(positive_scores):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(positive_scores)}")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    predictions = (positive_scores >= threshold).astype(np.int64)
    return binary_metrics(
        y_true,
        predictions,
        label_names=LABELS,
        label_values=(0, 1),
    )


def tune_binary_threshold(
    y_true,
    positive_scores,
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.005,
) -> dict:
    """Select a threshold on development data using macro-F1 only.

    Ties are resolved by higher negative-class F1 and then by proximity to
    the default threshold of 0.5. Test labels must never be passed here.
    """
    if not 0.0 < minimum <= maximum < 1.0:
        raise ValueError("threshold range must be inside (0, 1)")
    if step <= 0:
        raise ValueError("threshold step must be positive")

    n_steps = int(round((maximum - minimum) / step))
    thresholds = minimum + np.arange(n_steps + 1) * step
    thresholds = thresholds[thresholds <= maximum + 1e-12]

    y_true = np.asarray(y_true)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    if len(y_true) != len(positive_scores):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(positive_scores)}")

    best = None
    for threshold in thresholds:
        predictions = (positive_scores >= threshold).astype(np.int64)
        _p, _r, per_class_f1, _support = precision_recall_fscore_support(
            y_true, predictions, labels=[0, 1], zero_division=0
        )
        rank = (
            float(per_class_f1.mean()),
            float(per_class_f1[0]),
            -abs(float(threshold) - 0.5),
        )
        if best is None or rank > best[0]:
            best = (rank, float(threshold))

    best_metrics = metrics_at_threshold(y_true, positive_scores, best[1])
    return {
        "threshold": round(best[1], 6),
        "dev_metrics": best_metrics,
        "objective": "macro_f1",
    }


def discordant_counts(y_true, preds_a, preds_b, model_a: str, model_b: str) -> tuple[int, int]:
    """Return (n01, n10): a-wrong/b-right and a-right/b-wrong counts.

    McNemar's test uses only these discordant pairs - agreements carry no
    information about *relative* performance.
    """
    y_true = np.asarray(y_true)
    a_wrong_b_right = int(np.sum((preds_a != y_true) & (preds_b == y_true)))
    a_right_b_wrong = int(np.sum((preds_a == y_true) & (preds_b != y_true)))
    print(f"[{model_a} vs {model_b}] a-wrong/b-right={a_wrong_b_right}, a-right/b-wrong={a_right_b_wrong}")
    return a_wrong_b_right, a_right_b_wrong
