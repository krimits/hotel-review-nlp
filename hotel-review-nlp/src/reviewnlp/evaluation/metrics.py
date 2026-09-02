"""Classification metrics - thin, explicit wrappers with zero surprises.

We compute from raw counts (sklearn) but return a plain dict so results are
JSON-serializable for the benchmark artifact and README tables.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

LABELS = ("negative", "positive")


def binary_metrics(y_true, y_pred, label_names=LABELS) -> dict:
    """Accuracy, macro/micro/weighted F1, per-class P/R/F1, confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("empty evaluation set")

    acc = float((y_true == y_pred).mean())
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(label_names), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[2]
    cm = confusion_matrix(y_true, y_pred, labels=list(label_names))

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
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": cm.tolist(),  # rows = true, cols = predicted
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
