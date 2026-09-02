"""Classification metrics - thin, explicit wrappers with zero surprises.

We compute from raw counts (sklearn) but return a plain dict so results are
JSON-serializable for the benchmark artifact and README tables.

Labels may be passed either as strings ("negative"/"positive", the form the
classical baselines and the benchmark use) or as integer class ids (0/1, the
form the torch training loops carry). Integers are mapped onto ``label_names``
by position, so every model family writes the same string-keyed artifact.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

LABELS = ("negative", "positive")


def as_label_names(y, label_names=LABELS) -> np.ndarray:
    """Normalize class ids to label strings; leave string labels untouched.

    ``sklearn`` is asked for per-class stats with ``labels=label_names``, which
    fails outright when ``y`` holds integer ids (``confusion_matrix`` raises
    "At least one label specified must be in y_true"). The torch loops label
    with ids, so normalize here rather than at every call site.
    """
    y = np.asarray(y)
    if y.dtype.kind not in "iub":  # already strings/objects
        return y.astype(str)
    ids = y.astype(int)
    if ids.size and (ids.min() < 0 or ids.max() >= len(label_names)):
        raise ValueError(
            f"class id out of range for label_names={tuple(label_names)}: "
            f"got ids in [{ids.min()}, {ids.max()}]"
        )
    lookup = np.asarray(label_names)
    return lookup[ids]


def binary_metrics(y_true, y_pred, label_names=LABELS) -> dict:
    """Accuracy, macro/micro/weighted F1, per-class P/R/F1, confusion matrix.

    ``y_true``/``y_pred`` accept label strings or integer ids interchangeably.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("empty evaluation set")
    y_true = as_label_names(y_true, label_names)
    y_pred = as_label_names(y_pred, label_names)

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
