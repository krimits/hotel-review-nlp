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


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels) -> float:
    scores = []
    for label in labels:
        truth, guess = y_true == label, y_pred == label
        true_pos = np.sum(truth & guess)
        wrong = np.sum(~truth & guess) + np.sum(truth & ~guess)
        scores.append(2 * true_pos / (2 * true_pos + wrong) if true_pos or wrong else 0.0)
    return float(np.mean(scores))


def bootstrap_ci(y_true, y_pred, labels=LABELS, n_resamples: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of macro-F1, resampling test reviews with replacement."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("need two non-empty arrays of the same length")
    rng = np.random.default_rng(seed)
    values = [_macro_f1(y_true[rows], y_pred[rows], labels)
              for rows in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_resamples))]
    low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    return round(float(low), 4), round(float(high), 4)


def paired_bootstrap_diff(y_true, pred_a, pred_b, labels=LABELS, n_resamples: int = 2000, seed: int = 0,
                          level: float = 0.95) -> tuple[float, float, float]:
    """Macro-F1 of b minus a, with a percentile interval that resamples the same reviews for both."""
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    if not len(y_true) == len(pred_a) == len(pred_b) or len(y_true) == 0:
        raise ValueError("need three non-empty arrays of the same length")
    point = _macro_f1(y_true, pred_b, labels) - _macro_f1(y_true, pred_a, labels)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_resamples):
        rows = rng.integers(0, len(y_true), len(y_true))
        differences.append(_macro_f1(y_true[rows], pred_b[rows], labels) - _macro_f1(y_true[rows], pred_a[rows], labels))
    low, high = np.quantile(differences, [(1 - level) / 2, (1 + level) / 2])
    return round(float(point), 4), round(float(low), 4), round(float(high), 4)


Z95 = 1.959964


def wilson_interval(successes: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, clipped to [0, 1]."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError(f"need 0 <= successes <= total and total > 0, got {successes}/{total}")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def newcombe_interval(x1: int, n1: int, x2: int, n2: int, z: float = Z95) -> tuple[float, float]:
    """Newcombe's hybrid score interval (method 10) for the difference p2 - p1."""
    p1, p2 = x1 / n1, x2 / n2
    low1, high1 = wilson_interval(x1, n1, z)
    low2, high2 = wilson_interval(x2, n2, z)
    difference = p2 - p1
    return (float(difference - np.sqrt((p2 - low2) ** 2 + (high1 - p1) ** 2)),
            float(difference + np.sqrt((high2 - p2) ** 2 + (p1 - low1) ** 2)))


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
