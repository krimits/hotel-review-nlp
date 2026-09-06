"""Metric and McNemar correctness checks (including known-answer cases)."""

from __future__ import annotations

import numpy as np
import pytest

from reviewnlp.evaluation.metrics import binary_metrics, discordant_counts
from reviewnlp.evaluation.significance import mcnemar_exact, pairwise_mcnemar


def test_perfect_predictions():
    y = ["negative", "positive", "negative", "positive"]
    m = binary_metrics(y, y)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_known_answer_case():
    y = ["negative", "negative", "negative", "negative", "positive", "positive"]
    p = ["negative", "positive", "negative", "negative", "negative", "positive"]
    m = binary_metrics(y, p)
    # negative: 4 true / 3 correct -> recall 0.75; 4 predicted / 3 correct -> precision 0.75
    # positive: 2 true / 1 correct -> recall 0.5; 2 predicted / 1 correct -> precision 0.5
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["per_class"]["negative"]["recall"] == pytest.approx(0.75, abs=1e-4)
    assert m["per_class"]["negative"]["precision"] == pytest.approx(0.75, abs=1e-4)
    assert m["per_class"]["positive"]["precision"] == pytest.approx(0.5, abs=1e-4)
    assert m["per_class"]["positive"]["recall"] == pytest.approx(0.5, abs=1e-4)
    assert m["confusion_matrix"] == [[3, 1], [1, 1]]


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        binary_metrics(["negative"], ["negative", "positive"])


def test_numeric_labels_keep_human_readable_class_names():
    metrics = binary_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        label_names=("negative", "positive"),
        label_values=(0, 1),
    )
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert set(metrics["per_class"]) == {"negative", "positive"}
    assert metrics["per_class"]["negative"]["support"] == 2


def test_mcnemar_known_distribution():
    # 0 discordant -> no evidence of difference
    assert mcnemar_exact(0, 0)["p_value"] == 1.0
    # perfectly balanced discordant pairs -> p = 1.0
    assert mcnemar_exact(5, 5)["p_value"] == pytest.approx(1.0)
    # 18 vs 2 is a classic significant case (exact binomial p < 0.001)
    res = mcnemar_exact(18, 2)
    assert res["significant_at_0.05"] and res["p_value"] < 0.001
    # 7 vs 3 is NOT significant at 0.05 (two-sided exact p ~ 0.34)
    assert not mcnemar_exact(7, 3)["significant_at_0.05"]


def test_discordant_counts():
    y = np.array([1, 1, 0, 0, 1])
    a = np.array([1, 0, 0, 1, 1])  # A wrong on idx1 (y=1,a=0) AND idx3 (y=0,a=1)
    b = np.array([1, 1, 0, 1, 1])  # B right on idx1, wrong on idx3
    n01, n10 = discordant_counts(y, a, b, "A", "B")
    assert n01 == 1  # A wrong / B right  (idx1 only)
    assert n10 == 0  # no A-right/B-wrong case (both are wrong on idx3)


def test_pairwise_mcnemar_structure():
    y = np.array([0, 1] * 10)
    preds = {"m1": y.copy(), "m2": np.array([1, 0] * 10), "m3": y.copy()}
    out = pairwise_mcnemar(y, preds)
    assert set(out) == {"m1 vs m2", "m1 vs m3", "m2 vs m3"}
    assert out["m1 vs m3"]["p_value"] == 1.0
    assert out["m1 vs m2"]["n_discordant"] == 20
