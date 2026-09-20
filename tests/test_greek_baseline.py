"""Offline tests for the Greek classical baseline — sklearn only, no network."""

from __future__ import annotations

import pandas as pd
import pytest

from reviewnlp.greek.baseline import build_pipeline, train_baseline
from reviewnlp.greek.config import BaselineConfig


def _frames() -> dict[str, pd.DataFrame]:
    """Separable toy corpus: one vocabulary per class."""

    def block(n: int, offset: int = 0) -> pd.DataFrame:
        rows = []
        for index in range(n):
            rows.append({"text": f"εξαιρετικό τέλειο άριστο {index + offset}", "label": 1})
            rows.append({"text": f"απαίσιο χάλια κακό {index + offset}", "label": 0})
        return pd.DataFrame(rows)

    return {"train": block(40), "validation": block(8, 100), "test": block(8, 200)}


def test_pipeline_is_built_from_the_config():
    config = BaselineConfig(max_features=123, ngram_range=(1, 3), min_df=1, C=0.25)
    pipeline = build_pipeline(config)
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    assert tfidf.max_features == 123
    assert tfidf.ngram_range == (1, 3)
    assert tfidf.min_df == 1
    assert clf.C == 0.25
    assert clf.class_weight == "balanced"
    assert clf.random_state == config.seed


def test_train_baseline_reports_both_splits_in_the_shared_schema():
    metrics = train_baseline(_frames(), BaselineConfig(min_df=1))

    for split in ("validation", "test"):
        assert set(metrics[split]) >= {"accuracy", "macro_f1", "per_class", "confusion_matrix"}
        assert set(metrics[split]["per_class"]) == {"negative", "positive"}
    # A separable corpus must be solved, or the wiring is wrong.
    assert metrics["test"]["macro_f1"] == pytest.approx(1.0)
    assert metrics["vocabulary_size"] > 0
    assert metrics["settings"]["class_weight"] == "balanced"


def test_baseline_never_sees_the_test_split_during_fit():
    """Guards the leakage the English pipeline already tests for."""
    frames = _frames()
    pipeline = build_pipeline(BaselineConfig(min_df=1))
    pipeline.fit(frames["train"]["text"], frames["train"]["label"])
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    # "200" only appears in the test block's texts.
    assert not any(token.startswith("200") for token in vocabulary)
