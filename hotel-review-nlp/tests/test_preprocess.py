"""Tests for the label-construction rules - the most bug-prone data step."""

from __future__ import annotations

import pandas as pd

from reviewnlp.data.preprocess import clean_text, extract_labeled_reviews, split_and_cap


def test_label_rule_uses_only_unambiguous_rows(tiny_reviews):
    out = extract_labeled_reviews(tiny_reviews, max_chars=1200, min_chars=15)
    # rows 1,3,5 (positive-only text) -> positive; rows 2,4,6 -> negative
    assert set(out["label"]) == {"positive", "negative"}
    assert (out["label"] == "positive").sum() == 3
    assert (out["label"] == "negative").sum() == 3
    assert out["text"].str.contains("dirty|Awful|Bad service", regex=True).sum() == 3


def test_mixed_and_marker_rows_dropped():
    df = pd.DataFrame(
        {
            "Positive_Review": ["Great stay", "Good but noisy", "No Positive", "Nothing special"],
            "Negative_Review": ["No Negative", "Walls are thin", "Bad bed", "No Negative"],
        }
    )
    out = extract_labeled_reviews(df, max_chars=1200, min_chars=5)
    # row 0: clean positive; row 1: mixed -> dropped; row 2: clean negative;
    # row 3: positive field written but not "No Positive" -> positive
    assert len(out) == 3
    assert sorted(out["label"]) == ["negative", "positive", "positive"]


def test_clean_text_strips_tags_and_whitespace():
    assert clean_text("hello  <br> world </p>", 100) == "hello world"
    assert clean_text("x" * 500, 10) == "x" * 10


def test_deduplication():
    df = pd.DataFrame(
        {
            "Positive_Review": ["Same review text here"] * 2,
            "Negative_Review": ["No Negative"] * 2,
        }
    )
    out = extract_labeled_reviews(df, max_chars=1200, min_chars=5)
    assert len(out) == 1


def test_split_is_stratified_and_capped():
    n_per_class = 1000
    df = pd.concat(
        [
            pd.DataFrame({"text": [f"pos text {i}" for i in range(n_per_class)], "label": ["positive"] * n_per_class}),
            pd.DataFrame({"text": [f"neg text {i}" for i in range(n_per_class)], "label": ["negative"] * n_per_class}),
        ]
    )
    train, dev, test = split_and_cap(df, seed=42, train_frac=0.8, dev_frac=0.1,
                                     train_cap=100, test_cap=50)
    assert len(test) == 100          # 50 per class
    assert len(train) == 200         # capped to 100 per class
    assert len(dev) == 210           # 10/90 of the post-test remainder: 105 per class
    assert set(train["label"]) == {"positive", "negative"}
    assert len(set(train["text"]) & set(test["text"])) == 0  # no leakage


def test_split_is_reproducible(tiny_reviews):
    out = extract_labeled_reviews(tiny_reviews, max_chars=1200, min_chars=1)
    t1, d1, s1 = split_and_cap(out, seed=7, train_frac=0.8, dev_frac=0.1, train_cap=0, test_cap=0)
    t2, d2, s2 = split_and_cap(out, seed=7, train_frac=0.8, dev_frac=0.1, train_cap=0, test_cap=0)
    assert list(t1["text"]) == list(t2["text"])
    assert list(s1["text"]) == list(s2["text"])
