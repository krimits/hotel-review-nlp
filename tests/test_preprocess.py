"""Tests for the label-construction rules - the most bug-prone data step."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from reviewnlp.data.preprocess import (
    build_dataset,
    clean_text,
    extract_labeled_reviews,
    split_and_cap,
)


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


def test_deduplication_normalizes_case_whitespace_and_unicode():
    raw = pd.DataFrame(
        {
            "Positive_Review": [
                "Great location",
                "great  LOCATION",
                "ＦＡＮＴＡＳＴＩＣ stay",
                "fantastic stay",
            ],
            "Negative_Review": ["No Negative"] * 4,
        }
    )

    reviews = extract_labeled_reviews(raw, max_chars=1200, min_chars=5)

    assert reviews["text"].tolist() == ["Great location", "ＦＡＮＴＡＳＴＩＣ stay"]


def test_conflicting_normalized_labels_are_all_excluded():
    raw = pd.DataFrame(
        {
            "Positive_Review": ["Mixed service", "MIXED SERVICE", "No Positive", "Great stay"],
            "Negative_Review": ["No Negative", "No Negative", "mixed  service", "No Negative"],
        }
    )

    reviews = extract_labeled_reviews(raw, max_chars=1200, min_chars=5)

    assert reviews[["text", "label"]].to_dict("records") == [
        {"text": "Great stay", "label": "positive"}
    ]


def test_missing_source_text_is_not_a_review():
    raw = pd.DataFrame(
        {
            "Positive_Review": [None, "", "No Positive", "No Positive", "Good stay"],
            "Negative_Review": ["No Negative", "No Negative", pd.NA, "", "No Negative"],
        }
    )

    reviews = extract_labeled_reviews(raw, max_chars=1200, min_chars=1)

    assert reviews["text"].tolist() == ["Good stay"]


def test_review_id_is_stable_across_normalized_variants():
    first = pd.DataFrame({"Positive_Review": ["Great location"], "Negative_Review": ["No Negative"]})
    second = pd.DataFrame({"Positive_Review": ["great  LOCATION"], "Negative_Review": ["No Negative"]})

    original = extract_labeled_reviews(first, max_chars=1200, min_chars=1)
    variant = extract_labeled_reviews(second, max_chars=1200, min_chars=1)

    expected = hashlib.sha256(b"great location").hexdigest()
    assert original["review_id"].tolist() == variant["review_id"].tolist() == [expected]


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


def test_split_honors_fractions_and_zero_means_unlimited_cap():
    reviews = pd.DataFrame(
        [(f"{label} review {index}", label) for label in ("negative", "positive") for index in range(100)],
        columns=["text", "label"],
    )

    splits = split_and_cap(reviews, seed=42, train_frac=0.6, dev_frac=0.2, train_cap=0, test_cap=0)

    assert [len(frame) for frame in splits] == [120, 40, 40]


def test_clean_dataset_writes_a_verifiable_manifest(tmp_path, tiny_reviews):
    raw = tmp_path / "source.csv"
    tiny_reviews.to_csv(raw, index=False)
    output = tmp_path / "processed"
    config = tmp_path / "config.yaml"
    config.write_text(
        "seed: 42\n"
        f"data:\n  raw_csv: {raw}\n  processed_dir: {output}\n"
        "  max_chars: 1200\n  min_chars: 5\n  train_frac: 0.5\n"
        "  dev_frac: 0.25\n  train_cap: 0\n  test_cap: 0\n"
    )
    build_dataset(str(config))
    manifest = json.loads((output / "data_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["raw_csv_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert manifest["cross_split_overlap"]["train_test"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_frac": 0},
        {"dev_frac": 0},
        {"train_frac": 0.9, "dev_frac": 0.2},
        {"train_frac": float("nan")},
        {"train_cap": -1},
        {"test_cap": -1},
    ],
)
def test_split_rejects_invalid_settings(tiny_reviews, overrides):
    reviews = extract_labeled_reviews(tiny_reviews, max_chars=1200, min_chars=1)
    settings = dict(seed=42, train_frac=0.8, dev_frac=0.1, train_cap=0, test_cap=0)
    settings.update(overrides)

    with pytest.raises(ValueError):
        split_and_cap(reviews, **settings)


def _dated_reviews(hotels: int = 10, per_hotel: int = 20) -> pd.DataFrame:
    rows = []
    for hotel in range(hotels):
        for index in range(per_hotel):
            day = (hotel * per_hotel + index) % 60
            label = "positive" if index % 2 else "negative"
            rows.append({"text": f"{label} review {hotel}-{index}", "label": label, "hotel": f"Hotel {hotel}",
                         "review_date": f"2016-{1 + day // 28:02d}-{1 + day % 28:02d}"})
    return pd.DataFrame(rows)


def test_time_split_trains_on_older_reviews_and_tests_on_newer():
    reviews = _dated_reviews()
    train, dev, test = split_and_cap(reviews, seed=1, train_frac=0.6, dev_frac=0.2, train_cap=0, test_cap=0,
                                     policy="time")
    assert train["review_date"].max() < dev["review_date"].min()
    assert dev["review_date"].max() < test["review_date"].min()
    assert len(train) + len(dev) + len(test) == len(reviews)


def test_hotel_split_keeps_each_hotel_in_one_split():
    reviews = _dated_reviews()
    splits = split_and_cap(reviews, seed=1, train_frac=0.6, dev_frac=0.2, train_cap=0, test_cap=0, policy="hotel")
    train, dev, test = (set(frame["hotel"]) for frame in splits)
    assert not (train & dev or train & test or dev & test)
    assert [len(frame) for frame in splits] == [120, 40, 40]


def test_hotel_split_caps_within_its_hotels():
    reviews = _dated_reviews()
    train, _, test = split_and_cap(reviews, seed=1, train_frac=0.6, dev_frac=0.2, train_cap=10, test_cap=5,
                                   policy="hotel")
    assert test["label"].value_counts().to_dict() == {"negative": 5, "positive": 5}
    assert len(train) == 20
    assert not set(train["hotel"]) & set(test["hotel"])


@pytest.mark.parametrize("policy", ["time", "hotel", "weekday"])
def test_a_split_policy_needs_its_column(policy):
    reviews = pd.DataFrame({"text": [f"review {i}" for i in range(8)], "label": ["positive", "negative"] * 4})
    with pytest.raises(ValueError):
        split_and_cap(reviews, seed=1, train_frac=0.5, dev_frac=0.25, train_cap=0, test_cap=0, policy=policy)


def test_extraction_keeps_hotel_date_and_the_earliest_copy():
    raw = pd.DataFrame(
        {
            "Hotel_Name": ["B", "A", "C"],
            "Review_Date": ["8/3/2017", "1/15/2016", "5/1/2016"],
            "Positive_Review": ["Lovely staff and room", "lovely staff  and room", "Quiet and clean place"],
            "Negative_Review": ["No Negative"] * 3,
        }
    )
    out = extract_labeled_reviews(raw, max_chars=1200, min_chars=5)
    assert len(out) == 2
    kept = out.loc[out["text"].str.casefold().str.startswith("lovely")].iloc[0]
    assert (kept["hotel"], kept["review_date"]) == ("A", "2016-01-15")


def test_hotel_dataset_manifest_records_the_policy(tmp_path):
    rows = []
    for hotel in range(8):
        for index in range(6):
            positive = index % 2 == 0
            rows.append({"Hotel_Name": f"Hotel {hotel}", "Review_Date": f"{1 + index}/1/2016",
                         "Positive_Review": f"Great stay number {hotel} {index}" if positive else "No Positive",
                         "Negative_Review": "No Negative" if positive else f"Poor stay number {hotel} {index}"})
    raw = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    output = tmp_path / "processed"
    config = tmp_path / "config.yaml"
    config.write_text(
        "seed: 3\n"
        f"data:\n  raw_csv: {raw}\n  processed_dir: {output}\n  split: hotel\n"
        "  max_chars: 1200\n  min_chars: 5\n  train_frac: 0.5\n"
        "  dev_frac: 0.25\n  train_cap: 0\n  test_cap: 0\n"
    )
    build_dataset(str(config))
    manifest = json.loads((output / "data_manifest.json").read_text())
    assert manifest["split"] == "hotel"
    assert manifest["hotels_in_more_than_one_split"] == 0
    assert sum(manifest["hotels"].values()) == 8
    assert manifest["raw_csv_bytes"] == raw.stat().st_size
