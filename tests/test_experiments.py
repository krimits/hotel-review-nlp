from __future__ import annotations

import pandas as pd
import pytest

from reviewnlp.utils.experiments import fingerprint_splits, prepare_experiment_splits


def _write_splits(directory):
    directory.mkdir()
    offset = 0
    for split, rows in (("train", 40), ("dev", 20), ("test", 20)):
        frame = pd.DataFrame(
            {
                "text": [f"{split} review {index}" for index in range(rows)],
                "label": ["negative" if index % 4 == 0 else "positive" for index in range(rows)],
            }
        )
        frame.index += offset
        frame.to_parquet(directory / f"{split}.parquet", index=False)
        offset += rows


def test_prepare_experiment_splits_is_deterministic_and_stratified(tmp_path):
    source = tmp_path / "source"
    first = tmp_path / "first" / "data"
    second = tmp_path / "second" / "data"
    _write_splits(source)
    caps = {"train": 20, "dev": 10, "test": None}

    manifest_a = prepare_experiment_splits(source, first, caps, seed=42)
    manifest_b = prepare_experiment_splits(source, second, caps, seed=42)

    assert manifest_a["splits"] == manifest_b["splits"]
    assert manifest_a["splits"]["train"]["rows"] == 20
    assert manifest_a["splits"]["train"]["negative"] == 5
    assert manifest_a["splits"]["test"]["rows"] == 20
    assert fingerprint_splits(first) == manifest_a["splits"]


def test_prepare_experiment_splits_rejects_cross_split_overlap(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "experiment" / "data"
    _write_splits(source)
    dev = pd.read_parquet(source / "dev.parquet")
    dev.loc[0, "text"] = "train review 0"
    dev.to_parquet(source / "dev.parquet", index=False)

    with pytest.raises(ValueError, match="overlaps"):
        prepare_experiment_splits(
            source,
            destination,
            {"train": None, "dev": None, "test": None},
            seed=42,
        )
