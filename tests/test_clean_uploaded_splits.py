from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from reviewnlp.data.preprocess import load_processed
from reviewnlp.utils.experiments import assert_clean_splits, fingerprint_splits
from scripts.clean_uploaded_splits import clean_uploaded_splits


def _uploaded(tmp_path):
    source = tmp_path / "uploaded"
    source.mkdir()
    frames = {
        "train": pd.DataFrame({
            "text": ["conflicting room", "CONFLICTING  ROOM", "great staff", "great staff", "very bad staff"],
            "label": ["positive", "negative", "positive", "positive", "negative"],
        }),
        "dev": pd.DataFrame({"text": ["bright lobby", "loud room"], "label": ["positive", "negative"]}),
        "test": pd.DataFrame({"text": ["kind chef", "dirty carpet"], "label": ["positive", "negative"]}),
    }
    for name, frame in frames.items():
        frame.to_parquet(source / f"{name}.parquet")
    return source


def test_cleaning_keeps_legacy_files_and_removes_conflicting_labels(tmp_path):
    source = _uploaded(tmp_path)
    before = hashlib.sha256((source / "train.parquet").read_bytes()).hexdigest()
    output = tmp_path / "clean"
    manifest = clean_uploaded_splits(source, output)
    assert manifest["source_files_sha256"]["train"] == before
    assert hashlib.sha256((source / "train.parquet").read_bytes()).hexdigest() == before
    assert manifest["source_audit"]["train"]["conflicting_rows_removed"] == 2
    assert manifest["source_audit"]["train"]["duplicate_rows_removed"] == 1
    assert manifest["raw_csv_provenance_available"] is False
    assert_clean_splits(load_processed(str(output)))
    assert fingerprint_splits(output) == manifest["splits"]
    assert len(pd.read_parquet(output / "train.parquet")) == 2
    with pytest.raises(ValueError, match="new directory"):
        clean_uploaded_splits(source, output)


def test_cleaning_refuses_cross_split_leak_before_writing(tmp_path):
    source = _uploaded(tmp_path)
    pd.DataFrame({"text": ["GREAT  STAFF", "loud room"],
                  "label": ["positive", "negative"]}).to_parquet(source / "dev.parquet")
    output = tmp_path / "clean"
    with pytest.raises(ValueError, match="overlaps"):
        clean_uploaded_splits(source, output)
    assert not output.exists()
