"""The raw-data fetcher refuses any file that is not the project's source."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_booking_515k", ROOT / "scripts" / "fetch_booking_515k.py")
fetch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fetch
_spec.loader.exec_module(fetch)


def test_the_pinned_source_is_the_audited_kaggle_file():
    assert fetch.SIZE == 238_154_765
    assert fetch.SHA256.startswith("a4810c27")
    assert len(fetch.REVISION) == 40


def test_verify_accepts_the_expected_bytes(tmp_path):
    source = tmp_path / "reviews.csv"
    source.write_bytes(b"Hotel_Name,Review_Date\n")
    fetch.verify(source, size=source.stat().st_size, expected=hashlib.sha256(source.read_bytes()).hexdigest())


@pytest.mark.parametrize("change", ["size", "content"])
def test_verify_refuses_another_file(tmp_path, change):
    source = tmp_path / "reviews.csv"
    source.write_bytes(b"Hotel_Name,Review_Date\n")
    size, digest = source.stat().st_size, hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"Hotel_Name,Review_Date,Extra\n" if change == "size" else b"Hotel_Name,Review_Datf\n")
    with pytest.raises(SystemExit):
        fetch.verify(source, size=size, expected=digest)
