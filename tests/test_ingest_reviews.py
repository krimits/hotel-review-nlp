from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    source = Path(__file__).resolve().parents[1] / "scripts" / "ingest_reviews.py"
    spec = importlib.util.spec_from_file_location("ingest_reviews", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_import_rejects_cross_hotel_and_duplicate_external_ids(tmp_path):
    script = _load_script()
    csv = tmp_path / "reviews.csv"
    csv.write_text("hotel_id,review_id,source,text\nother,r1,booking,dirty room\n")
    with pytest.raises(ValueError, match="another hotel"):
        script.load_reviews(csv, "my-hotel")
    csv.write_text("review_id,source,text\nr1,booking,dirty room\nr1,booking,kind staff\n")
    with pytest.raises(ValueError, match="repeated"):
        script.load_reviews(csv, "my-hotel")


def test_non_local_import_refuses_plaintext_http():
    script = _load_script()
    with pytest.raises(ValueError, match="HTTPS"):
        script.import_reviews("http://some-hotel.example", "hotel", "token", [], 8)


def test_csv_import_refuses_unvalidated_greek_aspects(tmp_path):
    script = _load_script()
    csv = tmp_path / "greek.csv"
    csv.write_text("review_id,text,language\nr1,Άψογο προσωπικό,el\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only English"):
        script.load_reviews(csv, "hotel")
