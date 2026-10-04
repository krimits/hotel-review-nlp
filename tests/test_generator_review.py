"""Ratings are matched to the actual exported output, not guessed from shuffled IDs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from reviewnlp.triage.generator_experiment import RATINGS, digest_bytes, export_annotation
from reviewnlp.triage.generator_review import audit_ratings, csv_rows, write_audit


def setup_run(root: Path):
    root.mkdir()
    records = [{"candidate": "C", "id": "dev-case", "actions": [], "raw": '{"actions":[]}',
                "error": None, "hit_token_budget": False}]
    export_annotation(root, [{"id": "dev-case", "text": "Our stay was pleasant."}], records)
    (root / "results.jsonl").write_text(json.dumps(records[0]) + "\n")
    (root / "upstream.json").write_text("{}\n")
    info = {"split": "dev", "execution_complete": True, "candidates": {"C": {}}, "files": {
        name: digest_bytes((root / name).read_bytes()) for name in
        ("human_review.csv", "annotation_key.json", "results.jsonl", "upstream.json")}}
    (root / "run.json").write_text(json.dumps(info))
    rated = csv_rows(root / "human_review.csv")
    rated[0].update(useful="yes", grounded="n/a", department_correct="n/a", no_invented_facts="n/a",
                    execution_issue="no", notes="Provided review")
    return rated


def save(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_normalization_restores_the_machine_column_in_a_separate_file_and_preserves_sources(tmp_path):
    run, ratings = tmp_path / "run", tmp_path / "ratings.csv"
    rated = setup_run(run)
    save(ratings, rated)
    before = ratings.read_bytes(), (run / "human_review.csv").read_bytes()
    report, sheet = audit_ratings(run, ratings)
    assert report["ready_for_freeze"] and report["execution_issue_overwrites"] == [rated[0]["blind_id"]]
    assert sheet[0]["execution_issue"] == "" and sheet[0]["useful"] == "1"
    assert all(sheet[0][field] == "na" for field in RATINGS[1:])
    write_audit(tmp_path / "audit", report, sheet)
    assert (tmp_path / "audit" / "human_review_normalized.csv").exists()
    assert before == (ratings.read_bytes(), (run / "human_review.csv").read_bytes())
    with pytest.raises(FileExistsError):
        write_audit(tmp_path / "audit", report, sheet)


def test_partial_is_not_inferred_and_no_freeze_ready_sheet_is_exported(tmp_path):
    run, ratings = tmp_path / "run", tmp_path / "ratings.csv"
    rated = setup_run(run)
    rated[0]["useful"] = "partial"
    save(ratings, rated)
    report, sheet = audit_ratings(run, ratings)
    assert not report["ready_for_freeze"] and sheet[0]["useful"] == "partial"
    assert report["pending_judgments"][0]["field"] == "useful"
    write_audit(tmp_path / "audit", report, sheet)
    assert (tmp_path / "audit" / "adjudication.csv").exists()
    assert not (tmp_path / "audit" / "human_review_normalized.csv").exists()


@pytest.mark.parametrize("field,value", [("review", "Another review."), ("accepted_actions", "[{}]"),
                                        ("blind_id", "unknown")])
def test_changed_machine_content_or_ids_are_rejected(tmp_path, field, value):
    run, ratings = tmp_path / "run", tmp_path / "ratings.csv"
    rated = setup_run(run)
    rated[0][field] = value
    save(ratings, rated)
    with pytest.raises(ValueError):
        audit_ratings(run, ratings)


def test_edited_archive_and_duplicated_rows_are_rejected(tmp_path):
    run, ratings = tmp_path / "run", tmp_path / "ratings.csv"
    rated = setup_run(run)
    save(ratings, rated * 2)
    with pytest.raises(ValueError):
        audit_ratings(run, ratings)
    save(ratings, rated)
    (run / "results.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="fingerprint"):
        audit_ratings(run, ratings)
