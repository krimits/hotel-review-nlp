"""Human coverage counts cannot be scored with changed outputs or inconsistent paired ground truth."""

from __future__ import annotations

import csv
import json

import pytest

from reviewnlp.triage.generator_experiment import digest_bytes, export_annotation
from reviewnlp.triage.workflow_review import COUNTS, export_coverage_review, score_coverage


def fixture(tmp_path):
    rows = [{"id": "one", "text": "The cupboard was dusty."}, {"id": "two", "text": "We loved the quiet room."}]
    action = {"problem": "Dusty cupboard", "excerpt": rows[0]["text"], "measure": "Clean the cupboard.",
              "department": "housekeeping", "to_confirm": []}
    records = [{"candidate": candidate, "id": row["id"], "actions": [action] if row["id"] == "one" else [],
                "error": None, "workflow_error": None, "hit_token_budget": False}
               for candidate in ("F", "G") for row in rows]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    export_annotation(tmp_path, rows, records)
    export_coverage_review(tmp_path, rows, records)
    files = {name: digest_bytes((tmp_path / name).read_bytes()) for name in
             ("results.jsonl", "coverage_review.csv", "annotation_key.json")}
    (tmp_path / "run.json").write_text(json.dumps({"split": "dev", "execution_complete": True, "files": files}))
    with (tmp_path / "coverage_review.csv").open(newline="") as handle:
        ratings = list(csv.DictReader(handle))
    for row in ratings:
        row.update(dict.fromkeys(COUNTS, "0"))
        if row["review"] == rows[0]["text"]:
            row.update(actual_problem_count="1", useful_action_count="1")
    return ratings


def save(tmp_path, rows):
    path = tmp_path / "ratings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_scores_only_complete_human_counts_and_preserves_zero_denominators(tmp_path):
    rows = fixture(tmp_path)
    result = score_coverage(tmp_path, save(tmp_path, rows))
    assert not result["selection_frozen"]
    assert result["summary"]["F"]["human_issue_coverage"] == 1
    assert result["summary"]["G"]["actual_problem_count"] == 1
    for row in rows:
        row["actual_problem_count"] = row["useful_action_count"] = "0"
    result = score_coverage(tmp_path, save(tmp_path, rows))
    assert result["summary"]["G"]["human_issue_coverage"] is None


@pytest.mark.parametrize("field,value", [("actual_problem_count", ""), ("missed_problem_count", "-1"),
                                        ("missed_problem_count", "3"), ("useful_action_count", "8"),
                                        ("false_exclusion_count", "1"), ("review", "Changed review")])
def test_incomplete_invalid_or_changed_ratings_are_rejected(tmp_path, field, value):
    rows = fixture(tmp_path)
    rows[0][field] = value
    with pytest.raises(ValueError):
        score_coverage(tmp_path, save(tmp_path, rows))


def test_candidates_must_share_human_ground_truth_and_source_artifacts(tmp_path):
    rows = fixture(tmp_path)
    next(row for row in rows if row["review"] == "The cupboard was dusty.")["actual_problem_count"] = "2"
    with pytest.raises(ValueError, match="same human ground truth"):
        score_coverage(tmp_path, save(tmp_path, rows))
    rows = fixture(tmp_path)
    (tmp_path / "results.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="source artifact differs"):
        score_coverage(tmp_path, save(tmp_path, rows))
