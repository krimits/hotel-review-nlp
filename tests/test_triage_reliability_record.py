"""Preserve the real update and its failed development cases without manufacturing human scores."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from reviewnlp.triage.generator_experiment import RATINGS
from reviewnlp.triage.workflow_review import COUNTS
from scripts.score_triage_review import reconcile

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/experiments/triage_space_publication/e1e37e5"


def read(path):
    return json.loads((RECORD / path).read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archived_upload_matches_the_source_receipt_and_independently_read_live_snapshot():
    archive, inputs = read("record_manifest.json"), read("input_members.json")
    assert archive["input_archive_sha256"] == inputs["input_archive_sha256"]
    assert len(inputs["members"]) == 14
    for item in inputs["members"].values():
        assert sha(RECORD / item["archived_path"]) == item["sha256"]
    for name, expected in archive["files"].items():
        assert sha(RECORD / name) == expected
    manifest, receipt, smoke, verification = [read(path) for path in
        ("source_manifest.json", "published.json", "colab_smoke.json", "verification.json")]
    for report in (receipt, smoke, verification):
        assert report["source_commit"] == manifest["source_commit"] == archive["source_commit"]
        assert report["source_manifest_sha256"] == sha(RECORD / "source_manifest.json")
        assert report["quality_evaluated"] is False
    assert receipt["space_commit"] == verification["space_commit"] == archive["space_commit"]
    assert verification["runtime_commit"] == verification["current_hub_sha"] == receipt["space_commit"]
    assert receipt["upload_verified"] and receipt["runtime_verified"]
    assert receipt["live_model_info"]["source_snapshot"] == manifest
    assert verification["snapshot_verified"] and verification["loaded_snapshot_matches"]
    assert verification["public"] and verification["runtime_stage"] == "RUNNING"
    assert verification["hardware"] == receipt["hardware_requested"] == "zero-a10g"
    assert verification["verified_files_sha256"] == {**manifest["files"],
        "source_manifest.json": sha(RECORD / "source_manifest.json")}
    assert set(verification["hub_files"]) - {".gitattributes"} == set(verification["verified_files_sha256"])
    for name, expected in manifest["files"].items():
        path = "src/" + name if name.startswith("reviewnlp/") else "spaces/hotel-triage-demo/" + name
        data = subprocess.check_output(["git", "show", archive["source_commit"] + ":" + path], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == expected
    assert smoke["passed"] and smoke["real_weights_loaded"] and smoke["regression_checks_passed"]
    assert len(smoke["records"]) == 16 and smoke["problems"] == []
    assert {row["case"] for row in receipt["functional_checks"]} == set(smoke["required_regression_ids"])
    assert len(receipt["functional_checks"]) == 7
    assert all(row["status"] == "complete" and row["stage_failure"] is None and row["stored"] is False
               for row in receipt["functional_checks"])
    assert verification["functional_checks_repeated"] is False
    assert verification["additional_qwen_inference_calls"] == verification["additional_jev_calls"] == 0


def test_full_capture_keeps_four_failures_and_blank_human_judgments_instead_of_claiming_quality():
    folder = RECORD / "review"
    report, findings = read("review/run.json"), read("machine_findings.json")
    records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    for name, expected in report["files"].items():
        assert sha(folder / name) == expected
    assert (folder / "source_manifest.json").read_bytes() == (RECORD / "source_manifest.json").read_bytes()
    assert report["execution_complete"] and report["planned_cases"] == report["executed_cases"] == len(records) == 24
    assert report["failures"] == len([row for row in records if row["workflow_error"]]) == 4
    assert dict(Counter(row["status"] for row in records)) == findings["statuses"] == {"complete": 20, "partial": 4}
    assert sum(len(row["actions"]) for row in records) == findings["accepted_actions"] == 5
    assert Counter(issue["status"] for row in records for issue in row["extracted_issues"]) == findings["issue_statuses"]
    assert findings["issue_statuses"] == {"REAL_PENDING": 5, "UNCERTAIN": 11}
    assert all(row["api_cost"]["attempts"] == 0 and row["complaints"]["status"] == "disabled" for row in records)
    key = read("review/annotation_key.json")
    assert {row["review_id"] for row in key} == {row["id"] for row in records}
    assert len({row["blind_id"] for row in key}) == 24
    for name in ("coverage_A.csv", "coverage_B.csv"):
        with (folder / name).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 24 and all(row[field] == "" for row in rows for field in (*COUNTS, *RATINGS))
    with pytest.raises(ValueError, match="complete every coverage count"):
        reconcile(folder, folder / "coverage_A.csv", folder / "coverage_B.csv")
    assert report["selection_frozen"] is report["quality_evaluated"] is report["reserved_evaluation_used"] is False
    assert findings["promoted"] is findings["human_sheets_completed"] is False


def test_submitted_adjudication_matches_both_v3_sheets_and_the_unchanged_capture():
    folder = RECORD / "human_review_v3"
    manifest = read("human_review_v3/manifest.json")
    for name, expected in manifest["files"].items():
        assert sha(folder / name) == expected
    assert manifest["source_run_sha256"] == sha(RECORD / "review/run.json")
    assert manifest["space_commit"] == read("published.json")["space_commit"]
    assert manifest["source_commit"] == read("source_manifest.json")["source_commit"]
    assert manifest["judgments_reconciled"] and manifest["all_submitted_adjudications_match_v3"]
    assert manifest["reviewer_independence_verified"] is manifest["reserved_evaluation_used"] is False
    assert manifest["selection_frozen"] is manifest["promoted"] is False
    previous = reconcile(RECORD / "review", folder / "completed_A_v2.csv", folder / "completed_B_v2.csv")
    report = reconcile(RECORD / "review", folder / "completed_A_v3.csv", folder / "completed_B_v3.csv")
    assert previous == read("human_review_v3/scored_v2.json")
    assert report == read("human_review_v3/scored_v3.json")
    assert len(previous["disagreements"]) == manifest["prior_disagreement_fields"] == 15
    assert len({row["blind_id"] for row in previous["disagreements"]}) == manifest["prior_disagreement_reviews"] == 11
    assert report["judgments_reconciled"] and report["disagreements"] == []
    expected = {(row["blind_id"], row["field"]): (str(row["A"]), str(row["B"]))
                for row in previous["disagreements"]}
    sheets = []
    for reviewer in ("A", "B"):
        with (folder / f"completed_{reviewer}_v3.csv").open(encoding="utf-8-sig", newline="") as handle:
            sheets.append({row["blind_id"]: row for row in csv.DictReader(handle)})
        with (folder / f"completed_{reviewer}_v2.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for field, value in row.items():
                    if value != sheets[-1][row["blind_id"]][field]:
                        assert field == "notes" or (row["blind_id"], field) in expected
    with (folder / "triage_review_disagreements.csv").open(encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    assert len(decisions) == len(expected)
    assert {(row["blind_id"], row["field"]) for row in decisions} == set(expected)
    for decision in decisions:
        identity = decision["blind_id"], decision["field"]
        assert expected[identity] == (decision["A"], decision["B"])
        assert decision["agreed_value"] == sheets[0][identity[0]][identity[1]] == sheets[1][identity[0]][identity[1]]
        assert decision["adjudication_notes"].strip()
        for field in ("review", "issue_assessments", "accepted_actions"):
            assert decision[field] == sheets[0][identity[0]][field] == sheets[1][identity[0]][field]
