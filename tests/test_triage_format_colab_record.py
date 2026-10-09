"""Preserve the incomplete real Colab receipt; never mistake planned rows for a completed evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from reviewnlp.triage.workflow_review import COUNTS, FIXED, score_coverage
from scripts.compare_triage_formats import frozen_inputs, schema_fingerprint

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/experiments/triage_generator/format_v1/colab_incomplete_20261009"


def test_real_colab_archive_integrity_and_actual_versions_retain_the_failed_condition():
    manifest = json.loads((RECORD / "receipt_manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((RECORD / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["local_diagnostics"].items():
        assert hashlib.sha256((RECORD / name).read_bytes()).hexdigest() == expected
    report = json.loads((RECORD / "comparison/run.json").read_bytes())
    for name, expected in report["files"].items():
        assert hashlib.sha256((RECORD / "comparison" / name).read_bytes()).hexdigest() == expected
    pin = manifest["expected_source_commit"]
    for name, field in (("scripts/compare_triage_formats.py", "script_sha256"),
                        ("src/reviewnlp/triage/structured_span_generator.py", "format_source_sha256")):
        source = subprocess.check_output(["git", "show", pin + ":" + name], cwd=ROOT)
        assert hashlib.sha256(source).hexdigest() == report[field]
    assert report["runtime"]["transformers"] == "5.18.0"
    assert report["runtime"]["gpu_name"] == "Tesla T4"
    assert report["runtime"]["real_weights_loaded"] and not report["execution_complete"]
    assert manifest["human_scoring_allowed"] is report["promoted"] is report["reserved_evaluation_used"] is False


def test_captured_pairs_hints_schemas_and_csvs_are_complete_in_shape_but_not_execution():
    report = json.loads((RECORD / "comparison/run.json").read_bytes())
    rows, provenance = frozen_inputs(scope="all")
    assert all(report[name] == value for name, value in provenance.items())
    assert report["schema_sha256_by_review"] == {row["id"]: schema_fingerprint(row["text"]) for row in rows}
    upstream = json.loads((RECORD / "comparison/upstream.json").read_bytes())
    assert upstream == [{"id": row["id"], "signals": row["signals"].__dict__,
                         "hosted_error": row["hosted_error"]} for row in rows]
    records = [json.loads(line) for line in (RECORD / "comparison/results.jsonl").read_text().splitlines()]
    assert Counter(row["status"] for row in records) == {"partial": 1, "execution_failed": 1, "not_executed": 46}
    keys = {(item["candidate"], item["id"]): item for item in records}
    assert len(keys) == 48 and not any(row["actions"] for row in records)
    key = json.loads((RECORD / "comparison/annotation_key.json").read_bytes())
    assert len(key) == len({item["blind_id"] for item in key}) == 48
    assert {(item["candidate"], item["review_id"]) for item in key} == keys.keys()
    with (RECORD / "comparison/coverage_review.csv").open(newline="", encoding="utf-8") as handle:
        sheets = {row["blind_id"]: row for row in csv.DictReader(handle)}
    texts = {row["id"]: row["text"] for row in rows}
    for item in key:
        sheet, capture = sheets[item["blind_id"]], keys[item["candidate"], item["review_id"]]
        assert sheet["review"] == texts[item["review_id"]]
        assert json.loads(sheet["issue_assessments"]) == capture["extracted_issues"]
        assert json.loads(sheet["accepted_actions"]) == capture["actions"]
        assert sheet["execution_issue"] == (capture["error"] or capture["workflow_error"] or "")
        assert all(sheet[name] == "" for name in COUNTS)
        assert set(FIXED) <= sheet.keys()
    assert (RECORD / "comparison/coverage_A.csv").read_bytes() == (RECORD / "comparison/coverage_B.csv").read_bytes()
    launcher = json.loads((RECORD / "launcher.json").read_bytes())
    assert launcher["exit_code"] == 1 and not launcher["execution_complete"]
    with pytest.raises(ValueError, match="completed dev run"):
        score_coverage(RECORD / "comparison", RECORD / "comparison/coverage_A.csv")


def test_real_local_import_failure_and_pinned_integration_probe_are_limited_to_dependency_diagnosis():
    failure = json.loads((RECORD / "local_import_reproduction.json").read_bytes())
    repaired = json.loads((RECORD / "local_pinned_preflight.json").read_bytes())
    tiny = json.loads((RECORD / "local_tiny_generation.json").read_bytes())
    assert failure["transformers"] == "5.18.0" and not failure["integration_import_ok"]
    assert failure["error_type"] == "ImportError" and not failure["model_loaded"]
    assert repaired["packages"]["transformers"] == "4.56.2" and repaired["format_integration_import_ok"]
    assert repaired["model_loaded"] is repaired["hub_requests"] is False
    assert tiny["integration_generation_ok"] and tiny["prefix_filter_calls"] > 0
    assert json.loads(tiny["output"]) == {"result": "ok"} and tiny["eos_seen"]
    assert tiny["pinned_hotel_weights_loaded"] is tiny["quality_evaluated"] is False
