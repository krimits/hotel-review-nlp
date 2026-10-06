"""The real-weight formatting probe retains failures and does not erase remaining semantic defects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reviewnlp.triage.span_evidence_generator import parse_span_issues

RECORD = Path(__file__).resolve().parents[1] / "docs/experiments/triage_generator/format_v1"


def test_real_weight_probe_bytes_results_and_scope_are_preserved():
    manifest = json.loads((RECORD / "probe_manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((RECORD / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((RECORD / manifest["archived_experimental_source"]).read_bytes()).hexdigest() == \
        manifest["source_files_sha256"]["src/reviewnlp/triage/structured_span_generator.py"]
    old, new = [json.loads((RECORD / name).read_bytes()) for name in
                ("cpu_baseline_extraction.json", "cpu_structured_extraction.json")]
    for report in (old, new):
        assert report["real_weights_loaded"] and report["model_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        assert report["device"] == "cpu" and report["dtype"] == "torch.float32"
        assert report["quality_evaluated"] is report["reserved_evaluation_used"] is report["precision_matches_hosted"] is False
        assert report["do_sample"] is False and report["max_new_tokens"] == 400
    assert old["prompt_sha256"] == new["prompt_sha256"]
    assert [row["id"] for row in old["records"]] == [row["id"] for row in new["records"]] == \
        ["dev-01", "dev-05", "dev-08", "dev-11"]
    assert all(row["parse_error"] and not row["hit_token_budget"] for row in old["records"])
    assert old["records"][2]["hosted_error"] == "issues:invalid_json"
    assert old["records"][2]["parse_error"] == "invalid_evidence_issue_schema"
    for baseline, candidate in zip(old["records"], new["records"], strict=True):
        assert baseline["review"] == candidate["review"] and baseline["signals"] == candidate["signals"]
        assert candidate["parse_error"] is None and not candidate["hit_token_budget"]
        assert parse_span_issues(candidate["raw"], candidate["review"]) == candidate["issues"]
    assert new["records"][1]["issues"][0]["status"] == new["records"][3]["issues"][0]["status"] == "UNCERTAIN"
    assert new["records"][1]["issues"][0]["department"] == "management"
    assert new["records"][3]["issues"][0]["department"] == "housekeeping"


def test_full_real_weight_cpu_workflow_preserves_both_actions_and_remaining_uncertainty():
    folder = RECORD / "cpu_workflow"
    report = json.loads((folder / "run.json").read_bytes())
    assert hashlib.sha256((RECORD / "cpu_workflow_driver.py.txt").read_bytes()).hexdigest() == report["script_sha256"]
    for name, expected in report["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    assert report["runtime"]["real_weights_loaded"] and report["runtime"]["device"] == "cpu"
    assert report["execution_complete"] and not report["quality_evaluated"] and not report["promoted"]
    old, new = [report["summary"][name] for name in report["candidates"]]
    assert old["workflow_errors"] == 4 and new["workflow_errors"] == 0
    assert new["accepted_actions"] == new["uncertain_issues"] == 2
    records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    candidate = [row for row in records if row["candidate"] == "G-source-spans+json-schema"]
    assert [row["id"] for row in candidate if row["actions"]] == ["dev-01", "dev-08"]
    assert all([stage["stage"] for stage in row["stages"]] == ["issues", "measures"]
               for row in candidate if row["actions"])
    assert all(row["actions"] == [] for row in candidate if row["id"] in {"dev-05", "dev-11"})
