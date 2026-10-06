"""Integrity, interruption and human-rating gates for the new deployed-workflow capture."""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from reviewnlp.triage.generator_experiment import RATINGS
from reviewnlp.triage.workflow_review import COUNTS
from scripts.capture_triage_review import capture
from scripts.score_triage_review import reconcile

SOURCE = "a" * 40
ROWS = [{"id": "dev-01", "text": "The lobby was comfortable."},
        {"id": "dev-02", "text": "The room was comfortable."}]
MANIFEST = {"source_commit": SOURCE, "selection_status": "experimental_not_selected_not_promoted"}


class FakeClient:
    def __init__(self, *, failed=False, timeout=False, changed=False):
        self.calls, self.cancelled = [], False
        self.failed, self.timeout, self.changed = failed, timeout, changed
        self.info_reads = 0

    def predict(self, *, api_name):
        assert api_name == "/model_info"
        self.info_reads += 1
        return {"source_snapshot": {} if self.changed and self.info_reads == 2 else MANIFEST,
                "prompt_version": "actions-v6-source-spans", "jev_enabled": True, "runtime": {"ready": True}}

    def submit(self, text, jev, *, api_name):
        self.calls.append((text, jev))
        assert api_name == "/analyze"
        def result(timeout):
            assert timeout == 180
            if self.timeout:
                raise TimeoutError("SECRET visitor token")
            failure = {"stage": "qwen_runtime", "status": "error", "error": "gpu_quota_exceeded"} if self.failed else None
            stages = [failure] if failure else [{"stage": "issues", "status": "ok", "error": None}]
            return "summary", [], [], {"deployment": {"source_snapshot": MANIFEST}, "stored": False,
                "api_cost": {"attempts": 0}, "complaints": {"status": "disabled"}, "sentiment": {"label": "positive"},
                "issue_assessments": [], "actions": {"actions": []}, "status": "partial" if failure else "complete",
                "stage_reports": stages, "stage_failure": failure, "timings": {"total_ms": 10}}
        def cancel():
            self.cancelled = True
        return SimpleNamespace(result=result, cancel=cancel)


def filled(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(dict.fromkeys(COUNTS, "0"))
        row.update(dict.fromkeys(RATINGS, "na"))
        row["useful"] = "1"
    write(path, rows)
    return rows


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_capture_records_every_case_but_leaves_all_human_judgments_blank_and_never_promotes(tmp_path):
    client, folder = FakeClient(), tmp_path / "capture"
    report = capture(client, ROWS, folder, source_commit=SOURCE)
    assert report["execution_complete"] and report["planned_cases"] == report["executed_cases"] == 2
    assert report["failures"] == 0 and client.calls == [(row["text"], False) for row in ROWS]
    assert report["selection_frozen"] is report["quality_evaluated"] is report["reserved_evaluation_used"] is False
    with (folder / "coverage_A.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row[field] == "" for row in rows for field in (*COUNTS, *RATINGS))
    assert (folder / "coverage_A.csv").read_bytes() == (folder / "coverage_B.csv").read_bytes()
    with pytest.raises(FileExistsError):
        capture(client, ROWS, folder, source_commit=SOURCE)


@pytest.mark.parametrize("timeout", [False, True])
def test_runtime_failures_stop_further_calls_keep_unexecuted_cases_and_block_scoring(tmp_path, timeout):
    client, folder = FakeClient(failed=not timeout, timeout=timeout), tmp_path / "capture"
    report = capture(client, ROWS, folder, source_commit=SOURCE)
    assert not report["execution_complete"] and report["planned_cases"] == 2 and report["executed_cases"] == 1
    assert len(client.calls) == 1
    records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    assert records[1]["status"] == "not_executed" and records[1]["error"] == "not_executed_after_failure"
    assert report["stop_reason"]["error"] == ("client_timeout" if timeout else "qwen_runtime:gpu_quota_exceeded")
    assert "SECRET" not in str(report) and "SECRET" not in str(records)
    assert client.cancelled is timeout
    with pytest.raises(ValueError, match="completed dev run"):
        reconcile(folder, folder / "coverage_A.csv", folder / "coverage_B.csv")


def test_snapshot_changes_invalidate_run_and_jev_opt_in_must_be_exercised(tmp_path):
    report = capture(FakeClient(changed=True), ROWS, tmp_path / "changed", source_commit=SOURCE)
    assert not report["execution_complete"] and report["stop_reason"]["error"] == "snapshot_changed"
    report = capture(FakeClient(), ROWS, tmp_path / "jev", source_commit=SOURCE, use_jev=True)
    assert not report["execution_complete"] and report["stop_reason"]["error"] == "jev_opt_in_not_exercised"


def test_two_human_sheets_require_complete_immutable_outputs_and_explicit_agreement(tmp_path):
    folder = tmp_path / "capture"
    capture(FakeClient(), ROWS, folder, source_commit=SOURCE)
    a, b = tmp_path / "completed_A.csv", tmp_path / "completed_B.csv"
    a.write_bytes((folder / "coverage_A.csv").read_bytes())
    b.write_bytes((folder / "coverage_B.csv").read_bytes())
    with pytest.raises(ValueError, match="complete every coverage count"):
        reconcile(folder, a, b)
    filled(a)
    b_rows = filled(b)
    report = reconcile(folder, a, b)
    assert report["judgments_reconciled"] and report["agreed_summary"]
    assert report["selection_frozen"] is report["promoted"] is report["reviewer_independence_verified"] is False
    b_rows[0]["useful"] = "0"
    write(b, b_rows)
    report = reconcile(folder, a, b)
    assert not report["judgments_reconciled"] and report["agreed_summary"] is None
    assert report["disagreements"][0]["field"] == "useful"
    b_rows[0]["review"] = "edited review"
    write(b, b_rows)
    with pytest.raises(ValueError, match="output changed"):
        reconcile(folder, a, b)
    (folder / "source_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        reconcile(folder, a, b)


@pytest.mark.parametrize("useful, unaddressed", [("1", "1"), ("0", "0")])
def test_missing_actions_cannot_be_scored_as_resolved_or_fully_useful(tmp_path, useful, unaddressed):
    folder = tmp_path / "capture"
    capture(FakeClient(), ROWS, folder, source_commit=SOURCE)
    a, b = tmp_path / "completed_A.csv", tmp_path / "completed_B.csv"
    a.write_bytes((folder / "coverage_A.csv").read_bytes())
    b.write_bytes((folder / "coverage_B.csv").read_bytes())
    rows = filled(a)
    filled(b)
    rows[0].update(actual_problem_count="1", useful=useful, unaddressed_problem_count=unaddressed)
    write(a, rows)
    with pytest.raises(ValueError, match="contradicts|remain unaddressed"):
        reconcile(folder, a, b)
