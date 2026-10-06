"""Frozen-input guards and paired-output provenance for development-only format experiments."""

from __future__ import annotations

import copy
import csv
import json
import shutil

import pytest

from reviewnlp.triage.generator_experiment import RATINGS
from reviewnlp.triage.qwen_generator import GenerationResult
from reviewnlp.triage.workflow_review import COUNTS
from scripts.compare_triage_formats import BASELINE, DEVELOPMENT, ROOT, compare, frozen_inputs
from scripts.score_triage_review import reconcile


def test_original_failed_inputs_and_all_cases_are_loaded_without_any_human_labels():
    failures, provenance = frozen_inputs()
    assert [row["id"] for row in failures] == ["dev-01", "dev-05", "dev-08", "dev-11"]
    rows, _ = frozen_inputs(scope="all")
    assert len(rows) == 24
    assert all(set(row) == {"id", "text", "signals", "hosted_error"} for row in rows)
    assert all(row["signals"].flagged_topics == [] for row in rows)
    assert provenance["deployed_source_commit"] == "321c20689418c45267122f537ea6451f3ef30417"
    assert failures[-1]["signals"].sentiment_label == "positive"


@pytest.mark.parametrize("change", ["capture", "corpus", "backend"])
def test_changed_reviews_outputs_or_frozen_backend_stop_before_model_loading(tmp_path, change):
    baseline, development = tmp_path / "review", tmp_path / "dev.json"
    shutil.copytree(BASELINE, baseline)
    shutil.copyfile(DEVELOPMENT, development)
    root = ROOT
    if change == "capture":
        (baseline / "results.jsonl").write_text("changed")
    elif change == "corpus":
        corpus = json.loads(development.read_bytes())
        corpus["reviews"][0]["text"] = "An unrelated review."
        development.write_text(json.dumps(corpus))
    else:
        root = tmp_path / "repo"
        shutil.copytree(ROOT / "src/reviewnlp", root / "src/reviewnlp")
        (root / "src/reviewnlp/triage/span_evidence_generator.py").write_text("changed")
    with pytest.raises(ValueError):
        frozen_inputs(baseline=baseline, development=development, root=root)


class FakeWorkflow:
    def __init__(self, error=None, crash=False):
        self.workflow_error, self.crash, self.calls = error, crash, []
        self.issues, self.last_stages = [], []

    def generate(self, text, signals):
        self.calls.append((text, signals))
        self.last_stages = [{"stage": "issues", "raw": '{"issues":[]}',
                             "hit_token_budget": False, "error": None, "seconds": 0.001}]
        if self.crash:
            raise RuntimeError("untrusted exception contents are not retained")
        return GenerationResult('{"actions":[]}', False, "fake", "fake")


def test_paired_outputs_have_fresh_blind_judgments_and_preserve_expected_workflow_errors(tmp_path):
    rows, provenance = frozen_inputs()
    plain, constrained = FakeWorkflow("issues:invalid_evidence_span_id"), FakeWorkflow()
    output = tmp_path / "comparison"
    report = compare(rows, {"plain": plain, "structured": constrained}, output,
                     provenance=provenance, runtime={"real_weights_loaded": False})
    assert report["execution_complete"] and len(plain.calls) == len(constrained.calls) == 4
    assert report["summary"]["plain"]["workflow_errors"] == 4
    assert report["summary"]["structured"]["workflow_errors"] == 0
    assert report["failures"] == 4
    assert report["runtime"]["real_weights_loaded"] is False
    assert report["quality_evaluated"] is report["reserved_evaluation_used"] is report["hub_writes"] is False
    assert report["selection_frozen"] is report["promoted"] is False
    assert (output / "coverage_A.csv").read_bytes() == (output / "coverage_B.csv").read_bytes()
    with (output / "coverage_A.csv").open(newline="", encoding="utf-8") as handle:
        ratings = list(csv.DictReader(handle))
    assert len(ratings) == 8
    assert all(row[field] == "" for row in ratings for field in (*COUNTS, *RATINGS))
    assert sum(bool(row["execution_issue"]) for row in ratings) == 4
    assert all(text == row["text"] and signals is row["signals"]
               for row, (text, signals) in zip(rows, constrained.calls, strict=True))
    # Exercise the real scorer's report contract with artificial judgments of these fake outputs.
    for row in ratings:
        row.update(dict.fromkeys(COUNTS, "0"))
        row.update(dict.fromkeys(RATINGS, "na"))
        row["useful"] = "0"
    for name in ("completed_A.csv", "completed_B.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ratings[0]))
            writer.writeheader()
            writer.writerows(ratings)
    scored = reconcile(output, tmp_path / "completed_A.csv", tmp_path / "completed_B.csv")
    assert scored["judgments_reconciled"] and scored["machine_failures"] == 4
    assert scored["promoted"] is False
    with pytest.raises(FileExistsError):
        compare(rows, {}, output, provenance=provenance, runtime={})


def test_generation_crash_is_not_a_successful_empty_output_and_stops_remaining_calls(tmp_path):
    rows, provenance = frozen_inputs()
    plain, constrained = FakeWorkflow(), FakeWorkflow(crash=True)
    output = tmp_path / "failed"
    report = compare(rows, {"plain": plain, "structured": constrained}, output,
                     provenance=provenance, runtime={"real_weights_loaded": False})
    assert report["execution_complete"] is False
    assert report["stop_reason"] == {"candidate": "structured", "id": "dev-01", "error": "RuntimeError"}
    assert len(plain.calls) == len(constrained.calls) == 1
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert records[1]["status"] == "execution_failed"
    assert all(row["status"] == "not_executed" for row in records[2:])
    assert "untrusted exception" not in (output / "results.jsonl").read_text()


def test_measures_remain_subject_to_the_same_parser_and_are_not_retried(tmp_path):
    rows, provenance = frozen_inputs()
    workflow = FakeWorkflow("measures:hit_token_budget")
    report = compare(rows[:1], {"structured": workflow}, tmp_path / "cutoff",
                     provenance=copy.deepcopy(provenance), runtime={"real_weights_loaded": False})
    assert report["summary"]["structured"]["workflow_errors"] == 1
    assert report["summary"]["structured"]["accepted_actions"] == 0
    assert len(workflow.calls) == 1
