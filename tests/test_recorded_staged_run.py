"""Replay immutable model responses; no inference or human quality labels are fabricated."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from reviewnlp.triage.generator_experiment import digest_bytes, structural_summary
from reviewnlp.triage.qwen_generator import ActionSignals, GenerationResult, parse_actions
from reviewnlp.triage.staged_generator import TwoStageGenerator, assemble_actions, parse_issues

RUN = Path(__file__).resolve().parents[1] / "docs/experiments/triage_generator/runs/20261003T145901Z_a29e8d68"
FENCED_IDS = {"dev-01", "dev-02", "dev-03", "dev-10", "dev-14"}


def recorded():
    records = [json.loads(line) for line in (RUN / "results.jsonl").read_text().splitlines()]
    key = json.loads((RUN / "annotation_key.json").read_text())
    sheet = {row["blind_id"]: row for row in csv.DictReader(io.StringIO((RUN / "human_review.csv").read_text()))}
    reviews = {(row["candidate"], row["review_id"]): sheet[row["blind_id"]]["review"] for row in key}
    return records, reviews


def test_original_run_bytes_and_failed_results_remain_immutable():
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert len(manifest["files"]) == 9
    for name, expected in manifest["files"].items():
        assert digest_bytes((RUN / name).read_bytes()) == expected
    info = json.loads((RUN / "run.json").read_text())
    for name, expected in info["files"].items():
        assert digest_bytes((RUN / name).read_bytes()) == expected
    assert info["source_commit"] == "7aa6bf14afb419add1953afa256ccc2a18a9c3f1"
    assert info["execution_complete"] and not info["quality_evaluated"]
    assert info["api_cost"]["attempts"] == 0
    assert info["summary"]["F"]["accepted_actions"] == 0
    assert info["summary"]["F"]["workflow_failures"] == 9
    assert info["summary"]["C"]["outputs_with_duplicate_keys"] == 0


def test_all_five_recorded_measure_blocks_have_valid_payloads_and_issue_links():
    records, reviews = recorded()
    seen = set()
    for row in records:
        if row["candidate"] != "F" or len(row["stages"]) != 2:
            continue
        pending = parse_issues(row["stages"][0]["raw"], reviews[("F", row["id"])])
        raw = row["stages"][1]["raw"]
        assert raw.startswith("```json\n") and raw.endswith("\n```")
        action, = assemble_actions(raw, pending)
        issue, = pending
        assert issue["status"] == "REAL_PENDING"
        assert all(action[name] == issue[name] for name in ("problem", "excerpt", "department"))
        assert action["excerpt"] in reviews[("F", row["id"])]
        assert action["measure"] and not row["actions"]
        seen.add(row["id"])
    assert seen == FENCED_IDS


class Replay:
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    _bundle = ("recorded tokenizer", "recorded weights")

    def __init__(self, raw):
        self.raw = raw

    def generate(self, review, signals):
        return GenerationResult(self.raw, False, self.model_name)


def test_offline_replay_changes_only_the_five_wrapper_rejections():
    records, reviews = recorded()
    recovered, failures = set(), []
    for row in records:
        if row["candidate"] != "F":
            continue

        def measures(pending, saved=row):
            assert len(saved["stages"]) == 2
            return Replay(saved["stages"][1]["raw"])

        generator = TwoStageGenerator(Replay(row["stages"][0]["raw"]), measures)
        result = generator.generate(reviews[("F", row["id"])], ActionSignals("negative", 0.9))
        parsed = parse_actions(result.raw, reviews[("F", row["id"])])
        assert [stage["raw"] for stage in generator.last_stages] == [stage["raw"] for stage in row["stages"]]
        if row["id"] in FENCED_IDS:
            assert parsed.json_valid and len(parsed.actions) == 1 and generator.workflow_error is None
            recovered.add(row["id"])
        else:
            assert generator.workflow_error == row["workflow_error"]
            assert len(parsed.actions) == len(row["actions"]) == 0
        if generator.workflow_error:
            failures.append(row["id"])
    assert recovered == FENCED_IDS
    assert set(failures) == {"dev-09", "dev-11", "dev-12", "dev-13"}


def test_syntax_diagnostics_include_rejected_outputs():
    records, _ = recorded()
    c = structural_summary([row for row in records if row["candidate"] == "C"])
    f = structural_summary([row for row in records if row["candidate"] == "F"])
    assert c["outputs_with_duplicate_keys"] == 3
    assert c["workflow_failures"] == 3 and c["reviews_with_accepted_actions"] == 16
    assert f["full_json_valid"] == 19
    assert f["workflow_failures"] == 9 and f["reviews_with_accepted_actions"] == 0
