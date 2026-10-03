"""Reproduce the duplicate-key defect against the actual immutable Qwen outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reviewnlp.triage.generator_experiment import digest_bytes, json_diagnostics
from reviewnlp.triage.qwen_generator import parse_actions

RUN = Path(__file__).resolve().parents[1] / "docs/experiments/triage_generator/runs/20261003T121429Z_876f0781"


def test_all_recorded_bytes_and_the_historical_counts_are_preserved():
    manifest = json.loads((RUN / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert digest_bytes((RUN / name).read_bytes()) == expected
    info = json.loads((RUN / "run.json").read_text())
    for name, expected in info["files"].items():
        assert digest_bytes((RUN / name).read_bytes()) == expected
    assert set(info["candidates"]) == {"C", "E"} and info["execution_complete"]
    assert info["api_cost"]["attempts"] == 0
    assert info["summary"]["E"]["reviews_with_accepted_actions"] == 5
    assert info["summary"]["E"]["outputs_with_duplicate_keys"] == 12


def test_every_actual_ambiguous_output_is_rejected_without_reinterpreting_the_saved_sheet():
    key = json.loads((RUN / "annotation_key.json").read_text())
    with (RUN / "human_review.csv").open(newline="") as handle:
        sheet = {row["blind_id"]: row for row in csv.DictReader(handle)}
    reviews = {(row["candidate"], row["review_id"]): sheet[row["blind_id"]]["review"] for row in key}
    records = [json.loads(line) for line in (RUN / "results.jsonl").read_text().splitlines()]
    rejected = {"C": 0, "E": 0}
    for row in records:
        if json_diagnostics(row["raw"])["duplicate_json_keys"]:
            parsed = parse_actions(row["raw"], reviews[(row["candidate"], row["id"])])
            assert not parsed.json_valid and not parsed.actions and parsed.error == "duplicate JSON keys"
            rejected[row["candidate"]] += 1
    assert rejected == {"C": 3, "E": 12}
