"""The Jev benchmark's committed results: no review text, the locked inputs, and every number reproduced.

The run was made on 1 October 2026 (docs/experiments/jev_topic_benchmark/results/README.md). The
texts are not committed, so these checks use only the saved answers, the committed labels and key,
and the script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs" / "experiments" / "jev_topic_benchmark"
RESULTS = FOLDER / "results"
SCRIPT = "scripts/benchmark_jev_topics.py"

# SHA-256 of the three files as the script wrote them on the project owner's Windows machine, with CRLF
# line ends. The repository stores them with LF (.gitattributes), so the check restores CRLF first.
RECEIVED = {
    "summary.json": "d68ec45a93ebe1523ec3d253f551083c6f64a412ac703096593f52ac3f17fbb9",
    "predictions.csv": "1120910bbade4904d042194c60cf0e54911ab18c3882b34571d4624664dfe51a",
    "responses.jsonl": "d520274c805ff97fc4fa5f4aac9775df5e952c064417643acf1159235f3b6e76",
}

_spec = importlib.util.spec_from_file_location("benchmark_jev_topics", ROOT / SCRIPT)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)

SUMMARY = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
KEY = bench.read_key(bench.RESULTS / "key.csv")
GOLD = bench.read_gold(bench.RESULTS / "labels_final.csv")
RANDOM = sorted(item for item, row in KEY.items() if row["in_R"])


def _records() -> list[dict]:
    return [json.loads(line) for line in (RESULTS / "responses.jsonl").read_text(encoding="utf-8").splitlines()]


def _flatten(value, path: str = "") -> dict:
    if isinstance(value, dict):
        return {key: leaf for name, item in value.items() for key, leaf in _flatten(item, f"{path}/{name}").items()}
    if isinstance(value, list):
        return {key: leaf for i, item in enumerate(value) for key, leaf in _flatten(item, f"{path}/{i}").items()}
    return {path: value}


def test_only_the_three_files_and_the_readme_are_committed():
    assert sorted(path.name for path in RESULTS.iterdir()) == [
        "README.md", "predictions.csv", "responses.jsonl", "summary.json"]


def test_the_files_are_the_ones_received_apart_from_their_line_endings():
    for name, received in RECEIVED.items():
        stored = (RESULTS / name).read_bytes()
        assert b"\r" not in stored, name
        assert hashlib.sha256(stored.replace(b"\n", b"\r\n")).hexdigest() == received, name


def test_the_saved_answers_hold_no_review_text_and_no_key():
    records = _records()
    assert len(records) == 200
    for record in records:
        assert set(record) == {"item", "questions_sha256", "sent_utc", "seconds", "attempts", "response"}
        response = record["response"]
        assert set(response) == {"answers", "id", "model", "provider", "usage"}
        assert re.fullmatch(r"gen-[\w-]+", response["id"]) and response["provider"] == "TypeSafe"
        assert set(response["usage"]) == {"input_tokens", "output_tokens", "cost"}
        assert set(response["answers"]) == set(bench.TOPICS)
        for answer in response["answers"].values():
            assert set(answer) == {"type", "choice", "probabilities", "confidence"}
            assert answer["type"] == "choice" and answer["choice"] in bench.LABELS
            assert set(answer["probabilities"]) == set(bench.LABELS)
    for name in RECEIVED:
        text = (RESULTS / name).read_text(encoding="utf-8")
        assert not any(marker in text for marker in ("sk-", "Bearer", "API_KEY")), name
    with open(RESULTS / "predictions.csv", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == ["item", *(f"{t}_{field}" for t in bench.TOPICS
                                for field in ("choice", "p_choice", "p1", "confidence"))]


def test_the_answers_are_to_the_locked_questions_for_exactly_the_random_texts():
    records = _records()
    assert [record["item"] for record in records] == RANDOM  # sent in item order, none twice, none missing
    assert {record["questions_sha256"] for record in records} == {bench.QUESTIONS_SHA256}
    assert {record["response"]["model"] for record in records} == {"typesafe/jev-1.13-20260917"}
    assert {record["attempts"] for record in records} == {1}
    assert f"`{bench.QUESTIONS_SHA256}`" in (FOLDER / "DECISION_v2.md").read_text(encoding="utf-8")


def test_the_inputs_recorded_are_the_committed_files():
    inputs = SUMMARY["inputs"]
    agreement = json.loads((bench.RESULTS / "agreement.json").read_text(encoding="utf-8"))
    assert inputs["texts_sha256"] == agreement["sheets"]["A"]["sha256"]
    assert inputs["texts_are_the_handed_in_sheet_a"] is True
    assert inputs["key_sha256"] == bench.sha256_file(bench.RESULTS / "key.csv")
    assert inputs["gold_sha256"] == bench.sha256_file(bench.RESULTS / "labels_final.csv")
    assert inputs["questions_sha256"] == bench.QUESTIONS_SHA256
    assert (inputs["route"], inputs["endpoint"], inputs["model_requested"]) == (
        "openrouter", "https://openrouter.ai/api/v1/systemone", "jev-latest")
    assert SUMMARY["design"] == "docs/experiments/jev_topic_benchmark/DECISION_v2.md"


def test_the_summary_is_reproduced_from_the_saved_answers(tmp_path):
    (tmp_path / "responses.jsonl").write_bytes((RESULTS / "responses.jsonl").read_bytes())
    args = argparse.Namespace(price_per_million_input_tokens=None, price_per_million_output_tokens=0.0,
                              currency="USD")
    bench.write_results(RANDOM, 200, KEY, GOLD, tmp_path, SUMMARY["inputs"], args)
    again = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    committed, recomputed = _flatten(SUMMARY), _flatten(again)
    assert committed.keys() == recomputed.keys()
    for path, value in committed.items():
        if path == "/script_sha256":
            continue  # which script ran is checked below
        if isinstance(value, float):
            # Python 3.12 sums floats more exactly than 3.11, which moves the calibration errors
            # in the 16th digit; every count, interval and p-value is exact.
            assert recomputed[path] == pytest.approx(value, rel=0, abs=1e-12), path
        else:
            assert recomputed[path] == value, path
    produced = (tmp_path / "predictions.csv").read_bytes().replace(b"\r\n", b"\n")
    assert produced == (RESULTS / "predictions.csv").read_bytes()


def test_the_script_that_ran_is_a_version_in_the_history():
    log = subprocess.run(["git", "log", "--format=%H", "--", SCRIPT], cwd=ROOT, capture_output=True)
    commits = log.stdout.decode().split()
    if log.returncode != 0 or not commits:
        pytest.skip("no history of the script in this clone")
    blobs = {hashlib.sha256(subprocess.run(["git", "show", f"{commit}:{SCRIPT}"], cwd=ROOT,
                                           capture_output=True).stdout).hexdigest() for commit in commits}
    assert SUMMARY["script_sha256"] in blobs


def test_the_outcome_is_the_one_the_locked_rule_gives():
    decision = SUMMARY["decision"]
    assert bench.decide(SUMMARY["topics"], []) == decision
    assert (decision["only_jev"], decision["only_lexicon"], decision["mcnemar_p"]) == (11, 0, 0.001)
    assert decision["jev_better"] and decision["guard_holds"] and decision["rule_applies"]
    responsiveness = SUMMARY["topics"]["responsiveness"]
    assert (responsiveness["lexicon"]["true_positives"], responsiveness["jev"]["true_positives"],
            responsiveness["jev"]["false_positives"]) == (2, 13, 10)
    assert SUMMARY["requests"] == {**SUMMARY["requests"], "sent": 200, "unreadable_answers": 0, "retried": 0}
    assert SUMMARY["texts"]["answered"] == SUMMARY["texts"]["random"] == 200
