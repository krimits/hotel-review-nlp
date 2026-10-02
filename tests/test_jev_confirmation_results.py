"""The confirmation run's committed results: no review text, the locked inputs, and every number reproduced.

The run was made on 2 October 2026 (docs/experiments/jev_topic_benchmark/confirmation/results/README.md).
The texts are not committed, so these checks use only the saved answers, the committed labels and key,
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
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs" / "experiments" / "jev_topic_benchmark"
CONFIRMATION = FOLDER / "confirmation"
RESULTS = CONFIRMATION / "results"
SCRIPT = "scripts/benchmark_jev_topics.py"
PROTOCOL = ROOT / "docs" / "annotation" / "confirmation_protocol.md"

# SHA-256 of the three files as the script wrote them on the project owner's Windows machine, with CRLF
# line ends. The repository stores them with LF (.gitattributes), so the check restores CRLF first.
RECEIVED = {
    "summary.json": "685cdc65ab4d4d5114bb6b8bcbf7665724fa0d47acac9388d35cb993d992b99d",
    "predictions.csv": "b733094916673fdfb971c0f011d8e852a1fd6215ab7796cb2d2eef0c0da8a7a4",
    "responses.jsonl": "067d2d0d863689a4c62708d9f2399c98b5a57a9a5577a9761ecc5aa0d30412b2",
}

_spec = importlib.util.spec_from_file_location("benchmark_jev_topics", ROOT / SCRIPT)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)

TOPICS = bench.STAGES["confirmation"]["topics"]
SUMMARY = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
KEY = bench.read_key(CONFIRMATION / "key.csv", TOPICS)
GOLD = bench.read_gold(CONFIRMATION / "labels.csv", TOPICS)
ITEMS = bench.random_items(KEY, GOLD, None)


def _records() -> list[dict]:
    return [json.loads(line) for line in (RESULTS / "responses.jsonl").read_text(encoding="utf-8").splitlines()]


def _flatten(value, path: str = "") -> dict:
    if isinstance(value, dict):
        return {key: leaf for name, item in value.items() for key, leaf in _flatten(item, f"{path}/{name}").items()}
    if isinstance(value, list):
        return {key: leaf for i, item in enumerate(value) for key, leaf in _flatten(item, f"{path}/{i}").items()}
    return {path: value}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


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
    assert len(records) == 400
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


def test_the_answers_are_to_the_locked_questions_for_exactly_the_400_texts():
    records = _records()
    assert ITEMS == list(range(1, 401))
    assert [record["item"] for record in records] == ITEMS  # sent in item order, none twice, none missing
    assert {record["questions_sha256"] for record in records} == {bench.QUESTIONS_SHA256}
    assert {record["response"]["model"] for record in records} == {"typesafe/jev-1.13-20260917"}
    assert {record["attempts"] for record in records} == {1}
    assert f"`{bench.QUESTIONS_SHA256}`" in PROTOCOL.read_text(encoding="utf-8")


def test_the_inputs_recorded_are_the_committed_files():
    inputs = SUMMARY["inputs"]
    record = json.loads((CONFIRMATION / "sheet_record.json").read_text(encoding="utf-8"))
    assert inputs["texts_sha256"] == record["sheet_sha256"]
    assert inputs["texts_are_the_handed_in_sheet_a"] is True
    assert inputs["key_sha256"] == bench.sha256_file(CONFIRMATION / "key.csv")
    assert inputs["gold_sha256"] == bench.sha256_file(CONFIRMATION / "labels.csv")
    assert inputs["questions_sha256"] == bench.QUESTIONS_SHA256
    assert (inputs["route"], inputs["endpoint"], inputs["model_requested"]) == (
        "openrouter", "https://openrouter.ai/api/v1/systemone", "jev-latest")
    assert (inputs["stage"], inputs["topics_scored"]) == ("confirmation", ["responsiveness"])
    assert SUMMARY["design"] == "docs/annotation/confirmation_protocol.md"
    manifest = json.loads((CONFIRMATION / "sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_sha256"] == bench.sha256_file(PROTOCOL) == record["protocol_sha256"]


def test_the_labels_were_committed_before_the_first_request():
    # The protocol: the labels, the key and the manifest are committed before Jev is run.
    if _git("rev-parse", "--is-shallow-repository").stdout.strip() != b"false":
        pytest.skip("a shallow clone has no history to date the labels from")
    log = _git("log", "--diff-filter=A", "--format=%cI", "--", str(CONFIRMATION.relative_to(ROOT) / "labels.csv"))
    dates = log.stdout.decode().split()
    if log.returncode != 0 or not dates:
        pytest.skip("no history of the labels in this clone")
    committed = datetime.fromisoformat(dates[-1])
    first_request = datetime.fromisoformat(SUMMARY["requests"]["first_utc"])
    assert committed < first_request


def test_the_summary_is_reproduced_from_the_saved_answers(tmp_path):
    (tmp_path / "responses.jsonl").write_bytes((RESULTS / "responses.jsonl").read_bytes())
    args = argparse.Namespace(price_per_million_input_tokens=None, price_per_million_output_tokens=0.0,
                              currency="USD")
    bench.write_results(ITEMS, 400, KEY, GOLD, tmp_path, SUMMARY["inputs"], args, TOPICS, "confirmation")
    again = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    committed, recomputed = _flatten(SUMMARY), _flatten(again)
    assert committed.keys() == recomputed.keys()
    for path, value in committed.items():
        if path == "/script_sha256":
            continue  # which script ran is checked below
        if isinstance(value, float):
            # Python 3.12 sums floats more exactly than 3.11, which can move the calibration error
            # in the 16th digit; every count, interval and p-value is exact.
            assert recomputed[path] == pytest.approx(value, rel=0, abs=1e-12), path
        else:
            assert recomputed[path] == value, path
    produced = (tmp_path / "predictions.csv").read_bytes().replace(b"\r\n", b"\n")
    assert produced == (RESULTS / "predictions.csv").read_bytes()


def test_the_script_that_ran_is_the_one_locked_with_the_protocol():
    manifest = json.loads((CONFIRMATION / "sample_manifest.json").read_text(encoding="utf-8"))
    shown = _git("show", f"{manifest['code_commit']}:{SCRIPT}")
    if shown.returncode != 0:
        pytest.skip("the commit that locked the protocol is not in this clone")
    assert hashlib.sha256(shown.stdout).hexdigest() == SUMMARY["script_sha256"]


def test_the_outcome_is_the_one_the_locked_rule_gives():
    decision = SUMMARY["decision"]
    assert bench.decide(SUMMARY["topics"], [], bench.STAGES["confirmation"]["outcomes"]) == decision
    assert (decision["only_jev"], decision["only_lexicon"]) == (17, 0)
    # The summary rounds p-values to four decimals, so it shows 0.0; the exact value is 2 / 2**17.
    assert decision["mcnemar_p"] == 0.0 and bench.mcnemar_p(17, 0) == pytest.approx(2 / 2**17)
    assert decision["jev_better"] and decision["guard_holds"] and decision["rule_applies"]
    assert decision["outcome"].startswith("confirmed:")
    responsiveness = SUMMARY["topics"]["responsiveness"]
    assert (responsiveness["lexicon"]["true_positives"], responsiveness["jev"]["true_positives"],
            responsiveness["jev"]["false_positives"]) == (2, 19, 7)
    assert (responsiveness["texts"], responsiveness["gold_unsure_left_out"]) == (395, 5)
    assert SUMMARY["requests"] == {**SUMMARY["requests"], "sent": 400, "unreadable_answers": 0, "retried": 0}
    assert SUMMARY["texts"]["answered"] == SUMMARY["texts"]["random"] == SUMMARY["texts"]["selected"] == 400


def test_the_readme_states_the_counts_the_summary_holds():
    readme = " ".join((RESULTS / "README.md").read_text(encoding="utf-8").split())  # line breaks do not matter
    responsiveness = SUMMARY["topics"]["responsiveness"]
    positives = responsiveness["jev"]["recall"]["denominator"]
    flags = responsiveness["jev"]["precision"]["denominator"]
    assert f"Jev found {responsiveness['jev']['true_positives']} of the {positives}" in readme
    assert f"the lexicon {responsiveness['lexicon']['true_positives']}" in readme
    assert f"{responsiveness['jev']['true_positives']} of the {flags} texts Jev flagged" in readme
    assert f"{SUMMARY['decision']['only_jev']} found only by Jev" in readme
    assert SUMMARY["decision"]["outcome"] in readme
