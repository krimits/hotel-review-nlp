"""The real experiment runner wires both calls to one pinned set of weights and no upstream requests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reviewnlp.triage.evidence_generator import ISSUE_SYSTEM as EVIDENCE_SYSTEM
from reviewnlp.triage.evidence_generator import MEASURES_SYSTEM
from reviewnlp.triage.generator_experiment import CANDIDATES, digest_bytes, prompt_fingerprint
from reviewnlp.triage.qwen_generator import GenerationResult
from reviewnlp.triage.staged_generator import ISSUE_SYSTEM, MEASURE_SYSTEM

ROOT = Path(__file__).resolve().parents[1]
TEXT = "The shelf was dusty."


@pytest.mark.parametrize("candidates", [("C", "F"), ("F", "G")])
def test_cached_comparison_uses_one_model_load_and_exports_both_real_stages(tmp_path, candidates):
    spec = importlib.util.spec_from_file_location("staged_comparison", ROOT / "scripts/compare_triage_generators.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    dataset = {"split": "dev", "provenance": "synthetic-authored", "reviews": [{"id": "one", "text": TEXT}]}
    (tmp_path / "dev.json").write_text(json.dumps(dataset))
    dataset_sha = digest_bytes((tmp_path / "dev.json").read_bytes())
    reference = tmp_path / "reference"
    reference.mkdir()
    cache = {"dataset_sha256": dataset_sha, "config": script.upstream_config(), "api_cost": {"attempts": 1},
             "records": [{"id": "one", "sentiment": {"label": "negative", "confidence": 0.9,
                                                     "probabilities": {"negative": 0.9, "positive": 0.1}},
                          "complaints": {"status": "ok", "model": "jev-fixed", "topics": [],
                                         "other_complaint": {"topic": "other", "answer": "no"}},
                          "routing": {"qwen_triggered": True}}]}
    (reference / "upstream.json").write_text(json.dumps(cache))
    info = {"split": "dev", "execution_complete": True, "reviews": 1, "dataset_sha256": dataset_sha,
            "max_new_tokens": 400, "do_sample": False, "upstream_config": script.upstream_config(),
            "jev_model_resolved": "jev-fixed", "files": {
                "upstream.json": digest_bytes((reference / "upstream.json").read_bytes())}, "candidates": {
                    "C": {**CANDIDATES["C"], "revision": "a" * 40, "prompt_sha256": prompt_fingerprint("C")}}}
    (reference / "run.json").write_text(json.dumps(info))
    output = tmp_path / "output"
    output.mkdir()

    class FakeQwen:
        loads, calls, revisions = 0, [], []

        def __init__(self, model, **kwargs):
            self.model_name, self._bundle, self.build = model, None, kwargs["message_builder"]
            self.revisions.append(kwargs["revision"])

        def generate(self, review, signals):
            if self._bundle is None:
                type(self).loads += 1
                self._bundle = ("tokenizer", "weights")
            system = self.build(review, signals)[0]["content"]
            self.calls.append(system)
            action = {"problem": "Dusty shelf", "excerpt": TEXT, "measure": "Clean the shelf.",
                      "department": "housekeeping", "to_confirm": []}
            if system == ISSUE_SYSTEM:
                value = {"issues": [{"problem": action["problem"], "excerpt": TEXT,
                                     "status": "REAL_PENDING", "department": "housekeeping"}]} \
                    if review == TEXT else {"issues": []}
            elif system == EVIDENCE_SYSTEM:
                value = {"issues": [{"problem": action["problem"], "excerpt": TEXT,
                                     "status": "REAL_PENDING", "category": "cleanliness",
                                     "evidence": {"reported": TEXT, "hypothetical": None, "resolved": None}}]} \
                    if review == TEXT else {"issues": []}
            elif system in {MEASURE_SYSTEM, MEASURES_SYSTEM}:
                value = {"actions": [{"issue_id": 1, "measure": action["measure"], "to_confirm": []}]}
            else:
                value = {"actions": [action]}
            return GenerationResult(json.dumps(value), False, self.model_name)

    torch = SimpleNamespace(__version__="fake", cuda=SimpleNamespace(is_available=lambda: True,
        empty_cache=lambda: None, synchronize=lambda: None, get_device_name=lambda index: "fake GPU"))
    with patch.dict(sys.modules, {"torch": torch, "transformers": SimpleNamespace(__version__="fake")}), \
            patch.object(script, "DATA", tmp_path), patch.object(script, "QwenActionGenerator", FakeQwen), \
            patch.object(script, "prepare_upstream", side_effect=AssertionError("no new upstream calls")), \
            patch.object(script.subprocess, "check_output", return_value="b" * 40):
        assert script.compare(output, "dev", candidates=candidates, reference_run=reference) == 0
    result = json.loads((output / "run.json").read_text())
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert FakeQwen.loads == 1 and FakeQwen.revisions == ["a" * 40] * (3 if candidates[0] == "C" else 4)
    assert result["summary"]["F"]["issue_calls"] == result["summary"]["F"]["measure_calls"] == 1
    if candidates[0] == "C":
        assert result["generation_budget"]["C"]["max_calls_per_review"] == 1
    else:
        assert result["generation_budget"]["F"] == result["generation_budget"]["G"]
        assert (output / "coverage_review.csv").is_file()
        assert result["files"]["coverage_review.csv"] == digest_bytes((output / "coverage_review.csv").read_bytes())
        assert records[1]["extracted_issues"][0]["evidence"]["reported"] == TEXT
    assert result["generation_budget"]["F"]["max_calls_per_review"] == 2
    assert result["api_cost"]["attempts"] == 0
    assert result["annotation_shuffle_seed"] != 20261003
    assert len(records[1]["stages"]) == 2 and records[1]["actions"][0]["department"] == "housekeeping"
