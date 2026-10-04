"""Offline experiment guards: no model download or external API."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reviewnlp.triage.generator_experiment import (
    CANDIDATES,
    DEFAULT_CANDIDATES,
    EXAMPLES,
    RATINGS,
    RecordingTransport,
    builder,
    check_disjoint,
    cost_summary,
    digest_bytes,
    export_annotation,
    freeze_selection,
    json_diagnostics,
    prompt_fingerprint,
    read_dataset,
    run_candidate,
    structural_summary,
    validate_upstream,
)
from reviewnlp.triage.qwen_generator import (
    ActionSignals,
    GenerationResult,
    QwenActionGenerator,
    build_messages,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/experiments/triage_generator"
RUN = ROOT / "docs/experiments/triage_smoke/20261003T071410Z_18020ced"
TEXT = "The shelf was dusty."
ACTION = {"problem": "Dust on shelf", "excerpt": TEXT, "measure": "Clean the shelf.",
          "department": "housekeeping", "to_confirm": []}


def upstream(rows):
    return {"dataset_sha256": "dataset", "config": {"fixed": True}, "records": [
        {"id": row["id"], "sentiment": {"label": "positive", "confidence": 0.9,
                                      "probabilities": {"negative": 0.1, "positive": 0.9}},
         "complaints": {"status": "ok", "model": "jev-fixed", "topics": [],
                        "other_complaint": {"topic": "other", "answer": "no"}},
         "routing": {"qwen_triggered": False}} for row in rows]}


class Generator:
    def __init__(self, raw, budget=False):
        self.raw, self.budget, self.calls = raw, budget, []

    def generate(self, review, signals):
        self.calls.append((review, signals))
        return GenerationResult(self.raw, self.budget, "fake")


class ExperimentTests(unittest.TestCase):
    def test_new_cases_are_disjoint_from_old_cases_and_prompt_examples(self):
        dev, held = read_dataset(DATA / "dev.json", "dev"), read_dataset(DATA / "holdout.json", "holdout")
        smoke = json.loads((RUN.parent / "reviews.json").read_text())
        check_disjoint(dev, held, smoke, extra_texts=[text for text, _ in EXAMPLES])
        self.assertEqual((len(dev["reviews"]), len(held["reviews"])), (24, 24))
        with self.assertRaises(ValueError):
            check_disjoint(dev, dev)
        with self.assertRaises(ValueError):
            read_dataset(DATA / "holdout.json", "dev")

    def test_baseline_is_exact_and_ablation_changes_only_sentiment_metadata(self):
        signals = ActionSignals("negative", 0.93, ["bathroom"])
        self.assertEqual(builder("A")(TEXT, signals), build_messages(TEXT, signals))
        baseline, ablated = builder("B")(TEXT, signals), builder("D")(TEXT, signals)
        self.assertEqual(baseline[:-1], ablated[:-1])
        self.assertEqual(baseline[-1]["content"].split("\nOverall sentiment:")[0], ablated[-1]["content"])
        self.assertIn("negative", baseline[-1]["content"])
        self.assertNotIn("negative", ablated[-1]["content"])
        self.assertEqual(builder("B")(TEXT, signals), builder("C")(TEXT, signals))
        for text, example in EXAMPLES:
            self.assertTrue(text)
            self.assertIn("actions", example)

    def test_loader_receives_pinned_revision_without_changing_production_defaults(self):
        with patch("reviewnlp.triage.qwen_generator._load_qwen", return_value=("tokenizer", "weights")) as load:
            generator = QwenActionGenerator("fake/model", "cuda", revision="a" * 40,
                                            message_builder=builder("B"), prompt_version="actions-v2")
            self.assertEqual(generator._models(), ("tokenizer", "weights"))
            load.assert_called_once_with("fake/model", "cuda", "a" * 40)
            self.assertEqual(generator.prompt_version, "actions-v2")
        self.assertEqual(QwenActionGenerator().prompt_version, "actions-v1")

    def test_v3_changes_only_system_rules_at_the_same_capacity_and_examples(self):
        signals = ActionSignals("negative", 0.93, ["bathroom"])
        old, new = builder("C")(TEXT, signals), builder("E")(TEXT, signals)
        self.assertEqual(old[1:], new[1:])
        self.assertNotEqual(old[0], new[0])
        self.assertEqual(CANDIDATES["C"]["model"], CANDIDATES["E"]["model"])
        self.assertEqual(DEFAULT_CANDIDATES, ("A", "B", "C", "D"))
        self.assertEqual(prompt_fingerprint("C"), "00479231af814c9ded4f8d934d908952e9e04c2ada04eea1b4375096fce3f3d5")
        rules = new[0]["content"]
        for required in ("REAL_PENDING", "REAL_RESOLVED", "HYPOTHETICAL", "POSITIVE_COMMENT",
                         "hints, not evidence", "hotel can take", "preserve negation", "no reasoning"):
            self.assertIn(required, rules)
        self.assertNotIn("<thinking>", rules)
        self.assertNotIn('"proposals"', rules)

    def test_full_json_diagnostic_does_not_turn_salvage_into_compliance(self):
        value = json.dumps({"actions": [ACTION]})
        self.assertTrue(json_diagnostics(value)["full_json_valid"])
        for invalid in ('{"actions":[],"actions":[]}', "```json\n" + value + "\n```",
                        value + " trailing text", '{"actions":[],"score":NaN}'):
            with self.subTest(raw=invalid):
                self.assertFalse(json_diagnostics(invalid)["full_json_valid"])
        rows = [{"id": "one", "text": TEXT}]
        records = run_candidate("E", rows, upstream(rows), Generator("```json\n" + value + "\n```"))
        self.assertTrue(records[0]["json_valid"])
        self.assertTrue(records[0]["actions"])
        self.assertFalse(records[0]["full_json_valid"])
        self.assertEqual(json_diagnostics('{"actions":[],"actions":[]}')["duplicate_json_keys"], ["actions"])

    def test_cost_absence_partial_cost_and_retries_are_not_free(self):
        responses = iter([(429, {}, b'{"usage":{}}'), (200, {}, b'{"usage":{"cost":0.02,"total_tokens":8}}')])
        recorder = RecordingTransport(lambda *args: next(responses))
        recorder("url", {"private": TEXT}, "SECRET", 1)
        recorder("url", {}, "SECRET", 1)
        result = cost_summary(recorder.attempts)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["reported_cost_sum"], 0.02)
        self.assertFalse(result["complete_cost_available"])
        self.assertNotIn("SECRET", json.dumps(recorder.attempts))
        self.assertNotIn(TEXT, json.dumps(recorder.attempts))
        self.assertIsNone(cost_summary([{"cost_reported": None}])["reported_cost_sum"])

    def test_transport_exception_is_counted(self):
        def failed(*args):
            raise TimeoutError

        recorder = RecordingTransport(failed)
        with self.assertRaises(TimeoutError):
            recorder("url", {}, "SECRET", 1)
        self.assertEqual(len(recorder.attempts), 1)
        self.assertIsNone(recorder.attempts[0]["cost_reported"])

    def test_cache_rejects_other_ids_config_probabilities_or_jev_model_changes(self):
        rows = [{"id": "one", "text": TEXT}, {"id": "two", "text": TEXT + " Again."}]
        cache = upstream(rows)
        validate_upstream(cache, rows, "dataset", {"fixed": True})
        for mutation in ("id", "config", "probability", "model"):
            changed = json.loads(json.dumps(cache))
            if mutation == "id":
                changed["records"][0]["id"] = "other"
            elif mutation == "config":
                changed["config"] = {}
            elif mutation == "probability":
                changed["records"][0]["sentiment"]["probabilities"]["positive"] = 0.5
            else:
                changed["records"][1]["complaints"]["model"] = "new-jev"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_upstream(changed, rows, "dataset", {"fixed": True})

    def test_variants_receive_identical_upstream_without_author_labels(self):
        rows = [{"id": "one", "text": TEXT, "author_scenario": "THIS IS NOT GOLD"}]
        cache, generators, records = upstream(rows), [], []
        for name in CANDIDATES:
            generator = Generator(json.dumps({"actions": [ACTION]}))
            records.extend(run_candidate(name, rows, cache, generator))
            generators.append(generator)
        self.assertEqual(len({record["upstream_sha256"] for record in records}), 1)
        self.assertTrue(all(generator.calls == generators[0].calls for generator in generators))
        self.assertTrue(all(not record["production_routed"] for record in records))
        self.assertNotIn("THIS IS NOT GOLD", repr(generators[0].calls))
        self.assertIsNone(structural_summary(records)["human_quality"])

    def test_budget_exhaustion_does_not_count_as_accepted_complete_answer(self):
        rows = [{"id": "one", "text": TEXT}]
        records = run_candidate("A", rows, upstream(rows), Generator(json.dumps({"actions": [ACTION]}), True))
        self.assertTrue(records[0]["actions"])
        counts = structural_summary(records)
        self.assertEqual(counts["token_budget_hits"], 1)
        self.assertEqual(counts["accepted_actions"], 0)

    def test_unavailable_model_stops_instead_of_retrying_each_review(self):
        class Failed:
            def generate(self, *args):
                raise RuntimeError("failed")

        rows = [{"id": "one", "text": TEXT}, {"id": "two", "text": TEXT + " Again."}]
        records = run_candidate("A", rows, upstream(rows), Failed())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["error"], "RuntimeError")

    def test_freeze_requires_complete_human_judgments_and_unchanged_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": "one", "text": TEXT}]
            records = run_candidate("B", rows, upstream(rows), Generator(json.dumps({"actions": [ACTION]})))
            (root / "results.jsonl").write_text(json.dumps(records[0]) + "\n")
            export_annotation(root, rows, records)
            info = {"split": "dev", "execution_complete": True, "candidates": {"B": CANDIDATES["B"]},
                    "jev_model_resolved": "jev-fixed",
                    "upstream_config": {"fixed": True}, "files": {
                        name: digest_bytes((root / name).read_bytes()) for name in
                        ("results.jsonl", "annotation_key.json", "human_review.csv")}}
            (root / "run.json").write_text(json.dumps(info))
            path = root / "human_review.csv"
            with self.assertRaises(ValueError):
                freeze_selection(root, path, "B")
            with path.open(newline="") as handle:
                sheet = list(csv.DictReader(handle))
            sheet[0].update(dict.fromkeys(RATINGS, "1"))
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sheet[0]))
                writer.writeheader()
                writer.writerows(sheet)
            frozen = freeze_selection(root, path, "B")
            self.assertEqual(frozen["human_counts"]["B"]["useful_grounded_measures"], 1)
            self.assertEqual(frozen["jev_model_resolved"], "jev-fixed")
            (root / "results.jsonl").write_text("changed")
            with self.assertRaises(ValueError):
                freeze_selection(root, path, "B")

    def test_original_smoke_archive_is_preserved_and_zero_measures_is_explicit(self):
        manifest = json.loads((RUN / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            self.assertEqual(digest_bytes((RUN / name).read_bytes()), expected)
        records = [json.loads(line) for line in (RUN / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 28)
        self.assertFalse(any(row["result"]["actions"]["actions"] for row in records))

    def test_runner_exports_all_variants_and_preserves_a_warmup_failure(self):
        spec = importlib.util.spec_from_file_location("comparison", ROOT / "scripts/compare_triage_generators.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        torch = SimpleNamespace(__version__="fake", cuda=SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: None,
            synchronize=lambda: None, get_device_name=lambda index: "fake GPU"))
        transformers = SimpleNamespace(__version__="fake")

        class FakeQwen(Generator):
            failed_model = None

            def __init__(self, model, **kwargs):
                super().__init__(json.dumps({"actions": [ACTION]}))
                self.model_name, self._bundle = model, ("tokenizer", "weights")
                if kwargs["prompt_version"].startswith("actions-v4"):
                    payload = {"actions": [{"issue_id": 1, "measure": ACTION["measure"], "to_confirm": []}]} \
                        if kwargs["prompt_version"].endswith(":measures") else {"issues": [{
                            "problem": ACTION["problem"], "excerpt": ACTION["excerpt"],
                            "department": ACTION["department"], "status": "REAL_PENDING"}]}
                    self.raw = json.dumps(payload)

            def generate(self, review, signals):
                if self.model_name == self.failed_model:
                    raise RuntimeError("load failed")
                return super().generate(review, signals)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {"split": "dev", "provenance": "synthetic-authored", "reviews": [{"id": "one", "text": TEXT}]}
            (root / "dev.json").write_text(json.dumps(data))
            cache = upstream(data["reviews"])
            cache["api_cost"] = cost_summary([])
            configs = {name: {**value, "revision": "a" * 40} for name, value in CANDIDATES.items()}
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                    patch.object(script, "DATA", root), patch.object(script, "QwenActionGenerator", FakeQwen), \
                    patch.object(script, "prepare_upstream", return_value=cache), \
                    patch.object(script, "pinned_configs", return_value=configs), \
                    patch.object(script.subprocess, "check_output", return_value="a" * 40):
                success, failure = root / "success", root / "failure"
                success.mkdir()
                self.assertEqual(script.compare(success, "dev"), 0)
                info = json.loads((success / "run.json").read_text())
                self.assertTrue(info["execution_complete"])
                self.assertFalse(info["quality_evaluated"])
                self.assertEqual(set(info["summary"]), set(CANDIDATES))
                FakeQwen.failed_model = CANDIDATES["C"]["model"]
                failure.mkdir()
                self.assertEqual(script.compare(failure, "dev"), 1)
                info = json.loads((failure / "run.json").read_text())
                self.assertFalse(info["execution_complete"])
                self.assertEqual(info["summary"]["C"]["generation_errors"], 1)
                self.assertTrue((failure / "human_review.csv").exists())
                with self.assertRaises(ValueError):
                    script.compare(root, "holdout")

    def test_reference_reuse_verifies_hashes_config_and_old_weights_without_hub(self):
        spec = importlib.util.spec_from_file_location("reference_comparison", ROOT / "scripts/compare_triage_generators.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        rows = [{"id": "one", "text": TEXT}]
        cache = upstream(rows)
        cache["config"] = script.upstream_config()
        cache["api_cost"] = cost_summary([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_bytes = json.dumps(cache).encode()
            (root / "upstream.json").write_bytes(cache_bytes)
            info = {"split": "dev", "execution_complete": True, "dataset_sha256": "dataset", "reviews": 1,
                    "max_new_tokens": 400, "do_sample": False, "upstream_config": script.upstream_config(),
                    "jev_model_resolved": "jev-fixed", "files": {"upstream.json": digest_bytes(cache_bytes)},
                    "candidates": {"C": {**CANDIDATES["C"], "revision": "a" * 40,
                                          "prompt_sha256": prompt_fingerprint("C")}}}
            (root / "run.json").write_text(json.dumps(info))
            reference = script.read_reference_run(root, rows, "dataset")
            configs = script.pinned_configs(None, ("C", "E"), reference)
            self.assertEqual(set(configs), {"C", "E"})
            self.assertEqual(configs["C"]["revision"], configs["E"]["revision"])
            self.assertEqual(configs["E"]["revision"], "a" * 40)
            for field in ("split", "execution_complete", "dataset_sha256", "max_new_tokens", "do_sample", "jev_model_resolved"):
                changed = dict(info)
                changed[field] = None
                (root / "run.json").write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    script.read_reference_run(root, rows, "dataset")
            (root / "run.json").write_text(json.dumps(info))
            changed_info = json.loads(json.dumps(info))
            changed_info["candidates"]["C"]["prompt_sha256"] = "bad"
            with self.assertRaises(ValueError):
                script.pinned_configs(None, ("C", "E"), {**reference, "info": changed_info})
            (root / "upstream.json").write_bytes(cache_bytes + b"\n")
            with self.assertRaises(ValueError):
                script.read_reference_run(root, rows, "dataset")

    def test_cached_comparison_runs_only_C_and_E_with_one_weight_load_and_no_upstream_calls(self):
        spec = importlib.util.spec_from_file_location("cached_comparison", ROOT / "scripts/compare_triage_generators.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        torch = SimpleNamespace(__version__="fake", cuda=SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: None,
            synchronize=lambda: None, get_device_name=lambda index: "fake GPU"))

        class FakeQwen(Generator):
            loads = 0
            revisions = []

            def __init__(self, model, **kwargs):
                super().__init__(json.dumps({"actions": [ACTION]}))
                self._bundle = None
                self.revisions.append(kwargs["revision"])

            def generate(self, review, signals):
                if self._bundle is None:
                    type(self).loads += 1
                    self._bundle = ("tokenizer", "weights")
                return super().generate(review, signals)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": "one", "text": TEXT}]
            dataset_bytes = json.dumps({"split": "dev", "provenance": "synthetic-authored", "reviews": rows}).encode()
            (root / "dev.json").write_bytes(dataset_bytes)
            reference_dir = root / "reference"
            reference_dir.mkdir()
            cache = upstream(rows)
            cache.update(config=script.upstream_config(), dataset_sha256=digest_bytes(dataset_bytes),
                         api_cost=cost_summary([{"cost_reported": 0.02}]))
            cache_bytes = json.dumps(cache).encode()
            (reference_dir / "upstream.json").write_bytes(cache_bytes)
            info = {"split": "dev", "execution_complete": True, "dataset_sha256": digest_bytes(dataset_bytes),
                    "reviews": 1, "max_new_tokens": 400, "do_sample": False,
                    "upstream_config": script.upstream_config(), "jev_model_resolved": "jev-fixed",
                    "files": {"upstream.json": digest_bytes(cache_bytes)}, "candidates": {
                        "C": {**CANDIDATES["C"], "revision": "a" * 40, "prompt_sha256": prompt_fingerprint("C")}}}
            (reference_dir / "run.json").write_text(json.dumps(info))
            output = root / "result"
            output.mkdir()
            with patch.dict(sys.modules, {"torch": torch, "transformers": SimpleNamespace(__version__="fake")}), \
                    patch.object(script, "DATA", root), patch.object(script, "QwenActionGenerator", FakeQwen), \
                    patch.object(script, "prepare_upstream", side_effect=AssertionError("no new upstream calls")), \
                    patch.object(script.subprocess, "check_output", return_value="b" * 40):
                self.assertEqual(script.compare(output, "dev", candidates=("C", "E"), reference_run=reference_dir), 0)
                with self.assertRaises(ValueError):
                    script.compare(output, "holdout", {"candidate": "C"}, reference_run=reference_dir)
            result = json.loads((output / "run.json").read_text())
            self.assertEqual(set(result["candidates"]), {"C", "E"})
            self.assertEqual(FakeQwen.revisions, ["a" * 40, "a" * 40])
            self.assertEqual(FakeQwen.loads, 1)
            self.assertEqual(result["api_cost"]["attempts"], 0)
            self.assertEqual(result["reference_run"]["historical_api_cost"]["reported_cost_sum"], 0.02)
            self.assertEqual((output / "upstream.json").read_bytes(), cache_bytes)
            self.assertFalse(result["quality_evaluated"])


if __name__ == "__main__":
    unittest.main()
