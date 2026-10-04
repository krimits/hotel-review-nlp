"""Run the real Gradio app and service with offline stage fakes; no model or Jev downloads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from triage_fakes import POSITIVE, FakeGenerator, FakeWrapper

from reviewnlp.triage.demo_service import DISTILBERT_REVISION, DemoService, PinnedSentiment
from reviewnlp.triage.evidence_generator import EvidenceFirstGenerator
from reviewnlp.triage.jev_client import JevConfig
from reviewnlp.triage.space_runtime import SpaceRuntime, WorkerGenerator, run_worker

ROOT = Path(__file__).resolve().parents[1]
TEXT = "The cupboard was dusty."


def service(status="REAL_PENDING", raw=None):
    issue = {"problem": "Dusty cupboard", "excerpt": TEXT, "category": "cleanliness", "status": status,
             "evidence": {"reported": TEXT, "hypothetical": None, "resolved": None}}
    extractor = FakeGenerator(json.dumps({"issues": [issue]}))
    extractor._bundle = ("tokenizer", "weights")
    actioner = FakeGenerator(raw if raw is not None else json.dumps({"actions": [
        {"issue_id": 1, "measure": "Clean the cupboard.", "to_confirm": []}]}))
    actioner._bundle = None
    generator = EvidenceFirstGenerator(extractor, lambda pending: actioner)
    return DemoService(FakeWrapper(POSITIVE), generator, jev_config=JevConfig()), extractor


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_service_retains_positive_review_complaint_and_clears_model_diagnostics_after_response():
    demo, extractor = service()
    result = demo.analyze(TEXT)
    assert len(extractor.calls) == 1 and result["routing"]["qwen_triggered"]
    assert result["issue_assessments"][0]["status"] == "REAL_PENDING"
    assert result["actions"]["actions"][0]["department"] == "housekeeping"
    assert result["api_cost"]["attempts"] == 0 and not result["stored"]
    assert demo.generator.issues == demo.generator.last_stages == demo.generator.review_reasons == []


def test_jev_is_optional_and_cannot_be_requested_without_configuration():
    demo, _ = service()
    with pytest.raises(ValueError, match="jev_not_configured"):
        demo.analyze(TEXT, True)
    for text in ("", "x", "a" * 4001, " " * 4000 + TEXT):
        with pytest.raises(ValueError, match="one_review"):
            demo.analyze(text)


def test_failed_jev_retains_measures_and_reports_partial_with_unknown_cost():
    demo, _ = service()
    secret = "FAKE-SPACE-SECRET"
    demo.jev_config = JevConfig(enabled=True, route="openrouter", api_key=secret, max_attempts=1)
    demo.transport = lambda *args: (503, {}, b'{"error":"unavailable"}')
    result = demo.analyze(TEXT, True)
    assert result["status"] == "partial" and result["complaints"]["status"] == "error"
    assert result["actions"]["actions"]
    assert result["api_cost"]["attempts"] == 1 and result["api_cost"]["reported_cost_sum"] is None
    assert secret not in json.dumps(result)


@pytest.mark.parametrize("stage,error,expected", [
    ("issues", '{"issues":[', "invalid_json"),
    ("issues", '{"issues":[{"extra":"SECRET"}]}', "invalid_evidence_issue_schema"),
    ("measures", "SECRET broken JSON", "invalid_json"),
])
def test_failures_name_the_stage_without_raw_output_or_exception_text(stage, error, expected):
    demo, extractor = service(raw=error if stage == "measures" else None)
    if stage == "issues":
        extractor.raw = error
    result = demo.analyze(TEXT)
    assert result["stage_failure"] == {"stage": stage, "status": "error", "error": expected}
    assert result["status"] == "partial"
    assert "SECRET" not in json.dumps(result) and "raw" not in json.dumps(result)
    assert demo.generator.last_stages == [] and demo.generator.workflow_error is None


def test_worker_returns_issue_assessments_even_when_its_local_state_is_cleared():
    original, _ = service()
    def worker(review, signals):
        return json.loads(json.dumps(run_worker(original.generator, review, signals)))
    adapter = WorkerGenerator(worker)
    demo = DemoService(FakeWrapper(POSITIVE), adapter)
    result = demo.analyze(TEXT)
    assert original.generator.issues == [] and adapter.issues == []
    assert result["issue_assessments"][0]["department"] == "housekeeping"
    assert result["actions"]["actions"][0]["measure"] == "Clean the cupboard."
    assert {item["stage"] for item in result["stage_reports"]} == {"sentiment", "jev", "issues", "measures"}
    assert not result["stage_failure"]


def test_gpu_timeout_becomes_a_visible_partial_response_without_error_details():
    def fail(*args):
        raise RuntimeError("SECRET provider exception")

    demo = DemoService(FakeWrapper(POSITIVE), WorkerGenerator(fail))
    result = demo.analyze(TEXT)
    assert result["stage_failure"]["stage"] == "qwen_runtime"
    assert result["stage_failure"]["error"] == "gpu_unavailable_or_timeout"
    assert result["status"] == "partial" and "SECRET" not in json.dumps(result)


def test_worker_generation_exception_names_the_failing_substage_and_retains_issues():
    original, _ = service()
    original.generator.action_factory = lambda pending: FakeGenerator(error=RuntimeError("SECRET"))
    demo = DemoService(FakeWrapper(POSITIVE), WorkerGenerator(
        lambda review, signals: run_worker(original.generator, review, signals)))
    result = demo.analyze(TEXT)
    assert result["stage_failure"] == {"stage": "measures", "status": "error", "error": "generation_failed"}
    assert result["issue_assessments"] and "SECRET" not in json.dumps(result)


def test_startup_loading_failure_is_recorded_once_without_model_error_text():
    demo, extractor = service()
    wrapper = SimpleNamespace(_models=lambda: None)
    loads = []

    def load():
        loads.append(1)
        raise RuntimeError("SECRET model path")

    extractor._models = load
    runtime = SpaceRuntime(lambda *args: None, wrapper=wrapper, generator=demo.generator)
    runtime.preload()
    runtime.preload()
    assert len(loads) == 1
    assert runtime.information()["startup_failure"] == {"stage": "qwen_load", "status": "error", "error": "model_load_failed"}
    assert not runtime.information()["ready"]
    assert "SECRET" not in json.dumps(runtime.information())


def test_real_gradio_build_has_endpoints_and_displays_uncertainty_empty_and_failure_separately():
    app = load_script("triage_space_app", ROOT / "spaces/hotel-triage-demo/app.py")
    assert {"analyze", "model_info"} <= {item.get("api_name") for item in app.demo.config["dependencies"]}
    for status, raw, phrase in (("UNCERTAIN", None, "Αβέβαιο"),
                               ("REAL_PENDING", '{"actions":[]}', "Δεν προέκυψε αποδεκτό μέτρο"),
                               ("REAL_PENDING", "broken JSON", "δεν ολοκληρώθηκε έγκυρα")):
        demo, _ = service(status, raw)
        with patch.object(app, "service", return_value=demo):
            summary, issues, actions, result = app.analyze(TEXT, False)
        assert phrase in summary or any(phrase in str(row) for row in issues)
        assert result["validation_status"] == "unvalidated"
        if status == "UNCERTAIN":
            assert issues and not actions
            assert "ανθρώπινος έλεγχος" in summary
    app.demo.close()


def test_pinned_sentiment_loader_checks_model_labels_and_pins_both_downloads():
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "negative", 1: "positive"}))
    model.to = lambda device: model
    model.eval = lambda: model

    def tokenizer_load(*args, **kwargs):
        calls.append(kwargs)
        return "tokenizer"

    def model_load(*args, **kwargs):
        calls.append(kwargs)
        return model

    transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=model_load))
    with patch.dict(sys.modules, {"torch": SimpleNamespace(float32="fp32"), "transformers": transformers}):
        assert PinnedSentiment()._models()[2] == {0: "negative", 1: "positive"}
        assert all(item["revision"] == DISTILBERT_REVISION for item in calls)
        model.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
        with pytest.raises(ValueError, match="label_contract"):
            PinnedSentiment()._models()


def test_space_package_contains_only_allowed_sources_and_imports_without_the_original_repository(tmp_path):
    script = load_script("prepare_triage_space", ROOT / "scripts/prepare_triage_space.py")
    package = tmp_path / "package"
    manifest = script.build_package(ROOT, package, source_commit="a" * 40)
    assert manifest["hub_writes"] is False
    assert manifest["runtime_snapshot"]["candidate"] == "G"
    assert manifest["selection_status"] == "experimental_not_selected_not_promoted"
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected
    assert not any("data/" in name or "runs/" in name or "token" in name for name in manifest["files"])
    # A subprocess is an independent import check, without src/ on its module search path.
    import os
    import subprocess

    process = subprocess.run([sys.executable, "-c", "import app, json; print(json.dumps([x.get('api_name') for x in app.demo.config['dependencies']]))"],
        cwd=package, env={name: value for name, value in os.environ.items() if name != "PYTHONPATH"},
        capture_output=True, text=True, check=True)
    assert {"analyze", "model_info"} <= set(json.loads(process.stdout))
    with pytest.raises(FileExistsError):
        script.build_package(ROOT, package, source_commit="a" * 40)


def test_real_tiny_distilbert_prediction_uses_pinned_loader_contract(tmp_path, monkeypatch):
    # CI runs this with real HF APIs and random tiny weights; it is a wiring check, not task accuracy.
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\ncupboard\nwas\ndusty\n.\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(tmp_path)
    model = transformers.DistilBertForSequenceClassification(transformers.DistilBertConfig(
        vocab_size=10, n_layers=1, n_heads=2, dim=16, hidden_dim=32, num_labels=2,
        id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1}))
    model.save_pretrained(tmp_path)
    token_load = transformers.AutoTokenizer.from_pretrained
    model_load = transformers.AutoModelForSequenceClassification.from_pretrained

    def load_local(loader, **kwargs):
        assert kwargs["revision"] == DISTILBERT_REVISION
        return loader(tmp_path, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda repo, **kwargs: load_local(token_load, **kwargs))
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", lambda repo, **kwargs: load_local(model_load, **kwargs))
    wrapper = PinnedSentiment()
    (probabilities,) = wrapper.distribution_batch([TEXT])
    assert set(probabilities) == {"negative", "positive"}
    assert sum(probabilities.values()) == pytest.approx(1)
    assert next(wrapper._bundle[1].parameters()).dtype == torch.float32
