"""Meaningful offline checks; none substitutes for human labels or a provider inference run."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from reviewnlp.experiments import topic_benchmark as task
from reviewnlp.experiments import topic_models as models
from scripts import benchmark_decision_topics as cli

ROOT = Path(__file__).resolve().parents[1]


def distribution(label, confidence=0.8):
    return {key: confidence if key == label else (1 - confidence) / 4 for key in task.LABELS}


@pytest.fixture
def sample(tmp_path):
    folder = tmp_path / "sample"
    task.prepare_sample(
        cli.SOURCE,
        folder,
        protocol=cli.PROTOCOL,
        size=50,
        train_size=30,
        dev_size=10,
        second_size=10,
    )
    rows, _ = task.load_sample(folder, protocol=cli.PROTOCOL)
    return folder, rows


def complete_sheet(source, target, labels):
    rows = task.read_csv(source, task.SHEET_COLUMNS)
    for row in rows:
        row.update(
            label=labels[row["review_id"]],
            multiple_topics="yes",
            done="yes",
            note="test fixture, not ground truth",
        )
    task.write_sheet(target, rows)
    return target


@pytest.fixture
def human_fixture(sample, tmp_path):
    folder, rows = sample
    labels = {row["review_id"]: task.LABELS[index % 5] for index, row in enumerate(rows)}
    a = complete_sheet(folder / "sheet_A.csv", tmp_path / "a.csv", labels)
    b = complete_sheet(folder / "sheet_B.csv", tmp_path / "b.csv", labels)
    final = complete_sheet(folder / "labels_final.csv", tmp_path / "final.csv", labels)
    out = tmp_path / "labels.json"
    record = task.finish_labels(folder, a, b, final, out, protocol=cli.PROTOCOL)
    return folder, rows, out, record, a, b, final


def test_real_corpus_sampling_is_deterministic_blind_and_300_100_100(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    a = task.prepare_sample(cli.SOURCE, first, protocol=cli.PROTOCOL)
    b = task.prepare_sample(cli.SOURCE, second, protocol=cli.PROTOCOL)
    assert a == b
    assert a["sample_size"] == 500 and a["population_size"] == 658
    assert a["split_counts"] == {"train": 300, "dev": 100, "test": 100}
    rows, _ = task.load_sample(first, protocol=cli.PROTOCOL)
    assert len({r["review_sha256"] for r in rows}) == 500
    assert len(task.read_csv(first / "sheet_B.csv", task.SHEET_COLUMNS)) == 100
    sheet = task.read_csv(first / "sheet_A.csv", task.SHEET_COLUMNS)
    assert set(sheet[0]) == set(task.SHEET_COLUMNS)
    assert all(not r["label"] and not r["done"] for r in sheet)
    with pytest.raises(FileExistsError):
        task.prepare_sample(cli.SOURCE, first, protocol=cli.PROTOCOL)


def test_duplicate_texts_are_removed_before_sampling(tmp_path):
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Hotel", "Date", "Review Title", "Review"])
        writer.writeheader()
        for text in ("Staff helpful", " STAFF  helpful ", "Great place", "Good price"):
            writer.writerow({"Hotel": "X", "Date": "2009", "Review Title": "title", "Review": text})
    report = task.prepare_sample(
        source,
        tmp_path / "dedup",
        protocol=cli.PROTOCOL,
        size=3,
        train_size=1,
        dev_size=1,
        second_size=1,
    )
    assert report["population_size"] == 3


def test_modified_blank_original_cannot_be_used(sample):
    folder, _ = sample
    with (folder / "sheet_A.csv").open("a") as handle:
        handle.write("changed")
    with pytest.raises(ValueError, match="sample_artifact_changed"):
        task.load_sample(folder, protocol=cli.PROTOCOL)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "unknown",
        "changed_text",
        "changed_hash",
        "blank",
        "label",
        "done",
        "multi",
    ],
)
def test_labels_reject_bad_sheets(human_fixture, tmp_path, mutation):
    _, expected, _, _, a, _, _ = human_fixture
    rows = task.read_csv(a, task.SHEET_COLUMNS)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "unknown":
        rows[0]["review_id"] = "alien"
    elif mutation == "changed_text":
        rows[0]["review_text"] += " changed"
    elif mutation == "changed_hash":
        rows[0]["review_sha256"] = "wrong"
    elif mutation == "blank":
        rows[0]["label"] = ""
    elif mutation == "label":
        rows[0]["label"] = "positive"
    elif mutation == "done":
        rows[0]["done"] = "no"
    elif mutation == "multi":
        rows[0]["multiple_topics"] = "maybe"
    bad = tmp_path / "bad.csv"
    task.write_sheet(bad, rows)
    with pytest.raises(ValueError):
        task.checked_labels(bad, expected)


def test_agreement_measured_before_final_judgments_and_independence_not_inferred(
    human_fixture, tmp_path
):
    folder, _, _, record, a, b, final = human_fixture
    assert record["kappa"] == 1 and record["independence"] == "unverified"
    rows = task.read_csv(b, task.SHEET_COLUMNS)
    rows[0]["label"] = next(label for label in task.LABELS if label != rows[0]["label"])
    task.write_sheet(b, rows)
    report = task.finish_labels(
        folder,
        a,
        b,
        final,
        tmp_path / "adjudicated.json",
        protocol=cli.PROTOCOL,
        independent_reviewers=True,
    )
    assert report["agreements"] == 9 and report["kappa"] < 1
    assert report["disagreement_ids"] == [rows[0]["review_id"]]
    assert report["independence"] == "owner_attested"
    assert "review_text" not in json.dumps(report)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {"positive": 1},
        {k: 0.3 for k in task.LABELS},
        {k: float("nan") for k in task.LABELS},
        {k: True for k in task.LABELS},
        {k: "0.2" for k in task.LABELS},
        {**distribution("service"), "service": -1},
    ],
)
def test_probabilities_require_full_valid_distribution(bad):
    with pytest.raises(ValueError):
        task.probability_vector(bad)


def test_known_ece_brier_and_selective_risk_including_confidence_one():
    p = [distribution("cleanliness", 0.8), distribution("service", 0.8), distribution("value", 1.0)]
    m = task.classification_metrics(
        [task.probability_vector(v) for v in p], ["cleanliness", "other", "value"]
    )
    # Two .8 predictions have accuracy .5; one certain correct prediction has zero calibration gap.
    assert m["ece"] == pytest.approx(0.2)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["selective_error_rate"] == pytest.approx(1 / 3)
    assert m["retained"] == 3
    assert sum(b["n"] for b in m["reliability_bins"]) == 3
    assert m["reliability_bins"][-1]["n"] == 1
    assert m["brier_multiclass"] == pytest.approx((0.05 + 1.55) / 3)
    empty = task.classification_metrics([[0.2] * 5] * 2, ["service", "value"])
    assert empty["retained"] == 0 and empty["selective_error_rate"] is None
    assert empty["human_review"] == 2


def test_calibration_changes_confidence_preserves_argmax_and_reduces_dev_log_loss():
    p = np.array([task.probability_vector(distribution("cleanliness", 0.98))] * 10)
    y = ["cleanliness"] * 6 + ["service", "location", "value", "other"]
    t = task.fit_temperature(p, y)
    calibrated = task.temperature_probabilities(p, t)
    assert t > 1
    assert np.all(calibrated.argmax(axis=1) == p.argmax(axis=1))
    assert (
        task.classification_metrics(calibrated, y)["log_loss"]
        < task.classification_metrics(p, y)["log_loss"]
    )


def test_logistic_uses_only_train_text_and_labels(human_fixture):
    _, rows, _, record, *_ = human_fixture
    rows = copy.deepcopy(rows)
    for row in rows:
        if row["partition"] != "train":
            row["review_text"] += " unseenuniquetoken "
    model = task.fit_logistic(rows, record["labels"])
    assert "unseenuniquetoken" not in model[0].vocabulary_
    changed = copy.deepcopy(record["labels"])
    for row in rows:
        if row["partition"] != "train":
            changed[row["review_id"]]["label"] = "other"
    other = task.fit_logistic(rows, changed)
    assert np.array_equal(model[1].coef_, other[1].coef_)
    assert task.probability_vector(
        task.logistic_predict(model, "Helpful staff")["probabilities"]
    ).sum() == pytest.approx(1)


def response():
    return {
        "model": models.DECISION_MODEL,
        "provider": "Azure",
        "id": "synthetic-contract-response",
        "answers": {
            "primary_topic": {
                "type": "choice",
                "choice": "service",
                "probabilities": distribution("service", 0.78),
                "confidence": 0.67,
            }
        },
        "usage": {"input_tokens": 476, "output_tokens": 0, "cost": 0.000019992},
    }


def test_decisions_contract_uses_new_endpoint_and_only_review_data():
    def transport(url, body, key, timeout):
        assert url == "https://openrouter.ai/api/alpha/decisions"
        assert body == {
            "model": models.DECISION_MODEL,
            "state": {"review": "Helpful staff"},
            "questions": {"primary_topic": task.QUESTION},
        }
        assert key == "fake-key" and timeout == 30
        return response()

    result = models.decision_predict("Helpful staff", "fake-key", transport=transport)
    assert result["cost_status"] == "reported_usd"
    assert max(result["probabilities"].values()) == pytest.approx(0.78)
    assert result["provider_confidence"] == 0.67


@pytest.mark.parametrize(
    "mutation", ["model", "type", "choice", "missing_probabilities", "bad_sum", "cost"]
)
def test_decisions_response_rejects_revision_drift_or_malformed_answers(mutation):
    body = response()
    answer = body["answers"]["primary_topic"]
    if mutation == "model":
        body["model"] = "microsoft/microsoft-decision-1"
    elif mutation == "type":
        answer["type"] = "noul"
    elif mutation == "choice":
        answer["choice"] = "other"
    elif mutation == "missing_probabilities":
        answer.pop("probabilities")
    elif mutation == "bad_sum":
        answer["probabilities"]["value"] = 0.5
    elif mutation == "cost":
        body["usage"]["cost"] = -1
    with pytest.raises(ValueError):
        models.parse_decision(body)


def test_missing_cost_remains_unknown_not_zero():
    body = response()
    body["usage"].pop("cost")
    parsed = models.parse_decision(body)
    assert parsed["cost_usd"] is None and parsed["cost_status"] == "provider_cost_missing"


def test_failure_and_budget_stop_keep_planned_denominator(tmp_path):
    rows = [
        {"review_id": f"r{i}", "review_sha256": "x", "review_text": "helpful"} for i in range(3)
    ]
    calls = []

    def decision(text):
        calls.append(text)
        return {"probabilities": distribution("service"), "cost_usd": 0.01}

    records = cli.capture(rows, {"decision": decision}, tmp_path, max_cost_usd=0.01)
    assert len(calls) == 1 and len(records) == 3
    gold = {row["review_id"]: {"label": "service"} for row in rows}
    summary = cli.summarize(records, gold, ["decision"], {})["decision"]
    assert summary["planned"] == 3 and summary["not_executed"] == 2
    assert summary["raw"] is None and summary["calibrated"] is None
    assert summary["human_review_due_to_failure"] == 2
    assert summary["cost_usd_per_1000_attempts"] == 10


def test_error_with_unknown_cost_stops_external_calls_and_does_not_leak_exception(tmp_path):
    rows = [
        {"review_id": f"r{i}", "review_sha256": "x", "review_text": "helpful"} for i in range(2)
    ]
    calls = []

    def failed(text):
        calls.append(text)
        raise RuntimeError("SECRET should never be written")

    records = cli.capture(rows, {"decision": failed}, tmp_path, max_cost_usd=1)
    assert len(calls) == 1
    assert [r["status"] for r in records] == ["error", "not_executed"]
    assert "SECRET" not in (tmp_path / "predictions.jsonl").read_text()
    summary = cli.summarize(
        records, {r["review_id"]: {"label": "service"} for r in rows}, ["decision"], {}
    )["decision"]
    assert summary["cost_usd_per_1000_attempts"] is None


def test_paid_work_is_blocked_before_predictor_loading(human_fixture, tmp_path, monkeypatch):
    folder, _, labels, *_ = human_fixture

    def forbidden(*args, **kwargs):
        pytest.fail("must not load or call a model")

    monkeypatch.setattr(task, "fit_logistic", forbidden)
    with pytest.raises(ValueError, match="opt_in"):
        cli.benchmark(
            folder, labels, tmp_path / "blocked", partition="dev", arms=["decision"], device="cpu"
        )


def test_real_logistic_dev_freeze_then_test_never_fits_calibration_on_test(
    human_fixture, tmp_path, monkeypatch
):
    folder, _, labels, *_ = human_fixture
    dev = tmp_path / "dev"
    report = cli.benchmark(folder, labels, dev, partition="dev", arms=["logistic"], device="cpu")
    assert report["execution_complete"] and (dev / "freeze.json").exists()

    def forbidden(*args, **kwargs):
        pytest.fail("temperature must not be fit on test")

    monkeypatch.setattr(task, "fit_temperature", forbidden)
    result = cli.benchmark(
        folder,
        labels,
        tmp_path / "test",
        partition="test",
        arms=["logistic"],
        device="cpu",
        freeze_path=dev / "freeze.json",
    )
    assert result["summary"]["logistic"]["raw"]["n"] == 10
    assert result["summary"]["logistic"]["calibrated"]["n"] == 10
    assert result["summary"]["logistic"]["cost_usd_per_1000_attempts"] is None
    assert result["temperatures"] == report["temperatures"]
    with pytest.raises(ValueError, match="new_capture"):
        cli.benchmark(
            folder,
            labels,
            tmp_path / "test",
            partition="test",
            arms=["logistic"],
            device="cpu",
            freeze_path=dev / "freeze.json",
        )


def test_changed_development_capture_blocks_test(human_fixture, tmp_path):
    folder, _, labels, *_ = human_fixture
    dev = tmp_path / "dev"
    cli.benchmark(folder, labels, dev, partition="dev", arms=["logistic"], device="cpu")
    with (dev / "predictions.jsonl").open("a") as handle:
        handle.write("changed")
    with pytest.raises(ValueError, match="development_capture_changed"):
        cli.benchmark(
            folder,
            labels,
            tmp_path / "test",
            partition="test",
            arms=["logistic"],
            device="cpu",
            freeze_path=dev / "freeze.json",
        )


def test_three_arms_share_ids_and_queue_uses_max_probability_not_provider_confidence(
    human_fixture, tmp_path, monkeypatch
):
    folder, _, labels, *_ = human_fixture
    body = response()
    calls = []

    def decision(text, key):
        calls.append(text)
        return models.parse_decision(body)

    class FakeModel:
        device = "cpu"
        dtype = "float32"

    class Scorer:
        model = FakeModel()

        def predict(self, text):
            return {"probabilities": distribution("service"), "cost_usd": None}

    monkeypatch.setattr(models, "decision_predict", decision)
    monkeypatch.setattr(models.QwenTopicScorer, "load", lambda **kwargs: Scorer())
    out = tmp_path / "three_arms"
    report = cli.benchmark(
        folder,
        labels,
        out,
        partition="dev",
        arms=list(cli.ARMS),
        device="cpu",
        allow_external=True,
        key="fake-key",
        max_cost_usd=1,
    )
    assert report["execution_complete"] and len(calls) == 10
    records = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    ids_by_arm = [{r["review_id"] for r in records if r["arm"] == arm} for arm in cli.ARMS]
    assert ids_by_arm[0] == ids_by_arm[1] == ids_by_arm[2]
    assert len(records) == 30
    assert report["summary"]["decision"]["raw"]["retained"] == 10  # .78, not provider .67
    assert report["summary"]["decision"]["cost_usd_per_1000_attempts"] == pytest.approx(0.019992)
    queue = task.read_csv(out / "human_queue.csv", ["review_id", "arm", "reason"])
    decision_referrals = [r for r in queue if r["arm"] == "decision"]
    assert len(decision_referrals) == report["summary"]["decision"]["calibrated"]["human_review"]


def test_dependency_drift_blocks_frozen_test_before_any_provider_call(human_fixture, tmp_path):
    folder, _, labels, *_ = human_fixture
    dev = tmp_path / "version_dev"
    cli.benchmark(folder, labels, dev, partition="dev", arms=["logistic"], device="cpu")
    freeze = json.loads((dev / "freeze.json").read_bytes())
    freeze["runtime_signature"]["scikit-learn"] = "incorrect-version"
    task.write_json(dev / "freeze.json", freeze)
    with pytest.raises(ValueError, match="frozen_dependency_versions_changed"):
        cli.benchmark(
            folder,
            labels,
            tmp_path / "version_test",
            partition="test",
            arms=["logistic"],
            device="cpu",
            freeze_path=dev / "freeze.json",
        )


def test_real_tiny_qwen_forward_returns_five_option_distribution_without_generation():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tok = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<pad>": 1,
                **{letter: i + 2 for i, letter in enumerate(models.OPTION_LETTERS)},
            },
            unk_token="<unk>",
        )
    )
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=7,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
    )
    scorer = models.QwenTopicScorer(
        model, tokenizer, revision="tiny-test", model_id="tiny-random-test"
    )
    with torch.inference_mode():
        result = scorer.predict("Friendly staff, dirty room.")
    p = task.probability_vector(result["probabilities"])
    assert p.sum() == pytest.approx(1)
    assert 0 < result["full_vocabulary_option_mass"] <= 1
    assert result["revision"] == "tiny-test" and result["input_tokens"] > 5
