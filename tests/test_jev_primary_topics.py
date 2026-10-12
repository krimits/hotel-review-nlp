"""Offline guards for the pinned Jev extension; no real provider or model downloads."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_jev_primary_topics as cli


def probabilities(label):
    return {name: 0.8 if name == label else 0.05 for name in cli.task.LABELS}


def result(label="service", cost=0.00001):
    return {
        "model": cli.JEV_MODEL,
        "probabilities": probabilities(label),
        "cost_usd": cost,
        "cost_status": "reported_usd" if cost is not None else "provider_cost_missing",
        "usage": {"cost": cost, "input_tokens": 100, "output_tokens": 1},
    }


def dev_rows(n=100):
    return [
        {
            "review_id": f"topic-{i + 301:04d}",
            "review_sha256": f"hash-{i}",
            "review_text": f"Synthetic review {i}; helpful staff.",
        }
        for i in range(n)
    ]


def microsoft_rows(dev):
    return [
        {
            "review_id": r["review_id"],
            "review_sha256": r["review_sha256"],
            "arm": "decision",
            "status": "ok",
            "error": None,
            "probabilities": probabilities("service"),
            "milliseconds": 50.0,
            "cost_usd": 0.00002,
            "model": cli.adapters.DECISION_MODEL,
            "usage": {"cost": 0.00002},
            "cost_status": "reported_usd",
        }
        for r in dev
    ]


@pytest.fixture
def ready(tmp_path, monkeypatch):
    rows = dev_rows()
    gold = {r["review_id"]: {"label": "service"} for r in rows}
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "labels_rechecked.json").write_text("locked-test-fixture\n")
    annotation = {"sample_manifest_sha256": "sample-test-sha", "independence": "unverified"}
    monkeypatch.setattr(
        cli,
        "checked_inputs",
        lambda path: (
            rows,
            gold,
            microsoft_rows(rows),
            1.0,
            {"run_sha256": "historic-sha"},
            annotation,
        ),
    )
    return inputs, rows, gold


def test_jev_request_is_the_same_five_way_rubric_with_a_dated_build_and_no_metadata():
    calls = []

    def transport(url, body, key, timeout):
        calls.append(body)
        assert url == cli.adapters.DECISIONS_URL and key == "synthetic-key" and timeout == 30
        assert body == {
            "model": cli.JEV_MODEL,
            "state": {"review": "Helpful staff"},
            "questions": {"primary_topic": cli.task.QUESTION},
        }
        return {
            "model": cli.JEV_MODEL,
            "answers": {
                "primary_topic": {
                    "type": "choice",
                    "choice": "service",
                    "confidence": 0.2,
                    "probabilities": probabilities("service"),
                }
            },
            "usage": {"cost": 0.00001},
        }

    parsed = cli.jev_predict("Helpful staff", "synthetic-key", transport=transport)
    assert len(calls) == 1 and parsed["model"] == cli.JEV_MODEL
    assert parsed["probabilities"]["service"] == pytest.approx(0.8)
    assert parsed["provider_confidence"] == 0.2  # separate metadata, never the routing confidence


@pytest.mark.parametrize("mutation", ["alias", "model", "missing_class", "choice", "cost"])
def test_jev_response_rejects_alias_drift_and_invalid_distributions(mutation):
    body = {
        "model": cli.JEV_MODEL,
        "answers": {
            "primary_topic": {
                "type": "choice",
                "choice": "service",
                "probabilities": probabilities("service"),
            }
        },
        "usage": {"cost": 0.00001},
    }
    if mutation == "alias":
        body["model"] = "~typesafe/jev-latest"
    if mutation == "model":
        body["model"] = cli.adapters.DECISION_MODEL
    if mutation == "missing_class":
        del body["answers"]["primary_topic"]["probabilities"]["other"]
    if mutation == "choice":
        body["answers"]["primary_topic"]["choice"] = "value"
    if mutation == "cost":
        body["usage"]["cost"] = -1
    with pytest.raises(ValueError):
        cli.jev_predict("Synthetic review", "synthetic-key", transport=lambda *args: body)


@pytest.mark.parametrize("budget", [None, 0, -1, math.inf, math.nan, True])
def test_invalid_budget_fails_before_inputs_or_provider_work(tmp_path, monkeypatch, budget):
    monkeypatch.setattr(cli, "checked_inputs", lambda *args: pytest.fail("must fail before inputs"))
    with pytest.raises(ValueError, match="positive_usd_budget"):
        cli.benchmark(
            tmp_path / "unused",
            tmp_path / "out",
            allow_external=True,
            key="synthetic-key",
            max_cost_usd=budget,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("enabled,key", [(False, "synthetic-key"), (True, "")])
def test_no_provider_without_opt_in_and_key(tmp_path, enabled, key):
    with pytest.raises(ValueError, match="external_opt_in"):
        cli.benchmark(tmp_path, tmp_path / "out", allow_external=enabled, key=key, max_cost_usd=0.1)


def test_complete_jev_run_reuses_microsoft_and_preserves_inputs(ready, tmp_path, monkeypatch):
    inputs, rows, _ = ready
    calls = []
    original = (inputs / "labels_rechecked.json").read_bytes()
    monkeypatch.setattr(cli, "jev_predict", lambda text, key: (calls.append(text), result())[1])
    out = tmp_path / "complete"
    report = cli.benchmark(inputs, out, allow_external=True, key="synthetic-key", max_cost_usd=0.1)
    assert len(calls) == len(rows) == 100
    assert report["execution_complete"] and report["summary"]["jev"]["raw"]["accuracy"] == 1
    assert report["summary"]["decision"]["raw"]["accuracy"] == 1
    assert report["paired_comparison"]["delta_macro_f1_jev_minus_microsoft"] == 0
    assert report["paired_comparison"]["automatic_winner"] is None
    assert (
        not report["test_predictions_made"]
        and not report["promoted"]
        and not report["space_updated"]
    )
    assert report["summary"]["jev"]["cost_usd_per_1000_attempts"] == pytest.approx(0.01)
    assert (inputs / "labels_rechecked.json").read_bytes() == original
    freeze = json.loads((out / "freeze.json").read_text())
    assert "not a final four-arm test freeze" in freeze["scope"]
    for name, sha in freeze["development_files"].items():
        assert cli.task.digest(out / name) == sha
    saved = (out / "jev_predictions.jsonl").read_text()
    assert "Synthetic review" not in saved and "synthetic-key" not in saved


@pytest.mark.parametrize("failure", ["exception", "unknown_cost", "drift", "malformed"])
def test_first_failure_or_unknown_cost_stops_requests_and_partial_quality_is_null(
    ready, tmp_path, monkeypatch, capsys, failure
):
    inputs, _, _ = ready
    calls = []

    def predict(text, key):
        calls.append(text)
        if failure == "exception":
            raise RuntimeError("secret-canary; review text canary")
        value = result(cost=None if failure == "unknown_cost" else 0.001)
        if failure == "drift":
            value["model"] = "~typesafe/jev-latest"
        if failure == "malformed":
            value["probabilities"] = {"service": 1}
        return value

    monkeypatch.setattr(cli, "jev_predict", predict)
    out = tmp_path / failure
    report = cli.benchmark(inputs, out, allow_external=True, key="synthetic-key", max_cost_usd=0.1)
    assert len(calls) == 1
    summary = report["summary"]["jev"]
    assert summary["planned"] == 100 and summary["attempts"] == 1 and summary["not_executed"] == 99
    assert summary["raw"] is None and summary["calibrated"] is None
    assert not report["execution_complete"] and report["paired_comparison"] is None
    assert not (out / "freeze.json").exists()
    saved = "\n".join(p.read_text() for p in out.iterdir()) + capsys.readouterr().out
    assert "secret-canary" not in saved and "review text canary" not in saved
    queue = list(csv.DictReader((out / "human_queue.csv").open()))
    assert sum(row["arm"] == "jev" and row["reason"] == "not_executed" for row in queue) == 99
    assert summary["cost_usd_per_1000_attempts"] is None


def test_observed_budget_stops_at_the_boundary_and_keeps_every_planned_record(
    ready, tmp_path, monkeypatch
):
    inputs, _, _ = ready
    calls = []
    monkeypatch.setattr(
        cli, "jev_predict", lambda text, key: (calls.append(text), result(cost=0.001))[1]
    )
    report = cli.benchmark(
        inputs, tmp_path / "cap", allow_external=True, key="synthetic-key", max_cost_usd=0.002
    )
    assert len(calls) == 2
    s = report["summary"]["jev"]
    assert s["known_cost_usd_lower_bound"] == pytest.approx(0.002)
    assert s["cost_usd_per_1000_attempts"] == pytest.approx(1.0)
    assert s["not_executed"] == 98 and s["raw"] is None


def test_existing_output_is_not_reused_before_a_provider_call(ready, tmp_path, monkeypatch):
    inputs, _, _ = ready
    out = tmp_path / "existing"
    out.mkdir()
    monkeypatch.setattr(cli, "jev_predict", lambda *args: pytest.fail("provider called"))
    with pytest.raises(ValueError, match="new_capture_directory"):
        cli.benchmark(inputs, out, allow_external=True, key="synthetic-key", max_cost_usd=0.1)


def test_paired_comparison_aligns_by_id_not_row_order():
    dev = dev_rows()
    gold = {r["review_id"]: {"label": cli.task.LABELS[i % 5]} for i, r in enumerate(dev)}
    jev = [
        {
            **r,
            "arm": "jev",
            "status": "ok",
            "probabilities": probabilities(gold[r["review_id"]]["label"]),
        }
        for r in dev
    ]
    ms = [{**r, "probabilities": dict(r["probabilities"])} for r in jev]
    for r in ms[:2]:
        r["probabilities"] = probabilities("other")
    comparison = cli.paired_comparison(jev, list(reversed(ms)), gold)
    assert comparison["jev_only_correct"] == 2 and comparison["microsoft_only_correct"] == 0
    assert comparison["delta_macro_f1_jev_minus_microsoft"] > 0
    assert comparison["bootstrap"]["samples"] == 2000
    assert comparison["automatic_winner"] is None
    jev[0]["status"] = "error"
    assert cli.paired_comparison(jev, ms, gold) is None


def reference_bundle(folder: Path, dev, gold):
    folder.mkdir()
    records = microsoft_rows(dev)
    (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    cli.task.write_json(
        folder / "run.json",
        {
            "partition": "dev",
            "execution_complete": True,
            "sample_sha256": "sample",
            "labels_sha256": "labels",
            "configuration": {
                "question": cli.task.QUESTION,
                "decision_model": cli.adapters.DECISION_MODEL,
                "threshold": 0.7,
            },
            "runtime": {"code_commit": "fixture"},
            "temperatures": {"decision": 1.0},
            "summary": cli.original.summarize(records, gold, ["decision"], {"decision": 1.0}),
        },
    )
    cli.task.write_json(
        folder / "freeze.json",
        {
            "development_files": {
                name: cli.task.digest(folder / name) for name in ["run.json", "predictions.jsonl"]
            }
        },
    )
    cli.task.write_json(
        folder / "verification.json",
        {
            "archived_sha256": {
                name: cli.task.digest(folder / name)
                for name in ["run.json", "predictions.jsonl", "freeze.json"]
            }
        },
    )


@pytest.mark.parametrize(
    "mutation", [None, "capture", "gold_hash", "question", "identity", "scores"]
)
def test_microsoft_reference_must_be_intact_paired_and_recomputed(tmp_path, mutation):
    dev = dev_rows(5)
    gold = {r["review_id"]: {"label": "service"} for r in dev}
    folder = tmp_path / "reference"
    reference_bundle(folder, dev, gold)
    if mutation == "capture":
        (folder / "predictions.jsonl").write_text("changed\n")
    if mutation in {"question", "scores"}:
        report = json.loads((folder / "run.json").read_text())
        if mutation == "question":
            report["configuration"]["question"] = {"different": "rubric"}
        else:
            report["summary"]["decision"]["raw"]["accuracy"] = 0
        cli.task.write_json(folder / "run.json", report)
        freeze = json.loads((folder / "freeze.json").read_text())
        freeze["development_files"]["run.json"] = cli.task.digest(folder / "run.json")
        cli.task.write_json(folder / "freeze.json", freeze)
        verification = json.loads((folder / "verification.json").read_text())
        for name in ["run.json", "freeze.json"]:
            verification["archived_sha256"][name] = cli.task.digest(folder / name)
        cli.task.write_json(folder / "verification.json", verification)
    if mutation == "identity":
        dev[0]["review_sha256"] = "changed"
    if mutation is None:
        records, temperature, provenance = cli.checked_reference(
            dev, gold, "sample", "labels", reference=folder
        )
        assert len(records) == 5 and temperature == 1 and provenance["code_commit"] == "fixture"
    else:
        with pytest.raises(ValueError):
            cli.checked_reference(
                dev,
                gold,
                "sample",
                "changed" if mutation == "gold_hash" else "labels",
                reference=folder,
            )


def test_check_command_never_calls_a_provider(ready, monkeypatch, capsys):
    inputs, _, _ = ready
    monkeypatch.setattr(cli, "jev_predict", lambda *args: pytest.fail("provider called"))
    monkeypatch.setattr(cli.sys, "argv", ["benchmark", "check", "--inputs", str(inputs)])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["provider_requests_made"] == 0


def test_macro_f1_keeps_all_five_classes_when_gold_classes_are_absent():
    assert cli.macro_f1_from_indices(np.array([1, 1]), np.array([1, 1])) == pytest.approx(0.2)
