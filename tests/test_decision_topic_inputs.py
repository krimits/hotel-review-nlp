"""Offline safety and reproducibility checks for the human-label Colab handoff."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from reviewnlp.experiments import topic_benchmark as task
from scripts import unpack_decision_topic_inputs as handoff

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(tmp_path):
    sample = tmp_path / "sample"
    task.prepare_sample(
        ROOT / "data/raw/hotel_reviews.csv",
        sample,
        protocol=handoff.PROTOCOL,
        size=15,
        train_size=5,
        dev_size=5,
        second_size=5,
    )
    rows, _ = task.load_sample(sample, protocol=handoff.PROTOCOL)
    lookup = {row["review_id"]: task.LABELS[i % 5] for i, row in enumerate(rows)}
    files = {"sample/" + p.name: p.read_bytes() for p in sample.iterdir()}
    for original, completed in (
        ("sheet_A.csv", "completed_A.csv"),
        ("sheet_B.csv", "completed_B.csv"),
        ("labels_final.csv", "completed_final.csv"),
    ):
        judgments = task.read_csv(sample / original, task.SHEET_COLUMNS)
        for row in judgments:
            row.update(label=lookup[row["review_id"]], multiple_topics="yes", done="yes")
        path = tmp_path / completed
        task.write_sheet(path, judgments)
        files[completed] = path.read_bytes()
    labels = tmp_path / "labels.json"
    task.finish_labels(
        sample,
        tmp_path / "completed_A.csv",
        tmp_path / "completed_B.csv",
        tmp_path / "completed_final.csv",
        labels,
        protocol=handoff.PROTOCOL,
    )
    files["labels.json"] = labels.read_bytes()
    files["source_record.json"] = b'{"provenance":"offline fake labels; not quality evidence"}\n'
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    manifest = {
        "version": "decision-topics-inputs-v1",
        "files": expected,
        "labels_sha256": task.digest(labels),
        "sample_manifest_sha256": task.digest(sample / "sample_manifest.json"),
    }
    files["input_manifest.json"] = json.dumps(manifest).encode()
    lock = {**manifest, "input_files_sha256": expected, "sample_size": 15}
    lock_path = tmp_path / "lock.json"
    task.write_json(lock_path, lock)
    return files, lock_path


def zip_files(path, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def test_locked_labels_are_revalidated_without_provider_or_model_work(bundle, tmp_path):
    files, lock = bundle
    package = tmp_path / "inputs.zip"
    zip_files(package, files)
    report = handoff.unpack_inputs(package, tmp_path / "checked", lock_path=lock)
    assert report["sample_size"] == 15 and report["all_labels_valid"]
    assert report["independence"] == "unverified"
    assert (
        task.digest(tmp_path / "checked/labels_rechecked.json")
        == json.loads(lock.read_text())["labels_sha256"]
    )
    with pytest.raises(ValueError, match="new_input_directory"):
        handoff.unpack_inputs(package, tmp_path / "checked", lock_path=lock)


@pytest.mark.parametrize(
    "mutation", ["content", "manifest", "extra", "missing", "traversal", "duplicate"]
)
def test_bad_archives_never_extract_or_load_models(bundle, tmp_path, mutation):
    files, lock = bundle
    if mutation == "content":
        files["completed_final.csv"] += b"changed"
    elif mutation == "manifest":
        manifest = json.loads(files["input_manifest.json"])
        manifest["labels_sha256"] = "forged"
        files["input_manifest.json"] = json.dumps(manifest).encode()
    elif mutation == "extra":
        files["unexpected.txt"] = b"no"
    elif mutation == "missing":
        files.pop("labels.json")
    elif mutation == "traversal":
        files["../escape.txt"] = b"no"
    package = tmp_path / "bad.zip"
    zip_files(package, files)
    if mutation == "duplicate":
        with zipfile.ZipFile(package, "a") as archive:
            archive.writestr("labels.json", files["labels.json"])
    with pytest.raises(ValueError):
        handoff.unpack_inputs(package, tmp_path / "unsafe", lock_path=lock)
    assert not (tmp_path / "unsafe").exists()


def test_oversize_inputs_are_refused_before_extraction(bundle, tmp_path, monkeypatch):
    files, lock = bundle
    package = tmp_path / "large.zip"
    zip_files(package, files)
    monkeypatch.setattr(handoff, "MAX_UNCOMPRESSED_BYTES", 10)
    with pytest.raises(ValueError, match="too_large"):
        handoff.unpack_inputs(package, tmp_path / "large", lock_path=lock)


def test_real_archived_logistic_dev_reports_majority_collapse_and_no_test_predictions():
    folder = ROOT / "docs/experiments/decision_topic_benchmark/results/logistic_dev_20261012"
    report = json.loads((folder / "run.json").read_bytes())
    lock = json.loads(handoff.LOCK.read_bytes())
    assert report["partition"] == "dev" and report["execution_complete"]
    assert report["arms"] == ["logistic"] and report["labels_sha256"] == lock["labels_sha256"]
    assert report["sample_sha256"] == lock["sample_manifest_sha256"]
    for name, sha in report["files"].items():
        assert task.digest(folder / name) == sha
    records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert len(records) == 100 and len({r["review_id"] for r in records}) == 100
    assert all(
        r["status"] == "ok" and max(r["probabilities"], key=r["probabilities"].get) == "other"
        for r in records
    )
    raw = report["summary"]["logistic"]["raw"]
    assert raw["accuracy"] == raw["support"]["other"] / raw["n"]
    assert raw["macro_f1"] == pytest.approx((2 * raw["accuracy"] / (1 + raw["accuracy"])) / 5)
    assert report["summary"]["logistic"]["cost_usd_per_1000_attempts"] is None
    freeze = json.loads((folder / "freeze.json").read_bytes())
    for name, sha in freeze["development_files"].items():
        assert task.digest(folder / name) == sha
    assert lock["actual_test_predictions_made"] is False
    assert set(lock) & {"labels", "review_text"} == set()
    assert not (folder / "labels.json").exists()
