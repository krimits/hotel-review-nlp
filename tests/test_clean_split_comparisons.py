"""The scripts that will apply the BiLSTM (v3) and DistilBERT-seed protocols, run offline.

Synthetic runs with known accuracies stand in for the Colab artifacts, so the rules, the
alignment checks and the measured frontier are exercised before any real number exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reviewnlp.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bilstm_compare = load("compare_bilstm_clean")
seeds_compare = load("compare_seeds")
frontier_bench = load("benchmark_frontier")
phase2 = bilstm_compare.phase2

ROWS = 2000
LABELS = np.random.default_rng(0).integers(0, 2, ROWS)


def predictions(error_rate: float, seed: int) -> np.ndarray:
    flips = np.random.default_rng(seed).random(ROWS) < error_rate
    return np.where(flips, 1 - LABELS, LABELS)


def fingerprint(view: str) -> dict:
    return {"rows": ROWS, "negative": int((LABELS == 0).sum()), "positive": int((LABELS == 1).sum()),
            "sha256": f"fake-{view}"}


def split_views() -> dict:
    nb = predictions(0.20, 99)
    views = {}
    for view in ("random", "time", "hotel"):
        views[view] = {"splits": {"test": fingerprint(view)},
                       "test_predictions": ["p" if p else "n" for p in nb],
                       "test": {"macro_f1": binary_metrics(LABELS, nb, label_values=(0, 1))["macro_f1"]}}
    return views


def write_run(directory: Path, view: str, predicted: np.ndarray, seed: int, labels=LABELS) -> None:
    directory.mkdir(parents=True)
    logits = np.stack([1.0 - predicted, predicted.astype(float)], axis=1)
    np.save(directory / "test_logits.npy", logits)
    np.save(directory / "test_labels.npy", labels)
    (directory / "metrics.json").write_text(json.dumps({"data": {"test": fingerprint(view)},
                                                        "settings": {"seed": seed}}), encoding="utf-8")


def bundles(tmp_path: Path, bilstm_errors=(0.10, 0.11, 0.12)) -> tuple[Path, Path]:
    bilstm, distilbert = tmp_path / "bilstm_v2", tmp_path / "distilbert_v2"
    for view in ("random", "time", "hotel"):
        write_run(bilstm / view / "bilstm", view, predictions(bilstm_errors[0], 1), 42)
    for extra, error in zip((43, 44), bilstm_errors[1:], strict=True):
        write_run(bilstm / "time" / f"bilstm_seed{extra}", "time", predictions(error, extra), extra)
    write_run(distilbert / "time" / "distilbert", "time", predictions(0.04, 7), 42)
    write_run(distilbert / "time" / "distilbert_lora_scratch", "time", predictions(0.08, 8), 42)
    return bilstm, distilbert


def section_conditions(protocol: Path, start: str, end: str, cells: int) -> list[str]:
    text = protocol.read_text(encoding="utf-8").split(start, 1)[1].split(end, 1)[0]
    rows = [line.strip().strip("|").split("|") for line in text.splitlines() if line.strip().startswith("| `")]
    return [row[0].strip().strip("`") for row in rows if len(row) == cells]


def test_the_scripts_use_the_protocols_conditions_verbatim():
    bilstm_protocol = ROOT / "docs/experiments/bilstm_clean_splits/PROTOCOL.md"
    assert [c for c, _ in bilstm_compare.RULE] == section_conditions(
        bilstm_protocol, "## 4. Decision rule", "## 5. Limits", 2)
    assert [bilstm_compare.DOMINANCE] == section_conditions(
        bilstm_protocol, "### Secondary: speed, size", "## 4. Decision rule", 1)
    assert [c for c, _ in seeds_compare.RULE] == section_conditions(
        ROOT / "docs/experiments/distilbert_seed_robustness/PROTOCOL.md", "## 4. Decision rule", "## 5. Limits", 2)


def latency_report(medians: dict[str, float]) -> dict:
    return {"test": fingerprint("time"), "hardware": {"cpu": "fake"},
            "models": {name: {"size_mb": 1.0, "cpu": {"batch_1": {"ms_per_batch_median": ms}}}
                       for name, ms in medians.items()}}


def test_the_bilstm_comparison_applies_the_rule_the_seed_check_and_the_frontier(tmp_path):
    bilstm, distilbert = bundles(tmp_path)
    latency = latency_report({"tfidf_nb": 1.2, "bilstm": 4.0, "distilbert": 57.0, "distilbert_lora_scratch": 57.0})
    result = bilstm_compare.compare(bilstm, split_views(), latency, distilbert, resamples=300)
    primary = result["primary"]
    low, high = primary["paired_ci95"]
    assert primary["branch"] == bilstm_compare.branch(low, primary["d_bilstm_minus_distilbert"], high)
    assert primary["branch"] == "DistilBERT meaningfully better"  # 90% against 96% correct
    assert result["seed_check"]["same_branch"] and len(result["seeds"]) == 3
    # Naive Bayes is fastest, the BiLSTM sits between, DistilBERT is most accurate;
    # the LoRA is as slow as DistilBERT and less accurate, so it is dominated.
    assert result["frontier"]["frontier"] == ["tfidf_nb", "bilstm", "distilbert"]
    assert result["selection_frozen"] is False and result["promoted"] is False


def test_a_seed_in_another_branch_is_reported_as_seed_dependent(tmp_path):
    bilstm, distilbert = bundles(tmp_path, bilstm_errors=(0.10, 0.04, 0.10))
    result = bilstm_compare.compare(bilstm, split_views(), None, distilbert, resamples=300)
    assert not result["seed_check"]["same_branch"]
    assert result["seed_check"]["statement"] == "the conclusion depends on the BiLSTM's seed"
    assert "frontier" not in result


def test_runs_that_are_misaligned_or_mislabelled_are_refused(tmp_path):
    bilstm, distilbert = bundles(tmp_path)
    np.save(bilstm / "time" / "bilstm_seed43" / "test_labels.npy", LABELS[::-1])
    with pytest.raises(phase2.ComparisonError, match="test labels differ"):
        bilstm_compare.compare(bilstm, split_views(), None, distilbert, resamples=50)
    bilstm, distilbert = bundles(tmp_path / "other")
    metrics = bilstm / "time" / "bilstm_seed44" / "metrics.json"
    metrics.write_text(json.dumps({"data": {"test": fingerprint("time")}, "settings": {"seed": 42}}), encoding="utf-8")
    with pytest.raises(phase2.ComparisonError, match="seed 42, not 44"):
        bilstm_compare.compare(bilstm, split_views(), None, distilbert, resamples=50)


def seed_bundle(tmp_path: Path, full_errors, lora_errors) -> Path:
    artifacts = tmp_path / "distilbert_v2"
    for seed, full, lora in zip((42, 43, 44), full_errors, lora_errors, strict=True):
        names = seeds_compare.run_names(seed)
        write_run(artifacts / "time" / names[0], "time", predictions(full, seed), seed)
        write_run(artifacts / "time" / names[1], "time", predictions(lora, seed + 100), seed)
    return artifacts


def test_the_seed_comparison_finds_a_lead_that_holds(tmp_path):
    artifacts = seed_bundle(tmp_path, (0.04, 0.04, 0.04), (0.08, 0.08, 0.08))
    result = seeds_compare.compare(artifacts, split_views(), resamples=300)
    assert result["outcome"] == "the lead holds across seeds" and result["size"] == "meaningful"
    assert set(result["seeds"]) == {"42", "43", "44"} and result["phase2_primary_stays"] == "seed 42"


def test_a_reversed_seed_means_no_consistent_lead(tmp_path):
    artifacts = seed_bundle(tmp_path, (0.04, 0.08, 0.04), (0.08, 0.04, 0.08))
    result = seeds_compare.compare(artifacts, split_views(), resamples=300)
    assert result["outcome"] == "no consistent lead across seeds" and result["size"] is None


def test_the_frontier_benchmark_times_each_model_on_one_review_at_a_time(tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from test_bilstm_artifacts import test_a_training_run_writes_aligned_artifacts

    test_a_training_run_writes_aligned_artifacts(tmp_path)  # a tiny BiLSTM run in tmp_path/run
    train = pd.read_parquet(tmp_path / "data" / "train.parquet")
    test = pd.read_parquet(tmp_path / "data" / "test.parquet")
    pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB()).fit(train["text"], train["label"])
    import joblib

    joblib.dump(pipeline, tmp_path / "nb.joblib")
    nb_scored = (pipeline.predict(test["text"]) == "positive").astype(int)
    report = frontier_bench.frontier(test, nb_scored, tmp_path / "nb.joblib", tmp_path / "run", {}, sample_size=20)
    assert set(report["models"]) == {"tfidf_nb", "bilstm"}
    assert report["method"]["threads"] == 1 and report["method"]["batch_size"] == 1
    for entry in report["models"].values():
        timing = entry["cpu"]["batch_1"]
        assert timing["batch_size"] == 1 and timing["reviews"] == 20 and timing["agrees"] >= 19
    assert report["models"]["bilstm"]["size_mb"] >= 0


def test_the_bilstm_predictor_keeps_review_order_in_a_batch(tmp_path):
    from test_bilstm_artifacts import test_a_training_run_writes_aligned_artifacts

    test_a_training_run_writes_aligned_artifacts(tmp_path)
    texts = pd.read_parquet(tmp_path / "data" / "test.parquet")["text"].tolist()
    predicted = frontier_bench.BiLSTMPredictor(tmp_path / "run")(texts)  # one batch of every length
    assert predicted.tolist() == np.load(tmp_path / "run" / "test_logits.npy").argmax(axis=1).tolist()
