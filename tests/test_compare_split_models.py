"""The phase 2 comparison on synthetic runs whose answers are known."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from reviewnlp.evaluation.metrics import binary_metrics
from reviewnlp.utils.seed import load_config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_split_models", ROOT / "scripts" / "compare_split_models.py")
compare = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = compare
_spec.loader.exec_module(compare)

N = 4000


def _flip(labels: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    wrong = rng.random(len(labels)) < share
    return np.where(wrong, 1 - labels, labels)


def _fingerprint(labels: np.ndarray, tag: str) -> dict:
    return {"rows": len(labels), "negative": int((labels == 0).sum()), "positive": int((labels == 1).sum()),
            "sha256": f"test-{tag}"}


def _write_run(directory: Path, labels: np.ndarray, predictions: np.ndarray, fingerprint: dict,
               seed: int = 42) -> None:
    directory.mkdir(parents=True)
    logits = np.stack([-predictions, predictions], axis=1).astype(np.float32)
    np.save(directory / "test_logits.npy", logits)
    np.save(directory / "test_labels.npy", labels)
    (directory / "metrics.json").write_text(json.dumps({
        "data": {"test": fingerprint}, "training_seconds": 600.0, "peak_cuda_memory_mb": 2500.4,
        "trainable_params": 66955010, "total_params": 66955010, "best_dev_macro_f1": 0.95,
        "settings": {"seed": seed}}))


def _artifacts(tmp_path: Path, nb_error: float = 0.10, full_error: float = 0.03, lora_error: float = 0.04,
               seed: int = 0) -> tuple[Path, dict]:
    """Two splits; each model errs on a random share of reviews."""
    rng = np.random.default_rng(seed)
    views = {}
    for split in compare.SPLITS:
        labels = (rng.random(N) < 0.75).astype(int)
        fingerprint = _fingerprint(labels, split)
        nb = _flip(labels, nb_error, rng)
        views[split] = {"splits": {"test": fingerprint}, "test_predictions": "".join("np"[v] for v in nb),
                        "test": {"macro_f1": binary_metrics(labels, nb, label_values=(0, 1))["macro_f1"]}}
        _write_run(tmp_path / split / "distilbert", labels, _flip(labels, full_error, rng), fingerprint)
        _write_run(tmp_path / split / "distilbert_lora_scratch", labels, _flip(labels, lora_error, rng), fingerprint)
    return tmp_path, views


def test_a_clearly_better_model_meets_the_rule(tmp_path):
    artifacts, views = _artifacts(tmp_path)
    report = compare.compare(artifacts, views, resamples=200)
    primary = report["primary"]
    assert (primary["split"], primary["a"], primary["b"]) == ("time", "tfidf_nb", "distilbert")
    assert primary["macro_f1_difference_b_minus_a"] > 0.01 and primary["paired_ci95"][0] > 0
    assert primary["verdict"] == {"better": "distilbert", "gain": "meaningful"}
    drop = report["drop_random_to_time"]["distilbert"]
    assert drop["drop"] == pytest.approx(drop["random"] - drop["time"], abs=1e-4)
    assert report["cost"]["training"]["random"]["distilbert"]["training_minutes"] == 10.0
    assert report["cost"]["inference"] == "not measured"
    assert report["primary_other_seeds"] == {}


def test_a_second_seed_repeats_the_primary_comparison(tmp_path):
    artifacts, views = _artifacts(tmp_path)
    labels = np.load(artifacts / "time" / "distilbert" / "test_labels.npy")
    second = _flip(labels, 0.035, np.random.default_rng(5))
    _write_run(artifacts / "time" / "distilbert_seed43", labels, second, views["time"]["splits"]["test"], seed=43)
    report = compare.compare(artifacts, views, resamples=200)
    other = report["primary_other_seeds"]["distilbert_seed43"]
    assert other["seed"] == 43 and (other["a"], other["b"]) == ("tfidf_nb", "distilbert_seed43")
    assert other["macro_f1"] == binary_metrics(labels, second, label_values=(0, 1))["macro_f1"]
    assert other["verdict"] == {"better": "distilbert_seed43", "gain": "meaningful"}
    assert report["primary"]["b"] == "distilbert"


def test_mcnemar_counts_the_discordant_reviews(tmp_path):
    artifacts, views = _artifacts(tmp_path)
    report = compare.compare(artifacts, views, resamples=50)
    pair = report["splits"]["random"]["pairs"]["distilbert vs tfidf_nb"]
    labels = np.load(artifacts / "random" / "distilbert" / "test_labels.npy")
    nb = compare.nb_predictions(views["random"])
    full = np.load(artifacts / "random" / "distilbert" / "test_logits.npy").argmax(axis=1)
    assert pair["mcnemar"]["a_wrong_b_right"] == int(np.sum((nb != labels) & (full == labels)))
    assert pair["mcnemar"]["a_right_b_wrong"] == int(np.sum((nb == labels) & (full != labels)))
    assert pair["mcnemar"]["significant_at_0.05"]


def test_a_different_test_set_is_refused(tmp_path):
    artifacts, views = _artifacts(tmp_path)
    views["time"]["splits"]["test"] = dict(views["time"]["splits"]["test"], sha256="another test set")
    with pytest.raises(compare.ComparisonError, match="fingerprint"):
        compare.compare(artifacts, views, resamples=10)


def test_labels_in_another_order_are_refused(tmp_path):
    artifacts, views = _artifacts(tmp_path)
    path = artifacts / "random" / "distilbert_lora_scratch" / "test_labels.npy"
    np.save(path, np.load(path)[::-1])
    with pytest.raises(compare.ComparisonError, match="labels differ"):
        compare.compare(artifacts, views, resamples=10)


def test_a_missing_run_is_refused(tmp_path):
    artifacts, views = _artifacts(tmp_path)
    (artifacts / "time" / "distilbert" / "metrics.json").unlink()
    with pytest.raises(compare.ComparisonError, match="missing run"):
        compare.compare(artifacts, views, resamples=10)


def test_naive_bayes_must_score_as_recorded(tmp_path):
    artifacts, views = _artifacts(tmp_path)
    views["random"]["test"]["macro_f1"] = 0.5
    with pytest.raises(compare.ComparisonError, match="Naive Bayes scores"):
        compare.compare(artifacts, views, resamples=10)


@pytest.mark.parametrize(("difference", "low", "high", "expected"), [
    (0.02, 0.012, 0.028, {"better": "b", "gain": "meaningful"}),
    (0.005, 0.002, 0.008, {"better": "b", "gain": "clear but small"}),
    (-0.015, -0.02, -0.01, {"better": "a", "gain": "meaningful"}),
    (0.02, -0.001, 0.04, {"better": None, "gain": "no clear difference"}),
])
def test_the_rule_for_a_meaningful_gain(difference, low, high, expected):
    assert compare.verdict("a", "b", difference, low, high) == expected


@pytest.mark.parametrize(("config", "data_config"), [("distilbert_v2.yaml", "baselines.yaml"),
                                                     ("distilbert_v2_time.yaml", "baselines_time.yaml")])
def test_the_phase2_configs_change_only_the_data(config, data_config):
    legacy = load_config(str(ROOT / "configs" / "distilbert.yaml"))
    phase2 = load_config(str(ROOT / "configs" / config))
    assert phase2["data"]["processed_dir"] == load_config(str(ROOT / "configs" / data_config))["data"]["processed_dir"]
    assert {key: value for key, value in phase2.items() if key not in ("data", "output")} == \
        {key: value for key, value in legacy.items() if key != "output"}
    assert phase2["output"]["model_dir"] != legacy["output"]["model_dir"]
