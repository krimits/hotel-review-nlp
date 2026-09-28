"""The phase 2 latency benchmark, with a fake clock and a tiny local DistilBERT."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("benchmark_latency", ROOT / "scripts" / "benchmark_latency.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)

GOOD = ["great", "clean", "friendly", "lovely", "quiet", "helpful"]
BAD = ["dirty", "noisy", "rude", "broken", "cold", "small"]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _reviews(count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.where(rng.random(count) < 0.7, "positive", "negative")
    texts = [" ".join(rng.choice(GOOD if label == "positive" else BAD, size=rng.integers(2, 12)))
             for label in labels]
    return pd.DataFrame({"text": texts, "label": labels})


def _naive_bayes(path: Path) -> Pipeline:
    train = _reviews(200, seed=1)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]).fit(train["text"], train["label"])
    joblib.dump(pipeline, path)
    return pipeline


def _tiny_distilbert(directory: Path, texts: list[str]) -> np.ndarray:
    """A random tiny checkpoint, and its predictions made one review at a time, unpadded."""
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    directory.mkdir(parents=True)
    (directory / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *GOOD, *BAD]) + "\n")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
    tokenizer.save_pretrained(directory)
    torch.manual_seed(0)
    model = DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=5 + len(GOOD) + len(BAD), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
        max_position_embeddings=64)).eval()
    model.save_pretrained(directory)
    with torch.inference_mode():
        return np.array([int(model(**tokenizer(text, return_tensors="pt")).logits.argmax()) for text in texts])


def test_time_batches_times_only_the_timed_pass():
    clock, calls = FakeClock(), []

    def predict(batch):
        calls.append(len(batch))
        clock.now += 0.002 * len(batch)  # 2 ms per review
        return [len(text) % 2 for text in batch]

    texts = [f"review {'x' * index}" for index in range(70)]
    report, predictions = bench.time_batches(predict, texts, batch_size=32, warmup=1, clock=clock)
    assert calls == [32, 32, 32, 6]  # one untimed warm-up batch, then all 70 reviews
    assert predictions.tolist() == [len(text) % 2 for text in texts]
    assert report["reviews"] == 70 and report["warmup_batches"] == 1
    assert report["seconds"] == pytest.approx(0.14)
    assert report["ms_per_review"] == pytest.approx(2.0)
    assert report["ms_per_batch_median"] == pytest.approx(64.0)
    assert report["reviews_per_second"] == pytest.approx(500.0)


def test_every_model_is_timed_on_the_same_reviews():
    rows = bench.sample_rows(10_000, 500, seed=0)
    assert len(set(rows.tolist())) == 500 and list(rows) == sorted(rows)
    assert np.array_equal(rows, bench.sample_rows(10_000, 500, seed=0))
    assert not np.array_equal(rows, bench.sample_rows(10_000, 500, seed=1))
    assert bench.sample_rows(300, 500).tolist() == list(range(300))


def test_benchmark_times_the_scored_models(tmp_path):
    test = _reviews(60, seed=2)
    pipeline = _naive_bayes(tmp_path / "best_classical.joblib")
    nb_scored = (pipeline.predict(test["text"]) == "positive").astype(int)
    scored = _tiny_distilbert(tmp_path / "distilbert", test["text"].tolist())
    np.save(tmp_path / "distilbert" / "test_logits.npy", np.eye(2)[scored])

    report = bench.benchmark(test, nb_scored, tmp_path / "best_classical.joblib",
                             {"distilbert": tmp_path / "distilbert"}, sample_size=40, gpu=False)
    assert report["sample"]["reviews"] == 40 and report["test"]["rows"] == 60
    nb, encoder = report["models"]["tfidf_nb"], report["models"]["distilbert"]
    for timing in (*nb["cpu"].values(), *encoder["cpu"].values()):
        assert timing["reviews"] == 40 and timing["agrees"] == 40
    assert nb["cpu_whole_test"]["agrees"] == 60 and nb["cpu_whole_test"]["batch_size"] == 60
    assert encoder["cpu"]["batch_32"]["batch_size"] == 32
    assert encoder["gpu_whole_test"] == "no GPU" and "adapter_mb" not in encoder
    assert encoder["size_mb"] >= 0 and nb["size_mb"] >= 0


def test_a_model_that_was_not_scored_is_refused(tmp_path):
    test = _reviews(60, seed=2)
    pipeline = _naive_bayes(tmp_path / "best_classical.joblib")
    nb_scored = (pipeline.predict(test["text"]) == "positive").astype(int)
    with pytest.raises(bench.BenchmarkError, match="not the model that was scored"):
        bench.benchmark(test, 1 - nb_scored, tmp_path / "best_classical.joblib", {}, sample_size=40, gpu=False)


def test_the_size_of_naive_bayes_leaves_out_the_dropped_terms(tmp_path):
    pipeline = _naive_bayes(tmp_path / "best_classical.joblib")
    size = bench.classical_size_mb(pipeline)
    pipeline.named_steps["tfidf"].stop_words_ = {f"dropped term {index}" for index in range(50_000)}
    assert bench.classical_size_mb(pipeline) == size
