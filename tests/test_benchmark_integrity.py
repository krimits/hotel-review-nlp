from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from reviewnlp.evaluation.benchmark import _check_cached_test, run_benchmark
from reviewnlp.llm import predict as inference
from reviewnlp.utils.experiments import frame_fingerprint


def test_cached_logits_cannot_be_attached_to_a_different_ordered_test(tmp_path):
    frame = pd.DataFrame({"text": ["one good", "one bad"],
                          "label": ["positive", "negative"]})
    fingerprint = frame_fingerprint(frame)
    (tmp_path / "metrics.json").write_text(json.dumps({"data": {"test": fingerprint}}))
    np.save(tmp_path / "test_labels.npy", np.array([1, 0]))
    _check_cached_test(str(tmp_path), frame["label"].to_numpy(), fingerprint)

    changed = frame.copy()
    changed.loc[0, "text"] = "another good"
    with pytest.raises(ValueError, match="fingerprint"):
        _check_cached_test(str(tmp_path), changed["label"].to_numpy(),
                           frame_fingerprint(changed))

    np.save(tmp_path / "test_labels.npy", np.array([0, 1]))
    with pytest.raises(ValueError, match="ordered test labels"):
        _check_cached_test(str(tmp_path), frame["label"].to_numpy(), fingerprint)


def test_qwen_benchmark_reuses_one_loaded_adapter(monkeypatch):
    loads = []
    seen_bundles = []
    monkeypatch.setattr(inference, "load_qwen_qlora", lambda path: loads.append(path) or object())

    def fake_predict(path, texts, bundle):
        seen_bundles.append(bundle)
        return np.array(["positive"] * len(texts))

    monkeypatch.setattr(inference, "predict_qwen_qlora", fake_predict)
    predict = inference.load_predict_fn("qwen_qlora", "adapter")
    predict(["first"])
    predict(["second"])
    assert loads == ["adapter"]
    assert seen_bundles[0] is seen_bundles[1]


def test_benchmark_refuses_consistent_manifest_with_leaking_reviews(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    splits = tmp_path / "processed"
    splits.mkdir()
    frames = {
        "train": pd.DataFrame({"text": ["Good room", "Bad room"], "label": ["positive", "negative"]}),
        "dev": pd.DataFrame({"text": ["Other good", "Other bad"], "label": ["positive", "negative"]}),
        "test": pd.DataFrame({"text": [" good  room ", "Worst stay"], "label": ["positive", "negative"]}),
    }
    for name, frame in frames.items():
        frame.to_parquet(splits / f"{name}.parquet")
    (splits / "data_manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "raw_csv_sha256": "a" * 64,
        "splits": {name: frame_fingerprint(frame) for name, frame in frames.items()},
    }))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"seed: 42\ndata:\n  processed_dir: {splits}\n")
    with pytest.raises(ValueError, match="overlaps"):
        run_benchmark(str(cfg), models={})
