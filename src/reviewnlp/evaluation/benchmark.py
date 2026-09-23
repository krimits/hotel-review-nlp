"""Unified benchmark: every model, identical test set, one artifact.

Collects predictions from each available model family, computes metrics,
runs all-pairs McNemar tests, and writes ``runs/benchmark/`` containing:

    results.json       metrics + significance + timing per model
    confusion.png      side-by-side confusion matrices
    latency.png        p50/p95 inference latency per model (CPU and/or GPU)

Missing models are skipped with a warning - the benchmark degrades
gracefully so it can run at any point of the 4-week plan.

Run:  python -m reviewnlp.evaluation.benchmark --config configs/baselines.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from reviewnlp.evaluation.metrics import binary_metrics
from reviewnlp.evaluation.plots import plot_confusion_matrices, plot_latency
from reviewnlp.evaluation.significance import pairwise_mcnemar
from reviewnlp.llm.predict import load_predict_fn, load_test_set
from reviewnlp.utils.experiments import assert_clean_splits, fingerprint_splits, frame_fingerprint
from reviewnlp.utils.seed import load_config, set_seed


def _measure_latency(predict_fn, sample_texts: list[str], repeats: int = 3) -> dict:
    """Rough single-request latency over a small sample (reporting only)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(sample_texts)
        times.append(time.perf_counter() - t0)
    per_text = [t / len(sample_texts) for t in times]
    return {
        "p50_ms_per_text": round(1000 * float(np.percentile(per_text, 50)), 2),
        "p95_ms_per_text": round(1000 * float(np.percentile(per_text, 95)), 2),
    }


def run_benchmark(config_path: str, models: dict[str, tuple[str, str]] | None = None) -> dict:
    """models: {'name': (model_type, path)}. Auto-detected when None."""
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    processed_dir = cfg["data"]["processed_dir"]

    test_df = load_test_set(processed_dir)
    texts, gold = test_df["text"].tolist(), test_df["label"].values
    manifest_path = Path(processed_dir) / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: build clean versioned splits with `make data`")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    raw_provenance = version == 2 and bool(manifest.get("raw_csv_sha256"))
    source_hashes = manifest.get("source_files_sha256", {})
    uploaded_provenance = (version == 3 and
                           manifest.get("source_type") == "uploaded_processed_parquets_without_raw_csv" and
                           isinstance(source_hashes, dict) and
                           set(source_hashes) == {"train", "dev", "test"} and
                           all(isinstance(value, str) and len(value) == 64 and
                               all(char in "0123456789abcdef" for char in value)
                               for value in source_hashes.values()))
    if not (raw_provenance or uploaded_provenance):
        raise ValueError("benchmark needs a versioned clean-split manifest")
    # A fingerprint proves identity, not independence: reject a consistent
    # manifest that still points to overlapping or duplicated reviews.
    from reviewnlp.data.preprocess import load_processed

    assert_clean_splits(load_processed(processed_dir))
    actual_splits = fingerprint_splits(processed_dir)
    if manifest.get("splits") != actual_splits:
        raise ValueError("processed splits differ from data_manifest.json; refusing comparison")
    test_fingerprint = frame_fingerprint(test_df)
    print(f"benchmark on {len(texts):,} test reviews")

    models = models or _discover_models(cfg)

    predictions, results, skipped = {}, {}, {}
    for name, (model_type, path) in models.items():
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found")
            skipped[name] = "checkpoint not found"
            continue
        if model_type == "cached_logits":
            _check_cached_test(path, gold, test_fingerprint)
        elif model_type == "classical":
            metrics_path = Path(path).with_name("metrics.json")
            if not metrics_path.is_file() or json.loads(metrics_path.read_text()).get("_data", {}).get("test") != test_fingerprint:
                raise ValueError(f"{name}: missing or mismatched test fingerprint in {metrics_path}")
        try:
            predict_fn = load_predict_fn(model_type, path)
            t0 = time.perf_counter()
            preds = predict_fn(texts)
            wall = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - benchmark must not die on one model
            print(f"skip {name}: {exc}")
            skipped[name] = str(exc)
            continue

        predictions[name] = np.asarray(preds)
        m = binary_metrics(gold, preds)
        m["wall_seconds"] = round(wall, 2)
        # Cached logits have no callable inference path; dividing a full-test
        # cache by a 64-row sample used to raise and abort the entire benchmark.
        if model_type != "cached_logits":
            m["latency"] = _measure_latency(predict_fn, texts[:64])
        results[name] = m
        print(f"[{name:>18}] macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")

    significance = pairwise_mcnemar(gold, predictions)
    for pair, res in significance.items():
        verdict = "SIGNIFICANT" if res["significant_at_0.05"] else "not significant"
        print(f"McNemar {pair}: p={res['p_value']:.4g} ({verdict})")

    out_dir = "runs/benchmark"
    os.makedirs(out_dir, exist_ok=True)
    artifact = {"n_test": len(texts), "test_fingerprint": test_fingerprint,
                "data_manifest": manifest,
                "data_source_type": manifest.get("source_type", "raw_csv"),
                "latency_protocol": {"batch_size": min(64, len(texts)), "repeats": 3,
                                     "model_preloaded": True, "includes_http": False},
                "results": results, "mcnemar": significance, "skipped": skipped}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(artifact, f, indent=2)

    if results:
        plot_confusion_matrices(results, os.path.join(out_dir, "confusion.png"))
    if any("latency" in r for r in results.values()):
        plot_latency(results, os.path.join(out_dir, "latency.png"))

    print(f"benchmark artifact -> {out_dir}/results.json")
    return artifact


def _check_cached_test(path: str, gold: np.ndarray, test_fingerprint: dict) -> None:
    """Require both ordered labels and the source review fingerprint to match."""
    directory = Path(path)
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        raise ValueError(f"{directory}: no metrics.json carrying the training split identity")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("data", {}).get("test") != test_fingerprint:
        raise ValueError(f"{directory}: test fingerprint differs from current dataset")
    labels = np.load(directory / "test_labels.npy", allow_pickle=False)
    expected = np.array([{"negative": 0, "positive": 1}[label] for label in gold])
    if not np.array_equal(labels, expected):
        raise ValueError(f"{directory}: cached labels differ from ordered test labels")


def _discover_models(cfg: dict) -> dict[str, tuple[str, str]]:
    """Find whatever is available on disk - order defines benchmark columns."""
    classical_dir = Path(cfg.get("output", {}).get("model_dir", "runs/classical"))
    return {
        "Best classical (NB/LR-SGD)": ("classical", str(classical_dir / "best_classical.joblib")),
        "BiLSTM": ("cached_logits", "runs/bilstm"),
        "DistilBERT (full FT)": ("cached_logits", "runs/distilbert"),
        "DistilBERT + scratch LoRA": ("cached_logits", "runs/distilbert_lora_scratch"),
        "Qwen2.5-0.5B + QLoRA": ("qwen_qlora", "runs/qwen_qlora/adapter"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
