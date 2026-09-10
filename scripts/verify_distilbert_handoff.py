"""Verify the notebook 05 handoff without loading model weights or using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from reviewnlp.evaluation.metrics import binary_metrics
from reviewnlp.evaluation.significance import pairwise_mcnemar
from reviewnlp.utils.experiments import frame_fingerprint

MODELS = ("distilbert", "distilbert_lora_scratch")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "docs/experiments/results/distilbert_legacy_full_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(bundle: Path, processed_dir: Path | None = None) -> dict:
    bundle = bundle.resolve()
    artifact_manifest = read_json(bundle / "artifact_manifest.json")
    data = read_json(bundle / "data_manifest.json")
    environment = read_json(bundle / "environment.json")
    canonical = read_json(ROOT / "docs/experiments/legacy_dataset_manifest.json")
    commit = artifact_manifest["git_commit"]
    require(environment["git_commit"] == commit, "Conflicting training commits")

    verified_files = []
    omitted_files = []
    for entry in artifact_manifest["files"]:
        path = (bundle / entry["path"]).resolve()
        path.relative_to(bundle)
        if not path.is_file():
            omitted_files.append(entry["path"])
            continue
        content = path.read_bytes()
        require(len(content) == entry["bytes"], f"Size mismatch: {entry['path']}")
        require(
            hashlib.sha256(content).hexdigest() == entry["sha256"],
            f"SHA256 mismatch: {entry['path']}",
        )
        verified_files.append(entry["path"])

    required = {
        "data_manifest.json", "environment.json", "distilbert_comparison.json",
        "distilbert_comparison.csv", "mcnemar_full_vs_lora.json", "trackio_final_metrics.json",
        "distilbert_lora_scratch/lora_scratch_config.json",
    }
    for name in MODELS:
        required.update(f"{name}/{file}" for file in (
            "metrics.json", "request.json", "run_config.yaml", "train.log",
            "dev_labels.npy", "dev_logits.npy", "test_labels.npy", "test_logits.npy",
        ))
    require(required.issubset(verified_files), "Incomplete or unmanifested evaluation handoff")

    source_count = None
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, capture_output=True,
    )
    if available.returncode == 0:
        for path, expected in environment["source_sha256"].items():
            content = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
            require(hashlib.sha256(content).hexdigest() == expected, f"Source mismatch: {path}")
        source_count = len(environment["source_sha256"])

    frozen_labels = {}
    for split in ("train", "dev", "test"):
        fingerprint = data["semantic_fingerprints"][split]
        original = canonical["splits"][split]
        require(fingerprint == {
            "rows": original["rows"], "negative": original["negative"],
            "positive": original["positive"], "sha256": original["content_sha256"],
        }, f"Unexpected declared fingerprint: {split}")
        require(data["parquet_file_sha256"][split] == original["file_sha256"],
                f"Unexpected declared parquet hash: {split}")
        if processed_dir is not None:
            path = processed_dir / f"{split}.parquet"
            require(hashlib.sha256(path.read_bytes()).hexdigest() == original["file_sha256"],
                    f"Local parquet mismatch: {split}")
            frame = pd.read_parquet(path)
            require(frame_fingerprint(frame) == fingerprint, f"Local fingerprint mismatch: {split}")
            frozen_labels[split] = frame["label"].map({"negative": 0, "positive": 1}).to_numpy()

    labels_by_split = {}
    predictions = {}
    recomputed = {}
    records = {}
    for name in MODELS:
        records[name] = read_json(bundle / name / "metrics.json")
        record = records[name]
        require(record["data"] == data["semantic_fingerprints"], f"Run fingerprint mismatch: {name}")
        for split in ("dev", "test"):
            labels = np.load(bundle / name / f"{split}_labels.npy", allow_pickle=False)
            logits = np.load(bundle / name / f"{split}_logits.npy", allow_pickle=False)
            size = data["semantic_fingerprints"][split]["rows"]
            require(labels.shape == (size,) and np.isin(labels, [0, 1]).all(),
                    f"Invalid labels: {name}/{split}")
            require(logits.shape == (size, 2) and np.isfinite(logits).all(),
                    f"Invalid logits: {name}/{split}")
            require(int((labels == 0).sum()) == data["semantic_fingerprints"][split]["negative"],
                    f"Label count mismatch: {name}/{split}")
            if split in frozen_labels:
                require(np.array_equal(labels, frozen_labels[split]),
                        f"Labels differ from ordered local data: {name}/{split}")
            if split in labels_by_split:
                require(np.array_equal(labels, labels_by_split[split]),
                        f"Models have different ordered labels: {split}")
            labels_by_split[split] = labels
            measured = binary_metrics(
                labels, logits.argmax(1), label_names=("negative", "positive"), label_values=(0, 1),
            )
            if split == "dev":
                require(measured["macro_f1"] == record["best_dev_macro_f1"],
                        f"Dev score mismatch: {name}")
            else:
                require(measured == record["test"], f"Test metrics mismatch: {name}")
                recomputed[name] = measured
                predictions[name] = logits.argmax(1)

    paired = pairwise_mcnemar(labels_by_split["test"], predictions)
    saved_paired = read_json(bundle / "mcnemar_full_vs_lora.json")
    require(list(paired.values()) == list(saved_paired.values()), "McNemar mismatch")
    full_correct = predictions[MODELS[0]] == labels_by_split["test"]
    lora_correct = predictions[MODELS[1]] == labels_by_split["test"]
    full, lora = (records[name] for name in MODELS)
    return {
        "experiment_id": artifact_manifest["experiment_id"],
        "training_git_commit": commit,
        "verified_manifest_files": len(verified_files),
        "source_files_matching_training_commit": source_count,
        "local_frozen_parquets_verified": processed_dir is not None,
        "omitted_full_model_files": omitted_files,
        "recomputed_test_metrics": recomputed,
        "mcnemar": {
            **next(iter(paired.values())),
            "full_correct_lora_wrong": int((full_correct & ~lora_correct).sum()),
            "full_wrong_lora_correct": int((~full_correct & lora_correct).sum()),
        },
        "reported_training_cost_comparison": {
            "gpu": environment["gpu"],
            "training_time_reduction_pct": 100 * (1 - lora["training_seconds"] / full["training_seconds"]),
            "peak_memory_reduction_pct": 100 * (1 - lora["peak_cuda_memory_mb"] / full["peak_cuda_memory_mb"]),
            "macro_f1_difference_pp": 100 * (lora["test"]["macro_f1"] - full["test"]["macro_f1"]),
            "lora_trainable_pct": 100 * lora["trainable_params"] / lora["total_params"],
        },
        "limitations": [
            "The legacy splits contain the cross-split overlaps recorded in data_manifest.json.",
            "Training cost and parameter counts are reported run metadata; GPU training was not repeated.",
            "This handoff omits weights and tokenizers, so checkpoint reload and latency are not verified.",
            "McNemar compares paired classification errors, not macro-F1 or variation across training seeds.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--processed-dir", type=Path, help="Also verify the local frozen parquet files")
    parser.add_argument("--output", type=Path, help="Write the verification report as JSON")
    args = parser.parse_args()
    report = verify(args.bundle, args.processed_dir)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
