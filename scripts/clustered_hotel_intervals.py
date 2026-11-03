"""Cluster-robust intervals for the unseen-hotels split.

Every macro-F1 interval published for the three v2 splits resamples *reviews*. The hotel
view is about drawing a different set of hotels, and reviews of one hotel are not
independent, so the row-level interval understates the uncertainty there.

This script rebuilds the hotel split from the hash-checked raw file (``make data`` does
the fetch), trains the dev-selected classical pipeline on it, and reports both intervals
next to the saved BiLSTM logits, which are matched to the same test rows by their
committed labels. It writes ``docs/experiments/results/clustered/hotel.json``.

    python scripts/clustered_hotel_intervals.py

Refuses to run if the rebuilt split's fingerprints do not match the committed manifest:
an interval over different rows is not an interval over these rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewnlp.baselines.classical import train_classical_baselines  # noqa: E402
from reviewnlp.data.preprocess import load_processed  # noqa: E402
from reviewnlp.evaluation.metrics import _macro_f1, binary_metrics, bootstrap_ci  # noqa: E402
from reviewnlp.evaluation.significance import cluster_bootstrap_ci  # noqa: E402

CONFIG = ROOT / "configs" / "baselines_hotel.yaml"
PROCESSED = ROOT / "data" / "processed_v2_hotel"
COMMITTED_MANIFEST = ROOT / "docs" / "experiments" / "results" / "v2_splits" / "hotel" / "data_manifest.json"
BILSTM_DIR = ROOT / "docs" / "experiments" / "results" / "bilstm_v2" / "hotel" / "bilstm"
OUTPUT = ROOT / "docs" / "experiments" / "results" / "clustered" / "hotel.json"
LABEL_CODES = {"negative": 0, "positive": 1}
N_RESAMPLES = 1000
SEED = 42


def verify_split() -> dict:
    """The rebuilt split must be the committed one, fingerprints included."""
    if not COMMITTED_MANIFEST.exists():
        raise SystemExit(f"missing {COMMITTED_MANIFEST}; this script needs the committed v2 manifest")
    committed = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    built_path = PROCESSED / "data_manifest.json"
    if not built_path.exists():
        raise SystemExit(f"missing {built_path}: run `make data` (or the hotel preprocess step) first")
    built = json.loads(built_path.read_text(encoding="utf-8"))
    for key in ("split", "split_policy", "splits", "raw_csv_sha256", "hotels", "review_dates"):
        if built.get(key) != committed.get(key):
            raise SystemExit(f"rebuilt split differs from the committed manifest at {key!r}; refusing")
    return committed


def intervals(labels: np.ndarray, predicted: np.ndarray, clusters: np.ndarray) -> dict:
    """Row-level and cluster-level 95% intervals for the same predictions."""
    values = np.column_stack([labels, predicted])

    def statistic(rows: np.ndarray) -> float:
        return _macro_f1(rows[:, 0], rows[:, 1], (0, 1))

    low, high = bootstrap_ci(
        [name for name in np.where(labels == 1, "positive", "negative")],
        [name for name in np.where(predicted == 1, "positive", "negative")],
        seed=SEED,
    )
    c_low, c_high = cluster_bootstrap_ci(values, clusters, n_resamples=N_RESAMPLES, seed=SEED, statistic=statistic)
    width, c_width = high - low, c_high - c_low
    return {
        "ci95_rows": [round(low, 4), round(high, 4)],
        "ci95_clusters": [round(c_low, 4), round(c_high, 4)],
        "width_ratio_clusters_vs_rows": round(c_width / width, 3) if width > 0 else None,
    }


def main() -> None:
    manifest = verify_split()
    test = load_processed(str(PROCESSED))["test"]
    clusters = test["hotel"].astype(str).to_numpy()
    y_true = np.array([LABEL_CODES[label] for label in test["label"]])

    best = train_classical_baselines(str(CONFIG))["_best"]
    predicted = np.array([LABEL_CODES[label] for label in joblib.load(best["path"]).predict(test["text"])])

    bilstm_logits = np.load(BILSTM_DIR / "test_logits.npy")
    bilstm_labels = np.load(BILSTM_DIR / "test_labels.npy")
    if bilstm_labels.shape != y_true.shape or not np.array_equal(bilstm_labels, y_true):
        raise SystemExit("the saved BiLSTM labels do not line up with the rebuilt test rows")
    bilstm_predicted = bilstm_logits.argmax(axis=1)

    record = {
        "split": "hotel",
        "cluster_key": "hotel",
        "rows": int(len(y_true)),
        "clusters": int(len(np.unique(clusters))),
        "rows_per_cluster_median": float(np.median(np.unique(clusters, return_counts=True)[1])),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "raw_csv_sha256": manifest["raw_csv_sha256"],
        "split_fingerprints": manifest["splits"],
        "models": {},
    }
    for name, predicted_labels in (
        (f"classical_{best['name']}", predicted),
        ("bilstm_seed42", bilstm_predicted),
    ):
        metrics = binary_metrics(np.where(y_true == 1, "positive", "negative"),
                                 np.where(predicted_labels == 1, "positive", "negative"))
        record["models"][name] = {
            "macro_f1": round(metrics["macro_f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            **intervals(y_true, predicted_labels, clusters),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"{'model':24} {'macro-F1':>8}  {'rows CI95':>18}  {'clusters CI95':>18}  ratio")
    for name, row in record["models"].items():
        print(f"{name:24} {row['macro_f1']:>8.4f}  {str(row['ci95_rows']):>18}  "
              f"{str(row['ci95_clusters']):>18}  x{row['width_ratio_clusters_vs_rows']}")
    print(f"-> {OUTPUT.relative_to(ROOT)}  ({record['rows']:,} rows, {record['clusters']} hotels)")


if __name__ == "__main__":
    main()
