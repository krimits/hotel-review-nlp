"""What the test set measures: the same classical pipeline on three splits of the same reviews.

    random  reviews like the training ones (stratified)
    time    reviews written after every review in training (out-of-time)
    hotel   hotels that have no review in training

For each split built by `make data`, this trains the classical candidates,
picks the best on dev (reviewnlp.baselines.classical), scores the test set and
gives macro-F1 with a 95% bootstrap interval over test reviews. The test
predictions are stored in test order (one character per review, "n" or "p"),
so the figures can be recomputed without the model.

    python scripts/split_views.py   # writes docs/experiments/results/split_views.json

With `--cluster-key hotel`, each view also reports a cluster-bootstrap interval next to
the row-level one: whole hotels are resampled, so the interval describes drawing a
different set of hotels rather than different reviews from the same hotels. That is the
quantity the unseen-hotels split is about, and the row-level interval understates it when
reviews of one hotel are correlated. The default run, and the committed artifact, are
unchanged when the flag is absent.

    python scripts/split_views.py --cluster-key hotel --out /tmp/split_views_clustered.json
"""

from __future__ import annotations

import argparse
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
from reviewnlp.utils.seed import load_config  # noqa: E402

CONFIGS = {
    "random": ROOT / "configs" / "baselines.yaml",
    "time": ROOT / "configs" / "baselines_time.yaml",
    "hotel": ROOT / "configs" / "baselines_hotel.yaml",
}
OUTPUT = ROOT / "docs" / "experiments" / "results" / "split_views.json"
LABEL_CODES = {"negative": 0, "positive": 1}
CLUSTER_RESAMPLES = 1000  # the iid call below uses bootstrap_ci's default of 1000


def _clustered_macro_f1_ci(labels, predicted, clusters, seed: int) -> tuple[float, float]:
    """Macro-F1 interval resampling whole clusters, with the same rows as the iid interval.

    `_macro_f1` is the function `metrics.bootstrap_ci` uses, so the two intervals differ
    only in what they resample: rows or clusters.
    """
    values = np.column_stack(
        [np.array([LABEL_CODES[label] for label in labels]), np.array([LABEL_CODES[label] for label in predicted])]
    )

    def statistic(rows: np.ndarray) -> float:
        return _macro_f1(rows[:, 0], rows[:, 1], (0, 1))

    return cluster_bootstrap_ci(
        values, clusters, n_resamples=CLUSTER_RESAMPLES, seed=seed, statistic=statistic
    )


def split_view(config: Path, cluster_key: str | None = None) -> dict:
    cfg = load_config(str(config))
    processed = Path(cfg["data"]["processed_dir"])
    if not processed.is_absolute():
        processed = ROOT / processed
    manifest = json.loads((processed / "data_manifest.json").read_text(encoding="utf-8"))
    best = train_classical_baselines(str(config))["_best"]
    test = load_processed(str(processed))["test"]
    predicted = joblib.load(best["path"]).predict(test["text"])
    metrics = binary_metrics(test["label"].values, predicted)
    view = {
        "split_policy": manifest["split_policy"],
        "raw_csv_sha256": manifest["raw_csv_sha256"],
        "splits": manifest["splits"],
        "review_dates": manifest.get("review_dates"),
        "hotels": manifest.get("hotels"),
        "selected_on_dev": {"model": best["name"], "dev_macro_f1": best["dev_macro_f1"]},
        "test": metrics,
        "test_macro_f1_ci95": bootstrap_ci(test["label"].values, predicted, seed=cfg["seed"]),
        "test_predictions": "".join(label[0] for label in predicted),
    }
    if cluster_key and cluster_key in test.columns:
        clusters = test[cluster_key].astype(str).to_numpy()
        low, high = _clustered_macro_f1_ci(test["label"].values, predicted, clusters, cfg["seed"])
        iid_low, iid_high = view["test_macro_f1_ci95"]
        iid_width = iid_high - iid_low
        view["clustered"] = {
            "key": cluster_key,
            "clusters": int(len(np.unique(clusters))),
            "rows": int(len(clusters)),
            "test_macro_f1_ci95": [low, high],
            "width_ratio_vs_iid": round((high - low) / iid_width, 3) if iid_width > 0 else None,
        }
    elif cluster_key:
        view["clustered"] = None
        print(f"no {cluster_key!r} column in the test frame of {config.name}; clustered interval skipped")
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cluster-key",
        default=None,
        help="column in the test frame holding a cluster label (for example 'hotel'); "
        "adds a cluster-bootstrap interval next to the row-level one",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT, help="output JSON (default: the committed artifact)")
    args = parser.parse_args()
    views = {name: split_view(config, args.cluster_key) for name, config in CONFIGS.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(views, indent=1) + "\n", encoding="utf-8")
    print(f"{'split':8} {'model':22} {'test rows':>9}  macro-F1 (95% CI)")
    for name, view in views.items():
        low, high = view["test_macro_f1_ci95"]
        print(f"{name:8} {view['selected_on_dev']['model']:22} {view['splits']['test']['rows']:>9,}  "
              f"{view['test']['macro_f1']:.4f} ({low:.4f}-{high:.4f})")
        clustered = view.get("clustered")
        if clustered:
            c_low, c_high = clustered["test_macro_f1_ci95"]
            print(f"{'':8} {'':22} {'':>9}  clustered over {clustered['clusters']} {clustered['key']}s: "
                  f"({c_low:.4f}-{c_high:.4f}), width x{clustered['width_ratio_vs_iid']}")
    print(f"-> {args.out}")
    if args.cluster_key:
        print("clustered intervals resample whole clusters; they describe a different draw of "
              f"{args.cluster_key}s, not different rows from the same ones")


if __name__ == "__main__":
    main()
