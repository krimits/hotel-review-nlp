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
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewnlp.baselines.classical import train_classical_baselines  # noqa: E402
from reviewnlp.data.preprocess import load_processed  # noqa: E402
from reviewnlp.evaluation.metrics import binary_metrics, bootstrap_ci  # noqa: E402
from reviewnlp.utils.seed import load_config  # noqa: E402

CONFIGS = {
    "random": ROOT / "configs" / "baselines.yaml",
    "time": ROOT / "configs" / "baselines_time.yaml",
    "hotel": ROOT / "configs" / "baselines_hotel.yaml",
}
OUTPUT = ROOT / "docs" / "experiments" / "results" / "split_views.json"


def split_view(config: Path) -> dict:
    cfg = load_config(str(config))
    processed = Path(cfg["data"]["processed_dir"])
    if not processed.is_absolute():
        processed = ROOT / processed
    manifest = json.loads((processed / "data_manifest.json").read_text(encoding="utf-8"))
    best = train_classical_baselines(str(config))["_best"]
    test = load_processed(str(processed))["test"]
    predicted = joblib.load(best["path"]).predict(test["text"])
    metrics = binary_metrics(test["label"].values, predicted)
    return {
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


def main() -> None:
    views = {name: split_view(config) for name, config in CONFIGS.items()}
    OUTPUT.write_text(json.dumps(views, indent=1) + "\n", encoding="utf-8")
    print(f"{'split':8} {'model':22} {'test rows':>9}  macro-F1 (95% CI)")
    for name, view in views.items():
        low, high = view["test_macro_f1_ci95"]
        print(f"{name:8} {view['selected_on_dev']['model']:22} {view['splits']['test']['rows']:>9,}  "
              f"{view['test']['macro_f1']:.4f} ({low:.4f}-{high:.4f})")
    print(f"-> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
