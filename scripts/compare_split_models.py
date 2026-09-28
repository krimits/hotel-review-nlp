"""Compare TF-IDF + Naive Bayes, DistilBERT and the scratch LoRA on the same clean test sets.

Phase 2 of the project. The comparisons and the rule for a meaningful gain are
fixed in docs/experiments/phase2_analysis_plan.md.

For each split it reads:
- the Naive Bayes test predictions, from docs/experiments/results/split_views.json;
- for each DistilBERT run, metrics.json, test_logits.npy and test_labels.npy,
  from <artifacts>/<split>/<run>/.

It refuses to compare unless every model saw the same test reviews in the same
order. Then it reports:
- per model: macro-F1 with a bootstrap interval;
- per pair: the paired bootstrap difference, an exact McNemar test and the
  plan's verdict;
- each model's drop from the random to the out-of-time test;
- the costs, from metrics.json and from <artifacts>/latency.json when present.

A full fine-tune trained with another seed, in <artifacts>/time/distilbert_seed<N>/,
is compared with Naive Bayes in the same way and reported beside the primary result.

    python scripts/compare_split_models.py --artifacts docs/experiments/results/distilbert_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewnlp.evaluation.metrics import (  # noqa: E402
    binary_metrics,
    bootstrap_ci,
    paired_bootstrap_diff,
)
from reviewnlp.evaluation.significance import mcnemar_exact  # noqa: E402

SPLIT_VIEWS = ROOT / "docs" / "experiments" / "results" / "split_views.json"
ARTIFACTS = ROOT / "docs" / "experiments" / "results" / "distilbert_v2"
OUTPUT = ROOT / "docs" / "experiments" / "results" / "model_comparison_v2.json"
PLAN = "docs/experiments/phase2_analysis_plan.md"
SPLITS = ("random", "time")
RUNS = {"distilbert": "DistilBERT, full fine-tune", "distilbert_lora_scratch": "DistilBERT, scratch LoRA"}
MODELS = {"tfidf_nb": "TF-IDF + Naive Bayes", **RUNS}
PAIRS = (("tfidf_nb", "distilbert"), ("tfidf_nb", "distilbert_lora_scratch"), ("distilbert", "distilbert_lora_scratch"))
PRIMARY = {"split": "time", "a": "tfidf_nb", "b": "distilbert"}
MEANINGFUL = 0.01
RESAMPLES = 2000
LABEL_VALUES = (0, 1)  # negative, positive, as in reviewnlp.llm.encoder_trainer


class ComparisonError(ValueError):
    """The models did not see the same test reviews, so they cannot be compared."""


def nb_predictions(view: dict) -> np.ndarray:
    return np.array([{"n": 0, "p": 1}[code] for code in view["test_predictions"]])


def load_run(directory: Path) -> dict:
    if not (directory / "metrics.json").exists():
        raise ComparisonError(f"missing run: {directory}")
    return {"metrics": json.loads((directory / "metrics.json").read_text(encoding="utf-8")),
            "predictions": np.load(directory / "test_logits.npy").argmax(axis=1),
            "labels": np.load(directory / "test_labels.npy")}


def check_same_test(split: str, view: dict, runs: dict[str, dict]) -> np.ndarray:
    """The gold labels, once every run is shown to hold the Naive Bayes test set in the same order."""
    expected = view["splits"]["test"]
    labels = None
    for name, run in runs.items():
        if run["metrics"]["data"]["test"] != expected:
            raise ComparisonError(f"{split}/{name}: the test fingerprint differs from split_views.json")
        if labels is None:
            labels = run["labels"]
        elif not np.array_equal(labels, run["labels"]):
            raise ComparisonError(f"{split}/{name}: the test labels differ from the other runs")
    counts = {"rows": len(labels), "negative": int((labels == 0).sum()), "positive": int((labels == 1).sum())}
    if counts != {key: expected[key] for key in counts} or len(view["test_predictions"]) != len(labels):
        raise ComparisonError(f"{split}: the labels do not match the fingerprint's counts")
    recomputed = binary_metrics(labels, nb_predictions(view), label_values=LABEL_VALUES)["macro_f1"]
    if abs(recomputed - view["test"]["macro_f1"]) > 1e-4:
        raise ComparisonError(f"{split}: Naive Bayes scores {recomputed} on these labels, not "
                              f"{view['test']['macro_f1']} as recorded")
    return labels


def verdict(a: str, b: str, difference: float, low: float, high: float) -> dict:
    """The plan's rule: the paired interval excludes zero and the gain is at least one macro-F1 point."""
    if low <= 0 <= high:
        return {"better": None, "gain": "no clear difference"}
    return {"better": b if difference > 0 else a,
            "gain": "meaningful" if abs(difference) >= MEANINGFUL else "clear but small"}


def compare_split(split: str, view: dict, runs: dict[str, dict], seed: int = 0, resamples: int = RESAMPLES) -> dict:
    labels = check_same_test(split, view, runs)
    predictions = {"tfidf_nb": nb_predictions(view), **{name: run["predictions"] for name, run in runs.items()}}
    models = {}
    for name, predicted in predictions.items():
        metrics = binary_metrics(labels, predicted, label_values=LABEL_VALUES)
        models[name] = {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                        "macro_f1_ci95": list(bootstrap_ci(labels, predicted, LABEL_VALUES, resamples, seed)),
                        "per_class": metrics["per_class"], "confusion_matrix": metrics["confusion_matrix"]}
    pairs = {f"{b} vs {a}": compare_pair(labels, predictions, a, b, seed, resamples) for a, b in PAIRS}
    return {"test": view["splits"]["test"], "models": models, "pairs": pairs}


def compare_pair(labels: np.ndarray, predictions: dict[str, np.ndarray], a: str, b: str, seed: int = 0,
                 resamples: int = RESAMPLES) -> dict:
    difference, low, high = paired_bootstrap_diff(labels, predictions[a], predictions[b], LABEL_VALUES,
                                                  resamples, seed)
    a_wrong_b_right = int(np.sum((predictions[a] != labels) & (predictions[b] == labels)))
    a_right_b_wrong = int(np.sum((predictions[a] == labels) & (predictions[b] != labels)))
    return {"a": a, "b": b, "macro_f1_difference_b_minus_a": difference, "paired_ci95": [low, high],
            "mcnemar": {"a_wrong_b_right": a_wrong_b_right, "a_right_b_wrong": a_right_b_wrong,
                        **mcnemar_exact(a_wrong_b_right, a_right_b_wrong)},
            "verdict": verdict(a, b, difference, low, high)}


def other_seeds(artifacts: Path, split_views: dict, seed: int = 0, resamples: int = RESAMPLES) -> dict:
    """The primary comparison again, for each full fine-tune trained with another seed."""
    split, view, results = PRIMARY["split"], split_views[PRIMARY["split"]], {}
    for directory in sorted((artifacts / split).glob(f"{PRIMARY['b']}_seed*")):
        run = load_run(directory)
        labels = check_same_test(split, view, {directory.name: run})
        predictions = {PRIMARY["a"]: nb_predictions(view), directory.name: run["predictions"]}
        results[directory.name] = {
            "seed": run["metrics"]["settings"]["seed"],
            "macro_f1": binary_metrics(labels, run["predictions"], label_values=LABEL_VALUES)["macro_f1"],
            **compare_pair(labels, predictions, PRIMARY["a"], directory.name, seed, resamples)}
    return results


def drops(results: dict) -> dict:
    """Macro-F1 on the random test minus the out-of-time test; different reviews, so not tested."""
    return {name: {"random": results["random"]["models"][name]["macro_f1"],
                   "random_ci95": results["random"]["models"][name]["macro_f1_ci95"],
                   "time": results["time"]["models"][name]["macro_f1"],
                   "time_ci95": results["time"]["models"][name]["macro_f1_ci95"],
                   "drop": round(results["random"]["models"][name]["macro_f1"]
                                 - results["time"]["models"][name]["macro_f1"], 4)}
            for name in MODELS}


def costs(runs: dict[str, dict[str, dict]], latency: dict | None) -> dict:
    training = {split: {name: {"training_minutes": round(run["metrics"]["training_seconds"] / 60, 1),
                               "peak_gpu_memory_mb": round(run["metrics"]["peak_cuda_memory_mb"]),
                               "trainable_params": run["metrics"]["trainable_params"],
                               "total_params": run["metrics"]["total_params"],
                               "best_dev_macro_f1": run["metrics"]["best_dev_macro_f1"]}
                        for name, run in split_runs.items()}
                for split, split_runs in runs.items()}
    return {"training": training, "inference": latency if latency is not None else "not measured"}


def compare(artifacts: Path, split_views: dict, seed: int = 0, resamples: int = RESAMPLES) -> dict:
    runs = {split: {name: load_run(artifacts / split / name) for name in RUNS} for split in SPLITS}
    results = {split: compare_split(split, split_views[split], runs[split], seed, resamples) for split in SPLITS}
    latency_path = artifacts / "latency.json"
    latency = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.exists() else None
    primary = results[PRIMARY["split"]]["pairs"][f"{PRIMARY['b']} vs {PRIMARY['a']}"]
    return {"plan": PLAN, "models": MODELS, "meaningful_gain": MEANINGFUL, "resamples": resamples, "seed": seed,
            "primary": {"split": PRIMARY["split"], **primary},
            "primary_other_seeds": other_seeds(artifacts, split_views, seed, resamples), "splits": results,
            "drop_random_to_time": drops(results), "cost": costs(runs, latency)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--split-views", type=Path, default=SPLIT_VIEWS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    split_views = json.loads(args.split_views.read_text(encoding="utf-8"))
    try:
        report = compare(args.artifacts, split_views, args.seed, args.resamples)
    except ComparisonError as error:
        raise SystemExit(str(error)) from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for split, result in report["splits"].items():
        for name, model in result["models"].items():
            print(f"{split:6} {MODELS[name]:28} macro-F1 {model['macro_f1']:.4f} {model['macro_f1_ci95']}")
        for pair, comparison in result["pairs"].items():
            print(f"{split:6} {pair:36} {comparison['macro_f1_difference_b_minus_a']:+.4f} "
                  f"{comparison['paired_ci95']} -> {comparison['verdict']}")
    primary = report["primary"]
    print(f"primary ({primary['split']}): {primary['macro_f1_difference_b_minus_a']:+.4f} "
          f"{primary['paired_ci95']} -> {primary['verdict']}")
    for name, comparison in report["primary_other_seeds"].items():
        print(f"primary, {name}: {comparison['macro_f1_difference_b_minus_a']:+.4f} "
              f"{comparison['paired_ci95']} -> {comparison['verdict']}")


if __name__ == "__main__":
    main()
