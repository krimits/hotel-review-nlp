"""Apply the BiLSTM clean-split protocol (version 3) to the BiLSTM runs.

docs/experiments/bilstm_clean_splits/PROTOCOL.md fixes, before any run:
- the primary comparison: the seed-42 BiLSTM against the seed-42 DistilBERT full
  fine-tune, macro-F1 on the `time` test, paired bootstrap (2000 resamples, seed 0);
- the decision rule, whose conditions are the RULE strings below, verbatim;
- seeds 43 and 44 on the `time` view, each placed in the rule, and the seed check;
- the measured frontier, with the DOMINANCE condition, verbatim.

It reads the BiLSTM runs from <artifacts>/<view>/bilstm/ and <artifacts>/time/bilstm_seed<N>/,
the DistilBERT runs from docs/experiments/results/distilbert_v2/time/, the Naive Bayes
predictions from split_views.json and, when given, the session's latency_frontier.json.
It refuses to compare runs that did not see the same test reviews in the same order.

    python scripts/compare_bilstm_clean.py --artifacts docs/experiments/results/bilstm_v2 \\
        --latency docs/experiments/results/bilstm_v2/latency_frontier.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import compare_split_models as phase2  # noqa: E402

from reviewnlp.evaluation.metrics import (  # noqa: E402
    binary_metrics,
    bootstrap_ci,
    paired_bootstrap_diff,
)

PROTOCOL = "docs/experiments/bilstm_clean_splits/PROTOCOL.md"
ARTIFACTS = ROOT / "docs" / "experiments" / "results" / "bilstm_v2"
DISTILBERT = ROOT / "docs" / "experiments" / "results" / "distilbert_v2"
OUTPUT = ARTIFACTS / "bilstm_comparison_v2.json"
VIEWS = ("random", "time", "hotel")
EXTRA_SEEDS = (43, 44)
RESAMPLES = 2000
RULE = (
    ("hi < 0 and d <= -0.01", "DistilBERT meaningfully better"),
    ("hi < 0 and d > -0.01", "DistilBERT better, clear but small"),
    ("lo <= 0 <= hi", "indistinguishable: the transformer claim is not confirmed on clean data"),
    ("lo > 0 and d < 0.01", "BiLSTM better, clear but small"),
    ("lo > 0 and d >= 0.01", "BiLSTM meaningfully better"),
)
DOMINANCE = "lat_a <= lat_b and hi >= 0 and (lat_a < 0.9 * lat_b or lo > 0)"
FRONTIER_MODELS = ("tfidf_nb", "bilstm", "distilbert", "distilbert_lora_scratch")
NO_BUILTINS = {"__builtins__": {}}


def branch(low: float, difference: float, high: float) -> str:
    """The one conclusion of the protocol's rule for d = BiLSTM - DistilBERT."""
    scope = {"lo": low, "d": difference, "hi": high}
    fired = [conclusion for condition, conclusion in RULE if eval(condition, NO_BUILTINS, scope)]  # noqa: S307
    if len(fired) != 1:
        raise ValueError(f"the rule fired {len(fired)} branches for lo={low} d={difference} hi={high}")
    return fired[0]


def dominates(lat_a: float, lat_b: float, low: float, high: float) -> bool:
    """[low, high] is the paired interval of macro-F1(A) - macro-F1(B)."""
    return bool(eval(DOMINANCE, NO_BUILTINS, {"lat_a": lat_a, "lat_b": lat_b, "lo": low, "hi": high}))  # noqa: S307


def scored(labels: np.ndarray, predictions: np.ndarray, resamples: int, seed: int) -> dict:
    metrics = binary_metrics(labels, predictions, label_values=phase2.LABEL_VALUES)
    return {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
            "macro_f1_ci95": list(bootstrap_ci(labels, predictions, phase2.LABEL_VALUES, resamples, seed)),
            "per_class": metrics["per_class"]}


def seed_of(run: dict, expected: int, name: str) -> int:
    seed = run["metrics"]["settings"]["seed"]
    if seed != expected:
        raise phase2.ComparisonError(f"{name} was trained with seed {seed}, not {expected}")
    return seed


def against_distilbert(labels, distilbert, bilstm, resamples, seed) -> dict:
    pair = phase2.compare_pair(labels, {"distilbert": distilbert, "bilstm": bilstm}, "distilbert", "bilstm",
                               seed, resamples)
    low, high = pair["paired_ci95"]
    return {"d_bilstm_minus_distilbert": pair["macro_f1_difference_b_minus_a"], "paired_ci95": [low, high],
            "mcnemar": pair["mcnemar"], "branch": branch(low, pair["macro_f1_difference_b_minus_a"], high)}


def measured_frontier(latency: dict, labels: np.ndarray, predictions: dict[str, np.ndarray],
                      resamples: int, seed: int) -> dict:
    lat = {name: latency["models"][name]["cpu"]["batch_1"]["ms_per_batch_median"] for name in FRONTIER_MODELS}
    pairs = {}
    for a in FRONTIER_MODELS:
        for b in FRONTIER_MODELS:
            if a == b:
                continue
            difference, low, high = paired_bootstrap_diff(labels, predictions[b], predictions[a],
                                                          phase2.LABEL_VALUES, resamples, seed)  # A - B
            pairs[f"{a} vs {b}"] = {"macro_f1_a_minus_b": difference, "paired_ci95": [low, high],
                                    "a_dominates_b": dominates(lat[a], lat[b], low, high)}
    dominated = {b for b in FRONTIER_MODELS for a in FRONTIER_MODELS if a != b and pairs[f"{a} vs {b}"]["a_dominates_b"]}
    return {"rule": DOMINANCE, "median_ms_per_review": lat,
            "size_mb": {name: latency["models"][name]["size_mb"] for name in FRONTIER_MODELS},
            "hardware": latency["hardware"], "pairs": pairs,
            "frontier": [name for name in FRONTIER_MODELS if name not in dominated]}


def compare(artifacts: Path, split_views: dict, latency: dict | None = None, distilbert: Path = DISTILBERT,
            resamples: int = RESAMPLES, seed: int = 0) -> dict:
    views = {}
    for view in VIEWS:
        run = phase2.load_run(artifacts / view / "bilstm")
        seed_of(run, 42, f"{view}/bilstm")
        labels = phase2.check_same_test(view, split_views[view], {"bilstm": run})
        nb = phase2.nb_predictions(split_views[view])
        views[view] = {"bilstm": scored(labels, run["predictions"], resamples, seed),
                       "bilstm vs tfidf_nb": phase2.compare_pair(labels, {"tfidf_nb": nb, "bilstm": run["predictions"]},
                                                                 "tfidf_nb", "bilstm", seed, resamples)}

    time_view = split_views["time"]
    runs = {"bilstm": phase2.load_run(artifacts / "time" / "bilstm"),
            "distilbert": phase2.load_run(distilbert / "time" / "distilbert"),
            "distilbert_lora_scratch": phase2.load_run(distilbert / "time" / "distilbert_lora_scratch")}
    for extra in EXTRA_SEEDS:
        name = f"bilstm_seed{extra}"
        runs[name] = phase2.load_run(artifacts / "time" / name)
        seed_of(runs[name], extra, f"time/{name}")
    labels = phase2.check_same_test("time", time_view, runs)
    distil = runs["distilbert"]["predictions"]

    primary = against_distilbert(labels, distil, runs["bilstm"]["predictions"], resamples, seed)
    seeds = {42: {"macro_f1": views["time"]["bilstm"]["macro_f1"], **primary}}
    for extra in EXTRA_SEEDS:
        predictions = runs[f"bilstm_seed{extra}"]["predictions"]
        seeds[extra] = {"macro_f1": binary_metrics(labels, predictions, label_values=phase2.LABEL_VALUES)["macro_f1"],
                        **against_distilbert(labels, distil, predictions, resamples, seed)}
    scores = [entry["macro_f1"] for entry in seeds.values()]
    same = len({entry["branch"] for entry in seeds.values()}) == 1
    seed_check = {"branches": {str(s): entry["branch"] for s, entry in seeds.items()}, "same_branch": same,
                  "statement": ("the conclusion holds across three training seeds of the BiLSTM" if same else
                                "the conclusion depends on the BiLSTM's seed"),
                  "macro_f1_mean": float(np.mean(scores)), "macro_f1_sd": float(np.std(scores, ddof=1))}

    result = {"protocol": PROTOCOL, "resamples": resamples, "seed": seed, "primary": {"split": "time", **primary},
              "seeds": {str(s): entry for s, entry in seeds.items()}, "seed_check": seed_check, "views": views,
              "drop_random_to_time": round(views["random"]["bilstm"]["macro_f1"] - views["time"]["bilstm"]["macro_f1"], 4),
              "selection_frozen": False, "promoted": False}
    if latency is not None:
        if latency["test"] != time_view["splits"]["test"]:
            raise phase2.ComparisonError("the timed sample is not from the time test set of split_views.json")
        predictions = {"tfidf_nb": phase2.nb_predictions(time_view), "bilstm": runs["bilstm"]["predictions"],
                       "distilbert": distil,
                       "distilbert_lora_scratch": runs["distilbert_lora_scratch"]["predictions"]}
        result["frontier"] = measured_frontier(latency, labels, predictions, resamples, seed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--distilbert", type=Path, default=DISTILBERT)
    parser.add_argument("--split-views", type=Path, default=phase2.SPLIT_VIEWS)
    parser.add_argument("--latency", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    latency = json.loads(args.latency.read_text(encoding="utf-8")) if args.latency else None
    try:
        result = compare(args.artifacts, json.loads(args.split_views.read_text(encoding="utf-8")), latency,
                         args.distilbert)
    except phase2.ComparisonError as error:
        raise SystemExit(str(error)) from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    primary = result["primary"]
    print(f"primary: d = {primary['d_bilstm_minus_distilbert']:+.4f} {primary['paired_ci95']} -> {primary['branch']}")
    print(f"seeds: {result['seed_check']['statement']}")
    if "frontier" in result:
        print(f"measured frontier: {', '.join(result['frontier']['frontier'])}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
