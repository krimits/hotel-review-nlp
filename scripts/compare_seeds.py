"""Apply the DistilBERT seed protocol: the full fine-tune against the LoRA, within each seed.

docs/experiments/distilbert_seed_robustness/PROTOCOL.md fixes, before any run:
- for seeds 42, 43 and 44 on the `time` test, d = macro-F1(full) - macro-F1(LoRA) with its
  paired bootstrap interval (2000 resamples, seed 0);
- the decision rule, whose conditions are the RULE strings below, verbatim;
- beside it, each full fine-tune seed against Naive Bayes with the phase 2 plan's verdict.
  Seed 42 stays the primary result of that plan.

The seed-42 runs are docs/experiments/results/distilbert_v2/time/distilbert{,_lora_scratch}/;
the others are .../time/distilbert_seed<N>/ and .../time/distilbert_lora_scratch_seed<N>/.

    python scripts/compare_seeds.py
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

from reviewnlp.evaluation.metrics import binary_metrics  # noqa: E402

PROTOCOL = "docs/experiments/distilbert_seed_robustness/PROTOCOL.md"
OUTPUT = ROOT / "docs" / "experiments" / "results" / "distilbert_seed_robustness.json"
SEEDS = (42, 43, 44)
RESAMPLES = 2000
RULE = (
    ("all(x > 0 for x in lo)", "the lead holds across seeds"),
    ("all(x > 0 for x in d) and any(x <= 0 for x in lo)", "ahead in every seed, within seed variation in at least one"),
    ("any(x <= 0 for x in d)", "no consistent lead across seeds"),
)
MEANINGFUL = 0.01


def run_names(seed: int) -> tuple[str, str]:
    suffix = "" if seed == 42 else f"_seed{seed}"
    return f"distilbert{suffix}", f"distilbert_lora_scratch{suffix}"


def outcome(lows: list[float], differences: list[float]) -> str:
    scope = {"lo": lows, "d": differences}
    fired = [text for condition, text in RULE
             if eval(condition, {"__builtins__": {}, "all": all, "any": any}, scope)]  # noqa: S307
    if len(fired) != 1:
        raise ValueError(f"the rule fired {len(fired)} branches for lo={lows} d={differences}")
    return fired[0]


def compare(artifacts: Path, split_views: dict, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    view = split_views["time"]
    runs = {}
    for training_seed in SEEDS:
        for name in run_names(training_seed):
            runs[name] = phase2.load_run(artifacts / "time" / name)
            if runs[name]["metrics"]["settings"]["seed"] != training_seed:
                raise phase2.ComparisonError(f"time/{name} was not trained with seed {training_seed}")
    labels = phase2.check_same_test("time", view, runs)
    nb = phase2.nb_predictions(view)

    per_seed = {}
    for training_seed in SEEDS:
        full, lora = run_names(training_seed)
        predictions = {"tfidf_nb": nb, full: runs[full]["predictions"], lora: runs[lora]["predictions"]}
        pair = phase2.compare_pair(labels, predictions, lora, full, seed, resamples)  # full - LoRA
        per_seed[str(training_seed)] = {
            "full_macro_f1": binary_metrics(labels, predictions[full], label_values=phase2.LABEL_VALUES)["macro_f1"],
            "lora_macro_f1": binary_metrics(labels, predictions[lora], label_values=phase2.LABEL_VALUES)["macro_f1"],
            "d_full_minus_lora": pair["macro_f1_difference_b_minus_a"], "paired_ci95": pair["paired_ci95"],
            "full_vs_tfidf_nb": phase2.compare_pair(labels, predictions, "tfidf_nb", full, seed, resamples),
            "lora_vs_tfidf_nb": phase2.compare_pair(labels, predictions, "tfidf_nb", lora, seed, resamples)}

    differences = [entry["d_full_minus_lora"] for entry in per_seed.values()]
    lows = [entry["paired_ci95"][0] for entry in per_seed.values()]
    conclusion = outcome(lows, differences)
    size = None
    if conclusion == RULE[0][1]:
        size = "meaningful" if all(d >= MEANINGFUL for d in differences) else "clear but small"
    spread = {arm: {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
              for arm, values in (("full", [e["full_macro_f1"] for e in per_seed.values()]),
                                  ("lora", [e["lora_macro_f1"] for e in per_seed.values()]))}
    return {"protocol": PROTOCOL, "resamples": resamples, "seed": seed, "seeds": per_seed,
            "outcome": conclusion, "size": size, "spread": spread,
            "phase2_primary_stays": "seed 42", "selection_frozen": False, "promoted": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=phase2.ARTIFACTS)
    parser.add_argument("--split-views", type=Path, default=phase2.SPLIT_VIEWS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = compare(args.artifacts, json.loads(args.split_views.read_text(encoding="utf-8")))
    except phase2.ComparisonError as error:
        raise SystemExit(str(error)) from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for training_seed, entry in result["seeds"].items():
        print(f"seed {training_seed}: d = {entry['d_full_minus_lora']:+.4f} {entry['paired_ci95']}")
    print(f"outcome: {result['outcome']}" + (f" ({result['size']})" if result["size"] else ""))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
