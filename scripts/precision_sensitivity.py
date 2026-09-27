"""How far would the lexicon's precision have to fall to erase each rising complaint topic?

Main result, no labels needed. Within the same hotels the lexicon's rate goes
from L1 (base period) to L2 (recent period, at the base period's hotel mix).
If precision, the share of matches that are complaints about the topic, went
from p1 to p2, the genuine rates are p1 * L1 and p2 * L2. The genuine change
is zero when p2 / p1 = L1 / L2, so precision would have to fall by 1 - L1 / L2.

Supplementary and hypothetical. The counts from reading 20 matched quotes per
period give Beta posteriors for p1 and p2 (uniform prior) and a Katz interval
for p2 / p1. The output lists the assumptions this needs. The numbers do not
confirm that any trend is real.

    python scripts/precision_sensitivity.py --results docs/case_study/results/since_2016_02

Reads csv/04_within_hotel.csv and reading_counts.json in that folder and writes
precision_sensitivity.json next to them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSUMPTIONS = (
    "The quotes were read by Claude, the AI assistant that wrote the analysis, not by an independent annotator.",
    "The quotes were sampled from all hotels in each period, not from the compared hotels at the base "
    "period's hotel mix, so their precision may differ from the precision behind the within-hotel rates.",
    "Recall is unknown and taken as unchanged between the periods.",
    "Each count is binomial, with a uniform Beta(1, 1) prior on each precision.",
)
NOTE = ("Hypothetical sensitivity analysis under the assumptions listed. It shows how the reading bears on the "
        "precision fall that would erase a rise; it does not confirm that any trend is real.")


def needed_ratio(base_rate: float, recent_rate_base_mix: float) -> float:
    """The ratio p2 / p1 at which the genuine within-hotel change is zero."""
    return base_rate / recent_rate_base_mix


def katz_interval(x1: int, n1: int, x2: int, n2: int, z: float = 1.959964) -> tuple[float, float]:
    """Katz log interval for the ratio of two binomial proportions, (x2 / n2) / (x1 / n1)."""
    if x1 == 0 or x2 == 0:
        raise ValueError("the Katz interval needs at least one success in each sample")
    p1, p2 = x1 / n1, x2 / n2
    se = np.sqrt((1 - p1) / x1 + (1 - p2) / x2)
    centre = np.log(p2 / p1)
    return float(np.exp(centre - z * se)), float(np.exp(centre + z * se))


def beta_ratio_draws(x1: int, n1: int, x2: int, n2: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draws of p2 / p1 from independent Beta posteriors with uniform priors."""
    p1 = rng.beta(1 + x1, 1 + n1 - x1, draws)
    p2 = rng.beta(1 + x2, 1 + n2 - x2, draws)
    return p2 / p1


def sensitivity(within: pd.DataFrame, reading: dict, draws: int = 200_000, seed: int = 0) -> dict:
    counts = reading["counts"]
    missing = sorted(set(counts) - set(within["topic"]))
    if missing:
        raise ValueError(f"topics read but not in the within-hotel table: {missing}")
    rows = within.set_index("topic").loc[list(counts)]
    rng = np.random.default_rng(seed)
    main, supplementary = [], []
    for topic, row in rows.iterrows():
        (x1, n1), (x2, n2) = counts[topic]["base"], counts[topic]["recent"]
        ratio = needed_ratio(row["base_rate"], row["recent_rate_base_mix"])
        main.append({"topic": topic, "base_rate": float(row["base_rate"]),
                     "recent_rate_base_mix": float(row["recent_rate_base_mix"]),
                     "precision_ratio_that_erases_rise": round(ratio, 4),
                     "precision_fall_that_erases_rise": round(1 - ratio, 4),
                     "reading": {"base": [x1, n1], "recent": [x2, n2]}})
        simulated = beta_ratio_draws(x1, n1, x2, n2, draws, rng)
        low, high = katz_interval(x1, n1, x2, n2)
        supplementary.append({
            "topic": topic,
            "chance_precision_fell_that_far": round(float(np.mean(simulated <= ratio)), 4),
            "beta_ratio_interval95": [round(float(q), 3) for q in np.quantile(simulated, [0.025, 0.975])],
            "katz_ratio_interval95": [round(low, 3), round(high, 3)],
            "katz_interval_excludes_needed_ratio": bool(low > ratio),
        })
    return {"reader": reading["reader"], "rule": reading["rule"], "main": main,
            "supplementary": {"note": NOTE, "assumptions": list(ASSUMPTIONS), "draws": draws, "seed": seed,
                              "topics": supplementary}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True,
                        help="folder with csv/04_within_hotel.csv and reading_counts.json")
    parser.add_argument("--draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    within = pd.read_csv(args.results / "csv" / "04_within_hotel.csv")
    reading = json.loads((args.results / "reading_counts.json").read_text(encoding="utf-8"))
    report = sensitivity(within, reading, args.draws, args.seed)
    (args.results / "precision_sensitivity.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for main_row, extra in zip(report["main"], report["supplementary"]["topics"], strict=True):
        (x1, n1), (x2, n2) = main_row["reading"]["base"], main_row["reading"]["recent"]
        print(f"{main_row['topic']:22} fall that erases the rise {main_row['precision_fall_that_erases_rise']:6.1%}"
              f"   read {x1}/{n1} -> {x2}/{n2}   [hypothetical] chance {extra['chance_precision_fell_that_far']:.3f}"
              f"   Katz {extra['katz_ratio_interval95']}")


if __name__ == "__main__":
    main()
