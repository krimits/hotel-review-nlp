"""Planning input for the triage-diagnosis protocol: how widely the decision metric can vary on 500 reviews.

Simulates the inverse-probability-weighted (IPW) urgent recall under four allocations of a 500-review test
set across the strata H, N and B (see population.py). Population sizes are the measured ones. Urgent rates
come from the pilot's 200 random texts: 23 of 39 with a trigger word, 7 of 161 without; B is assumed 0.01.
They cover only cleanliness, pests and air conditioning, the urgent groups the pilot labelled, so the real
urgent rate is somewhat higher and these spreads are a little wide. One urgent problem per review.

    .venv/Scripts/python -I docs/experiments/triage_diagnosis/planning/power.py
"""
from __future__ import annotations

import numpy as np

N = {"H": 63325, "N": 285436, "B": 166412}
RATE = {"H": 23 / 39, "N": 7 / 161, "B": 0.01}
ALLOCATIONS = ({"H": 72, "N": 249, "B": 179},    # proportional to the population, i.e. a simple random draw
               {"H": 250, "N": 175, "B": 75},    # 50/35/15, as an outside review proposed
               {"H": 150, "N": 300, "B": 50},    # the draft protocol's allocation
               {"H": 100, "N": 375, "B": 25})
SCENARIOS = {"0.90 in every stratum": {"H": .90, "N": .90, "B": .90},
             "0.95 with a trigger word, 0.75 without": {"H": .95, "N": .75, "B": .75}}


def simulate(allocation: dict, recall: dict, sims: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    weighted, unweighted, support = [], [], []
    for _ in range(sims):
        caught_w = urgent_w = caught_u = urgent_u = 0.0
        for stratum, n in allocation.items():
            urgent = rng.binomial(n, RATE[stratum])
            caught = rng.binomial(urgent, recall[stratum])
            weight = N[stratum] / n
            caught_w, urgent_w = caught_w + weight * caught, urgent_w + weight * urgent
            caught_u, urgent_u = caught_u + caught, urgent_u + urgent
        if urgent_w:
            weighted.append(caught_w / urgent_w)
            unweighted.append(caught_u / urgent_u)
            support.append(urgent_u)
    weighted = np.array(weighted)
    truth = sum(N[s] * RATE[s] * recall[s] for s in N) / sum(N[s] * RATE[s] for s in N)
    low, high = np.percentile(weighted, [2.5, 97.5])
    return {"truth": truth, "ipw_mean": weighted.mean(), "ipw_2.5": low, "ipw_97.5": high,
            "unweighted_mean": float(np.mean(unweighted)), "urgent_in_sample": float(np.mean(support))}


def main() -> None:
    for name, recall in SCENARIOS.items():
        print("assumed recall:", name)
        for allocation in ALLOCATIONS:
            r = simulate(allocation, recall)
            print(f"  {allocation}: true {r['truth']:.3f} | IPW {r['ipw_mean']:.3f} "
                  f"[{r['ipw_2.5']:.3f}, {r['ipw_97.5']:.3f}] | unweighted {r['unweighted_mean']:.3f} "
                  f"| urgent in sample {r['urgent_in_sample']:.0f}")


if __name__ == "__main__":
    main()
