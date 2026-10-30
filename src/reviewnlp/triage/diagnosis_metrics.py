"""The metric and the guard of the triage diagnosis protocol (docs/experiments/triage_diagnosis/PROTOCOL.md, §3).

One implementation for every use: the teacher's choice on the development set and the scoring run on the test
set. Labels are 0/1 frames, one row per review and one column per catalogue type, on the same index.

- A (review, gold urgent type) pair is **caught** when the prediction for that review holds at least one urgent
  type of the same department.
- A predicted urgent type is a **false urgent flag** when the gold holds no urgent type of its department.
- Both are inverse-probability weighted by review: each review carries its stratum weight `N_s / n_s`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def urgent_departments(types: Iterable[Mapping]) -> dict[str, str]:
    """Urgent type -> its department, from the catalogue's `types` entries."""
    return {entry["id"]: entry["department"] for entry in types if entry["priority"] == "urgent"}


def per_review(gold: pd.DataFrame, pred: pd.DataFrame, urgent: Mapping[str, str]) -> pd.DataFrame:
    """For each review of `gold`: its gold urgent pairs, the pairs caught, and the false urgent flags.

    A review missing from `pred` predicts nothing, so it catches nothing and raises no flag.
    """
    names = list(urgent)
    gold = gold[names].astype(int)
    pred = pred.reindex(gold.index)[names].fillna(0).astype(int)
    departments = sorted(set(urgent.values()))
    in_department = {d: [t for t in names if urgent[t] == d] for d in departments}
    gold_has = {d: gold[ts].max(axis=1) for d, ts in in_department.items()}
    pred_has = {d: pred[ts].max(axis=1) for d, ts in in_department.items()}
    caught = sum(gold[t] * pred_has[urgent[t]] for t in names)
    false = sum(pred[t] * (1 - gold_has[urgent[t]]) for t in names)
    return pd.DataFrame({"pairs": gold.sum(axis=1), "caught": caught, "false_urgent": false})


def department_urgent_recall(counts: pd.DataFrame, weights: pd.Series) -> float:
    """IPW ratio of caught pairs to all gold urgent pairs; NaN when there are no pairs."""
    weights = weights.reindex(counts.index)
    pairs = float((weights * counts["pairs"]).sum())
    return float((weights * counts["caught"]).sum()) / pairs if pairs else float("nan")


def false_urgent_per_100(counts: pd.DataFrame, weights: pd.Series) -> float:
    """IPW false urgent flags per 100 reviews."""
    weights = weights.reindex(counts.index)
    return 100.0 * float((weights * counts["false_urgent"]).sum()) / float(weights.sum())
