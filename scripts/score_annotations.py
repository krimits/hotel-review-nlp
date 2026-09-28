"""Score the pilot labels of the complaint lexicon (docs/annotation/pilot_protocol.md).

Two steps, in this order:

    python scripts/score_annotations.py agreement --key key.csv --sheet-a sheet_A.csv \
        --sheet-b sheet_B.csv --out docs/case_study/results/annotation
    python scripts/score_annotations.py metrics --key key.csv --sheet-a sheet_A.csv \
        --adjudicated adjudicated.csv --out docs/case_study/results/annotation

- `agreement` reads the two sheets as they were handed in and computes Cohen's
  kappa per topic, before the annotators discuss anything. Its output is
  committed first.
- `metrics` takes the agreed labels for the doubly labelled texts and annotator
  A's for the rest. It computes precision and recall with the estimators and
  thresholds of the protocol.

A submission is refused as a whole if any label is blank or unknown, if `done`
is missing, or if an item is missing, repeated or unknown. The outputs hold ids
and labels only: no review text and no notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewnlp.evaluation.metrics import newcombe_interval, wilson_interval  # noqa: E402

TOPICS = ("bathroom", "cleanliness", "air_conditioning", "pests", "responsiveness")
PERIODS = ("base", "recent")
VALUES = {"1": 1, "0": 0, "unsure": "unsure"}
MIN_DENOMINATOR = 10
MIN_POSITIVES = 5
RESAMPLES = 2000


class SubmissionError(ValueError):
    """A sheet that cannot be scored. The message lists every problem found."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_key(path: Path) -> pd.DataFrame:
    key = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" in key.columns:
        raise SubmissionError("key.csv must not hold review text")
    for column in ["item", "in_R", "in_B", *(f"lex_{topic}" for topic in TOPICS)]:
        key[column] = key[column].astype(int)
    key["drawn_for"] = key["drawn_for"].map(lambda value: set(filter(None, value.split(";"))))
    return key.set_index("item", drop=False)


def check_sheet(sheet: pd.DataFrame, expected: set[int], name: str) -> pd.DataFrame:
    """Labels indexed by item, each 1, 0 or 'unsure'; refuses the whole sheet on any problem."""
    missing_columns = [column for column in ("item", *TOPICS, "done") if column not in sheet.columns]
    if missing_columns:
        raise SubmissionError(f"{name}: missing columns {missing_columns}")
    problems, labels, seen = [], {}, set()
    for record in sheet.to_dict("records"):
        raw_item = str(record["item"]).strip()
        if not raw_item.isdigit():
            problems.append(f"item {raw_item!r} is not a number")
            continue
        item = int(raw_item)
        if item in seen:
            problems.append(f"item {item} appears twice")
        elif item not in expected:
            problems.append(f"item {item} is not in this sheet's sample")
        seen.add(item)
        row = {}
        for topic in TOPICS:
            value = str(record[topic]).strip().lower()
            if value not in VALUES:
                problems.append(f"item {item}: {topic} is {record[topic]!r}, not 1, 0 or unsure")
            row[topic] = VALUES.get(value)
        if str(record["done"]).strip().lower() != "yes":
            problems.append(f"item {item}: done is {record['done']!r}, not yes")
        labels[item] = row
    problems += [f"item {item} is missing" for item in sorted(expected - seen)]
    if problems:
        listed = "\n".join(problems[:60]) + (f"\n... and {len(problems) - 60} more" if len(problems) > 60 else "")
        raise SubmissionError(f"{name}: {len(problems)} problem(s), nothing scored:\n{listed}")
    return pd.DataFrame.from_dict(labels, orient="index")[list(TOPICS)].sort_index()


def read_sheet(path: Path, expected: set[int], name: str) -> pd.DataFrame:
    return check_sheet(pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig"), expected, name)


def _kappa(x: np.ndarray, y: np.ndarray) -> float:
    observed = float(np.mean(x == y))
    expected = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return (observed - expected) / (1 - expected)


def _both_labels_used(x: np.ndarray, y: np.ndarray) -> bool:
    return 0 < x.sum() < len(x) and 0 < y.sum() < len(y)


def kappa_report(a: pd.Series, b: pd.Series, seed: int = 0, resamples: int = RESAMPLES) -> dict:
    """Cohen's kappa on the texts both annotators labelled 1 or 0, with the protocol's outcomes."""
    counted = a.isin([0, 1]) & b.isin([0, 1])
    x, y = a[counted].astype(int).to_numpy(), b[counted].astype(int).to_numpy()
    report = {"counted": int(len(x)), "left_out_unsure": int((~counted).sum()),
              "positives": {"A": int(x.sum()), "B": int(y.sum())},
              "table_rows_A_columns_B": [[int(np.sum((x == i) & (y == j))) for j in (0, 1)] for i in (0, 1)],
              "raw_agreement": round(float(np.mean(x == y)), 4) if len(x) else None}
    if not len(x) or not _both_labels_used(x, y):
        return {**report, "status": "undefined"}
    if min(x.sum(), y.sum()) < MIN_POSITIVES:
        return {**report, "status": "too few positives"}
    rng = np.random.default_rng(seed)
    values, dropped = [], 0
    for _ in range(resamples):
        rows = rng.integers(0, len(x), len(x))
        if _both_labels_used(x[rows], y[rows]):
            values.append(_kappa(x[rows], y[rows]))
        else:
            dropped += 1
    low, high = np.quantile(values, [0.025, 0.975])
    return {**report, "status": "ok", "kappa": round(_kappa(x, y), 4),
            "bootstrap95": [round(float(low), 4), round(float(high), 4)],
            "resamples": resamples, "resamples_dropped_undefined": dropped}


def _proportion(numerator: int, denominator: int) -> dict:
    result = {"numerator": int(numerator), "denominator": int(denominator)}
    if denominator < MIN_DENOMINATOR:
        return {**result, "status": "insufficient sample"}
    low, high = wilson_interval(numerator, denominator)
    return {**result, "status": "ok", "estimate": round(numerator / denominator, 4),
            "wilson95": [round(low, 4), round(high, 4)]}


def _change(base: dict, recent: dict) -> dict:
    if base["status"] != "ok" or recent["status"] != "ok":
        return {"status": "not reported: a period is below the threshold"}
    low, high = newcombe_interval(base["numerator"], base["denominator"], recent["numerator"], recent["denominator"])
    difference = recent["numerator"] / recent["denominator"] - base["numerator"] / base["denominator"]
    return {"status": "ok", "recent_minus_base": round(difference, 4), "newcombe95": [round(low, 4), round(high, 4)]}


def precision(key: pd.DataFrame, final: pd.DataFrame, topic: str) -> dict:
    """On (R ∩ A_t) ∪ M_t per period: texts drawn for another topic do not count."""
    result = {}
    for name in PERIODS:
        rows = key[(key["period"] == name)
                   & (((key["in_R"] == 1) & (key[f"lex_{topic}"] == 1)) | key["drawn_for"].map(lambda d: topic in d))]
        labels = final.loc[rows.index, topic]
        ones, zeros, unsure = int((labels == 1).sum()), int((labels == 0).sum()), int((labels == "unsure").sum())
        result[name] = {**_proportion(ones, ones + zeros), "unsure": unsure}
        if unsure and ones + zeros + unsure:
            result[name]["if_unsure_all_1"] = round((ones + unsure) / (ones + zeros + unsure), 4)
            result[name]["if_unsure_all_0"] = round(ones / (ones + zeros + unsure), 4)
    result["change"] = _change(result["base"], result["recent"])
    return result


def recall(key: pd.DataFrame, final: pd.DataFrame, topic: str) -> dict:
    """On the texts of R labelled 1 for the topic, per period; pooled when a period falls short."""
    result, totals = {}, np.zeros(2, dtype=int)
    for name in PERIODS:
        rows = key[(key["period"] == name) & (key["in_R"] == 1)]
        labels, matched = final.loc[rows.index, topic], rows[f"lex_{topic}"] == 1
        positives, unsure = labels == 1, labels == "unsure"
        found, total = int((positives & matched).sum()), int(positives.sum())
        totals += (found, total)
        result[name] = {**_proportion(found, total), "unsure": int(unsure.sum())}
        if unsure.any():
            result[name]["if_unsure_all_1"] = round((found + int((unsure & matched).sum()))
                                                    / (total + int(unsure.sum())), 4)
            result[name]["if_unsure_all_0"] = round(found / total, 4) if total else None
    if any(result[name]["status"] != "ok" for name in PERIODS):
        result["pooled"] = _proportion(*totals)
    result["change"] = _change(result["base"], result["recent"])
    return result


def agreement(key: pd.DataFrame, sheet_a: pd.DataFrame, sheet_b: pd.DataFrame, seed: int = 0) -> dict:
    double = sheet_b.index
    return {topic: kappa_report(sheet_a.loc[double, topic], sheet_b.loc[double, topic], seed) for topic in TOPICS}


def final_labels(sheet_a: pd.DataFrame, adjudicated: pd.DataFrame) -> pd.DataFrame:
    """The agreed labels for the doubly labelled texts, annotator A's for the rest."""
    final = sheet_a.copy()
    final.loc[adjudicated.index, list(TOPICS)] = adjudicated[list(TOPICS)]
    return final


def metrics(key: pd.DataFrame, final: pd.DataFrame) -> dict:
    return {topic: {"precision": precision(key, final, topic), "recall": recall(key, final, topic)}
            for topic in TOPICS}


def _labels_table(labels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [frame.assign(annotator=name).rename_axis("item").reset_index() for name, frame in labels.items()]
    return pd.concat(parts)[["item", "annotator", *TOPICS]]


def run_agreement(args: argparse.Namespace) -> None:
    key = read_key(args.key)
    sheet_a = read_sheet(args.sheet_a, set(key.index), "sheet A")
    sheet_b = read_sheet(args.sheet_b, set(key.index[key["in_B"] == 1]), "sheet B")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "docs/annotation/pilot_protocol.md", "step": "agreement, before any discussion",
              "sheets": {"A": {"sha256": _sha256(args.sheet_a), "items": len(sheet_a)},
                         "B": {"sha256": _sha256(args.sheet_b), "items": len(sheet_b)}},
              "topics": agreement(key, sheet_a, sheet_b, args.seed)}
    (args.out / "agreement.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    _labels_table({"A": sheet_a, "B": sheet_b}).to_csv(args.out / "labels_raw.csv", index=False)
    key.assign(drawn_for=key["drawn_for"].map(lambda d: ";".join(sorted(d)))).to_csv(args.out / "key.csv", index=False)
    for topic, result in report["topics"].items():
        print(f"{topic:18} {result['status']:18} kappa {result.get('kappa', '-')}  {result['positives']}")


def run_metrics(args: argparse.Namespace) -> None:
    key = read_key(args.key)
    sheet_a = read_sheet(args.sheet_a, set(key.index), "sheet A")
    adjudicated = read_sheet(args.adjudicated, set(key.index[key["in_B"] == 1]), "adjudicated")
    final = final_labels(sheet_a, adjudicated)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "docs/annotation/pilot_protocol.md",
              "labels": "adjudicated for the doubly labelled texts, annotator A for the rest",
              "adjudicated_sha256": _sha256(args.adjudicated),
              "thresholds": {"min_denominator": MIN_DENOMINATOR, "min_positives_for_kappa": MIN_POSITIVES},
              "topics": metrics(key, final)}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    _labels_table({"final": final}).to_csv(args.out / "labels_final.csv", index=False)
    for topic, result in report["topics"].items():
        for measure in ("precision", "recall"):
            cells = [f"{name} {result[measure][name].get('estimate', result[measure][name]['status'])}"
                     for name in PERIODS]
            print(f"{topic:18} {measure:9} " + "  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("agreement", help="kappa from the sheets as handed in")
    first.add_argument("--sheet-b", type=Path, required=True)
    second = steps.add_parser("metrics", help="precision and recall from the final labels")
    second.add_argument("--adjudicated", type=Path, required=True)
    for step in (first, second):
        step.add_argument("--key", type=Path, required=True)
        step.add_argument("--sheet-a", type=Path, required=True)
        step.add_argument("--out", type=Path, default=ROOT / "docs" / "case_study" / "results" / "annotation")
        step.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    step = run_agreement if args.step == "agreement" else run_metrics
    try:
        step(args)
    except SubmissionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
