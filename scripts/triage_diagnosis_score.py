"""Choose the student's thresholds on development, then score the test set once (PROTOCOL.md version 6, §3–§4).

    python scripts/triage_diagnosis_score.py thresholds   # development only; writes student/thresholds_locked.json
    git add ... && git commit ...                         # the thresholds are committed before any test score
    python scripts/triage_diagnosis_score.py score        # refuses unless labels and thresholds are committed

`thresholds` follows §4: one threshold per priority group on the grid 0.05, 0.10, ... 0.95. Urgent: the smallest
value whose development false urgent flags per 100 (IPW) are at most 5, else the largest. Routine and structural:
the value that maximises development micro-F1 within the group against `reference_chatgpt`; on a tie, the
smallest. A type is emitted when the sigmoid of its logit is at least its group's threshold.

`score` applies §3 and the rule of §4 to the 500 test reviews against `reference_chatgpt`: IPW department-urgent
recall `rec`, its bootstrap lower bound `lo` (2,000 resamples, seed 0, reviews resampled within each stratum)
and false urgent flags per 100 `fa`. It then computes the "reported, not deciding" items it can: per-type recall
and precision, strict type recall, department accuracy on emitted problems, the unweighted and per-stratum
recall, the teacher and `complaint-lexicon-v1` baselines on the same reviews, the second rater's agreement, the
teacher choice, and the owner's check. It writes student/score.json once and never again.

The clause evidence and the CPU latency need the student's weights, which notebook 29 did not keep; they are
listed as not produced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import triage_diagnosis_sample as sample  # noqa: E402

from reviewnlp.triage import diagnosis_metrics as metrics  # noqa: E402

FOLDER = sample.FOLDER
STUDENT, LABELS, SAMPLES = FOLDER / "student", FOLDER / "labels", FOLDER / "samples"
GRID = [round(0.05 * i, 2) for i in range(1, 20)]
F, L, TARGET = 5.0, 0.80, 0.90  # PROTOCOL.md §3–§4
BOOTSTRAP, BOOTSTRAP_SEED = 2000, 0
RULE = [  # PROTOCOL.md §4, in order
    ("rec >= 0.90 and lo >= 0.80 and fa <= 5", "Met"),
    ("rec >= 0.90 and lo >= 0.80 and fa > 5", "Recall met, over budget"),
    ("rec >= 0.90 and lo < 0.80", "Point estimate met, not established"),
    ("rec < 0.90", "Not met"),
]
EMISSION = "a type is emitted when sigmoid(logit) >= its priority group's threshold"


class Stop(RuntimeError):
    """The step cannot run; nothing is written."""


def catalogue() -> dict:
    return yaml.safe_load(sample.CATALOGUE.read_text(encoding="utf-8"))


def groups(data: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"urgent": [], "routine": [], "structural": []}
    for entry in data["types"]:
        out[entry["priority"]].append(entry["id"])
    return out


def emit(logits: pd.DataFrame, thresholds: dict[str, float], data: dict) -> pd.DataFrame:
    """0/1 emitted types from logits, by each type's group threshold."""
    by_type = {t: thresholds[g] for g, ts in groups(data).items() for t in ts}
    types = [entry["id"] for entry in data["types"]]
    probs = 1.0 / (1.0 + np.exp(-logits[types].to_numpy(dtype=float)))
    return pd.DataFrame((probs >= np.array([by_type[t] for t in types])).astype(int), index=logits.index, columns=types)


def micro_f1(gold: pd.DataFrame, pred: pd.DataFrame, columns: list[str]) -> float:
    g, p = gold[columns].to_numpy(), pred[columns].to_numpy()
    tp, fp, fn = int((g & p).sum()), int(((1 - g) & p).sum()), int((g & (1 - p)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def load_indexed(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("raw_row")


def choose_thresholds(logits: pd.DataFrame, gold: pd.DataFrame, weights: pd.Series, data: dict) -> dict:
    """§4 on the development set: per group, the grid search and the chosen value, with the values seen."""
    urgent = metrics.urgent_departments(data["types"])
    by_group = groups(data)
    seen: dict[str, dict] = {"urgent": {}, "routine": {}, "structural": {}}
    for value in GRID:
        pred = emit(logits, dict.fromkeys(by_group, value), data)
        counts = metrics.per_review(gold, pred, urgent)
        seen["urgent"][value] = round(metrics.false_urgent_per_100(counts, weights), 4)
        for group in ("routine", "structural"):
            seen[group][value] = round(micro_f1(gold, pred, by_group[group]), 6)
    within = [v for v in GRID if seen["urgent"][v] <= F]
    chosen = {"urgent": min(within) if within else max(GRID)}
    for group in ("routine", "structural"):
        best = max(seen[group].values())
        chosen[group] = min(v for v in GRID if seen[group][v] == best)
    return {"thresholds": chosen, "development_fa_per_100_by_urgent_value": seen["urgent"],
            "development_micro_f1_by_value": {g: seen[g] for g in ("routine", "structural")},
            "emission": EMISSION, "tie": "the smallest grid value among equal micro-F1", "grid": GRID}


def committed(path: Path) -> bool:
    """Tracked by git with no uncommitted change."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=ROOT)
    return tracked.returncode == 0 and clean.returncode == 0


def thresholds(student: Path = STUDENT, labels: Path = LABELS, samples: Path = SAMPLES) -> dict:
    out = student / "thresholds_locked.json"
    if out.exists():
        raise Stop(f"{out} exists; thresholds are chosen once")
    data = catalogue()
    key = pd.read_csv(samples / "key.csv")
    dev = key[key["set"] == "development"].set_index("raw_row")
    logits = load_indexed(student / "logits_development.csv").loc[dev.index]
    gold = load_indexed(labels / "reference_chatgpt.csv").loc[dev.index]
    result = choose_thresholds(logits, gold, dev["weight"], data)
    result.update({"set": "development", "reviews": len(dev), "against": "reference_chatgpt",
                   "code_commit": sample.pilot.git_commit()})
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return result


def bootstrap_lower(counts: pd.DataFrame, weights: pd.Series, strata: pd.Series, reps: int = BOOTSTRAP,
                    seed: int = BOOTSTRAP_SEED) -> tuple[float, int]:
    """2.5th percentile of IPW recall over resamples of reviews within each stratum, and resamples with no pair."""
    rng = np.random.default_rng(seed)
    caught = (weights * counts["caught"]).to_numpy()
    pairs = (weights * counts["pairs"]).to_numpy()
    positions = {s: np.flatnonzero(strata.to_numpy() == s) for s in sorted(strata.unique())}
    values, empty = [], 0
    for _ in range(reps):
        picked = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in positions.values()])
        denominator = pairs[picked].sum()
        if denominator == 0:
            empty += 1
            continue
        values.append(caught[picked].sum() / denominator)
    return float(np.percentile(values, 2.5)), empty


def outcome(rec: float, lo: float, fa: float) -> str:
    # eval only sees the four fixed RULE strings, with these three numbers.
    fired = [name for condition, name in RULE if eval(condition, {}, {"rec": rec, "lo": lo, "fa": fa})]  # noqa: S307
    assert len(fired) == 1, fired
    return fired[0]


def cohen_kappa(a: pd.Series, b: pd.Series) -> float | None:
    a, b = a.astype(int).to_numpy(), b.astype(int).to_numpy()
    observed = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    expected = float(pa * pb + (1 - pa) * (1 - pb))
    return None if expected == 1.0 else round((observed - expected) / (1 - expected), 4)


def agreement(rater: pd.DataFrame, gold: pd.DataFrame, types: list[str]) -> dict:
    rater = rater.loc[gold.index]
    return {t: {"kappa": cohen_kappa(rater[t], gold[t]), "gold_positive": int(gold[t].sum()),
                "rater_positive": int(rater[t].sum()), "reviews": len(gold)} for t in types}


def primary(pred: pd.DataFrame, gold: pd.DataFrame, weights: pd.Series, urgent: dict) -> dict:
    counts = metrics.per_review(gold, pred, urgent)
    return {"rec": round(metrics.department_urgent_recall(counts, weights), 6),
            "fa": round(metrics.false_urgent_per_100(counts, weights), 4),
            "pairs": int(counts["pairs"].sum()), "caught": int(counts["caught"].sum()),
            "unweighted_recall": round(int(counts["caught"].sum()) / max(int(counts["pairs"].sum()), 1), 6)}


def reported(pred: pd.DataFrame, gold: pd.DataFrame, weights: pd.Series, strata: pd.Series, data: dict) -> dict:
    types = [entry["id"] for entry in data["types"]]
    urgent = metrics.urgent_departments(data["types"])
    department = {entry["id"]: entry["department"] for entry in data["types"]}
    per_type = {}
    for t in types:
        support, emitted = int(gold[t].sum()), int(pred[t].sum())
        both = gold[t] * pred[t]
        row = {"support": support, "emitted": emitted, "true_positive": int(both.sum())}
        if support >= 10:
            row["recall"] = round(float((weights * both).sum() / (weights * gold[t]).sum()), 4)
        if emitted >= 10:
            row["precision"] = round(float((weights * both).sum() / (weights * pred[t]).sum()), 4)
        per_type[t] = row
    urgent_types = list(urgent)
    strict = (gold[urgent_types] * pred[urgent_types]).sum(axis=1)
    pairs = gold[urgent_types].sum(axis=1)
    gold_departments = gold[types].apply(lambda r: {department[t] for t in types if r[t]}, axis=1)
    right = sum(weights[i] * sum(1 for t in types if pred.at[i, t] and department[t] in gold_departments[i])
                for i in gold.index)
    emitted_w = float((weights * pred[types].sum(axis=1)).sum())
    counts = metrics.per_review(gold, pred, urgent)
    by_stratum = {s: {"pairs": int(counts.loc[strata == s, "pairs"].sum()),
                      "caught": int(counts.loc[strata == s, "caught"].sum())} for s in sorted(strata.unique())}
    for row in by_stratum.values():
        row["recall"] = round(row["caught"] / row["pairs"], 4) if row["pairs"] else None
    return {"per_type": per_type,
            "strict_type_recall": round(float((weights * strict).sum() / (weights * pairs).sum()), 6),
            "department_accuracy_on_emitted": round(right / emitted_w, 6) if emitted_w else None,
            "recall_per_stratum": by_stratum}


def lexicon_prediction(texts: pd.Series, data: dict) -> pd.DataFrame:
    """complaint-lexicon-v1 on each text, its topics answered `yes` mapped to types by the catalogue."""
    from reviewnlp.triage.lexicon import LexiconClassifier

    classifier, mapping = LexiconClassifier(), data["lexicon_mapping"]
    types = [entry["id"] for entry in data["types"]]
    rows = []
    for text in texts:
        result = classifier.classify(text)
        hit = {mapping[t.topic] for t in result.topics if t.answer == "yes"}
        rows.append([int(t in hit) for t in types])
    return pd.DataFrame(rows, index=texts.index, columns=types)


def test_texts(key: pd.DataFrame, csv: Path) -> pd.Series:
    raw = pd.read_csv(csv, usecols=["Negative_Review", "Positive_Review"]).fillna("").astype(str)
    rows = raw.loc[key.index]
    texts = pd.Series([sample.whole_review(n, p) for n, p in zip(rows["Negative_Review"], rows["Positive_Review"],
                                                                 strict=True)], index=key.index)
    if [sample.text_hash(t) for t in texts] != key["text_sha256"].tolist():
        raise Stop("a rebuilt test text differs from its key; nothing scored")
    return texts


def score(csv: Path, student: Path = STUDENT, labels: Path = LABELS, samples: Path = SAMPLES,
          require_commits: bool = True) -> dict:
    out = student / "score.json"
    if out.exists():
        raise Stop(f"{out} exists; the test set is scored once")
    needed = [student / "thresholds_locked.json", student / "logits_test.csv", labels / "reference_chatgpt.csv",
              labels / "owner_check.csv", labels / "second_rater.csv", labels / "second_rater_check.csv",
              labels / "teacher_haiku_test.csv"]
    if require_commits:
        missing = [p.name for p in needed if not committed(p)]
        if missing:
            raise Stop(f"not committed: {missing}; labels and thresholds are committed before the test is scored")
    data = catalogue()
    types = [entry["id"] for entry in data["types"]]
    urgent = metrics.urgent_departments(data["types"])
    chosen = json.loads((student / "thresholds_locked.json").read_text(encoding="utf-8"))["thresholds"]
    key = pd.read_csv(samples / "key.csv")
    test = key[key["set"] == "test"].set_index("raw_row")
    logits = load_indexed(student / "logits_test.csv").reindex(test.index)
    complete = len(test) == 500 and bool(logits[types].notna().all().all())
    gold = load_indexed(labels / "reference_chatgpt.csv").loc[test.index]
    weights, strata = test["weight"], test["stratum"]
    pred = emit(logits, chosen, data)
    main = primary(pred, gold, weights, urgent)
    lo, empty = bootstrap_lower(metrics.per_review(gold, pred, urgent), weights, strata)
    main.update({"lo": round(lo, 6), "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                                                   "resamples_without_pairs": empty}})
    main["outcome"] = outcome(main["rec"], main["lo"], main["fa"]) if complete else "not decided: incomplete run"

    teacher = load_indexed(labels / "teacher_haiku_test.csv").loc[test.index]
    lexicon = lexicon_prediction(test_texts(test, csv), data)
    check_key = pd.read_csv(samples / "check_key.csv").set_index("raw_row")
    check_pred = emit(load_indexed(student / "logits_check.csv").loc[check_key.index], chosen, data)
    owner = load_indexed(labels / "owner_check.csv").loc[check_key.index]
    eight_hundred = key.set_index("raw_row").index
    choice = json.loads((labels / "teacher_choice.json").read_text(encoding="utf-8"))
    result = {
        "protocol_sha256": sample.trends._sha256(sample.PROTOCOL), "code_commit": sample.pilot.git_commit(),
        "thresholds": chosen, "emission": EMISSION, "test_reviews": len(test), "complete": complete,
        "decision": main, "rule": [condition for condition, _ in RULE],
        "reported": {
            "student": reported(pred, gold, weights, strata, data),
            "baselines": {"teacher_claude_haiku_5_5": primary(teacher[types], gold, weights, urgent),
                          "complaint_lexicon_v1": primary(lexicon, gold, weights, urgent)},
            "agreement_second_rater_vs_reference_800": agreement(
                load_indexed(labels / "second_rater.csv"), load_indexed(labels / "reference_chatgpt.csv")
                .loc[eight_hundred], types),
            "teacher_choice": {k: choice[k] for k in ("rec_dev", "fa_dev", "cost_usd", "sonnet_lead", "model")},
            "owner_check_100": {
                "student_vs_owner": primary(check_pred, owner, check_key["weight"], urgent),
                "second_rater_vs_owner": agreement(load_indexed(labels / "second_rater_check.csv"), owner, types)},
            "not_produced": {"evidence": "needs the student's weights, which notebook 29 did not keep",
                             "cpu_latency": "needs the student's weights, which notebook 29 did not keep"},
        },
    }
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("thresholds", help="choose the three thresholds on the development set")
    second = steps.add_parser("score", help="score the test set once")
    second.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    args = parser.parse_args()
    try:
        if args.step == "thresholds":
            print("thresholds:", thresholds()["thresholds"])
        else:
            if sample.trends._sha256(args.csv) != sample.RAW_SHA256:
                raise Stop(f"{args.csv} is not the Kaggle file the protocol fixes")
            decision = score(args.csv)["decision"]
            print({k: decision[k] for k in ("rec", "lo", "fa", "outcome")})
    except Stop as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
