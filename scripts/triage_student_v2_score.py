"""Thresholds, the choice of variant, and the one test score of the triage student v2 protocol (PROTOCOL.md v1).

    python scripts/triage_student_v2_score.py thresholds   # development (and held-out) only
    git add ... && git commit ...                          # thresholds, figures and the choice before any test score
    python scripts/triage_student_v2_score.py score        # refuses unless they and every label are committed; once

`thresholds` applies §3 on the grid 0.05, 0.10, ... 0.95, against `reference_chatgpt_v2` on the 600 development
reviews, with the first protocol's code wherever the rule is the first protocol's:

- A and B: one threshold per priority group by the first protocol's §4 rule (urgent: the smallest value whose
  development false urgent flags per 100 are at most 5, else the largest; routine and structural: the value with
  the highest development micro-F1, ties to the smallest), on A's and on B's logits.
- C: B's logits, with B's routine and structural thresholds. Each urgent type gets the grid value that maximises
  its F1 against the teacher on the held-out reviews (ties to the smallest). Then, while the development `fa`
  exceeds 5, every urgent threshold rises by 0.05 together, each stopping at 0.95; the rise ends when `fa` is at
  most 5 or every urgent threshold is at 0.95.
- The choice: among the variants whose development `fa` is at most 5, the highest development `rec`; on a tie the
  simpler, in the order A, B, C; if none qualifies, A.

It writes student/thresholds_locked.json once: every variant's thresholds and development figures, the values the
grid searches saw, C's rise, and the choice.

`score` applies §4 to the chosen variant on the 500 test reviews against `reference_chatgpt_v2`, once: `rec`, `lo`
(2,000 resamples, seed 0, within strata) and `fa`, and the outcome by the rule. Then the reported items: per-type
recall and precision, strict type recall, department accuracy on emitted types, per-stratum and unweighted recall,
the teacher and `complaint-lexicon-v1` baselines on the same reviews, and the teacher against the student on the
held-out reviews. It writes student/score.json once and never again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import triage_diagnosis_score as first  # noqa: E402
import triage_student_v2 as v2  # noqa: E402

from reviewnlp.triage import diagnosis_metrics as metrics  # noqa: E402

sample = v2.sample
STUDENT, LABELS, SAMPLES = v2.FOLDER / "student", v2.LABELS, v2.SAMPLES
GRID, F, RULE, Stop = first.GRID, first.F, first.RULE, first.Stop
TOP = max(GRID)
VARIANTS = ("A", "B", "C")  # the order of simplicity, for ties
LOGITS_OF = {"A": "A", "B": "B", "C": "B"}  # C is B's model with other thresholds
REFERENCE = "reference_chatgpt_v2"
C_RULE = ("urgent: per type, the grid value with the highest F1 against the teacher on the held-out reviews (ties to "
          "the smallest); then, while development fa > 5, every urgent threshold + 0.05 together, each at most 0.95, "
          "until fa <= 5 or all are 0.95. Routine and structural: B's.")
CHOICE_RULE = ("among the variants with development fa <= 5, the highest development rec; on a tie the simpler (A, B, "
               "C); if none qualifies, A")


def catalogue() -> dict:
    return yaml.safe_load(v2.CATALOGUE.read_text(encoding="utf-8"))


def by_type(group_thresholds: dict[str, float], data: dict) -> dict[str, float]:
    return {t: group_thresholds[g] for g, ts in first.groups(data).items() for t in ts}


def emit_types(logits: pd.DataFrame, thresholds: dict[str, float], types: list[str]) -> pd.DataFrame:
    """0/1 emitted types: sigmoid(logit) >= the type's own threshold."""
    probs = 1.0 / (1.0 + np.exp(-logits[types].to_numpy(dtype=float)))
    return pd.DataFrame((probs >= np.array([thresholds[t] for t in types])).astype(int), index=logits.index,
                        columns=types)


def f1(gold: pd.Series, pred: pd.Series) -> float:
    g, p = gold.to_numpy().astype(int), pred.to_numpy().astype(int)
    tp, fp, fn = int((g & p).sum()), int(((1 - g) & p).sum()), int((g & (1 - p)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def held_out_thresholds(logits: pd.DataFrame, teacher: pd.DataFrame, urgent: list[str]) -> tuple[dict, dict]:
    """Per urgent type, the grid value with the highest F1 against the teacher; ties to the smallest."""
    chosen, seen = {}, {}
    for t in urgent:
        probs = 1.0 / (1.0 + np.exp(-logits[t].to_numpy(dtype=float)))
        seen[t] = {v: round(f1(teacher[t], pd.Series((probs >= v).astype(int))), 6) for v in GRID}
        best = max(seen[t].values())
        chosen[t] = min(v for v in GRID if seen[t][v] == best)
    return chosen, seen


def rise_within_budget(start: dict[str, float], rest: dict[str, float], logits: pd.DataFrame, gold: pd.DataFrame,
                       weights: pd.Series, data: dict) -> tuple[dict[str, float], list[dict]]:
    """C's rise: every urgent threshold + 0.05 per step, each at most 0.95, until development fa <= 5 or all are 0.95."""
    types = [entry["id"] for entry in data["types"]]
    urgent = metrics.urgent_departments(data["types"])
    steps, step = [], 0
    while True:
        current = {t: min(round(v + 0.05 * step, 2), TOP) for t, v in start.items()}
        counts = metrics.per_review(gold, emit_types(logits, {**rest, **current}, types), urgent)
        fa = round(metrics.false_urgent_per_100(counts, weights), 4)
        steps.append({"step": step, "fa": fa, "urgent": current})
        if fa <= F or all(v == TOP for v in current.values()):
            return current, steps
        step += 1


def choose_variant(figures: dict[str, dict]) -> dict:
    """§3: within the budget, the highest development rec; ties to the simpler; none within it, A."""
    qualified = [v for v in VARIANTS if figures[v]["fa"] <= F]
    if not qualified:
        return {"variant": "A", "qualified": [], "reason": "no variant has development fa <= 5: A by the rule"}
    best = max(figures[v]["rec"] for v in qualified)
    chosen = next(v for v in VARIANTS if v in qualified and figures[v]["rec"] == best)
    tied = [v for v in qualified if figures[v]["rec"] == best]
    return {"variant": chosen, "qualified": qualified,
            "reason": f"highest development rec ({best}) among {qualified}"
                      + (f"; tied with {[v for v in tied if v != chosen]}, the simpler wins" if len(tied) > 1 else "")}


def variant_thresholds(record: dict, variant: str, data: dict) -> dict[str, float]:
    """The per-type thresholds a variant emits with, from thresholds_locked.json."""
    entry = record["variants"][variant]
    if variant == "C":
        return {**by_type(entry["routine_and_structural"], data), **entry["urgent_per_type"]}
    return by_type(entry["thresholds"], data)


def inputs(paths: list[Path]) -> dict[str, str]:
    """Each file the step read, by its path in the repository where it is in one, with its SHA-256."""
    def name(path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return {name(p): sample.trends._sha256(p) for p in paths}


def thresholds(student: Path = STUDENT, labels: Path = LABELS, samples: Path = SAMPLES,
               require_commits: bool = True) -> dict:
    out = student / "thresholds_locked.json"
    if out.exists():
        raise Stop(f"{out} exists; thresholds and the choice are made once")
    needed = [student / f"logits_{v}_{s}.csv" for v in ("A", "B") for s in ("development", "held_out")]
    needed += [labels / f"{REFERENCE}.csv", labels / "teacher_haiku_teacher.csv", samples / "key.csv",
               samples / "teacher_key.csv"]
    if require_commits:
        missing = [p.name for p in needed if not first.committed(p)]
        if missing:
            raise Stop(f"not committed: {missing}")
    data = catalogue()
    types = [entry["id"] for entry in data["types"]]
    urgent = metrics.urgent_departments(data["types"])
    key = pd.read_csv(samples / "key.csv")
    dev = key[key["set"] == "development"].set_index("raw_row")
    gold = first.load_indexed(labels / f"{REFERENCE}.csv").loc[dev.index]
    weights = dev["weight"]
    logits = {v: first.load_indexed(student / f"logits_{v}_development.csv").loc[dev.index] for v in ("A", "B")}

    variants, figures = {}, {}
    for v in ("A", "B"):
        searched = first.choose_thresholds(logits[v], gold, weights, data)
        pred = first.emit(logits[v], searched["thresholds"], data)
        figures[v] = first.primary(pred, gold, weights, urgent)
        variants[v] = {"logits": v, "thresholds": searched["thresholds"],
                       "development_fa_per_100_by_urgent_value": searched["development_fa_per_100_by_urgent_value"],
                       "development_micro_f1_by_value": searched["development_micro_f1_by_value"],
                       "development": figures[v]}

    teacher_key = pd.read_csv(samples / "teacher_key.csv")
    held = teacher_key.loc[teacher_key["held_out"] == 1, "raw_row"]
    teacher = first.load_indexed(labels / "teacher_haiku_teacher.csv").loc[held]
    teacher = teacher[teacher["failed"] == 0]
    held_logits = first.load_indexed(student / "logits_B_held_out.csv").loc[teacher.index]
    start, f1_seen = held_out_thresholds(held_logits, teacher, list(urgent))
    rest = {t: v for t, v in by_type(variants["B"]["thresholds"], data).items() if t not in urgent}
    final, rise = rise_within_budget(start, rest, logits["B"], gold, weights, data)
    figures["C"] = first.primary(emit_types(logits["B"], {**rest, **final}, types), gold, weights, urgent)
    variants["C"] = {"logits": "B", "urgent_per_type": final, "urgent_from_held_out": start,
                     "routine_and_structural": {g: variants["B"]["thresholds"][g] for g in ("routine", "structural")},
                     "held_out_f1_by_value": f1_seen, "held_out_reviews": len(teacher), "rise": rise,
                     "development": figures["C"]}

    result = {"protocol_sha256": sample.trends._sha256(v2.PROTOCOL), "code_commit": sample.pilot.git_commit(),
              "set": "development", "reviews": len(dev), "against": REFERENCE, "grid": GRID,
              "emission": "a type is emitted when sigmoid(logit) >= its threshold", "rule_c": C_RULE,
              "variants": variants, "choice": {**choose_variant(figures), "rule": CHOICE_RULE},
              "inputs": inputs(needed)}
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return result


def versus_teacher(pred: pd.DataFrame, teacher: pd.DataFrame, weights: pd.Series, data: dict) -> dict:
    """The student against its teacher on the held-out reviews: what distillation loses, apart from the reference."""
    types = [entry["id"] for entry in data["types"]]
    urgent = metrics.urgent_departments(data["types"])
    per_type = {}
    for t in types:
        support, emitted, both = int(teacher[t].sum()), int(pred[t].sum()), int((teacher[t] * pred[t]).sum())
        per_type[t] = {"teacher": support, "student": emitted, "both": both,
                       "recall": round(both / support, 4) if support else None,
                       "precision": round(both / emitted, 4) if emitted else None}
    return {"reviews": len(teacher), **first.primary(pred, teacher[types], weights, urgent), "per_type": per_type}


def score(csv: Path, student: Path = STUDENT, labels: Path = LABELS, samples: Path = SAMPLES,
          require_commits: bool = True) -> dict:
    out = student / "score.json"
    if out.exists():
        raise Stop(f"{out} exists; the test set is scored once")
    locked = student / "thresholds_locked.json"
    if not locked.exists():
        raise Stop("no thresholds_locked.json: run `thresholds` and commit it first")
    record = json.loads(locked.read_text(encoding="utf-8"))
    variant = record["choice"]["variant"]
    source = LOGITS_OF[variant]
    needed = [locked, student / f"logits_{source}_test.csv", student / f"logits_{source}_held_out.csv",
              labels / f"{REFERENCE}.csv", labels / "teacher_haiku_test.csv", labels / "teacher_haiku_teacher.csv"]
    if require_commits:
        missing = [p.name for p in needed if not first.committed(p)]
        if missing:
            raise Stop(f"not committed: {missing}; thresholds and labels are committed before the test is scored")
    data = catalogue()
    types = [entry["id"] for entry in data["types"]]
    urgent = metrics.urgent_departments(data["types"])
    chosen = variant_thresholds(record, variant, data)
    key = pd.read_csv(samples / "key.csv")
    test = key[key["set"] == "test"].set_index("raw_row")
    logits = first.load_indexed(student / f"logits_{source}_test.csv").reindex(test.index)
    complete = len(test) == 500 and bool(logits[types].notna().all().all())
    gold = first.load_indexed(labels / f"{REFERENCE}.csv").loc[test.index]
    weights, strata = test["weight"], test["stratum"]
    pred = emit_types(logits.fillna(-np.inf), chosen, types)
    main = first.primary(pred, gold, weights, urgent)
    lo, empty = first.bootstrap_lower(metrics.per_review(gold, pred, urgent), weights, strata)
    main.update({"lo": round(lo, 6), "bootstrap": {"resamples": first.BOOTSTRAP, "seed": first.BOOTSTRAP_SEED,
                                                   "resamples_without_pairs": empty}})
    main["outcome"] = first.outcome(main["rec"], main["lo"], main["fa"]) if complete else "not decided: incomplete run"

    teacher_test = first.load_indexed(labels / "teacher_haiku_test.csv").loc[test.index]
    lexicon = first.lexicon_prediction(first.test_texts(test, csv), data)
    teacher_key = pd.read_csv(samples / "teacher_key.csv")
    held = teacher_key[teacher_key["held_out"] == 1].set_index("raw_row")
    teacher_held = first.load_indexed(labels / "teacher_haiku_teacher.csv").loc[held.index]
    teacher_held = teacher_held[teacher_held["failed"] == 0]
    held_pred = emit_types(first.load_indexed(student / f"logits_{source}_held_out.csv").loc[teacher_held.index],
                           chosen, types)
    result = {
        "protocol_sha256": sample.trends._sha256(v2.PROTOCOL), "code_commit": sample.pilot.git_commit(),
        "variant": variant, "logits": source, "thresholds": chosen, "test_reviews": len(test), "complete": complete,
        "against": REFERENCE, "decision": main, "rule": [condition for condition, _ in RULE],
        "reported": {
            "development_by_variant": {v: record["variants"][v]["development"] for v in VARIANTS},
            "choice": record["choice"],
            "student": first.reported(pred, gold, weights, strata, data),
            "baselines": {"teacher_claude_haiku_5_5": first.primary(teacher_test[types], gold, weights, urgent),
                          "complaint_lexicon_v1": first.primary(lexicon, gold, weights, urgent)},
            "student_vs_teacher_held_out": versus_teacher(held_pred, teacher_held, held.loc[teacher_held.index,
                                                                                            "weight"], data),
            "not_produced_here": {
                "evidence": "needs the chosen model's weights, kept in the owner's Google Drive (run.json); not run",
                "cpu_latency": "needs the chosen model's weights, kept in the owner's Google Drive (run.json); not run"},
        },
        "inputs": inputs(needed),
    }
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("thresholds", help="thresholds of A, B and C on development, and the choice")
    second = steps.add_parser("score", help="score the chosen variant on the test set, once")
    second.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    args = parser.parse_args()
    try:
        if args.step == "thresholds":
            result = thresholds()
            for v in VARIANTS:
                print(v, {k: result["variants"][v]["development"][k] for k in ("rec", "fa", "pairs", "caught")})
            print("choice:", result["choice"]["variant"], "-", result["choice"]["reason"])
        else:
            if sample.trends._sha256(args.csv) != sample.RAW_SHA256:
                raise Stop(f"{args.csv} is not the Kaggle file the protocol fixes")
            decision = score(args.csv)["decision"]
            print({k: decision[k] for k in ("rec", "lo", "fa", "pairs", "caught", "outcome")})
    except Stop as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
