"""Thresholds of A, B and C, the choice, and the one score of the triage student v2 protocol, on synthetic data."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reviewnlp.triage import diagnosis_metrics as metrics

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score = _load("triage_student_v2_score")
sample = score.sample
DATA = score.catalogue()
TYPES = [entry["id"] for entry in DATA["types"]]
URGENT = list(metrics.urgent_departments(DATA["types"]))


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def frame(rows: dict[int, dict[str, float]], fill: float) -> pd.DataFrame:
    table = pd.DataFrame(fill, index=list(rows), columns=TYPES, dtype=float)
    for index, values in rows.items():
        for column, value in values.items():
            table.loc[index, column] = value
    return table


def test_the_choice_is_the_protocols():
    def figures(*rows):
        return {v: {"rec": rec, "fa": fa} for v, (rec, fa) in zip("ABC", rows, strict=True)}
    assert score.choose_variant(figures((0.80, 4.0), (0.85, 6.0), (0.85, 5.0)))["variant"] == "C"
    tie = score.choose_variant(figures((0.85, 4.0), (0.85, 3.0), (0.80, 2.0)))
    assert tie["variant"] == "A" and "the simpler wins" in tie["reason"]
    none = score.choose_variant(figures((0.95, 5.1), (0.97, 9.0), (0.99, 6.0)))
    assert none["variant"] == "A" and none["qualified"] == []
    assert score.VARIANTS == ("A", "B", "C") and score.LOGITS_OF == {"A": "A", "B": "B", "C": "B"}


def test_c_starts_from_the_held_out_f1_and_ties_go_low():
    logits = frame({0: {"safety_fault": logit(0.62)}, 1: {"safety_fault": logit(0.58)},
                    2: {"safety_fault": logit(0.41)}}, logit(0.01))
    teacher = frame({0: {"safety_fault": 1}, 1: {"safety_fault": 1}, 2: {}}, 0).astype(int)
    chosen, seen = score.held_out_thresholds(logits, teacher, URGENT)
    assert chosen["safety_fault"] == 0.45  # 0.45, 0.5 and 0.55 all give F1 1; the smallest wins
    assert seen["safety_fault"][0.6] == pytest.approx(2 / 3)
    assert chosen["pests"] == 0.05  # no positive: every value ties at F1 0


def test_c_rises_together_by_005_until_within_the_budget_or_at_095():
    gold = frame({i: {} for i in range(20)}, 0).astype(int)  # no urgent problem in the reference
    weights = pd.Series(1.0, index=range(20))
    probs = {0: 0.97, 1: 0.92, 2: 0.87, 3: 0.82}
    logits = frame({i: {"cleanliness": logit(probs.get(i, 0.01))} for i in range(20)}, logit(0.01))
    rest = {t: 0.5 for t in TYPES if t not in URGENT}
    start = {t: 0.80 for t in URGENT}
    final, steps = score.rise_within_budget(start, rest, logits, gold, weights, DATA)
    assert [s["fa"] for s in steps] == [20.0, 15.0, 10.0, 5.0] and final["cleanliness"] == 0.95
    capped = frame({0: {"cleanliness": logit(0.99)}, 1: {"cleanliness": logit(0.98)}}, logit(0.01))
    final, steps = score.rise_within_budget({**start, "pests": 0.5}, rest, capped.reindex(range(20)).fillna(
        logit(0.01)), gold, weights, DATA)
    assert steps[-1]["fa"] == 10.0 and final["pests"] == 0.95 and all(v == 0.95 for v in final.values())
    assert steps[2]["urgent"]["pests"] == 0.6 and steps[2]["urgent"]["cleanliness"] == 0.9


@pytest.fixture
def layout(tmp_path):
    """10 development, 10 test and 8 held-out reviews, with every label and logits file the steps read."""
    rows = [{"Negative_Review": f"The room was dirty {i}", "Positive_Review": "No Positive"} for i in range(40)]
    raw = pd.DataFrame(rows)
    csv = tmp_path / "raw.csv"
    raw.to_csv(csv, index=False)
    student, labels, samples = tmp_path / "student", tmp_path / "labels", tmp_path / "samples"
    for folder in (student, labels, samples):
        folder.mkdir()

    def key(raw_rows, sets):
        texts = [sample.whole_review(raw.loc[r, "Negative_Review"], raw.loc[r, "Positive_Review"]) for r in raw_rows]
        return pd.DataFrame({"item": range(1, len(raw_rows) + 1), "set": sets, "raw_row": raw_rows,
                             "stratum": ["H", "N"] * (len(raw_rows) // 2), "weight": 2.0, "pool": 85,
                             "text_sha256": [sample.text_hash(t) for t in texts]})
    key(list(range(20)), ["development"] * 10 + ["test"] * 10).to_csv(samples / "key.csv", index=False)
    key(list(range(20, 36)), "teacher").assign(held_out=[1] * 8 + [0] * 8).to_csv(samples / "teacher_key.csv",
                                                                                  index=False)

    def labels_of(raw_rows, positive, extra=None):
        table = pd.DataFrame({"raw_row": list(raw_rows), "set": "x", **{t: 0 for t in TYPES}, "none": 0, **(extra or {})})
        table.loc[table["raw_row"].isin(positive), "cleanliness"] = 1
        table["none"] = 1 - table["cleanliness"]
        return table
    labels_of(range(20), set(range(0, 20, 2))).to_csv(labels / "reference_chatgpt_v2.csv", index=False)
    labels_of(range(10, 20), set(range(10, 20, 2)), {"failed": 0}).to_csv(labels / "teacher_haiku_test.csv",
                                                                          index=False)
    labels_of(range(20, 36), set(range(20, 36, 2)), {"failed": 0}).to_csv(labels / "teacher_haiku_teacher.csv",
                                                                          index=False)

    def logits_of(raw_rows, sure, p):
        table = pd.DataFrame(logit(0.01), index=list(raw_rows), columns=TYPES)
        table.loc[[r for r in raw_rows if r in sure], "cleanliness"] = logit(p)
        return table.rename_axis("raw_row").reset_index().assign(set="x")[["raw_row", "set", *TYPES]]
    for variant, p in (("A", 0.40), ("B", 0.90)):
        logits_of(range(10), set(range(0, 10, 2)), p).to_csv(student / f"logits_{variant}_development.csv", index=False)
        logits_of(range(10, 20), set(range(10, 20, 2)), p).to_csv(student / f"logits_{variant}_test.csv", index=False)
        logits_of(range(20, 28), set(range(20, 28, 2)), p).to_csv(student / f"logits_{variant}_held_out.csv",
                                                                  index=False)
    return csv, student, labels, samples


def test_thresholds_are_chosen_once_and_record_every_variant(layout):
    csv, student, labels, samples = layout
    with pytest.raises(score.Stop, match="not committed"):
        score.thresholds(student, labels, samples)
    result = score.thresholds(student, labels, samples, require_commits=False)
    assert set(result["variants"]) == {"A", "B", "C"} and result["against"] == "reference_chatgpt_v2"
    for variant in ("A", "B", "C"):
        assert result["variants"][variant]["development"]["pairs"] == 5
    # B's urgent probability 0.90 on the five reference pairs: perfect at its development threshold.
    assert result["variants"]["B"]["development"]["rec"] == 1.0
    assert result["variants"]["C"]["urgent_from_held_out"]["cleanliness"] in score.GRID
    assert result["variants"]["C"]["routine_and_structural"] == {
        g: result["variants"]["B"]["thresholds"][g] for g in ("routine", "structural")}
    assert result["choice"]["variant"] in ("A", "B", "C") and result["choice"]["rule"] == score.CHOICE_RULE
    with pytest.raises(score.Stop, match="made once"):
        score.thresholds(student, labels, samples, require_commits=False)


def test_score_uses_the_chosen_variants_logits_refuses_uncommitted_files_and_runs_once(layout):
    csv, student, labels, samples = layout
    with pytest.raises(score.Stop, match="run `thresholds`"):
        score.score(csv, student, labels, samples, require_commits=False)
    chosen = score.thresholds(student, labels, samples, require_commits=False)["choice"]["variant"]
    with pytest.raises(score.Stop, match="not committed"):
        score.score(csv, student, labels, samples)
    result = score.score(csv, student, labels, samples, require_commits=False)
    assert result["variant"] == chosen and result["logits"] == score.LOGITS_OF[chosen]
    decision = result["decision"]
    assert decision["outcome"] == "not decided: incomplete run" and not result["complete"]
    assert decision["pairs"] == 5 and set(decision) >= {"rec", "lo", "fa", "bootstrap"}
    reported = result["reported"]
    assert set(reported["development_by_variant"]) == {"A", "B", "C"}
    assert reported["baselines"]["teacher_claude_haiku_5_5"]["caught"] == 5
    assert reported["student_vs_teacher_held_out"]["reviews"] == 8
    assert json.loads((student / "score.json").read_text(encoding="utf-8")) == result
    with pytest.raises(score.Stop, match="scored once"):
        score.score(csv, student, labels, samples, require_commits=False)
