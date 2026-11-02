"""Thresholds on development and the one scoring run of the triage diagnosis protocol, on synthetic data."""

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


score = _load("triage_diagnosis_score")
DATA = score.catalogue()
TYPES = [entry["id"] for entry in DATA["types"]]


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def frame(rows: dict[int, dict[str, float]], fill: float) -> pd.DataFrame:
    table = pd.DataFrame(fill, index=list(rows), columns=TYPES, dtype=float)
    for index, values in rows.items():
        for column, value in values.items():
            table.loc[index, column] = value
    return table


def test_the_rule_and_its_boundaries():
    assert score.outcome(0.93, 0.85, 3.0) == "Met"
    assert score.outcome(0.90, 0.80, 5.0) == "Met"
    assert score.outcome(0.93, 0.85, 5.01) == "Recall met, over budget"
    assert score.outcome(0.95, 0.79, 1.0) == "Point estimate met, not established"
    assert score.outcome(0.899, 0.95, 0.0) == "Not met"
    assert [condition for condition, _ in score.RULE] == [
        "rec >= 0.90 and lo >= 0.80 and fa <= 5", "rec >= 0.90 and lo >= 0.80 and fa > 5",
        "rec >= 0.90 and lo < 0.80", "rec < 0.90"]


def test_a_type_is_emitted_at_its_group_threshold_or_above():
    logits = frame({1: {"cleanliness": logit(0.30), "wifi": logit(0.30), "price_value": logit(0.299)}}, logit(0.01))
    pred = score.emit(logits, {"urgent": 0.30, "routine": 0.35, "structural": 0.30}, DATA)
    assert pred.loc[1, "cleanliness"] == 1 and pred.loc[1, "wifi"] == 0 and pred.loc[1, "price_value"] == 0


def test_the_urgent_threshold_is_the_smallest_within_the_budget_and_ties_go_low():
    # 20 reviews, no urgent problem in the gold; review i carries an urgent logit of probability 0.02 + 0.05 i,
    # so a threshold t flags every review whose probability is at least t.
    rows = {i: {"cleanliness": logit(0.02 + 0.05 * i)} for i in range(20)}
    logits, gold = frame(rows, logit(0.01)), frame({i: {} for i in range(20)}, 0).astype(int)
    weights = pd.Series(1.0, index=range(20))
    result = score.choose_thresholds(logits, gold, weights, DATA)
    # At 0.95 one review (p = 0.97) is flagged: 5 per 100. At 0.90 two are: 10. So 0.95 is the smallest within 5.
    assert result["development_fa_per_100_by_urgent_value"][0.95] == 5.0
    assert result["development_fa_per_100_by_urgent_value"][0.9] == 10.0
    assert result["thresholds"]["urgent"] == 0.95
    # No routine or structural type is present or predicted: every value ties at micro-F1 0; the smallest wins.
    assert result["thresholds"]["routine"] == 0.05 and result["thresholds"]["structural"] == 0.05
    noisy = frame({i: {"cleanliness": logit(0.99)} for i in range(20)}, logit(0.01))
    assert score.choose_thresholds(noisy, gold, weights, DATA)["thresholds"]["urgent"] == 0.95  # none fits: largest


def test_micro_f1_picks_the_value_that_separates_best():
    rows = {0: {"wifi": logit(0.62)}, 1: {"wifi": logit(0.58)}, 2: {"wifi": logit(0.41)}}
    gold = frame({0: {"wifi": 1}, 1: {"wifi": 1}, 2: {}}, 0).astype(int)
    result = score.choose_thresholds(frame(rows, logit(0.01)), gold, pd.Series(1.0, index=[0, 1, 2]), DATA)
    assert result["thresholds"]["routine"] == 0.45  # 0.45 and 0.5, 0.55 all give F1 1; the smallest wins


def test_the_bootstrap_is_fixed_by_its_seed_and_perfect_predictions_give_one():
    counts = pd.DataFrame({"pairs": [1, 1, 0, 2, 1, 1], "caught": [1, 0, 0, 2, 1, 0]}, index=range(6))
    weights = pd.Series([2.0, 2.0, 2.0, 5.0, 5.0, 5.0], index=range(6))
    strata = pd.Series(["H", "H", "H", "N", "N", "N"], index=range(6))
    first, second = score.bootstrap_lower(counts, weights, strata, reps=500), score.bootstrap_lower(
        counts, weights, strata, reps=500)
    assert first == second and first[0] <= metrics.department_urgent_recall(counts, weights)
    perfect = counts.assign(caught=counts["pairs"])
    assert score.bootstrap_lower(perfect, weights, strata, reps=200)[0] == 1.0


def test_cohen_kappa():
    a = pd.Series([1, 1, 0, 0, 1, 0, 1, 0])
    assert score.cohen_kappa(a, a) == 1.0
    assert score.cohen_kappa(a, pd.Series([1, 0, 1, 0, 1, 0, 1, 0])) == 0.5
    assert score.cohen_kappa(pd.Series([0] * 4), pd.Series([0] * 4)) is None


@pytest.fixture
def layout(tmp_path):
    """12 reviews (6 development, 6 test), 4 check reviews, logits and every label file the score reads."""
    rows = [{"Negative_Review": f"The room was dirty {i}", "Positive_Review": "No Positive"} for i in range(20)]
    raw = pd.DataFrame(rows)
    csv = tmp_path / "raw.csv"
    raw.to_csv(csv, index=False)
    student, labels, samples = tmp_path / "student", tmp_path / "labels", tmp_path / "samples"
    for folder in (student, labels, samples):
        folder.mkdir()

    def key(raw_rows, sets, strata):
        texts = [score.sample.whole_review(raw.loc[r, "Negative_Review"], raw.loc[r, "Positive_Review"])
                 for r in raw_rows]
        return pd.DataFrame({"item": range(1, len(raw_rows) + 1), "set": sets, "raw_row": raw_rows,
                             "stratum": strata, "weight": [2.0 if s == "H" else 4.0 for s in strata], "pool": 90,
                             "text_sha256": [score.sample.text_hash(t) for t in texts]})
    main = key(list(range(12)), ["development", "test"] * 6, ["H", "H", "N", "N"] * 3)
    main.to_csv(samples / "key.csv", index=False)
    key(list(range(14, 18)), "check", ["H", "N"] * 2).to_csv(samples / "check_key.csv", index=False)

    def labels_of(raw_rows, positive):
        table = pd.DataFrame({"raw_row": raw_rows, "set": "x", **{t: 0 for t in TYPES}, "none": 0})
        table.loc[table["raw_row"].isin(positive), "cleanliness"] = 1
        table["none"] = 1 - table["cleanliness"]
        return table
    reference = labels_of(range(12), {0, 1, 2, 3, 4, 5, 6, 7})
    reference.to_csv(labels / "reference_chatgpt.csv", index=False)
    reference.to_csv(labels / "second_rater.csv", index=False)
    labels_of(range(1, 12, 2), {1, 3, 5}).assign(failed=0).to_csv(labels / "teacher_haiku_test.csv", index=False)
    labels_of(range(14, 18), {14, 15}).to_csv(labels / "owner_check.csv", index=False)
    labels_of(range(14, 18), {14}).to_csv(labels / "second_rater_check.csv", index=False)
    (labels / "teacher_choice.json").write_text(json.dumps(
        {"rec_dev": {}, "fa_dev": {}, "cost_usd": {}, "sonnet_lead": 0.0, "model": "m"}), encoding="utf-8")

    def logits_of(raw_rows, sure):
        table = pd.DataFrame(logit(0.01), index=list(raw_rows), columns=TYPES)
        table.loc[[r for r in raw_rows if r in sure], "cleanliness"] = logit(0.97)
        return table.rename_axis("raw_row").reset_index().assign(set="x")[["raw_row", "set", *TYPES]]
    logits_of(range(0, 12, 2), {0, 2, 4, 6}).to_csv(student / "logits_development.csv", index=False)
    logits_of(range(1, 12, 2), {1, 3, 5}).to_csv(student / "logits_test.csv", index=False)
    logits_of(range(14, 18), {14}).to_csv(student / "logits_check.csv", index=False)
    return csv, student, labels, samples


def test_score_refuses_labels_or_thresholds_that_are_not_committed(layout):
    csv, student, labels, samples = layout
    score.thresholds(student, labels, samples)
    with pytest.raises(score.Stop, match="not committed"):
        score.score(csv, student, labels, samples)
    assert not (student / "score.json").exists()


def test_a_run_short_of_500_test_reviews_is_not_decided_and_scores_once(layout):
    csv, student, labels, samples = layout
    chosen = score.thresholds(student, labels, samples)
    assert chosen["thresholds"]["urgent"] in score.GRID
    with pytest.raises(score.Stop, match="chosen once"):
        score.thresholds(student, labels, samples)
    result = score.score(csv, student, labels, samples, require_commits=False)
    decision = result["decision"]
    assert decision["outcome"] == "not decided: incomplete run" and not result["complete"]
    # Test raw rows 1, 3, 5, 7, 9, 11: the reference flags 1, 3, 5, 7; the student flags 1, 3, 5.
    assert (decision["pairs"], decision["caught"]) == (4, 3) and decision["fa"] == 0.0
    assert decision["rec"] == pytest.approx((2 + 4 + 2) / (2 + 4 + 2 + 4))  # weights H 2, N 4 by position
    assert result["reported"]["baselines"]["teacher_claude_haiku_5_5"]["caught"] == 3
    assert result["reported"]["owner_check_100"]["student_vs_owner"]["pairs"] == 2
    assert result["reported"]["agreement_second_rater_vs_reference_800"]["cleanliness"]["kappa"] == 1.0
    assert set(result["reported"]["not_produced"]) == {"evidence", "cpu_latency"}
    with pytest.raises(score.Stop, match="scored once"):
        score.score(csv, student, labels, samples, require_commits=False)
