"""Lock test for docs/experiments/bilstm_clean_splits/PROTOCOL.md.

The protocol is frozen once committed. This test fails if its five sections, its data
fingerprints, its primary comparison's reference values or its decision rule drift from
what the repository actually records — which is what "locked" has to mean to be worth
anything.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "experiments" / "bilstm_clean_splits" / "PROTOCOL.md"
VIEWS = ROOT / "docs" / "experiments" / "results" / "v2_splits"
COMPARISON = ROOT / "docs" / "experiments" / "results" / "model_comparison_v2.json"
ANALYSIS_PLAN = ROOT / "docs" / "experiments" / "phase2_analysis_plan.md"

RAW_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"
PRIMARY_VIEW = "time"


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prose(protocol) -> str:
    """The protocol with runs of whitespace collapsed, so a phrase that markdown wrapped
    across two lines still reads as one."""
    return " ".join(protocol.split())


def table_rows(text: str) -> list[list[str]]:
    """Body rows of every markdown table, bold markers and code ticks stripped.

    Header rows are skipped: a row counts as body only after its table's `---`
    separator, and a non-table line ends the table.
    """
    rows, in_body = [], False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            in_body = False
            continue
        if set(line) <= set("|- :"):
            in_body = True
            continue
        if not in_body:
            continue
        rows.append([c.strip().strip("*").strip().strip("`") for c in line.strip("|").split("|")])
    return rows


def test_the_five_required_sections_are_present(protocol):
    for heading in ("## 1. Question", "## 2. Data", "## 3. One primary comparison",
                    "## 4. Decision rule", "## 5. Limits"):
        assert heading in protocol, f"missing section: {heading}"


def test_the_raw_source_is_pinned_by_hash_and_matches_every_view(protocol):
    assert RAW_SHA256 in protocol
    assert "238,154,765" in protocol
    for view in ("random", "time", "hotel"):
        manifest = json.loads((VIEWS / view / "data_manifest.json").read_text(encoding="utf-8"))
        assert manifest["raw_csv_sha256"] == RAW_SHA256, view
        assert manifest["raw_csv_bytes"] == 238154765, view


def test_every_split_fingerprint_matches_the_recorded_manifest(protocol):
    recorded = {}
    for view in ("random", "time", "hotel"):
        manifest = json.loads((VIEWS / view / "data_manifest.json").read_text(encoding="utf-8"))
        assert manifest["cross_split_overlap"] == {"train_dev": 0, "train_test": 0, "dev_test": 0}, view
        for split, info in manifest["splits"].items():
            recorded[(view, split)] = info

    checked = 0
    for cells in table_rows(protocol):
        if len(cells) != 6 or (cells[0], cells[1]) not in recorded:
            continue
        info = recorded[(cells[0], cells[1])]
        assert int(cells[2]) == info["rows"], cells
        assert int(cells[3]) == info["negative"], cells
        assert int(cells[4]) == info["positive"], cells
        assert info["sha256"].startswith(cells[5]), cells
        checked += 1
    assert checked == 9, f"expected all nine split rows to be checked, matched {checked}"


def test_the_primary_comparison_quotes_the_recorded_numbers(protocol, prose):
    models = json.loads(COMPARISON.read_text(encoding="utf-8"))["splits"][PRIMARY_VIEW]["models"]
    quoted = {cells[0]: cells[1:] for cells in table_rows(protocol) if len(cells) == 3}
    for label, key in (("DistilBERT, full fine-tune", "distilbert"),
                       ("DistilBERT, scratch LoRA", "distilbert_lora_scratch"),
                       ("TF-IDF + Naive Bayes", "tfidf_nb")):
        assert label in quoted, label
        assert float(quoted[label][0]) == models[key]["macro_f1"], label
        low, high = (float(v) for v in re.findall(r"-?\d+\.\d+", quoted[label][1]))
        assert [low, high] == models[key]["macro_f1_ci95"], label
    # the primary pair and its test set must be named
    assert "13,921 test reviews" in prose
    assert "2000 resamples" in prose
    assert "BiLSTM versus DistilBERT (full fine-tune)" in prose


def test_the_decision_rule_partitions_every_outcome(protocol):
    rule = protocol.split("## 4. Decision rule", 1)[1].split("## 5. Limits", 1)[0]
    conditions = [cells[0] for cells in table_rows(rule) if len(cells) == 2]
    assert conditions == [
        "hi < 0 and d <= -0.01",
        "hi < 0 and d > -0.01",
        "lo <= 0 <= hi",
        "lo > 0 and d < 0.01",
        "lo > 0 and d >= 0.01",
    ], conditions

    # Machine-checkable: for any (lo, d, hi) with lo <= d <= hi exactly one branch fires,
    # and every branch is reachable. The 0.01 edges are included on purpose.
    outcomes = [
        (-0.030, -0.020, -0.015), (-0.012, -0.010, -0.008),  # meaningful, DistilBERT
        (-0.008, -0.005, -0.002), (-0.012, -0.009, -0.001),  # clear but small, DistilBERT
        (-0.010, 0.000, 0.010), (0.000, 0.000, 0.000), (-0.020, -0.010, 0.000),  # indistinguishable
        (0.001, 0.004, 0.007), (0.001, 0.009, 0.012),  # clear but small, BiLSTM
        (0.015, 0.020, 0.030), (0.008, 0.010, 0.012),  # meaningful, BiLSTM
    ]
    fired = {c: 0 for c in conditions}
    for low, diff, high in outcomes:
        assert low <= diff <= high
        scope = {"lo": low, "d": diff, "hi": high}
        matches = [c for c in conditions if eval(c, {"__builtins__": {}}, scope)]  # noqa: S307
        assert len(matches) == 1, f"lo={low} d={diff} hi={high} matched {matches}"
        fired[matches[0]] += 1
    assert all(fired.values()), f"a branch was never exercised: {fired}"


def test_the_rule_is_the_analysis_plans_rule(protocol, prose):
    """Version 1 of this protocol dropped the plan's 1-point criterion. Pin the link so
    the protocol and the plan cannot drift apart again without a test failing."""
    plan = " ".join(ANALYSIS_PLAN.read_text(encoding="utf-8").split())
    assert "the paired 95% interval of the macro-F1 difference excludes zero" in plan
    assert "the difference is at least 1 macro-F1 point (0.01)" in plan
    assert "clear but small" in plan
    assert "phase2_analysis_plan.md" in protocol
    assert "at least 1 macro-F1 point (0.01)" in prose
    assert "clear but small" in prose


def test_the_version_history_records_the_amendments(protocol):
    assert protocol.startswith("# BiLSTM on the clean splits — version 3")
    history = protocol.split("## Version history", 1)[1]
    assert "3ddb358" in history and "334eb82" in history
    assert "No run had been executed under version 1" in history
    assert "No run had been executed under version 2" in history


def test_the_limits_name_what_the_run_does_not_show(protocol):
    limits = protocol.split("## 5. Limits", 1)[1]
    for required in ("Qwen", "out of scope", "No production", "Three seeds", "min_freq",
                     "selection_frozen", "one CPU"):
        assert required in limits, f"limits must mention {required}"


def test_the_seeds_are_fixed_and_seed_42_decides(prose):
    assert "seeds 43 and 44 on the `time` view" in prose
    assert "Seed 42 stays the primary result" in prose
    assert "the decision rule in section 4 is applied to it alone" in prose
    assert "Neither can change the primary conclusion" in prose


def test_the_frontier_rule_is_machine_checkable(protocol):
    (condition,) = [cells[0] for cells in table_rows(protocol) if len(cells) == 1]
    assert condition == "lat_a <= lat_b and hi >= 0 and (lat_a < 0.9 * lat_b or lo > 0)"

    def dominates(lat_a, lat_b, lo, hi):
        return eval(condition, {"__builtins__": {}}, {"lat_a": lat_a, "lat_b": lat_b, "lo": lo, "hi": hi})  # noqa: S307

    assert dominates(1.2, 57.0, -0.01, 0.01)        # much faster, accuracy indistinguishable
    assert not dominates(55.0, 57.0, -0.01, 0.01)   # under 10% faster: a tie
    assert dominates(57.0, 57.0, 0.002, 0.01)       # as fast, significantly more accurate
    assert not dominates(1.2, 57.0, -0.04, -0.02)   # faster but significantly less accurate
    assert not dominates(80.0, 57.0, 0.02, 0.04)    # more accurate but slower


def test_the_correction_matches_the_distilbert_runs(protocol):
    """Version 2 said the DistilBERT artifacts had another seed; every phase 2 run used seed 42.

    The seed-robustness protocol later added `*_seed43` and `*_seed44` runs beside them; the
    sentence is about the four phase 2 runs.
    """
    limits = protocol.split("## 5. Limits", 1)[1].split("## Execution", 1)[0]
    assert "second seed" not in limits
    runs = sorted(path for path in (ROOT / "docs" / "experiments" / "results" / "distilbert_v2")
                  .glob("*/*/run_config.yaml") if "_seed" not in path.parent.name)
    assert len(runs) == 4
    for run in runs:
        assert "seed: 42" in run.read_text(encoding="utf-8").splitlines(), run
