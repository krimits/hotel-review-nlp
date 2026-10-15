"""Lock test for docs/experiments/distilbert_seed_robustness/PROTOCOL.md.

The protocol is frozen once committed. This test fails if its sections, its data
fingerprints, its seed-42 reference values or its decision rule drift from what the
repository records.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "experiments" / "distilbert_seed_robustness" / "PROTOCOL.md"
MANIFEST = ROOT / "docs" / "experiments" / "results" / "v2_splits" / "time" / "data_manifest.json"
COMPARISON = ROOT / "docs" / "experiments" / "results" / "model_comparison_v2.json"
RUNS = ROOT / "docs" / "experiments" / "results" / "distilbert_v2" / "time"
ANALYSIS_PLAN = ROOT / "docs" / "experiments" / "phase2_analysis_plan.md"
RAW_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prose(protocol) -> str:
    return " ".join(protocol.split())


def table_rows(text: str) -> list[list[str]]:
    """Body rows of every markdown table (after its `---` separator), markers stripped."""
    rows, in_body = [], False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            in_body = False
            continue
        if set(line) <= set("|- :"):
            in_body = True
            continue
        if in_body:
            rows.append([c.strip().strip("*").strip().strip("`") for c in line.strip("|").split("|")])
    return rows


def test_the_five_required_sections_are_present(protocol):
    for heading in ("## 1. Question", "## 2. Data", "## 3. One primary comparison",
                    "## 4. Decision rule", "## 5. Limits"):
        assert heading in protocol, f"missing section: {heading}"


def test_the_time_view_is_pinned_by_its_recorded_fingerprints(protocol):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert RAW_SHA256 in protocol and manifest["raw_csv_sha256"] == RAW_SHA256
    assert "238,154,765" in protocol and manifest["raw_csv_bytes"] == 238154765
    assert manifest["cross_split_overlap"] == {"train_dev": 0, "train_test": 0, "dev_test": 0}
    checked = 0
    for cells in table_rows(protocol):
        if len(cells) == 6 and cells[0] == "time":
            info = manifest["splits"][cells[1]]
            assert [int(cells[2]), int(cells[3]), int(cells[4])] == [info["rows"], info["negative"], info["positive"]]
            assert info["sha256"].startswith(cells[5]), cells
            checked += 1
    assert checked == 3


def test_the_seed_42_reference_is_the_recorded_pair(protocol, prose):
    pair = json.loads(COMPARISON.read_text(encoding="utf-8"))["splits"]["time"]["pairs"]["distilbert_lora_scratch vs distilbert"]
    (row,) = [cells for cells in table_rows(protocol) if len(cells) == 3 and cells[0] == "42"]
    # The recorded pair is LoRA minus full; the protocol states full minus LoRA.
    assert float(row[1]) == round(-pair["macro_f1_difference_b_minus_a"], 4)
    low, high = (float(v) for v in re.findall(r"-?\d+\.\d+", row[2]))
    assert [low, high] == [round(-pair["paired_ci95"][1], 4), round(-pair["paired_ci95"][0], 4)]
    assert "within each of three training seeds: 42, 43 and 44" in prose
    assert "13,921 test reviews" in prose and "2000 resamples" in prose
    for run in ("distilbert", "distilbert_lora_scratch"):
        assert "seed: 42" in (RUNS / run / "run_config.yaml").read_text(encoding="utf-8").splitlines()


def test_the_decision_rule_partitions_every_outcome(protocol):
    rule = protocol.split("## 4. Decision rule", 1)[1].split("## 5. Limits", 1)[0]
    conditions = [cells[0] for cells in table_rows(rule) if len(cells) == 2]
    assert conditions == [
        "all(x > 0 for x in lo)",
        "all(x > 0 for x in d) and any(x <= 0 for x in lo)",
        "any(x <= 0 for x in d)",
    ], conditions
    outcomes = [  # three seeds of (lo, d, hi)
        [(0.004, 0.007, 0.009), (0.002, 0.005, 0.008), (0.001, 0.004, 0.007)],    # holds
        [(0.010, 0.013, 0.016), (0.011, 0.014, 0.017), (0.012, 0.015, 0.018)],    # holds, meaningful size
        [(0.004, 0.007, 0.009), (-0.001, 0.002, 0.005), (0.001, 0.004, 0.007)],   # ahead, within noise
        [(0.000, 0.003, 0.006), (0.001, 0.004, 0.007), (0.002, 0.005, 0.008)],    # lo exactly 0
        [(0.004, 0.007, 0.009), (-0.004, -0.001, 0.002), (0.001, 0.004, 0.007)],  # not consistent
        [(-0.003, 0.000, 0.003), (0.002, 0.005, 0.008), (0.001, 0.004, 0.007)],   # d exactly 0
    ]
    fired = {c: 0 for c in conditions}
    for seeds in outcomes:
        assert all(low <= diff <= high for low, diff, high in seeds)
        scope = {"lo": [s[0] for s in seeds], "d": [s[1] for s in seeds]}
        matches = [c for c in conditions
                   if eval(c, {"__builtins__": {}, "all": all, "any": any}, scope)]  # noqa: S307
        assert len(matches) == 1, f"{seeds} matched {matches}"
        fired[matches[0]] += 1
    assert all(fired.values()), f"a branch was never exercised: {fired}"


def test_the_extra_seed_is_the_one_the_phase_2_plan_provides(protocol, prose):
    plan = " ".join(ANALYSIS_PLAN.read_text(encoding="utf-8").split())
    assert "The notebook can add a second seed for the primary comparison" in plan
    assert "which stays the primary result" in plan
    assert "phase2_analysis_plan.md" in protocol
    assert "which stays the primary result of that plan" in prose


def test_the_limits_name_what_the_runs_do_not_show(protocol):
    limits = protocol.split("## 5. Limits", 1)[1]
    for required in ("No production", "Three seeds", "One split", "fp16", "No tuning",
                     "selection_frozen"):
        assert required in limits, f"limits must mention {required}"
