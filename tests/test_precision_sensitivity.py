"""The precision fall that would erase a rise, and the hypothetical supplement, on hand-made numbers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("precision_sensitivity", ROOT / "scripts" / "precision_sensitivity.py")
sensitivity_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sensitivity_module
_spec.loader.exec_module(sensitivity_module)

WITHIN = pd.DataFrame({"topic": ["noise.general", "room.bed"], "base_rate": [0.10, 0.05],
                       "recent_rate_base_mix": [0.125, 0.05]})
READING = {"reader": "test", "rule": "test rule",
           "counts": {"noise.general": {"base": [17, 20], "recent": [19, 20]},
                      "room.bed": {"base": [10, 20], "recent": [10, 20]}}}


def test_a_rise_from_ten_to_twelve_and_a_half_percent_needs_a_fifth_less_precision():
    assert sensitivity_module.needed_ratio(0.10, 0.125) == pytest.approx(0.8)


def test_katz_interval_matches_the_formula():
    low, high = sensitivity_module.katz_interval(17, 20, 19, 20)
    se = np.sqrt(0.15 / 17 + 0.05 / 19)
    assert low == pytest.approx(0.95 / 0.85 * np.exp(-1.959964 * se))
    assert high == pytest.approx(0.95 / 0.85 * np.exp(1.959964 * se))
    with pytest.raises(ValueError):
        sensitivity_module.katz_interval(0, 20, 5, 20)


def test_equal_counts_put_half_the_draws_below_a_ratio_of_one():
    draws = sensitivity_module.beta_ratio_draws(10, 20, 10, 20, 100_000, np.random.default_rng(0))
    assert np.mean(draws <= 1.0) == pytest.approx(0.5, abs=0.01)


def test_report_keeps_the_hypothetical_part_apart():
    report = sensitivity_module.sensitivity(WITHIN, READING, draws=20_000, seed=1)
    noise, bed = report["main"]
    assert noise["precision_fall_that_erases_rise"] == pytest.approx(0.2)
    assert bed["precision_fall_that_erases_rise"] == pytest.approx(0.0)
    assert "chance_precision_fell_that_far" not in noise
    extra = report["supplementary"]
    assert "does not confirm" in extra["note"]
    assert any("Claude" in assumption for assumption in extra["assumptions"])
    assert [row["topic"] for row in extra["topics"]] == ["noise.general", "room.bed"]
    # 17/20 -> 19/20 makes a fall to 0.8 of the base precision unlikely; equal counts leave it open.
    assert extra["topics"][0]["chance_precision_fell_that_far"] < 0.05
    assert extra["topics"][1]["chance_precision_fell_that_far"] > 0.3


def test_topics_read_but_not_measured_are_refused():
    with pytest.raises(ValueError, match="room.view"):
        sensitivity_module.sensitivity(WITHIN, {**READING, "counts": {"room.view": {"base": [5, 20],
                                                                                     "recent": [6, 20]}}})


def test_committed_reading_and_sensitivity_agree():
    folder = ROOT / "docs" / "case_study" / "results" / "since_2016_02"
    reading = json.loads((folder / "reading_counts.json").read_text(encoding="utf-8"))
    committed = json.loads((folder / "precision_sensitivity.json").read_text(encoding="utf-8"))
    within = pd.read_csv(folder / "csv" / "04_within_hotel.csv")
    fresh = sensitivity_module.sensitivity(within, reading, committed["supplementary"]["draws"],
                                           committed["supplementary"]["seed"])
    assert fresh == committed
    for topic, periods in reading["not_about_topic"].items():
        for period, wrong in periods.items():
            assert reading["counts"][topic][period] == [20 - len(wrong), 20]
