"""The complaint-trend analysis on a small database whose answers are known by hand."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("complaint_trends", ROOT / "scripts" / "complaint_trends.py")
trends = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = trends
_spec.loader.exec_module(trends)

PARAMS = {"first_month": "2016-01", "last_month": "2017-01", "base_start": "2016-01", "base_end": "2016-01",
          "recent_start": "2017-01", "recent_end": "2017-01", "min_hotel_reviews": "2"}


def _reviews() -> tuple[pd.DataFrame, list[tuple[int, str, str]]]:
    """Hotel A: noise in 2 of 10 reviews, then 4 of 10. Hotel B: none, in 10 then 30 reviews.

    The raw rate stays at 10% because B's share grows; within the same hotels
    it rises from 10% to 20%.
    """
    rows, complaints = [], []
    for hotel, month, count, noisy in (("A", "2016-01", 10, 2), ("A", "2017-01", 10, 4),
                                       ("B", "2016-01", 10, 0), ("B", "2017-01", 30, 0)):
        for index in range(count):
            review_id = len(rows)
            rows.append({"hotel": f"Hotel {hotel}", "address": f"{hotel} street 1 Paris France", "city": "Paris",
                         "review_date": pd.Timestamp(f"{month}-15"), "nationality": "Greece",
                         "trip_type": "Leisure trip", "traveller_type": "Couple", "score": 8.0,
                         "negative": "noisy street" if index < noisy else "No Negative",
                         "has_complaint_text": index < noisy})
            if index < noisy:
                complaints.append((review_id, "noise.general", "noisy street"))
    return pd.DataFrame(rows), complaints


@pytest.fixture
def database() -> sqlite3.Connection:
    frame, complaints = _reviews()
    return trends.build_database(frame, complaints, PARAMS)


def test_within_hotel_change_separates_a_real_rise_from_the_hotel_mix(database):
    row = pd.read_sql_query((trends.SQL_DIR / "04_within_hotel.sql").read_text(), database).iloc[0]
    assert row["topic"] == "noise.general"
    assert row["base_rate"] == pytest.approx(0.10)
    assert row["recent_rate"] == pytest.approx(0.10)
    assert row["raw_change"] == pytest.approx(0.0)
    assert row["within_hotel_change"] == pytest.approx(0.10)
    assert row["mix_effect"] == pytest.approx(-0.10)


def test_python_and_sql_agree_on_the_within_hotel_change(database):
    counts = pd.read_sql_query((trends.SQL_DIR / "03_hotel_counts.sql").read_text(), database)
    summary = trends.within_hotel_intervals(counts, n_resamples=200, seed=1)
    assert summary.loc[0, "within_hotel_change"] == pytest.approx(0.10)
    low, high = summary.loc[0, "ci95"]
    assert low <= 0.10 <= high


def test_year_over_year_uses_every_review_as_denominator(database):
    row = pd.read_sql_query((trends.SQL_DIR / "02_year_over_year.sql").read_text(), database).iloc[0]
    assert (row["base_complaining"], row["base_reviews"]) == (2, 20)
    assert (row["recent_complaining"], row["recent_reviews"]) == (4, 40)


def test_monthly_rates_and_checks(database):
    monthly = pd.read_sql_query((trends.SQL_DIR / "01_monthly_rates.sql").read_text(), database)
    assert monthly.set_index("month")["rate"].to_dict() == pytest.approx({"2016-01": 0.10, "2017-01": 0.10})
    checks = pd.read_sql_query((trends.SQL_DIR / "06_checks.sql").read_text(), database)
    assert checks["value"].tolist() == [0] * len(checks)


def test_a_review_counts_once_per_topic(database):
    with pytest.raises(sqlite3.IntegrityError):
        with database:
            database.execute("INSERT INTO complaints VALUES (0, 'noise.general', 'again')")


def test_guest_mix_shares_sum_to_one(database):
    mix = pd.read_sql_query((trends.SQL_DIR / "05_guest_mix.sql").read_text(), database)
    for _, shares in mix.groupby("dimension"):
        assert shares["recent_share"].sum() == pytest.approx(1.0)


def test_analysis_windows_use_full_months_one_year_apart():
    dates = pd.Series(pd.to_datetime(["2015-08-04", "2016-03-10", "2017-08-03"]))
    assert trends.analysis_windows(dates) == {
        "first_month": "2015-09", "last_month": "2017-07", "base_start": "2015-09", "base_end": "2016-07",
        "recent_start": "2016-09", "recent_end": "2017-07", "min_hotel_reviews": "30"}


@pytest.mark.parametrize(("text", "complains"), [
    ("No Negative", False), ("Nothing", False), ("Nothing at all, lovely stay", False),
    ("Nothing really but the breakfast was cold", True), ("The bed was hard", True), ("", False)])
def test_has_complaint(text, complains):
    assert trends.has_complaint(text) is complains


def test_guest_type_city_and_topics():
    assert trends.guest_type("[' Leisure trip ', ' Couple ', ' Double Room ']") == ("Leisure trip", "Couple")
    assert trends.guest_type("not a list") == ("Unknown", "Unknown")
    assert trends.city_of("s Gravesandestraat 55 Oost 1092 AA Amsterdam Netherlands") == "Amsterdam"
    topics = dict(trends.complaint_topics("The room was tiny and the wifi kept dropping."))
    assert {"room.size", "facilities.wifi"} <= set(topics)


def test_every_topic_has_an_english_label():
    assert set(trends.TOPIC_LABELS) == set(trends.TRIAGE.TOPICS)


@pytest.mark.parametrize(("low", "high", "relative", "expected"), [
    (0.001, 0.02, 0.2, "rising"), (-0.02, -0.001, -0.2, "falling"),
    (-0.001, 0.02, 0.2, ""), (0.001, 0.002, 0.05, "")])
def test_flag_needs_a_clear_and_material_change(low, high, relative, expected):
    row = pd.Series({"ci99": [low, high], "relative_change": relative})
    assert trends.flag(row, "ci99") == expected
