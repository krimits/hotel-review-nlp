"""Offline tests for aspect analytics and the recommendations endpoint.

The endpoint previously answered every hotel with the same two hardcoded
aspects while echoing back the hotel_id and days it was asked for, so the reply
read as measured. These tests pin the replacement: every figure is computed
from rows, and with no rows to compute from the endpoint says so instead.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reviewnlp.analytics.recommendations import (
    RECOMMENDATION_TEXT,
    build_analytics,
    build_recommendation,
    calculate_priority_score,
    normalize_volume,
)
from reviewnlp.analytics.store import get_aspect_store
from reviewnlp.serving.app import app
from reviewnlp.serving.model_wrapper import ModelWrapper


def _rows(*triples) -> list[dict]:
    return [
        {"review_id": review_id, "aspect": aspect, "sentiment": sentiment}
        for review_id, aspect, sentiment in triples
    ]


# --------------------------------------------------------------------------
# scoring primitives
# --------------------------------------------------------------------------


def test_priority_score_follows_the_documented_weights():
    # 0.45 negative rate + 0.25 volume + 0.20 trend + 0.10 business weight.
    assert calculate_priority_score(1.0, 1.0, 1.0, 1.0) == 1.0
    assert calculate_priority_score(0.0, 0.0, 0.0, 0.0) == 0.0
    assert calculate_priority_score(1.0, 0.0, 0.0, 0.0) == 0.45
    assert calculate_priority_score(0.0, 1.0, 0.0, 0.0) == 0.25


def test_an_improving_aspect_never_scores_above_a_flat_one():
    # A falling negative rate is good news; it must not add urgency.
    improving = calculate_priority_score(0.5, 0.5, -0.9)
    flat = calculate_priority_score(0.5, 0.5, 0.0)
    assert improving == flat


def test_normalize_volume_handles_an_empty_corpus():
    assert normalize_volume(0, 0) == 0.0  # no division by zero
    assert normalize_volume(5, 10) == 0.5
    assert normalize_volume(20, 10) == 1.0  # clamped


def test_every_taxonomy_aspect_has_its_own_recommendation():
    from reviewnlp.absa.aspects import ASPECTS

    assert set(RECOMMENDATION_TEXT) == set(ASPECTS)
    fallback = build_recommendation("weather")
    assert fallback and fallback not in RECOMMENDATION_TEXT.values()


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_analytics_are_computed_from_the_rows_and_sorted_worst_first():
    rows = _rows(
        ("r1", "cleanliness", "negative"),
        ("r2", "cleanliness", "negative"),
        ("r3", "cleanliness", "positive"),
        ("r1", "staff", "positive"),
        ("r2", "staff", "positive"),
    )
    result = build_analytics(rows)

    assert [item["aspect"] for item in result] == ["cleanliness", "staff"]

    worst = result[0]
    assert (worst["review_count"], worst["negative_count"], worst["positive_count"]) == (3, 2, 1)
    assert worst["negative_rate"] == 0.6667
    assert worst["trend"] is None  # nothing to compare against
    # 0.45*(2/3) + 0.25*(3/3) + 0.20*0 + 0.10*0.5
    assert worst["priority_score"] == 0.6
    assert worst["recommendation"] == RECOMMENDATION_TEXT["cleanliness"]

    best = result[1]
    assert best["negative_rate"] == 0.0
    # 0.45*0 + 0.25*(2/3) + 0.20*0 + 0.10*0.5
    assert best["priority_score"] == 0.2167


def test_one_review_mentioning_an_aspect_twice_counts_once():
    # The same rule overall_from_aspects applies to the vote: a review
    # repeating itself is one signal, not several.
    rows = _rows(
        ("r1", "cleanliness", "negative"),
        ("r1", "cleanliness", "negative"),
        ("r2", "cleanliness", "positive"),
    )
    result = build_analytics(rows)
    assert result[0]["review_count"] == 2
    assert result[0]["negative_rate"] == 0.5


def test_trend_measures_the_change_in_negative_rate():
    now = _rows(("r1", "food", "negative"), ("r2", "food", "positive"))
    before = _rows(
        ("p1", "food", "negative"),
        ("p2", "food", "negative"),
        ("p3", "food", "negative"),
        ("p4", "food", "positive"),
    )
    result = build_analytics(now, before)

    assert result[0]["negative_rate"] == 0.5
    assert result[0]["trend"] == -0.25  # 0.50 now against 0.75 before: improving


def test_an_aspect_with_no_previous_data_has_no_trend():
    result = build_analytics(
        _rows(("r1", "noise", "negative")),
        _rows(("p1", "staff", "negative")),
    )
    assert result[0]["aspect"] == "noise"
    assert result[0]["trend"] is None


def test_no_rows_yields_no_recommendations():
    # The truthful answer to "what should this hotel fix" before anything has
    # been analyzed is nothing - not an example.
    assert build_analytics([]) == []
    assert build_analytics([], []) == []


def test_rows_outside_the_sentiment_vocabulary_are_ignored():
    rows = _rows(("r1", "staff", "mixed"), ("r2", "staff", "positive"))
    result = build_analytics(rows)
    assert result[0]["review_count"] == 1
    assert result[0]["positive_count"] == 1


# --------------------------------------------------------------------------
# endpoint
# --------------------------------------------------------------------------


class _FakeStore:
    """An in-memory stand-in, so the endpoint's wiring is covered end to end."""

    def __init__(self, rows, previous):
        self._rows = rows
        self._previous = previous
        self.calls: list[tuple] = []

    def aspect_rows(self, hotel_id: str, days: int) -> list[dict]:
        self.calls.append(("current", hotel_id, days))
        return self._rows

    def previous_period_rows(self, hotel_id: str, days: int) -> list[dict]:
        self.calls.append(("previous", hotel_id, days))
        return self._previous


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_TYPE", "stub")
    import reviewnlp.serving.app as app_module

    app_module.wrapper = ModelWrapper("stub", "")
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_recommendations_report_501_when_no_store_is_configured(client):
    response = client.get("/hotels/acme-athens/recommendations")

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "AspectStore" in detail  # the message names the seam to implement
    assert "schema.sql" in detail


def test_recommendations_serve_only_what_the_store_holds(client):
    store = _FakeStore(
        rows=_rows(
            ("r1", "noise", "negative"),
            ("r2", "noise", "negative"),
            ("r3", "staff", "positive"),
        ),
        previous=_rows(("p1", "noise", "positive")),
    )
    app.dependency_overrides[get_aspect_store] = lambda: store

    response = client.get("/hotels/acme-athens/recommendations?days=90")
    assert response.status_code == 200
    body = response.json()

    assert body["hotel_id"] == "acme-athens"
    assert body["period_days"] == 90
    # The hotel and window actually reach the store rather than being echoed.
    assert store.calls == [
        ("current", "acme-athens", 90),
        ("previous", "acme-athens", 90),
    ]

    worst = body["recommendations"][0]
    assert worst["aspect"] == "noise"
    assert worst["review_count"] == 2
    assert worst["negative_rate"] == 1.0
    assert worst["trend"] == 1.0  # 0% negative before, 100% now
    assert worst["recommendation"] == RECOMMENDATION_TEXT["noise"]


def test_an_empty_store_returns_an_empty_list_not_an_example(client):
    app.dependency_overrides[get_aspect_store] = lambda: _FakeStore([], [])

    response = client.get("/hotels/quiet-hotel/recommendations")
    assert response.status_code == 200
    assert response.json()["recommendations"] == []


def test_the_lookback_window_is_still_bounded(client):
    app.dependency_overrides[get_aspect_store] = lambda: _FakeStore([], [])

    assert client.get("/hotels/h/recommendations?days=6").status_code == 422
    assert client.get("/hotels/h/recommendations?days=366").status_code == 422
