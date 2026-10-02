"""Triage in the SQLite store, through the API, and on the dashboard: no review text kept, hotels kept apart."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from triage_fakes import NEGATIVE, POSITIVE, REVIEW, FakeGenerator, FakeJev, FakeWrapper

from reviewnlp.analytics.store import SqliteAspectStore, get_aspect_store
from reviewnlp.serving import triage_router
from reviewnlp.serving.app import app
from reviewnlp.triage.jev_client import JevError
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.schemas import TOPICS, TriageResponse

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "reviewnlp" / "serving" / "dashboard.html"


def result_for(text: str = REVIEW, wrapper=None, jev=None, generator=None) -> dict:
    """What POST /triage returns for a review, as the store receives it."""
    run = TriagePipeline(wrapper or FakeWrapper(NEGATIVE), jev if jev is not None else FakeJev(yes=["bathroom"]),
                         generator if generator is not None else FakeGenerator()).run(text)
    return TriageResponse(hotel_id="h", review_id="r", status=run.status, sentiment=run.sentiment,
                          complaints=run.complaints, routing=run.routing, actions=run.actions,
                          timings=run.timings).model_dump(mode="json")


@pytest.fixture
def store(tmp_path) -> SqliteAspectStore:
    return SqliteAspectStore(tmp_path / "triage.db")


def save(store, review_id="r-1", hotel="hotel-a", source="api", text=REVIEW, date=None, **stages):
    store.save_triage(hotel_id=hotel, review_id=review_id, source=source, text=text, language="en", review_date=date,
                      result=result_for(text, **stages))


# --- The store ---------------------------------------------------------------------------------------------------

def test_the_triage_tables_are_added_to_a_database_that_already_has_aspects(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""CREATE TABLE hotel_reviews (id INTEGER PRIMARY KEY, hotel_id TEXT NOT NULL,
            source TEXT NOT NULL, external_review_id TEXT NOT NULL, text_sha256 TEXT NOT NULL, language TEXT NOT NULL,
            review_date TEXT NOT NULL, json_valid INTEGER NOT NULL, salvaged INTEGER NOT NULL,
            entries_dropped INTEGER NOT NULL, updated_at TEXT NOT NULL);
            INSERT INTO hotel_reviews VALUES (1, 'h', 'api', 'old', 'x', 'en', '2026-01-01', 1, 0, 0, 'now');""")
    store = SqliteAspectStore(path)
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"triage_runs", "triage_complaints", "triage_actions"} <= tables
        assert connection.execute("SELECT COUNT(*) FROM hotel_reviews").fetchone()[0] == 1
    assert store.triage_summary("h", 30)["reviews"] == 0
    SqliteAspectStore(path)  # opening it again changes nothing and does not fail


def test_a_result_is_stored_without_the_review_and_read_back(store):
    save(store)
    summary = store.triage_summary("hotel-a", 30)
    assert (summary["reviews"], summary["sentiment"], summary["complaints_evaluated"]) == (1, {"negative": 1}, 1)
    assert summary["complaints"]["bathroom"] == {"yes": 1, "no": 0, "unsure": 0}
    assert set(summary["complaints"]) == {*TOPICS, "other"}
    (action,) = summary["actions"]
    assert action["excerpt"] in REVIEW and action["department"] == "maintenance" and action["to_confirm"] == ["room number"]
    assert (action["review_id"], action["source"]) == ("r-1", "api")
    assert summary["to_check"] == []
    with sqlite3.connect(store.path) as connection:
        dump = "\n".join(connection.iterdump())
        row = connection.execute("SELECT text_sha256, sentiment_label, questions_version, validation_status "
                                 "FROM triage_runs").fetchone()
    assert "SECRETWORD" not in dump and REVIEW not in dump  # only a span of the review, in an action, is kept
    assert row == (hashlib.sha256(REVIEW.encode()).hexdigest(), "negative", "full-review-v1", "unvalidated")


def test_saving_a_review_again_replaces_its_result(store):
    save(store)
    save(store, wrapper=FakeWrapper(POSITIVE), jev=FakeJev(), generator=FakeGenerator())
    summary = store.triage_summary("hotel-a", 30)
    assert (summary["reviews"], summary["sentiment"], summary["actions"]) == (1, {"positive": 1}, [])
    assert summary["complaints"]["bathroom"] == {"yes": 0, "no": 1, "unsure": 0}


def test_hotels_are_kept_apart_and_the_window_is_respected(store):
    now = datetime.now(timezone.utc)
    save(store, "recent", date=now - timedelta(days=2))
    save(store, "old", date=now - timedelta(days=40))
    save(store, "other-hotel", hotel="hotel-b")
    assert store.triage_summary("hotel-a", 30)["reviews"] == 1
    assert store.triage_summary("hotel-a", 90)["reviews"] == 2
    assert store.triage_summary("hotel-b", 30)["reviews"] == 1
    assert store.triage_summary("hotel-c", 30)["reviews"] == 0


def test_complaints_are_counted_only_from_reviews_where_they_were_asked_for(store):
    save(store, "asked")
    save(store, "not-asked", jev=FakeJev(enabled=False))
    summary = store.triage_summary("hotel-a", 30)
    assert (summary["reviews"], summary["complaints_evaluated"]) == (2, 1)
    assert summary["complaints"]["bathroom"] == {"yes": 1, "no": 0, "unsure": 0}


def test_reviews_that_need_a_look_are_listed_with_their_reasons(store):
    save(store, "failed", wrapper=FakeWrapper(POSITIVE), jev=FakeJev(error=JevError("timeout")), generator=FakeGenerator())
    save(store, "fine", wrapper=FakeWrapper(POSITIVE), jev=FakeJev(), generator=FakeGenerator())
    (check,) = store.triage_summary("hotel-a", 30)["to_check"]
    assert (check["review_id"], check["reasons"]) == ("failed", ["complaint_check_failed"])


def test_an_excerpt_that_is_not_in_the_review_is_refused_and_nothing_is_stored(store):
    bad = result_for()
    bad["actions"]["actions"][0]["excerpt"] = "words the guest never wrote"
    with pytest.raises(ValueError, match="unsupported excerpt"):
        store.save_triage(hotel_id="hotel-a", review_id="r", source="api", text=REVIEW, language="en",
                          review_date=None, result=bad)
    assert store.triage_summary("hotel-a", 30)["reviews"] == 0


def test_deleting_a_review_deletes_its_triage_result_and_says_if_there_was_nothing_to_delete(store):
    save(store)
    save(store, "kept")
    assert store.delete_review("hotel-a", "api", "r-1") is True
    summary = store.triage_summary("hotel-a", 30)
    assert summary["reviews"] == 1 and [a["review_id"] for a in summary["actions"]] == ["kept"]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM triage_complaints").fetchone()[0] == len(TOPICS) + 1
    assert store.delete_review("hotel-a", "api", "r-1") is False
    assert store.delete_review("hotel-b", "api", "kept") is False  # another hotel cannot delete it


# --- Through the API -----------------------------------------------------------------------------------------------------

@pytest.fixture
def api(store, monkeypatch):
    stages = (FakeWrapper(NEGATIVE), FakeJev(yes=["bathroom"]), FakeGenerator())
    monkeypatch.delenv("REVIEWNLP_API_KEYS_JSON", raising=False)
    app.dependency_overrides[get_aspect_store] = lambda: store
    app.dependency_overrides[triage_router.get_triage_pipeline] = lambda: TriagePipeline(*stages)
    try:
        yield TestClient(app), stages
    finally:
        app.dependency_overrides.pop(get_aspect_store, None)
        app.dependency_overrides.pop(triage_router.get_triage_pipeline, None)


def test_a_triage_request_is_stored_summarised_and_deletable(api):
    client, _ = api
    body = {"hotel_id": "hotel-a", "text": REVIEW}
    response = client.post("/triage", json=body)
    assert response.status_code == 200 and response.json()["stored"] is True
    review_id = response.json()["review_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", review_id)  # generated, as /absa does
    summary = client.get("/hotels/hotel-a/triage/summary").json()
    assert summary["reviews"] == 1 and summary["actions"][0]["review_id"] == review_id
    assert summary["validation_status"] == "unvalidated" and summary["complaints"]["bathroom"]["yes"] == 1
    assert client.delete(f"/hotels/hotel-a/reviews/{review_id}", params={"source": "api"}).status_code == 204
    assert client.get("/hotels/hotel-a/triage/summary").json()["reviews"] == 0


def test_the_summary_enforces_hotel_access(api, monkeypatch):
    client, _ = api
    token = "key-for-hotel-a"
    monkeypatch.setenv("REVIEWNLP_API_KEYS_JSON", json.dumps({hashlib.sha256(token.encode()).hexdigest(): ["hotel-a"]}))
    assert client.get("/hotels/hotel-a/triage/summary").status_code == 401
    assert client.get("/hotels/hotel-b/triage/summary", headers={"X-API-Key": token}).status_code == 403
    assert client.get("/hotels/hotel-a/triage/summary", headers={"X-API-Key": token}).status_code == 200
    assert client.get("/hotels/hotel-a/triage/summary", params={"days": 3}, headers={"X-API-Key": token}).status_code == 422


def test_without_a_database_a_triage_request_still_works_and_the_summary_says_there_is_none(monkeypatch):
    monkeypatch.delenv("REVIEWNLP_API_KEYS_JSON", raising=False)
    app.dependency_overrides[get_aspect_store] = lambda: None
    app.dependency_overrides[triage_router.get_triage_pipeline] = lambda: TriagePipeline(FakeWrapper(NEGATIVE))
    try:
        client = TestClient(app)
        body = client.post("/triage", json={"hotel_id": "hotel-a", "text": REVIEW}).json()
        assert body["stored"] is False and body["review_id"] is None
        assert client.get("/hotels/hotel-a/triage/summary").status_code == 501
    finally:
        app.dependency_overrides.pop(get_aspect_store, None)
        app.dependency_overrides.pop(triage_router.get_triage_pipeline, None)


def test_the_suggestion_stage_is_given_counts_from_the_hotels_own_stored_aspects(api, store):
    client, (_, _, generator) = api
    record = {"aspects": [{"aspect": "noise", "sentiment": "negative", "quote": "very loud"},
                          {"aspect": "staff", "sentiment": "positive", "quote": "kind staff"}],
              "json_valid": True, "salvaged": False, "entries_dropped": 0}
    store.save_review(hotel_id="hotel-a", review_id="old", source="api", text="very loud and kind staff",
                      language="en", review_date=None, record=record)
    client.post("/triage", json={"hotel_id": "hotel-a", "text": REVIEW})
    client.post("/triage", json={"hotel_id": "hotel-b", "text": REVIEW})
    assert [call[1].hotel_context for call in generator.signals] == [
        "Recent negative mentions at this hotel (last 30 days): noise 1", None]


# --- The dashboard -------------------------------------------------------------------------------------------------------

def test_the_dashboard_has_the_four_views_and_reads_the_summary():
    page = DASHBOARD.read_text(encoding="utf-8")
    for title in ("Συνολικό συναίσθημα", "Εντοπισμένα παράπονα", "Προτεινόμενα μέτρα", "Χρειάζεται έλεγχος"):
        assert f"<h3>{title}</h3>" in page
    assert "/triage/summary?days=" in page and "request('/triage'" in page
    assert "δεν έχει επικυρωθεί" in page.lower() and "unvalidated" not in page


def test_the_dashboards_script_writes_text_only_and_is_valid_javascript():
    page = DASHBOARD.read_text(encoding="utf-8")
    script = re.search(r"<script>(.*)</script>", page, re.S).group(1)
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script and "document.write" not in script
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    check = subprocess.run([node, "--check", "-"], input=script, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
