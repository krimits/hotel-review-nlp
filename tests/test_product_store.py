"""Product contracts: idempotent writes, hotel isolation, authorization and deletion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from reviewnlp.analytics.store import SqliteAspectStore, get_aspect_store
from reviewnlp.serving.absa_router import get_absa_model
from reviewnlp.serving.app import app
from reviewnlp.serving.security import validate_production_config


def _record(aspect="cleanliness", sentiment="negative", quote="dirty room"):
    return {"json_valid": True, "salvaged": False, "entries_dropped": 0,
            "quote_absent": 0, "quote_not_in_review": 0,
            "generation_hit_token_budget": False,
            "aspects": [{"aspect": aspect, "sentiment": sentiment, "quote": quote}]}


def test_store_scopes_upserts_and_deletes(tmp_path):
    store = SqliteAspectStore(tmp_path / "reviews.db")
    for hotel in ("a", "b"):
        store.save_review(hotel_id=hotel, source="booking", review_id="same-id", text="dirty room",
                          language="en", review_date=None, record=_record())
    store.save_review(hotel_id="a", source="booking", review_id="same-id", text="kind staff",
                      language="en", review_date=None, record=_record("staff", "positive", "kind staff"))
    assert [r["aspect"] for r in store.aspect_rows("a", 30)] == ["staff"]
    assert [r["aspect"] for r in store.aspect_rows("b", 30)] == ["cleanliness"]
    assert not store.delete_review("c", "booking", "same-id")
    assert store.delete_review("a", "booking", "same-id")
    assert store.aspect_rows("a", 30) == []
    assert len(store.aspect_rows("b", 30)) == 1


def test_store_never_persists_unsubstantiated_quote(tmp_path):
    store = SqliteAspectStore(tmp_path / "reviews.db")
    with pytest.raises(ValueError, match="unsupported quote"):
        store.save_review(hotel_id="hotel", source="api", review_id="r1", text="kind staff",
                          language="en", review_date=None, record=_record())
    assert store.aspect_rows("hotel", 30) == []


def test_store_compares_equal_length_windows(tmp_path):
    store = SqliteAspectStore(tmp_path / "reviews.db")
    for review_id, age in (("recent", 1), ("previous", 35), ("old", 90)):
        store.save_review(hotel_id="h", source="api", review_id=review_id, text="dirty room",
                          language="en", review_date=datetime.now(timezone.utc)-timedelta(days=age),
                          record=_record())
    assert len(store.aspect_rows("h", 30)) == 1
    assert len(store.previous_period_rows("h", 30)) == 1


def test_hotel_scoped_api_refuses_cross_hotel_access(tmp_path, monkeypatch):
    db = tmp_path / "reviews.db"
    monkeypatch.setenv("REVIEWNLP_DB_PATH", str(db))
    token = "long-random-test-token"
    monkeypatch.setenv("REVIEWNLP_API_KEYS_JSON", json.dumps({
        hashlib.sha256(token.encode()).hexdigest(): ["hotel-a"]
    }))
    store = SqliteAspectStore(db)

    class Model:
        def analyze(self, text):
            return _record()

        def analyze_batch(self, texts):
            return [_record() for _ in texts]

    app.dependency_overrides[get_absa_model] = lambda: Model()
    app.dependency_overrides[get_aspect_store] = lambda: store
    try:
        with TestClient(app) as client:
            dashboard = client.get("/dashboard")
            assert dashboard.status_code == 200
            assert "Hotel Review Operations" in dashboard.text
            payload = {"hotel_id": "hotel-a", "review_id": "r1", "text": "dirty room"}
            assert client.post("/absa", json=payload).status_code == 401
            assert client.post("/absa", json=payload,
                               headers={"X-API-Key": "invalid"}).status_code == 401
            allowed = {"X-API-Key": token}
            saved = client.post("/absa", json=payload, headers=allowed)
            assert saved.status_code == 200
            assert saved.json()["stored"] is True
            assert client.get("/hotels/hotel-a/recommendations", headers=allowed).json()[
                "recommendations"][0]["aspect"] == "cleanliness"
            assert client.get("/hotels/hotel-b/recommendations", headers=allowed).status_code == 403
            bad_batch = {"hotel_id": "hotel-a", "reviews": [
                {"hotel_id": "hotel-b", "text": "dirty room"}]}
            assert client.post("/absa/batch", json=bad_batch, headers=allowed).status_code == 422
            assert client.delete("/hotels/hotel-a/reviews/r1", headers=allowed).status_code == 204
            assert client.get("/hotels/hotel-a/recommendations", headers=allowed).json()[
                "recommendations"] == []
    finally:
        app.dependency_overrides.clear()


def test_production_refuses_missing_security_configuration(monkeypatch):
    monkeypatch.setenv("REVIEWNLP_ENV", "production")
    monkeypatch.delenv("REVIEWNLP_DB_PATH", raising=False)
    monkeypatch.delenv("REVIEWNLP_API_KEYS_JSON", raising=False)
    with pytest.raises(RuntimeError, match="production requires"):
        with TestClient(app):
            pass
    monkeypatch.setenv("REVIEWNLP_API_KEYS_JSON", '{"bad":"config"}')
    with pytest.raises(Exception, match="API key configuration is invalid"):
        validate_production_config()
