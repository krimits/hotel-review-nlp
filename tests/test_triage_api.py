"""POST /triage through the app: hotel access before anything runs, the contract, and off-by-default."""

from __future__ import annotations

import hashlib
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from reviewnlp.serving import triage_router
from reviewnlp.serving.app import app
from reviewnlp.serving.model_wrapper import ModelWrapper
from reviewnlp.triage import jev_client, questions
from reviewnlp.triage.jev_client import JevClient, JevConfig
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.qwen_generator import GenerationResult
from reviewnlp.triage.schemas import TOPICS

REVIEW = "SECRETWORD the shower was cold and nobody came to fix it. The staff at breakfast were lovely though."
ACTIONS = json.dumps({"actions": [{
    "problem": "Cold shower not repaired", "excerpt": "the shower was cold and nobody came to fix it",
    "measure": "Fix the boiler and tell the guest", "department": "maintenance", "to_confirm": ["room number"]}]})


class Wrapper:
    model_type, model_path = "fake", "models/fake"

    def __init__(self, error: Exception | None = None):
        self.calls, self.error = [], error

    def distribution_batch(self, texts):
        self.calls.append(texts)
        if self.error:
            raise self.error
        return [{"negative": 0.95, "positive": 0.05}]

    def predict(self, text):
        raise AssertionError("not needed")


class Jev:
    enabled = True

    def __init__(self):
        from types import SimpleNamespace
        self.calls, self.config = [], SimpleNamespace(route="typesafe")

    def classify(self, text):
        from reviewnlp.triage.jev_client import JevResult
        from reviewnlp.triage.schemas import OTHER, ComplaintTopic
        self.calls.append(text)
        topics = [ComplaintTopic(topic=t, answer="yes" if t == "bathroom" else "no",
                                 probability=0.97 if t == "bathroom" else 0.0) for t in TOPICS]
        return JevResult(topics, ComplaintTopic(topic=OTHER, answer="no", probability=0.0), "typesafe/jev-test", "typesafe")


class Generator:
    model_name = "fake/qwen"

    def __init__(self):
        self.calls = []

    def generate(self, review, signals):
        self.calls.append(review)
        return GenerationResult(raw=ACTIONS, hit_token_budget=False, model="fake/qwen")


@pytest.fixture
def stages():
    return Wrapper(), Jev(), Generator()


@pytest.fixture
def client(stages, monkeypatch):
    monkeypatch.delenv("REVIEWNLP_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("REVIEWNLP_DB_PATH", raising=False)
    wrapper, jev, generator = stages
    app.dependency_overrides[triage_router.get_triage_pipeline] = lambda: TriagePipeline(wrapper, jev, generator)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(triage_router.get_triage_pipeline, None)


def payload(**changes) -> dict:
    return {"hotel_id": "hotel-a", "review_id": "r-1", "text": REVIEW, "source": "manual", **changes}


def test_a_request_gets_the_whole_workflow(client, stages):
    response = client.post("/triage", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert (body["hotel_id"], body["review_id"], body["stored"], body["status"]) == ("hotel-a", "r-1", False, "complete")
    assert body["not_stored_reason"] == "no_database"  # the review has an id, but there is nowhere to keep the result
    assert body["validation_status"] == "unvalidated"
    assert body["sentiment"] == {"label": "negative", "confidence": 0.95, "model_type": "fake", "model_path": "models/fake",
                                 "probabilities": {"negative": 0.95, "positive": 0.05}}
    complaints = body["complaints"]
    assert complaints["status"] == "ok" and complaints["model"] == "typesafe/jev-test"
    assert [t["topic"] for t in complaints["topics"]] == list(TOPICS) and complaints["other_complaint"]["topic"] == "other"
    assert {t["topic"] for t in complaints["topics"] if t["answer"] == "yes"} == {"bathroom"}
    assert (complaints["questions_version"], complaints["questions_sha256"]) == (
        questions.QUESTIONS_VERSION, questions.QUESTIONS_SHA256)
    assert body["routing"]["qwen_triggered"] and body["routing"]["reasons"] == ["negative_sentiment", "complaint_detected"]
    assert body["routing"]["thresholds_status"] == "provisional"
    assert body["actions"]["status"] == "ok" and body["actions"]["actions"][0]["excerpt"] in REVIEW
    assert set(body["timings"]) == {"sentiment_ms", "complaints_ms", "routing_ms", "actions_ms", "total_ms"}
    assert all(isinstance(value, float) for value in body["timings"].values())
    assert "SECRETWORD" not in response.text


def test_the_route_is_in_the_schema_with_its_response(client):
    schema = client.get("/openapi.json").json()
    assert "/triage" in schema["paths"] and "TriageResponse" in schema["components"]["schemas"]


# --- Hotel access comes first --------------------------------------------------------------------------------------------

def test_the_hotel_key_is_checked_before_any_model_or_provider_is_called(client, stages, monkeypatch):
    token = "key-for-hotel-a"
    monkeypatch.setenv("REVIEWNLP_API_KEYS_JSON", json.dumps({hashlib.sha256(token.encode()).hexdigest(): ["hotel-a"]}))
    for headers, hotel, status in [({}, "hotel-a", 401), ({"X-API-Key": "wrong"}, "hotel-a", 401),
                                   ({"X-API-Key": token}, "hotel-b", 403)]:
        assert client.post("/triage", json=payload(hotel_id=hotel), headers=headers).status_code == status
    assert [len(stage.calls) for stage in stages] == [0, 0, 0]  # nothing ran, so nothing could have left the server
    allowed = client.post("/triage", json=payload(), headers={"X-API-Key": token})
    assert allowed.status_code == 200 and [len(stage.calls) for stage in stages] == [1, 1, 1]


@pytest.mark.parametrize("change", [{"language": "el"}, {"text": "ok"}, {"text": "x" * 8001}, {"hotel_id": ""},
                                    {"review_id": "r" * 201}])
def test_a_request_outside_the_contract_is_refused_before_anything_runs(client, stages, change):
    assert client.post("/triage", json=payload(**change)).status_code == 422
    assert [len(stage.calls) for stage in stages] == [0, 0, 0]
    assert client.post("/triage", json={"text": REVIEW}).status_code == 422


def test_a_failing_sentiment_model_is_an_error_that_does_not_echo_the_review(stages, monkeypatch):
    wrapper, jev, generator = Wrapper(error=RuntimeError("SECRETWORD in the message")), Jev(), Generator()
    monkeypatch.delenv("REVIEWNLP_API_KEYS_JSON", raising=False)
    app.dependency_overrides[triage_router.get_triage_pipeline] = lambda: TriagePipeline(wrapper, jev, generator)
    try:
        response = TestClient(app).post("/triage", json=payload())
    finally:
        app.dependency_overrides.pop(triage_router.get_triage_pipeline, None)
    assert response.status_code == 500 and response.json() == {"detail": "inference failed"}
    assert jev.calls == [] and generator.calls == []


# --- The app as configured, with no overrides ----------------------------------------------------------------------------------

@pytest.fixture
def default_app(monkeypatch):
    app_module = importlib.import_module("reviewnlp.serving.app")
    for name in ("REVIEWNLP_JEV_ENABLED", "REVIEWNLP_TRIAGE_QWEN_ENABLED", "REVIEWNLP_API_KEYS_JSON", "REVIEWNLP_DB_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app_module, "wrapper", ModelWrapper("stub", ""))
    monkeypatch.setattr(triage_router, "_pipeline", None)
    yield TestClient(app)
    triage_router._pipeline = None


def test_by_default_only_the_sentiment_runs_and_nothing_leaves_the_server(default_app, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a request was sent")

    monkeypatch.setattr(jev_client, "urllib_transport", forbidden)
    response = default_app.post("/triage", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"]["model_type"] == "stub" and set(body["sentiment"]["probabilities"]) == {"negative", "positive"}
    assert (body["complaints"]["status"], body["actions"]["status"], body["status"]) == ("disabled", "disabled", "complete")


def test_switching_jev_on_in_the_environment_sends_one_request_to_its_route_with_its_key(default_app, monkeypatch):
    sent = []

    def transport(url, body, key, timeout):
        sent.append((url, key, body["state"]))
        answers = {name: {"type": "choice", "choice": "0", "probabilities": {"1": 0.0, "0": 1.0, "unsure": 0.0}}
                   for name in questions.QUESTIONS}
        return 200, {}, json.dumps({"model": "typesafe/jev-test", "answers": answers}).encode()

    monkeypatch.setattr(jev_client, "urllib_transport", transport)
    monkeypatch.setenv("REVIEWNLP_JEV_ENABLED", "1")
    monkeypatch.setenv("REVIEWNLP_JEV_ROUTE", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-typesafe-test")
    response = default_app.post("/triage", json=payload())
    assert response.status_code == 200 and response.json()["complaints"]["status"] == "ok"
    assert response.json()["complaints"]["route"] == "openrouter"
    assert sent == [("https://openrouter.ai/api/v1/systemone", "sk-or-test", REVIEW)]


def test_jev_switched_on_without_its_key_is_reported_on_every_response_and_sends_nothing(default_app, monkeypatch):
    monkeypatch.setattr(jev_client, "urllib_transport", lambda *a: (_ for _ in ()).throw(AssertionError("sent")))
    monkeypatch.setenv("REVIEWNLP_JEV_ENABLED", "1")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    body = default_app.post("/triage", json=payload()).json()
    assert (body["complaints"]["status"], body["complaints"]["error"], body["status"]) == ("error", "not_configured", "partial")
    assert "complaint_check_failed" in body["routing"]["review_reasons"]  # the stub's confidence varies by process


def test_a_client_built_by_the_router_is_the_one_configured_from_the_environment(default_app, monkeypatch):
    monkeypatch.setenv("REVIEWNLP_JEV_ENABLED", "1")
    monkeypatch.setenv("REVIEWNLP_JEV_MODEL", "jev-pinned")
    default_app.post("/triage", json=payload())
    pipeline = triage_router._pipeline
    assert isinstance(pipeline.jev, JevClient) and pipeline.jev.config == JevConfig(
        enabled=True, route="typesafe", api_key="", model="jev-pinned")
    assert pipeline.generator is None
