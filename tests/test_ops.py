"""The operations demo: findings with evidence and status, tickets after confirmation, the outbox and its worker,
reply drafts from what was done, and the /ops routes. Offline: the model is a fake transport or the stand-ins.

The owner's acceptance checks for the demo (18 November 2026) are the three tests marked ACCEPTANCE:
the same submission does not double the tickets; a transient notification failure recovers; a complaint already
resolved does not open an urgent ticket.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from email import message_from_bytes
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reviewnlp.ops import notify
from reviewnlp.ops.catalogue import load_catalogue
from reviewnlp.ops.findings import (
    Analysis,
    FakeAnalyser,
    Finding,
    FindingsError,
    LLMAnalyser,
    check,
    findings_prompt,
)
from reviewnlp.ops.llm import LLMError, OpenRouterClient
from reviewnlp.ops.replies import LLMDrafter, TemplateDrafter
from reviewnlp.ops.service import NeedsInput, OpsService
from reviewnlp.ops.store import Conflict, SqliteOpsStore
from reviewnlp.serving import ops_router
from reviewnlp.serving.app import app

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = load_catalogue()
MIXED = ("The staff were wonderful and very kind. The air conditioning did not work until we left. "
         "The sheets were stained.")
T0 = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, start: datetime = T0):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class ScriptedAnalyser:
    """Returns the findings it is given, and counts calls: a second analysis of the same review is a bug."""

    name = "scripted"

    def __init__(self, *findings: Finding, language: str = "en"):
        self.findings, self.language, self.calls = findings, language, 0

    def analyse(self, text: str) -> Analysis:
        self.calls += 1
        return Analysis(self.language, self.findings, self.name)


def service_with(tmp_path, analyser, clock=None) -> OpsService:
    return OpsService(SqliteOpsStore(tmp_path / "ops.db"), analyser,
                      TemplateDrafter({t.id: t.label_el for t in CATALOGUE.types.values()}), CATALOGUE,
                      {"*": "test@example.invalid", "maintenance": "maint@example.invalid"}, clock or Clock())


AC = Finding("hvac_failure", "The air conditioning did not work until we left.", "pending")
SHEETS = Finding("cleanliness", "The sheets were stained.", "unclear")
FIXED_AC = Finding("hvac_failure", "the AC broke and the staff repaired it", "resolved")


# --- the catalogue ---------------------------------------------------------------------------------------------------

def test_the_packaged_catalogue_and_labels_are_the_approved_files():
    pairs = {"catalogue_v2.yaml": "docs/experiments/triage_diagnosis/catalogue_v2.yaml",
             "labels_el.yaml": "docs/experiments/hotelier_blind_eval/labels_el.yaml"}
    for packaged, approved in pairs.items():
        assert (ROOT / "src/reviewnlp/ops" / packaged).read_bytes() == (ROOT / approved).read_bytes(), packaged


def test_the_catalogue_assigns_and_says_what_to_confirm():
    assert len(CATALOGUE.types) == 21
    hvac, incident, price = (CATALOGUE.types[t] for t in ("hvac_failure", "safety_incident", "price_value"))
    assert (hvac.department, hvac.priority, hvac.needs_room, hvac.opens_ticket) == ("maintenance", "urgent", True, True)
    assert (incident.department, incident.needs_room) == ("management", False)
    assert price.priority == "structural" and not price.opens_ticket


# --- findings ----------------------------------------------------------------------------------------------------------

def test_an_excerpt_must_be_the_guests_own_words():
    good = {"language": "en", "findings": [{"type": "hvac_failure", "excerpt": "did NOT work  until", "status": "pending"}]}
    assert check(good, MIXED, CATALOGUE)[1] == (Finding("hvac_failure", "did NOT work  until", "pending"),)
    for bad in ({"type": "hvac_failure", "excerpt": "The AC was broken", "status": "pending"},
                {"type": "spa", "excerpt": "The sheets were stained.", "status": "pending"},
                {"type": "cleanliness", "excerpt": "The sheets were stained.", "status": "maybe"}):
        with pytest.raises(FindingsError):
            check({"language": "en", "findings": [bad]}, MIXED, CATALOGUE)


def test_the_prompt_carries_the_catalogue_and_the_three_statuses():
    prompt = findings_prompt(CATALOGUE)
    assert all(f"`{t}`" in prompt for t in CATALOGUE.types) and "A problem the hotel fixed" in prompt
    assert all(f"`{s}`" in prompt for s in ("pending", "resolved", "unclear"))


def answer(content: dict) -> bytes:
    return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()


def test_the_model_is_asked_again_when_an_excerpt_is_invented_and_waits_out_a_busy_server():
    answers = [(503, b""), (200, answer({"language": "en", "findings": [
        {"type": "hvac_failure", "excerpt": "AC broken", "status": "pending"}]})),
        (200, answer({"language": "en", "findings": [
            {"type": "hvac_failure", "excerpt": "did not work until we left", "status": "pending"}]}))]
    sent, waits = [], []

    def transport(url, body, key, timeout):
        sent.append(body)
        return answers.pop(0)

    client = OpenRouterClient("sk-test", transport=transport, sleep=waits.append)
    analysis = LLMAnalyser(client, CATALOGUE).analyse(MIXED)
    assert analysis.findings[0].status == "pending" and len(sent) == 3 and waits == [1.0]
    assert sent[0]["response_format"]["json_schema"]["strict"] is True
    assert "sk-test" not in json.dumps(sent)


def test_a_wrong_key_fails_at_once():
    client = OpenRouterClient("sk-test", transport=lambda *a: (401, b"no"), sleep=lambda s: None)
    with pytest.raises(LLMError) as error:
        LLMAnalyser(client, CATALOGUE).analyse(MIXED)
    assert error.value.kind == "status 401"


def test_the_keyword_stand_in_reads_the_owners_example():
    text = ("Το προσωπικό ήταν εξαιρετικό. Ο κλιματισμός δεν λειτούργησε μέχρι την αναχώρηση. "
            "Τα σεντόνια ήταν λερωμένα.")
    found = {f.type: f.status for f in FakeAnalyser(CATALOGUE).analyse(text).findings}
    assert found == {"hvac_failure": "pending", "cleanliness": "unclear"}
    fixed = FakeAnalyser(CATALOGUE).analyse("The AC broke and the staff repaired it the same day.")
    assert [(f.type, f.status) for f in fixed.findings] == [("hvac_failure", "resolved")]


# --- the rules: tickets ------------------------------------------------------------------------------------------------

def test_ACCEPTANCE_the_same_submission_does_not_double_the_tickets(tmp_path):
    analyser = ScriptedAnalyser(AC, SHEETS)
    service = service_with(tmp_path, analyser)
    card = service.submit("H1", "booking", "r1", MIXED)
    ac, sheets = card["findings"]
    service.confirm("H1", ac["id"], room="214")
    service.confirm("H1", sheets["id"], still_open=True, room="214")
    again = service.submit("H1", "booking", "r1", MIXED)
    service.confirm("H1", ac["id"], room="214")  # a double click: the same decision again
    assert analyser.calls == 1 and len(again["findings"]) == 2
    assert len(service.tickets("H1")) == 2 and len(service.store.outbox("H1")) == 2
    # The same sentence in another review is another finding, with its own ticket.
    service.submit("H1", "booking", "r2", MIXED)
    other = service.card("H1", "booking", "r2")["findings"][0]
    service.confirm("H1", other["id"], room="301")
    assert len(service.tickets("H1")) == 3


def test_ACCEPTANCE_a_complaint_already_resolved_opens_no_urgent_ticket(tmp_path):
    service = service_with(tmp_path, ScriptedAnalyser(FIXED_AC))
    card = service.submit("H1", "booking", "r1", "Good stay; the AC broke and the staff repaired it quickly.")
    finding = card["findings"][0]
    assert finding["priority"] == "urgent" and finding["status_label"] == "Αναφέρεται ως λυμένο"
    card = service.confirm("H1", finding["id"])
    assert card["findings"][0]["decision"] == "confirmed" and card["findings"][0]["ticket_id"] is None
    assert service.tickets("H1") == [] and service.store.outbox("H1") == []


def test_an_unclear_problem_waits_for_staff_and_a_room_for_reception(tmp_path):
    service = service_with(tmp_path, ScriptedAnalyser(SHEETS))
    finding = service.submit("H1", "api", "r1", MIXED)["findings"][0]
    with pytest.raises(NeedsInput, match="προσωπικό"):
        service.confirm("H1", finding["id"])
    with pytest.raises(NeedsInput, match="ρεσεψιόν"):
        service.confirm("H1", finding["id"], still_open=True)
    assert service.card("H1", "api", "r1")["findings"][0]["decision"] == "proposed"  # nothing recorded
    card = service.confirm("H1", finding["id"], still_open=False)  # staff: no longer open
    assert card["findings"][0]["ticket_id"] is None


def test_a_ticket_goes_to_the_catalogues_department_with_its_notice_in_the_same_transaction(tmp_path):
    service = service_with(tmp_path, ScriptedAnalyser(AC))
    finding = service.submit("H1", "booking", "r1", MIXED)["findings"][0]
    service.confirm("H1", finding["id"], room="214")
    ticket = service.tickets("H1", department="maintenance")[0]
    assert (ticket["priority"], ticket["state_label"], ticket["room"]) == ("urgent", "Ανατέθηκε", "214")
    item = service.store.outbox("H1")[0]
    assert item["send_id"] == f"ticket-{ticket['id']}-assigned" and item["recipient"] == "maint@example.invalid"
    assert "214" in item["subject"] and item["state"] == "pending"


def test_tickets_move_forward_only_and_stay_within_their_hotel(tmp_path):
    service = service_with(tmp_path, ScriptedAnalyser(AC))
    finding = service.submit("H1", "booking", "r1", MIXED)["findings"][0]
    service.confirm("H1", finding["id"], room="214")
    ticket_id = service.tickets("H1")[0]["id"]
    assert service.move_ticket("H1", ticket_id, "in_progress")["state_label"] == "Σε εξέλιξη"
    with pytest.raises(Conflict):
        service.move_ticket("H1", ticket_id, "assigned")
    with pytest.raises(KeyError):
        service.move_ticket("H2", ticket_id, "resolved")
    with pytest.raises(KeyError):
        service.confirm("H2", finding["id"], room="1")


def test_the_store_itself_keeps_one_review_per_identity_and_text(tmp_path):
    store = SqliteOpsStore(tmp_path / "ops.db")
    departments = {t.id: (t.department, t.priority) for t in CATALOGUE.types.values()}
    analysis = Analysis("en", (AC,), "scripted")
    first = store.save_review("H1", "booking", "r1", MIXED, analysis, departments, T0)
    second = store.save_review("H1", "booking", "r1", MIXED, Analysis("en", (AC, SHEETS), "other"), departments, T0)
    assert first == second and len(store.review("H1", "booking", "r1")["findings"]) == 1


def test_a_changed_text_is_refused_once_a_finding_was_decided(tmp_path):
    service = service_with(tmp_path, ScriptedAnalyser(AC))
    finding = service.submit("H1", "booking", "r1", MIXED)["findings"][0]
    service.submit("H1", "booking", "r1", MIXED + " Edited before any decision.")  # replaced: nothing decided yet
    finding = service.card("H1", "booking", "r1")["findings"][0]
    service.reject("H1", finding["id"])
    with pytest.raises(Conflict):
        service.submit("H1", "booking", "r1", MIXED + " Edited after a decision.")
    with pytest.raises(Conflict):
        service.confirm("H1", finding["id"], room="1")


# --- the outbox ----------------------------------------------------------------------------------------------------------

class FlakyTransport:
    name = "flaky"

    def __init__(self, failures: int):
        self.failures, self.sent = failures, []

    def send(self, send_id, recipient, subject, body):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("mail server unreachable")
        self.sent.append(send_id)


def confirmed_ticket(tmp_path, clock):
    service = service_with(tmp_path, ScriptedAnalyser(AC), clock)
    finding = service.submit("H1", "booking", "r1", MIXED)["findings"][0]
    service.confirm("H1", finding["id"], room="214")
    return service


def test_ACCEPTANCE_a_transient_notification_failure_recovers(tmp_path):
    clock = Clock()
    service = confirmed_ticket(tmp_path, clock)
    transport = FlakyTransport(failures=1)
    assert notify.deliver_due(service.store, transport, now=clock.now) == {"sent": 0, "retry": 1, "failed": 0}
    assert notify.deliver_due(service.store, transport, now=clock.now) == {"sent": 0, "retry": 0, "failed": 0}
    later = clock.now + timedelta(seconds=notify.FIRST_WAIT_SECONDS)
    assert notify.deliver_due(service.store, transport, now=later) == {"sent": 1, "retry": 0, "failed": 0}
    item = service.store.outbox("H1")[0]
    assert (item["state"], item["attempts"], transport.sent) == ("sent", 2, [item["send_id"]])
    assert notify.deliver_due(service.store, transport, now=later + timedelta(hours=1))["sent"] == 0


def test_a_notification_is_given_up_after_the_last_try_and_shown_failed(tmp_path):
    clock = Clock()
    service = confirmed_ticket(tmp_path, clock)
    transport, now = FlakyTransport(failures=99), clock.now
    for _ in range(notify.MAX_ATTEMPTS):
        notify.deliver_due(service.store, transport, now=now)
        now += timedelta(hours=2)
    item = service.store.outbox("H1")[0]
    assert (item["state"], item["attempts"]) == ("failed", notify.MAX_ATTEMPTS)
    assert "ConnectionError" in item["last_error"]


def test_a_claimed_notification_is_not_sent_twice_by_two_workers(tmp_path):
    clock = Clock()
    service = confirmed_ticket(tmp_path, clock)
    first = service.store.claim_due(clock.now, notify.LEASE_SECONDS)
    assert len(first) == 1 and service.store.claim_due(clock.now, notify.LEASE_SECONDS) == []


def test_test_email_is_one_file_per_send_id_with_that_message_id(tmp_path):
    clock = Clock()
    service = confirmed_ticket(tmp_path, clock)
    transport = notify.FileTransport(tmp_path / "mail")
    notify.deliver_due(service.store, transport, now=clock.now)
    [mail] = list((tmp_path / "mail").glob("*.eml"))
    parsed = message_from_bytes(mail.read_bytes())
    assert mail.stem == parsed["Message-ID"].strip("<>").split("@")[0] == "ticket-1-assigned"
    assert "#1" in parsed.get_payload(decode=True).decode("utf-8")


def test_smtp_verifies_the_servers_certificate_before_logging_in(monkeypatch):
    seen = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            seen["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self, context=None):
            seen["context"] = context

        def login(self, user, password):
            seen["login_after_tls"] = "context" in seen

        def send_message(self, mail):
            seen["message_id"] = mail["Message-ID"]

    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    notify.SmtpTransport("smtp.example.invalid", 587, "user", "pw", "from@example.invalid").send(
        "ticket-1-assigned", "to@example.invalid", "subject", "body")
    context = seen["context"]
    assert context is not None and context.check_hostname and context.verify_mode == notify.ssl.CERT_REQUIRED
    assert seen["login_after_tls"] and seen["message_id"] == "<ticket-1-assigned@reviewnlp.ops>"


def test_without_smtp_settings_the_transport_is_test_email(monkeypatch, tmp_path):
    monkeypatch.delenv("REVIEWNLP_OPS_SMTP_HOST", raising=False)
    monkeypatch.setenv("REVIEWNLP_OPS_MAIL_DIR", str(tmp_path))
    assert isinstance(notify.transport_from_env(), notify.FileTransport)
    monkeypatch.setenv("REVIEWNLP_OPS_SMTP_HOST", "smtp.example.invalid")
    assert isinstance(notify.transport_from_env(), notify.SmtpTransport)


# --- replies ---------------------------------------------------------------------------------------------------------

def test_the_reply_waits_for_every_decision_and_says_only_what_was_done(tmp_path):
    service = service_with(tmp_path, ScriptedAnalyser(AC, SHEETS))
    ac, sheets = service.submit("H1", "booking", "r1", MIXED)["findings"]
    service.confirm("H1", ac["id"], room="214")
    with pytest.raises(NeedsInput):
        service.draft_reply("H1", "booking", "r1")
    service.reject("H1", sheets["id"])
    draft = service.draft_reply("H1", "booking", "r1")["reply"]["draft"]
    assert "passed" in draft and "put right" not in draft and "214" not in draft and "sheets" not in draft
    service.move_ticket("H1", service.tickets("H1")[0]["id"], "resolved")
    assert "put right" in service.draft_reply("H1", "booking", "r1")["reply"]["draft"]
    card = service.approve_reply("H1", "booking", "r1", "Thank you. Edited by the manager.")
    assert card["reply"]["state"] == "approved" and card["reply"]["approved_text"].endswith("manager.")


def test_the_model_drafts_from_states_and_never_sees_the_room(tmp_path):
    seen = []

    def transport(url, body, key, timeout):
        seen.append(body)
        return 200, answer({"reply": "Ευχαριστούμε. Διαβιβάσαμε το θέμα."})

    service = service_with(tmp_path, ScriptedAnalyser(AC, language="el"))
    service.drafter = LLMDrafter(OpenRouterClient("sk-test", transport=transport, sleep=lambda s: None))
    finding = service.submit("H1", "booking", "r1", MIXED)["findings"][0]
    service.confirm("H1", finding["id"], room="214")
    assert service.draft_reply("H1", "booking", "r1")["reply"]["draft"].startswith("Ευχαριστούμε")
    sent = json.loads(seen[0]["messages"][1]["content"])
    assert sent["language"] == "el" and [p["state"] for p in sent["problems"]] == ["forwarded"]
    assert "214" not in json.dumps(seen)


# --- the routes ----------------------------------------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.delenv("REVIEWNLP_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("REVIEWNLP_DB_PATH", raising=False)
    service = service_with(tmp_path, FakeAnalyser(CATALOGUE))
    transport = notify.FileTransport(tmp_path / "mail")
    app.dependency_overrides[ops_router.get_ops_service] = lambda: service
    app.dependency_overrides[ops_router.get_ops_transport] = lambda: transport
    try:
        yield TestClient(app), tmp_path / "mail"
    finally:
        app.dependency_overrides.pop(ops_router.get_ops_service, None)
        app.dependency_overrides.pop(ops_router.get_ops_transport, None)


def test_the_demo_scenario_through_the_api(api):
    client, mail = api
    body = {"source": "booking", "review_id": "b-1",
            "text": "Το προσωπικό ήταν εξαιρετικό. Ο κλιματισμός δεν λειτούργησε μέχρι την αναχώρηση. "
                    "Τα σεντόνια ήταν λερωμένα."}
    card = client.post("/ops/hotels/H1/reviews", json=body).json()
    ac, sheets = card["findings"]
    assert (ac["department"], sheets["department"]) == ("maintenance", "housekeeping")
    refused = client.post(f"/ops/hotels/H1/findings/{ac['id']}/confirm", json={})
    assert refused.status_code == 422 and "ρεσεψιόν" in refused.json()["detail"]
    client.post(f"/ops/hotels/H1/findings/{ac['id']}/confirm", json={"room": "214"})
    client.post(f"/ops/hotels/H1/findings/{sheets['id']}/confirm", json={"still_open": True, "room": "214"})
    assert client.post("/ops/hotels/H1/reviews", json=body).status_code == 200  # resubmitted
    tickets = client.get("/ops/hotels/H1/tickets").json()
    assert sorted(t["department"] for t in tickets) == ["housekeeping", "maintenance"]
    assert len(list(mail.glob("*.eml"))) == 2  # sent at once by the confirm route
    assert all(item["state"] == "sent" for item in client.get("/ops/hotels/H1/outbox").json())
    reply = client.post("/ops/hotels/H1/reviews/booking/b-1/reply/draft").json()["reply"]["draft"]
    assert "διαβιβάσει" in reply and "214" not in reply
    moved = client.post(f"/ops/hotels/H1/tickets/{tickets[0]['id']}/state", json={"state": "in_progress"})
    assert moved.json()["state"] == "in_progress"
    assert client.post("/ops/hotels/H1/reviews/booking/b-1/reply/approve", json={"text": reply}).status_code == 200


def test_the_routes_enforce_hotel_access(api, monkeypatch):
    client, _ = api
    token = "key-for-H1"
    monkeypatch.setenv("REVIEWNLP_API_KEYS_JSON", json.dumps({hashlib.sha256(token.encode()).hexdigest(): ["H1"]}))
    body = {"review_id": "x", "text": "The sheets were stained."}
    assert client.post("/ops/hotels/H1/reviews", json=body).status_code == 401
    assert client.post("/ops/hotels/H2/reviews", json=body, headers={"X-API-Key": token}).status_code == 403
    assert client.post("/ops/hotels/H1/reviews", json=body, headers={"X-API-Key": token}).status_code == 200
    assert client.get("/ops/hotels/H2/tickets", headers={"X-API-Key": token}).status_code == 403


def test_without_a_database_or_an_analyser_the_routes_say_so(monkeypatch, tmp_path):
    monkeypatch.delenv("REVIEWNLP_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("REVIEWNLP_DB_PATH", raising=False)
    client = TestClient(app)
    assert client.get("/ops/hotels/H1/tickets").status_code == 501
    ops_router._service_for.cache_clear()
    monkeypatch.setenv("REVIEWNLP_DB_PATH", str(tmp_path / "ops.db"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("REVIEWNLP_OPS_DEMO", raising=False)
    monkeypatch.setenv("REVIEWNLP_API_KEYS_JSON", json.dumps({hashlib.sha256(b"k").hexdigest(): ["H1"]}))
    assert client.get("/ops/hotels/H1/tickets", headers={"X-API-Key": "k"}).status_code == 503
    ops_router._service_for.cache_clear()
    monkeypatch.setenv("REVIEWNLP_OPS_DEMO", "1")
    assert client.get("/ops/hotels/H1/tickets", headers={"X-API-Key": "k"}).json() == []
    ops_router._service_for.cache_clear()
