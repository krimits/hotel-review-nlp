"""The Jev client, offline: what it sends and to whom, how it fails, and that it fails without the review."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reviewnlp.triage import jev_client, questions
from reviewnlp.triage.jev_client import JevClient, JevConfig, JevError
from reviewnlp.triage.schemas import OTHER, TOPICS

REVIEW = "SECRETWORD the shower was cold but the staff were lovely"


def _answer(choice: str = "0", p_yes: float | None = None) -> dict:
    p_yes = (1.0 if choice == "1" else 0.0) if p_yes is None else p_yes
    rest = {"0": 1 - p_yes, "unsure": 0.0}
    if choice == "unsure":
        rest = {"0": 0.3, "unsure": 0.7 - p_yes}
    return {"type": "choice", "choice": choice, "confidence": 0.9,
            "probabilities": {"1": p_yes, **rest}}


def body_for(**choices: str) -> dict:
    """A System One answer: every question 0, except the ones named."""
    names = [*TOPICS, questions.OTHER_QUESTION]
    return {"model": "typesafe/jev-test", "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {name: _answer(choices.get(name, "0")) for name in names}}


class FakeTransport:
    """Records every call and answers from a script: a body for a 200, a (status, headers) pair, or an exception."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def __call__(self, url, body, api_key, timeout):
        self.calls.append({"url": url, "body": body, "key": api_key, "timeout": timeout})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            status, headers = item
            return status, headers, b'{"detail": "SECRETWORD must not be repeated"}'
        return 200, {}, json.dumps(item).encode()


class FakeTime:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def client(*script, enabled=True, route="typesafe", key="sk-test", **config):
    fake_time = FakeTime()
    transport = FakeTransport(*script)
    cfg = JevConfig(enabled=enabled, route=route, api_key=key, **config)
    return JevClient(cfg, transport, fake_time.sleep, fake_time.clock), transport, fake_time


# --- Configuration -----------------------------------------------------------------------------------------------

def test_it_is_off_unless_asked_for_and_then_sends_nothing():
    assert JevConfig.from_env({}).enabled is False
    assert JevConfig.from_env({"TYPESAFE_API_KEY": "sk-test"}).enabled is False  # a key alone does not switch it on
    off, transport, _ = client(body_for(), enabled=False)
    assert off.enabled is False
    with pytest.raises(JevError) as raised:
        off.classify(REVIEW)
    assert raised.value.kind == "disabled" and transport.calls == []


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("YES", True), ("0", False), ("", False),
                                             ("off", False)])
def test_the_switch(value, expected):
    assert JevConfig.from_env({"REVIEWNLP_JEV_ENABLED": value}).enabled is expected


def test_each_route_has_its_own_key_and_address():
    typesafe = JevConfig.from_env({"REVIEWNLP_JEV_ENABLED": "1", "TYPESAFE_API_KEY": " sk-t ", "OPENROUTER_API_KEY": "sk-o"})
    openrouter = JevConfig.from_env({"REVIEWNLP_JEV_ENABLED": "1", "REVIEWNLP_JEV_ROUTE": "OpenRouter",
                                     "TYPESAFE_API_KEY": "sk-t", "OPENROUTER_API_KEY": "sk-o"})
    assert (typesafe.route, typesafe.api_key, openrouter.route, openrouter.api_key) == (
        "typesafe", "sk-t", "openrouter", "sk-o")
    for route, address in (("typesafe", "https://api.typesafe.ai/v1/systemone"),
                           ("openrouter", "https://openrouter.ai/api/v1/systemone")):
        sender, transport, _ = client(body_for(), route=route, key="sk-" + route)
        sender.classify(REVIEW)
        assert transport.calls[0]["url"] == address and transport.calls[0]["key"] == "sk-" + route


def test_a_key_for_the_other_route_is_never_used_and_a_missing_key_sends_nothing():
    wrong = JevConfig.from_env({"REVIEWNLP_JEV_ENABLED": "1", "REVIEWNLP_JEV_ROUTE": "openrouter",
                                "TYPESAFE_API_KEY": "sk-typesafe-only"})
    assert wrong.api_key == ""
    for config in (wrong, JevConfig(enabled=True, route="elsewhere", api_key="sk-x")):
        transport = FakeTransport(body_for())
        with pytest.raises(JevError) as raised:
            JevClient(config, transport).classify(REVIEW)
        assert raised.value.kind == "not_configured" and transport.calls == []


def test_a_bad_number_in_the_environment_is_refused_loudly():
    for name in ("REVIEWNLP_JEV_TIMEOUT_S", "REVIEWNLP_JEV_DEADLINE_S", "REVIEWNLP_JEV_MAX_ATTEMPTS"):
        for value in ("soon", "0", "-1"):
            with pytest.raises(ValueError, match=name):
                JevConfig.from_env({name: value})
    assert JevConfig.from_env({"REVIEWNLP_JEV_MAX_ATTEMPTS": "0.5"}).max_attempts == 1


def test_the_key_is_not_in_the_configs_representation():
    assert "sk-secret" not in repr(JevConfig(enabled=True, api_key="sk-secret"))


# --- A successful call ---------------------------------------------------------------------------------------------

def test_one_call_sends_the_review_and_every_question_and_reads_every_answer():
    sender, transport, _ = client(body_for(bathroom="1", pests="unsure", other_complaint="1"))
    result = sender.classify(REVIEW)
    call = transport.calls[0]
    assert call["body"] == questions.request_body(REVIEW, "jev-latest") and len(transport.calls) == 1
    assert [t.topic for t in result.topics] == list(TOPICS) and result.other_complaint.topic == OTHER
    answers = {t.topic: (t.answer, t.probability) for t in result.topics}
    assert answers["bathroom"] == ("yes", 1.0) and answers["pests"][0] == "unsure"
    assert answers["cleanliness"] == ("no", 0.0) and result.other_complaint.answer == "yes"
    assert (result.model, result.route) == ("typesafe/jev-test", "typesafe")


@pytest.mark.parametrize("damage", [
    lambda body: body["answers"].pop("responsiveness"),
    lambda body: body["answers"].pop("other_complaint"),
    lambda body: body["answers"]["bathroom"].update(type="noul"),
    lambda body: body["answers"]["bathroom"].update(choice="maybe"),
    lambda body: body["answers"]["bathroom"]["probabilities"].pop("unsure"),
    lambda body: body["answers"]["bathroom"]["probabilities"].update({"1": 1.5}),
    lambda body: body["answers"]["bathroom"]["probabilities"].update({"1": "high"}),
    lambda body: body.update(answers=[]),
    lambda body: body.update(answers=None),
])
def test_an_answer_that_is_not_in_the_documented_form_is_refused_without_retrying(damage):
    body = body_for()
    damage(body)
    sender, transport, _ = client(body)
    with pytest.raises(JevError) as raised:
        sender.classify(REVIEW)
    assert raised.value.kind == "invalid_answer" and len(transport.calls) == 1


def test_an_answer_that_is_not_json_is_refused():
    class Garbage(FakeTransport):
        def __call__(self, *args):
            self.calls.append(args)
            return 200, {}, b"<html>busy</html>"

    transport = Garbage()
    with pytest.raises(JevError) as raised:
        JevClient(JevConfig(enabled=True, api_key="sk-test"), transport).classify(REVIEW)
    assert raised.value.kind == "invalid_answer"


# --- Failures --------------------------------------------------------------------------------------------------------

def test_a_busy_provider_is_retried_as_it_asks_and_then_answers():
    sender, transport, fake_time = client((429, {"Retry-After-Ms": "1500"}), (503, {}), body_for())
    assert sender.classify(REVIEW).model == "typesafe/jev-test"
    assert len(transport.calls) == 3 and fake_time.sleeps == [1.5, 1.0]


def test_retries_are_bounded_and_the_last_kind_is_reported():
    sender, transport, fake_time = client((503, {}), max_attempts=3)
    with pytest.raises(JevError) as raised:
        sender.classify(REVIEW)
    assert raised.value.kind == "http_503" and len(transport.calls) == 3 and fake_time.sleeps == [0.5, 1.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_refusal_is_not_retried(status):
    sender, transport, fake_time = client((status, {}))
    with pytest.raises(JevError) as raised:
        sender.classify(REVIEW)
    assert raised.value.kind == f"http_{status}" and len(transport.calls) == 1 and fake_time.sleeps == []


@pytest.mark.parametrize("failure, kind", [(TimeoutError(), "timeout"), (ConnectionResetError(), "network"),
                                           (OSError("down"), "network")])
def test_timeouts_and_network_failures_are_retried_then_named(failure, kind):
    sender, transport, _ = client(failure, max_attempts=2)
    with pytest.raises(JevError) as raised:
        sender.classify(REVIEW)
    assert raised.value.kind == kind and len(transport.calls) == 2


def test_the_deadline_stops_retries_and_shortens_each_attempts_timeout():
    sender, transport, fake_time = client((429, {"Retry-After": "4"}), deadline_s=7.0, timeout_s=10.0, max_attempts=5)
    with pytest.raises(JevError) as raised:
        sender.classify(REVIEW)
    assert raised.value.kind == "http_429"
    assert [round(call["timeout"], 1) for call in transport.calls] == [7.0, 3.0]  # the second wait would pass the deadline
    assert fake_time.sleeps == [4.0]


def test_a_failure_carries_neither_the_review_nor_the_providers_body(caplog):
    sender, _, _ = client((500, {}), max_attempts=1)
    with caplog.at_level(logging.DEBUG, logger="reviewnlp.triage"), pytest.raises(JevError) as raised:
        sender.classify(REVIEW)
    assert "SECRETWORD" not in str(raised.value) and "SECRETWORD" not in caplog.text and "sk-test" not in caplog.text
    assert "http_500" in caplog.text


# --- The real transport, against a local server ---------------------------------------------------------------------

@pytest.fixture
def servers(monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    seen = {"api": [], "other": []}

    def handler(name, redirect_to=None):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                seen[name].append((self.path, dict(self.headers), json.loads(body)))
                if redirect_to and self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", f"{redirect_to}/stolen")
                    self.end_headers()
                    return
                payload = json.dumps(body_for(responsiveness="1")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass
        return Handler

    other = ThreadingHTTPServer(("127.0.0.1", 0), handler("other"))
    api = ThreadingHTTPServer(("127.0.0.1", 0), handler("api", f"http://127.0.0.1:{other.server_port}"))
    threads = [threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
               for server in (api, other)]
    for thread in threads:
        thread.start()
    yield f"http://127.0.0.1:{api.server_port}", seen
    for server in (api, other):
        server.shutdown()
        server.server_close()


def test_the_real_transport_sends_the_documented_request(servers):
    base, seen = servers
    status, _, raw = jev_client.urllib_transport(f"{base}/v1/systemone", {"state": "hello"}, "sk-secret", 5)
    assert status == 200 and json.loads(raw)["answers"]["responsiveness"]["choice"] == "1"
    path, headers, body = seen["api"][0]
    assert (path, body) == ("/v1/systemone", {"state": "hello"})
    assert headers["Authorization"] == "Bearer sk-secret" and headers["Content-Type"] == "application/json"


def test_a_redirect_is_an_error_and_the_key_stays_put(servers, monkeypatch):
    base, seen = servers
    monkeypatch.setitem(jev_client.ROUTES, "typesafe", {"base_url": base, "key_env": "TYPESAFE_API_KEY"})
    monkeypatch.setattr(jev_client, "SYSTEM_ONE_PATH", "/redirect")
    sender = JevClient(JevConfig(enabled=True, api_key="sk-secret", max_attempts=2))
    with pytest.raises(JevError) as raised:
        sender.classify(REVIEW)
    assert raised.value.kind == "redirect" and seen["other"] == [] and len(seen["api"]) == 1  # not retried either


def test_a_connection_that_is_refused_is_a_network_failure():
    sender = JevClient(JevConfig(enabled=True, api_key="sk-secret", max_attempts=1, timeout_s=2),
                       transport=lambda url, body, key, timeout: jev_client.urllib_transport(
                           "http://127.0.0.1:9/v1/systemone", body, key, timeout))
    with pytest.raises(JevError) as raised:
        sender.classify(REVIEW)
    assert raised.value.kind == "network"
