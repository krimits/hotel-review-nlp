"""Calls TypeSafe's Jev (System One) for the complaint topics of one review.

Off unless asked for. Nothing leaves the server unless REVIEWNLP_JEV_ENABLED is set and the key for the
chosen route is present. The review goes to one of two places, TypeSafe's own API or OpenRouter's System One
endpoint, and a key is only ever sent to its own provider. A redirect is an error, so a key cannot be carried
to another address.

Time is bounded: each attempt has a timeout, there are few attempts, and the whole call has a deadline, so a
slow provider cannot hold a request open. A failure is reported as a kind (`timeout`, `http_503`,
`invalid_answer`, ...) and never carries the review or the provider's body.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from reviewnlp.triage.questions import LABELS, TOPIC_OF_QUESTION, request_body
from reviewnlp.triage.schemas import OTHER, ComplaintTopic

log = logging.getLogger("reviewnlp.triage")

SYSTEM_ONE_PATH = "/v1/systemone"
ROUTES = {
    "typesafe": {"base_url": "https://api.typesafe.ai", "key_env": "TYPESAFE_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api", "key_env": "OPENROUTER_API_KEY"},
}
RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
ANSWER_OF_LABEL = {"1": "yes", "0": "no", "unsure": "unsure"}


@dataclass(frozen=True)
class JevConfig:
    enabled: bool = False
    route: str = "typesafe"
    api_key: str = field(default="", repr=False)
    model: str = "jev-latest"
    timeout_s: float = 10.0
    deadline_s: float = 20.0
    max_attempts: int = 3

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> JevConfig:
        """REVIEWNLP_JEV_ENABLED, _ROUTE, _MODEL, _TIMEOUT_S, _DEADLINE_S, _MAX_ATTEMPTS, and the route's key."""
        env = os.environ if environ is None else environ
        route = env.get("REVIEWNLP_JEV_ROUTE", "typesafe").strip().lower()
        key_env = ROUTES.get(route, {}).get("key_env")
        return cls(
            enabled=env.get("REVIEWNLP_JEV_ENABLED", "").strip().lower() in {"1", "true", "yes"},
            route=route,
            api_key=env.get(key_env, "").strip() if key_env else "",
            model=env.get("REVIEWNLP_JEV_MODEL", "jev-latest").strip() or "jev-latest",
            timeout_s=_number(env, "REVIEWNLP_JEV_TIMEOUT_S", 10.0),
            deadline_s=_number(env, "REVIEWNLP_JEV_DEADLINE_S", 20.0),
            max_attempts=max(1, int(_number(env, "REVIEWNLP_JEV_MAX_ATTEMPTS", 3))),
        )


def _number(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class JevError(Exception):
    """A failed call, named by its kind alone."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True)
class JevResult:
    topics: list[ComplaintTopic]
    other_complaint: ComplaintTopic
    model: str | None
    route: str


Transport = Callable[[str, dict, str, float], tuple[int, dict[str, str], bytes]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def urllib_transport(url: str, body: dict, api_key: str, timeout: float) -> tuple[int, dict[str, str], bytes]:
    """One POST: (status, headers, body). Timeouts and network failures raise; a redirect is a status."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError from error
        raise


def retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    """The server's retry-after-ms or retry-after, else 0.5, 1, 2... seconds; never more than 5."""
    lowered = {name.lower(): value for name, value in headers.items()}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            return min(max(float(lowered[name]) * scale, 0.0), 5.0)
        except (KeyError, ValueError):
            continue
    return min(0.5 * 2.0 ** (attempt - 1), 5.0)


def _topic(item: object, topic: str) -> ComplaintTopic:
    """One answer, from the documented form: a choice among 1, 0 and unsure, with a probability for each."""
    if not isinstance(item, dict) or item.get("type") != "choice":
        raise JevError("invalid_answer")
    choice, probabilities = item.get("choice"), item.get("probabilities")
    if choice not in LABELS or not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
        raise JevError("invalid_answer")
    try:
        p_yes = float(probabilities["1"])
    except (TypeError, ValueError):
        raise JevError("invalid_answer") from None
    if not 0.0 <= p_yes <= 1.0:
        raise JevError("invalid_answer")
    return ComplaintTopic(topic=topic, answer=ANSWER_OF_LABEL[choice], probability=p_yes)


def parse_answer(raw: bytes, route: str) -> JevResult:
    try:
        body = json.loads(raw)
        answers = body["answers"]
        topics = [_topic(answers[question], topic) for question, topic in TOPIC_OF_QUESTION.items() if topic != OTHER]
        other = _topic(answers["other_complaint"], OTHER)
    except (ValueError, KeyError, TypeError):
        raise JevError("invalid_answer") from None
    model = body.get("model")
    return JevResult(topics=topics, other_complaint=other, model=model if isinstance(model, str) else None, route=route)


class JevClient:
    def __init__(self, config: JevConfig | None = None, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.config = config or JevConfig()
        self._transport, self._sleep, self._clock = transport or urllib_transport, sleep, clock

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def classify(self, text: str) -> JevResult:
        """The complaint topics of one review. Raises JevError; sends nothing unless it is enabled and configured."""
        config = self.config
        if not config.enabled:
            raise JevError("disabled")
        route = ROUTES.get(config.route)
        if route is None or not config.api_key:
            raise JevError("not_configured")
        url, body = route["base_url"] + SYSTEM_ONE_PATH, request_body(text, config.model)
        start = self._clock()
        kind = "timeout"
        for attempt in range(1, config.max_attempts + 1):
            remaining = config.deadline_s - (self._clock() - start)
            if remaining <= 0:
                raise JevError("timeout")
            headers: Mapping[str, str] = {}
            try:
                status, headers, raw = self._transport(url, body, config.api_key, min(config.timeout_s, remaining))
            except TimeoutError:
                kind, retryable = "timeout", True
            except (OSError, http.client.HTTPException):
                kind, retryable = "network", True
            else:
                if status == 200:
                    return parse_answer(raw, config.route)
                kind = "redirect" if 300 <= status < 400 else f"http_{status}"
                retryable = status in RETRY_STATUSES
            if not retryable or attempt == config.max_attempts:
                break
            delay = retry_delay(headers, attempt)
            if self._clock() - start + delay >= config.deadline_s:
                break
            self._sleep(delay)
        log.warning("jev call failed: %s", kind)
        raise JevError(kind)
