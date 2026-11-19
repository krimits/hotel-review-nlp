"""One structured-output call to OpenRouter, with the waits the hotel pilot's Space ran with.

A rate limit, a server error or a network failure is waited out a few times; any other status fails at once,
because a wrong key or an empty balance does not get better by asking again. The transport is injectable so tests
never touch the network. The key is read from the environment by the caller and never logged.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-haiku-5.5-20261007"  # the model of the hotel pilot
TRANSIENT = {408, 429, 500, 502, 503, 504}
WAITS = (1.0, 3.0, 6.0)
TIMEOUT = 120.0
Transport = Callable[[str, dict, str, float], tuple[int, bytes]]


class LLMError(RuntimeError):
    """The call gave no usable answer. `kind` is for logs, never for the hotel, and never holds the key."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


def urllib_transport(url: str, body: dict, api_key: str, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - a fixed https URL
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@dataclass
class OpenRouterClient:
    api_key: str
    model: str = DEFAULT_MODEL
    transport: Transport = urllib_transport
    sleep: Callable[[float], None] = field(default=time.sleep)

    @classmethod
    def from_env(cls) -> OpenRouterClient | None:
        """None without a key: the caller then refuses to analyse, or uses the demo's stand-in if asked to."""
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        return cls(api_key=key, model=os.getenv("REVIEWNLP_OPS_MODEL", DEFAULT_MODEL).strip()) if key else None

    def json_call(self, system: str, user: str, name: str, schema: dict) -> dict:
        """The parsed JSON object of one answer that follows `schema`."""
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": name, "strict": True, "schema": schema}}}
        raw = self._send(body)
        try:
            content = json.loads(raw)["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(content)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            raise LLMError(f"unparsable answer: {type(error).__name__}") from None
        if not isinstance(parsed, dict):
            raise LLMError("the answer is not a JSON object")
        return parsed

    def _send(self, body: dict) -> bytes:
        kind = "no call made"
        for wait in (*WAITS, None):
            try:
                status, raw = self.transport(URL, body, self.api_key, TIMEOUT)
            except (OSError, http.client.HTTPException) as error:
                kind = f"network: {type(error).__name__}"
            else:
                if status == 200:
                    return raw
                kind = f"status {status}"
                if status not in TRANSIENT:
                    raise LLMError(kind)
            if wait is None:
                break
            self.sleep(wait)
        raise LLMError(kind)
