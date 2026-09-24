"""Hotel-scoped API keys. Configure SHA256 digests, never raw keys, in env."""

from __future__ import annotations

import hashlib
import hmac
import json
import os

from fastapi import Header, HTTPException


def _configured_keys() -> dict[str, list[str]] | None:
    serialized = os.getenv("REVIEWNLP_API_KEYS_JSON")
    if not serialized:
        return None
    try:
        configured = json.loads(serialized)
        if not isinstance(configured, dict) or not configured:
            raise ValueError("expected a nonempty object")
        if not all(isinstance(k, str) and len(k) == 64 and
                   all(char in "0123456789abcdef" for char in k) and
                   isinstance(v, list) and all(isinstance(x, str) for x in v)
                   for k, v in configured.items()):
            raise ValueError("each digest must map to a list of hotel IDs")
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="API key configuration is invalid") from exc
    return configured


def validate_production_config() -> None:
    if not _configured_keys():
        raise ValueError("production requires valid REVIEWNLP_API_KEYS_JSON")


def _key_scope(token: str | None) -> list[str]:
    configured = _configured_keys()
    if configured is None:
        if os.getenv("REVIEWNLP_DB_PATH") or os.getenv("REVIEWNLP_ENV") == "production":
            raise HTTPException(status_code=503, detail="API keys are required when persistence is enabled")
        return []  # explicitly local demo without database
    if not token:
        raise HTTPException(status_code=401, detail="X-API-Key required")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    scope = None
    for saved, hotels in configured.items():
        if hmac.compare_digest(digest, saved):
            scope = hotels
    if scope is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return scope


def authorize_hotel(hotel_id: str, token: str | None) -> None:
    scope = _key_scope(token)
    if os.getenv("REVIEWNLP_API_KEYS_JSON") and hotel_id not in scope:
        raise HTTPException(status_code=403, detail="API key cannot access this hotel")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    _key_scope(x_api_key)
