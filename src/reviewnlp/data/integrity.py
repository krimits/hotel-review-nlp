"""Shared review identity rules for dataset construction and validation."""

from __future__ import annotations

import hashlib
import unicodedata

NORMALIZATION_VERSION = "nfkc-whitespace-casefold-v1"


def canonical_text(text: str) -> str:
    """Normalize identity without changing the text supplied to a model."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def review_id(text: str) -> str:
    """Stable identity shared by normalized variants, independent of the label."""
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()
