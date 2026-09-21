"""Fixed aspect taxonomy: exact-match normalization only (v9).

Change vs v8: the substring fallback is removed. It mapped "food and staff"
to "staff" purely by ASPECTS tuple order. Unrecognized aspect strings now
return None and are counted in entries_dropped — never silently coerced.
"""

from __future__ import annotations

ASPECTS = (
    "cleanliness", "staff", "location", "room",
    "food", "noise", "value", "facilities",
)
SENTIMENTS = ("positive", "negative", "neutral")

# Exact-match map: canonical names plus explicitly approved alternatives only.
ASPECT_ALIASES = {
    "cleanliness": "cleanliness", "clean": "cleanliness", "hygiene": "cleanliness",
    "staff": "staff", "service": "staff", "employees": "staff", "reception": "staff",
    "room": "room", "bed": "room", "bathroom": "room",
    "food": "food", "breakfast": "food", "restaurant": "food", "bar": "food",
    "noise": "noise", "quiet": "noise", "sound": "noise",
    "value": "value", "price": "value", "money": "value",
    "facilities": "facilities", "wifi": "facilities", "pool": "facilities",
    "parking": "facilities",
    "location": "location",
}


def normalize_aspect(raw: str) -> str | None:
    text = str(raw).strip().lower().replace("-", " ").replace("_", " ").strip()
    return ASPECT_ALIASES.get(text)


def normalize_sentiment(raw: str) -> str | None:
    text = str(raw).strip().lower()
    return text if text in SENTIMENTS else None
