"""Session-only review triage for the Hugging Face product demo.

All aggregates are calculated from successfully parsed, quoted model output.
There is no shared hotel database in this demo.
"""

from __future__ import annotations

from reviewnlp.absa.aspects import ASPECTS, SENTIMENTS
from reviewnlp.absa.extract import overall_from_aspects, review_key
from reviewnlp.analytics.recommendations import build_analytics

ASPECT_NAMES = {
    "cleanliness": "Καθαριότητα",
    "staff": "Προσωπικό",
    "location": "Τοποθεσία",
    "room": "Δωμάτιο",
    "food": "Φαγητό / πρωινό",
    "noise": "Θόρυβος",
    "value": "Σχέση τιμής-αξίας",
    "facilities": "Παροχές",
}
SENTIMENT_NAMES = {
    "positive": "Θετικό",
    "negative": "Αρνητικό",
    "neutral": "Ουδέτερο",
    None: "Μικτό / αβέβαιο",
}


def accept_record(history: list[dict] | None, text: str, record: dict) -> list[dict]:
    """Add one review only when generation produced valid structured output.

Reject incomplete generations and outputs repaired from surrounding prose.
An empty, valid JSON array is also excluded from the complaint counts.
"""
    if (not record.get("json_valid") or record.get("salvaged")
            or record.get("generation_hit_token_budget")):
        return list(history or [])
    grouped: dict[str, list[dict]] = {}
    for aspect in record.get("aspects") or []:
        if (aspect.get("aspect") not in ASPECTS
                or aspect.get("sentiment") not in SENTIMENTS
                or not aspect.get("quote")
                or review_key(aspect["quote"]) not in review_key(text)):
            continue
        grouped.setdefault(aspect["aspect"], []).append(aspect)
    aspects = []
    for name, entries in grouped.items():
        counts = {sentiment: sum(a["sentiment"] == sentiment for a in entries)
                  for sentiment in SENTIMENTS}
        winner = max(counts, key=counts.get)
        if list(counts.values()).count(counts[winner]) > 1:
            winner = "neutral"
        quote = next((a["quote"] for a in entries if a["sentiment"] == winner),
                     entries[0]["quote"])
        aspects.append({"aspect": name, "sentiment": winner, "quote": quote})
    if not aspects:
        return list(history or [])
    accepted = list(history or [])
    accepted.append({"review_id": len(accepted) + 1, "text": text, "aspects": aspects})
    return accepted


def render(history: list[dict] | None, latest: dict | None = None) -> tuple:
    """Return grounded UI rows: latest aspects, session reviews, complaints, evidence."""
    history = history or []
    latest_aspects = (latest or {}).get("aspects") or []
    aspect_rows = [
        [ASPECT_NAMES[a["aspect"]], SENTIMENT_NAMES[a["sentiment"]], a["quote"]]
        for a in latest_aspects
    ]
    review_rows = [
        [entry["review_id"], SENTIMENT_NAMES[overall_from_aspects(entry["aspects"])],
         len({a["aspect"] for a in entry["aspects"]})]
        for entry in history
    ]
    rows = [
        {"review_id": entry["review_id"], **aspect}
        for entry in history for aspect in entry["aspects"]
    ]
    analytics = [a for a in build_analytics(rows) if a["negative_count"]]
    complaint_rows = [
        [ASPECT_NAMES[item["aspect"]], item["negative_count"], item["review_count"],
         round(item["negative_rate"] * 100), item["recommendation"]]
        for item in analytics
    ]
    complaint_aspects = {a["aspect"] for a in analytics}
    evidence_rows = [
        [entry["review_id"], ASPECT_NAMES[a["aspect"]], a["quote"]]
        for entry in history for a in entry["aspects"]
        if a["sentiment"] == "negative" and a["aspect"] in complaint_aspects
    ]
    return aspect_rows, review_rows, complaint_rows, evidence_rows
