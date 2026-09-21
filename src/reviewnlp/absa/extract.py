"""Prompt construction, strict parsing, overall vote (v9).

Changes vs v8 (each addresses a verified review finding):
- format_absa_messages returns OpenAI-style dicts; chat formatting is done by
  tokenizer.apply_chat_template(..., add_generation_prompt=True) in pipeline.py.
  This is the ChatML format the adapter was fine-tuned on — the plain-text
  "system/user/assistant" prompt of v8 was not the model's format.
- parse_absa_output(raw, review): separates JSON salvage from compliance
  (salvaged flag), counts dropped entries instead of folding them into
  parse_ok, and validates quotes against the source review.
- overall_from_aspects: one vote per DISTINCT aspect per review (duplicate
  JSON entries can no longer flip the vote); mixed or all-neutral reviews
  abstain (None) — declared behavior, not a silent tie-break.
"""

from __future__ import annotations

import json
import re

from reviewnlp.absa.aspects import normalize_aspect, normalize_sentiment

ABSA_SYSTEM_PROMPT = (
    "You are a hotel review analysis assistant. "
    "Extract the aspects the review talks about. "
    "Aspects are exactly one of: cleanliness, staff, location, room, food, "
    "noise, value, facilities. "
    "For each aspect give the sentiment: positive, negative, or neutral, "
    "and a short quote from the review. "
    "Answer with a JSON array only, nothing else."
)

ABSA_USER_TEMPLATE = "Review:\n\n{review}\n\nExtract aspects as JSON."

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def review_key(text: str) -> str:
    """Normalization shared by quote validation — whitespace-collapsed, casefolded."""
    return " ".join(str(text).split()).casefold()


def format_absa_messages(review: str) -> list[dict]:
    review = " ".join(str(review).split())[:4000]
    return [
        {"role": "system", "content": ABSA_SYSTEM_PROMPT},
        {"role": "user", "content": ABSA_USER_TEMPLATE.format(review=review)},
    ]


def parse_absa_output(raw: str, review: str | None = None) -> dict:
    """Strict parse with evidence tracking.

    Returns: json_valid, salvaged (extra text around the array), entries_total,
    entries_kept, entries_dropped, quote_absent (no quote given),
    quote_not_in_review (quote not found in source), quote_truncated,
    empty_valid (valid JSON, zero valid entries — a legitimate outcome that
    v8 could not distinguish from all-entries-rejected), aspects, error.
    """
    text = str(raw).strip()
    out = {
        "json_valid": False, "salvaged": False,
        "entries_total": 0, "entries_kept": 0, "entries_dropped": 0,
        "quote_absent": 0, "quote_not_in_review": 0, "quote_truncated": 0,
        "aspects": [], "error": None,
    }
    if not text:
        out["error"] = "empty generation"
        return out
    match = _JSON_ARRAY_RE.search(text)
    if match is None:
        out["error"] = "no JSON array found"
        return out
    out["salvaged"] = bool(match.start() > 0 or match.end() < len(text))
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        out["error"] = f"invalid JSON: {exc}"
        return out
    if not isinstance(items, list):
        out["error"] = "JSON is not an array"
        return out

    out["json_valid"] = True
    out["entries_total"] = len(items)
    review_norm = review_key(review) if review is not None else None
    for item in items:
        if not isinstance(item, dict):
            out["entries_dropped"] += 1
            continue
        aspect = normalize_aspect(str(item.get("aspect", "")))
        sentiment = normalize_sentiment(str(item.get("sentiment", "")))
        if aspect is None or sentiment is None:
            out["entries_dropped"] += 1
            continue
        quote = " ".join(str(item.get("quote", "")).split())
        if not quote or quote.casefold() == "none":
            out["quote_absent"] += 1
        else:
            if len(quote) > 200:
                quote = quote[:200]
                out["quote_truncated"] += 1
            if review_norm is not None and quote.casefold() not in review_norm:
                out["quote_not_in_review"] += 1
        out["aspects"].append({"aspect": aspect, "sentiment": sentiment, "quote": quote})
        out["entries_kept"] += 1
    return out


def overall_from_aspects(aspects: list) -> str | None:
    """Vote once per distinct aspect (majority within aspect; ties -> neutral),
    then majority over aspects; mixed or all-neutral -> None (abstention)."""
    per_aspect: dict[str, list[str]] = {}
    for item in aspects:
        per_aspect.setdefault(item["aspect"], []).append(item["sentiment"])
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for sent_list in per_aspect.values():
        pos, neg, neu = sent_list.count("positive"), sent_list.count("negative"), sent_list.count("neutral")
        if pos > neg and pos > neu:
            counts["positive"] += 1
        elif neg > pos and neg > neu:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1
    if counts["positive"] > counts["negative"] and counts["positive"] > counts["neutral"]:
        return "positive"
    if counts["negative"] > counts["positive"] and counts["negative"] > counts["neutral"]:
        return "negative"
    return None
