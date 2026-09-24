"""Evaluate aspect extraction against human annotations, never binary labels."""

from __future__ import annotations

import json
from pathlib import Path

from reviewnlp.absa.aspects import ASPECTS, SENTIMENTS
from reviewnlp.absa.extract import review_key


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_annotations(gold: list[dict], predictions: list[dict]) -> dict:
    """Score one prediction per annotated review, aligned by frozen row_id.

    An empty *annotated* aspect list means the annotator found none. A template
    with status=unlabeled is rejected rather than accidentally scored as empty.
    Sentiment accuracy is conditional on a correctly detected aspect; the
    aspect F1 also penalizes missing or invented aspect categories.
    """
    if not gold:
        raise ValueError("no annotated reviews supplied")
    ids = [item["row_id"] for item in gold]
    predicted_ids = [item["row_id"] for item in predictions]
    if len(ids) != len(set(ids)) or len(predicted_ids) != len(set(predicted_ids)):
        raise ValueError("duplicate row_id in gold or predictions")
    if set(ids) != set(predicted_ids):
        raise ValueError("gold and prediction row_ids differ")

    by_id = {item["row_id"]: item for item in predictions}
    counts = {aspect: {"tp": 0, "fp": 0, "fn": 0} for aspect in ASPECTS}
    n_correct_sentiment = matched = invalid_quotes = 0
    for example in gold:
        if example.get("status") != "annotated":
            raise ValueError(f"row {example['row_id']}: annotation has not been completed")
        pred = by_id[example["row_id"]]
        if review_key(example["text"]) != review_key(pred["text"]):
            raise ValueError(f"row {example['row_id']}: review text differs")
        reference = {}
        for entry in example["aspects"]:
            aspect, sentiment, quote = entry["aspect"], entry["sentiment"], entry["quote"]
            if aspect not in ASPECTS or sentiment not in SENTIMENTS:
                raise ValueError(f"row {example['row_id']}: invalid gold label")
            if aspect in reference:
                raise ValueError(f"row {example['row_id']}: duplicate gold aspect {aspect}")
            if not quote or review_key(quote) not in review_key(example["text"]):
                raise ValueError(f"row {example['row_id']}: gold quote is absent from the review")
            reference[aspect] = sentiment
        extracted: dict[str, set[str]] = {}
        for entry in pred["aspects"]:
            aspect, sentiment = entry["aspect"], entry["sentiment"]
            if aspect not in ASPECTS or sentiment not in SENTIMENTS:
                raise ValueError(f"row {example['row_id']}: invalid predicted label")
            extracted.setdefault(aspect, set()).add(sentiment)
            if not entry.get("quote") or review_key(entry["quote"]) not in review_key(pred["text"]):
                invalid_quotes += 1
        for aspect in ASPECTS:
            if aspect in reference and aspect in extracted:
                counts[aspect]["tp"] += 1
                matched += 1
                n_correct_sentiment += extracted[aspect] == {reference[aspect]}
            elif aspect in reference:
                counts[aspect]["fn"] += 1
            elif aspect in extracted:
                counts[aspect]["fp"] += 1

    totals = {key: sum(item[key] for item in counts.values()) for key in ("tp", "fp", "fn")}
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    return {
        "n_reviews": len(gold),
        "aspect_micro_precision": round(precision, 4),
        "aspect_micro_recall": round(recall, 4),
        "aspect_micro_f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        "aspect_counts": counts,
        "matched_aspects": matched,
        "matched_sentiment_accuracy": round(n_correct_sentiment / matched, 4) if matched else None,
        "invalid_predicted_quotes": invalid_quotes,
    }
