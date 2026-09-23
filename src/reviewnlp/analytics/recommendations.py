"""Recommendation engine for hotel improvements based on aspect analytics."""

from __future__ import annotations

RECOMMENDATION_TEXT = {
    "cleanliness": (
        "Ενισχύστε τον τελικό έλεγχο καθαριότητας στα δωμάτια, "
        "στα μπάνια και στους κοινόχρηστους χώρους."
    ),
    "staff": (
        "Εξετάστε την εκπαίδευση προσωπικού και τη διαδικασία "
        "διαχείρισης αιτημάτων κατά το check-in και τη διαμονή."
    ),
    "location": (
        "Βελτιώστε την ενημέρωση για πρόσβαση, μετακινήσεις, "
        "θόρυβο και πραγματικές αποστάσεις από τα σημεία ενδιαφέροντος."
    ),
    "room": (
        "Δώστε προτεραιότητα σε συντήρηση δωματίων, κρεβάτια, "
        "μπάνια και εξοπλισμό."
    ),
    "food": (
        "Εξετάστε ποιότητα πρωινού, ποικιλία, θερμοκρασία φαγητού "
        "και συνέπεια στο service."
    ),
    "noise": (
        "Εντοπίστε πηγές θορύβου και εφαρμόστε μέτρα ηχομόνωσης "
        "ή καλύτερη κατανομή δωματίων."
    ),
    "value": (
        "Συγκρίνετε τιμή και παροχές με ανταγωνιστικά ξενοδοχεία "
        "και κάντε σαφέστερη την αξία της διαμονής."
    ),
    "facilities": (
        "Βελτιώστε τη διαθεσιμότητα, συντήρηση και ενημέρωση "
        "για Wi-Fi, πισίνα, parking και λοιπές παροχές."
    ),
}


def calculate_priority_score(
    negative_rate: float,
    normalized_volume: float,
    negative_trend: float,
    business_weight: float = 0.5,
) -> float:
    """Calculate priority score for an aspect based on multiple factors.

    The scoring formula weights:
    - 45% negative rate (proportion of negative mentions)
    - 25% normalized volume (how frequently the aspect is mentioned)
    - 20% negative trend (whether the aspect is getting worse)
    - 10% business weight (domain-specific importance)

    Args:
        negative_rate: Proportion of negative mentions (0.0 to 1.0)
        normalized_volume: Normalized mention volume (0.0 to 1.0)
        negative_trend: Trend in negative sentiment (-1.0 to 1.0)
        business_weight: Domain-specific importance (0.0 to 1.0)

    Returns:
        Priority score between 0.0 and 1.0
    """
    trend_score = max(0.0, min(1.0, negative_trend))
    return round(
        0.45 * negative_rate
        + 0.25 * normalized_volume
        + 0.20 * trend_score
        + 0.10 * business_weight,
        4,
    )


def build_recommendation(aspect: str) -> str:
    """Get a predefined recommendation text for an aspect.

    Args:
        aspect: The aspect category (e.g., 'cleanliness', 'staff')

    Returns:
        Recommendation text in Greek, or a generic fallback
    """
    return RECOMMENDATION_TEXT.get(
        aspect,
        "Εξετάστε τις αρνητικές κριτικές και δημιουργήστε συγκεκριμένο πλάνο βελτίωσης.",
    )


def normalize_volume(mention_count: int, max_count: int) -> float:
    """Normalize mention count to a 0-1 scale.

    Args:
        mention_count: Number of mentions for this aspect
        max_count: Maximum mention count across all aspects

    Returns:
        Normalized volume between 0.0 and 1.0
    """
    if max_count == 0:
        return 0.0
    return min(1.0, mention_count / max_count)


def _count_by_aspect(rows) -> dict[str, dict[str, int]]:
    """Tally sentiments per aspect, counting each aspect once per review.

    A review that mentions cleanliness three times is one negative signal, not
    three - the same rule overall_from_aspects and run_absa.py already apply.
    Rows without a review_id cannot be deduplicated and are counted as given.
    """
    seen: set[tuple] = set()
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        aspect = row.get("aspect")
        sentiment = row.get("sentiment")
        if aspect is None or sentiment not in ("positive", "negative", "neutral"):
            continue
        review_id = row.get("review_id")
        if review_id is not None:
            key = (review_id, aspect)
            if key in seen:
                continue
            seen.add(key)
        tally = counts.setdefault(
            aspect, {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
        )
        tally[sentiment] += 1
        tally["total"] += 1
    return counts


def build_analytics(rows, previous_rows=None) -> list[dict]:
    """Aggregate aspect rows into per-aspect analytics, worst first.

    Every number here is computed from `rows`. There is no fallback and no
    placeholder: no rows means an empty list, which is a truthful answer to
    "what should this hotel fix" when nothing has been analyzed yet.

    trend is the change in negative rate against `previous_rows` - positive
    means the aspect is getting worse. An aspect absent from the previous
    period has no trend (None) and contributes nothing to its priority, rather
    than being treated as a sudden regression.
    """
    counts = _count_by_aspect(rows)
    if not counts:
        return []
    previous = _count_by_aspect(previous_rows or [])
    max_total = max(tally["total"] for tally in counts.values())

    analytics = []
    for aspect, tally in counts.items():
        negative_rate = tally["negative"] / tally["total"]
        before = previous.get(aspect)
        trend = None
        if before is not None and before["total"]:
            trend = round(negative_rate - before["negative"] / before["total"], 4)
        analytics.append({
            "aspect": aspect,
            "review_count": tally["total"],
            "positive_count": tally["positive"],
            "negative_count": tally["negative"],
            "neutral_count": tally["neutral"],
            "negative_rate": round(negative_rate, 4),
            "trend": trend,
            "priority_score": calculate_priority_score(
                negative_rate=negative_rate,
                normalized_volume=normalize_volume(tally["total"], max_total),
                negative_trend=trend or 0.0,
            ),
            "recommendation": build_recommendation(aspect),
        })
    # Aspect name breaks ties so the ordering is stable run to run.
    analytics.sort(key=lambda item: (-item["priority_score"], item["aspect"]))
    return analytics
