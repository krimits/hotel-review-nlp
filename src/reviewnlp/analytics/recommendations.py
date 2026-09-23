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
