"""When is the suggestion stage asked for, and when does a person need to look?

Pure functions, so every rule can be tested without a model. The thresholds are provisional: they were
chosen by hand and not on a development sample, and the response says so. Choosing them is a step of the
evaluation in docs/TRIAGE.md that has not been done.
"""

from __future__ import annotations

from dataclasses import dataclass

from reviewnlp.triage.schemas import ComplaintsResult, RoutingDecision, SentimentResult


@dataclass(frozen=True)
class RoutingThresholds:
    sentiment_confidence_min: float = 0.80
    # A topic Jev answered 'no' to, with a probability of 'yes' inside this band, is treated as doubtful.
    doubtful_band: tuple[float, float] = (0.20, 0.50)

    def as_dict(self) -> dict[str, float | list[float]]:
        return {"sentiment_confidence_min": self.sentiment_confidence_min, "doubtful_band": list(self.doubtful_band)}


def is_negative(label: str) -> bool:
    return str(label).strip().lower().startswith("neg")


def decide(sentiment: SentimentResult, complaints: ComplaintsResult,
           thresholds: RoutingThresholds | None = None) -> RoutingDecision:
    """The reasons for asking the suggestion stage, and the reasons a person should look.

    The suggestion stage is asked for when the sentiment is negative, a complaint was found, or either signal is
    doubtful. A positive review with a complaint in it is therefore asked for: that is the case the sentiment
    alone would miss. If the complaint stage failed, a confidently positive review cannot be told from a
    positive review with a complaint, so a person is asked to look.
    """
    thresholds = thresholds or RoutingThresholds()
    reasons: list[str] = []
    review: list[str] = []

    if is_negative(sentiment.label):
        reasons.append("negative_sentiment")
    if sentiment.confidence is not None and sentiment.confidence < thresholds.sentiment_confidence_min:
        reasons.append("uncertain_sentiment")
        review.append("uncertain_sentiment")

    if complaints.status == "ok":
        answers = [*complaints.topics, *([complaints.other_complaint] if complaints.other_complaint else [])]
        low, high = thresholds.doubtful_band
        if any(item.answer == "yes" for item in answers):
            reasons.append("complaint_detected")
        if any(item.answer == "unsure" or (item.answer == "no" and low <= item.probability < high)
               for item in answers):
            reasons.append("uncertain_complaint")
            review.append("uncertain_complaint")
    elif complaints.status == "error":
        review.append("complaint_check_failed")

    return RoutingDecision(qwen_triggered=bool(reasons), reasons=reasons, needs_review=bool(review),
                           review_reasons=review, thresholds=thresholds.as_dict())
