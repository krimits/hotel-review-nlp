"""When is the suggestion stage asked for, and when does a person need to look?

Pure functions, so every rule can be tested without a model. The thresholds are provisional: they were
chosen by hand and not on a development sample, and the response says so. Choosing them is a step of the
evaluation in docs/TRIAGE.md that has not been done.
"""

from __future__ import annotations

from dataclasses import dataclass

from reviewnlp.triage.schemas import (
    ComplaintsResult,
    ComplaintTopic,
    RoutingDecision,
    SentimentResult,
)

# Every reason for which a person is asked to look at a review. The dashboard names each of them, and a test
# keeps that list, this one and docs/TRIAGE.md the same. The last three are added by the pipeline, once the
# suggestion stage has answered.
REVIEW_REASONS = ("uncertain_sentiment", "uncertain_complaint", "complaint_check_failed",
                  "actions_failed", "actions_ungrounded", "no_actions_suggested")


@dataclass(frozen=True)
class RoutingThresholds:
    sentiment_confidence_min: float = 0.80
    # A topic Jev answered 'no' to, with a probability of 'yes' inside this band, is treated as doubtful. So is a
    # 'yes' whose probability is below the band's upper edge: it won, but with less than half of the probability.
    doubtful_band: tuple[float, float] = (0.20, 0.50)

    def as_dict(self) -> dict[str, float | list[float]]:
        return {"sentiment_confidence_min": self.sentiment_confidence_min, "doubtful_band": list(self.doubtful_band)}


def is_negative(label: str) -> bool:
    return str(label).strip().lower().startswith("neg")


def _is_doubtful(item: ComplaintTopic, low: float, high: float) -> bool:
    """Whether Jev's answer on one topic should not be taken at face value.

    The answer is the label with the most probability, which is not the same as being sure of it: a 'yes' can win
    with 0.36 against 0.34 for 'no'. Only the probability of 'yes' is kept, so a 'yes' that wins narrowly with
    more than `high` is not caught.
    """
    if item.answer == "unsure":
        return True
    if item.answer == "no":
        return low <= item.probability < high
    return item.answer == "yes" and item.probability < high


def decide(sentiment: SentimentResult, complaints: ComplaintsResult,
           thresholds: RoutingThresholds | None = None) -> RoutingDecision:
    """The reasons for asking the suggestion stage, and the reasons a person should look.

    The suggestion stage is asked for when the sentiment is negative, a complaint was found, or either signal is
    doubtful. A positive review with a complaint in it is therefore asked for: that is the case the sentiment
    alone would miss. If the complaint stage failed, a confidently positive review cannot be told from a
    positive review with a complaint, so a person is asked to look. If the complaint stage is switched off,
    nothing was looked for: the review is not flagged, and the result says `disabled`, not 'no complaints'.
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
        if any(_is_doubtful(item, low, high) for item in answers):
            reasons.append("uncertain_complaint")
            review.append("uncertain_complaint")
    elif complaints.status == "error":
        review.append("complaint_check_failed")

    return RoutingDecision(qwen_triggered=bool(reasons), reasons=reasons, needs_review=bool(review),
                           review_reasons=review, thresholds=thresholds.as_dict())
