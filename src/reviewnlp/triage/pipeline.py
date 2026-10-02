"""Sentiment -> complaint topics -> suggested actions for one review.

Only the sentiment stage is required. Jev and Qwen are asked for when they are switched on, and a failure in
either gives a partial result with the kind of failure named, never an error for the whole request. The
review is never put in a result, a log line or an error.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from reviewnlp.triage.jev_client import JevClient, JevError
from reviewnlp.triage.questions import QUESTIONS_SHA256, QUESTIONS_VERSION
from reviewnlp.triage.qwen_generator import ActionGenerator, ActionSignals, parse_actions
from reviewnlp.triage.routing import RoutingThresholds, decide
from reviewnlp.triage.schemas import (
    ActionsResult,
    ComplaintsResult,
    RoutingDecision,
    SentimentResult,
    StageTimings,
)

log = logging.getLogger("reviewnlp.triage")


class SentimentModel(Protocol):
    model_type: str
    model_path: str

    def distribution_batch(self, texts: list[str]) -> list[dict[str, float] | None]: ...

    def predict(self, text: str) -> tuple[str, float | None]: ...


@dataclass(frozen=True)
class PipelineResult:
    status: str
    sentiment: SentimentResult
    complaints: ComplaintsResult
    routing: RoutingDecision
    actions: ActionsResult
    timings: StageTimings


def hotel_context(store, hotel_id: str, days: int = 30) -> str | None:
    """A line of counts from the hotel's own stored aspects, for the suggestion stage. Counts only, no review text."""
    from reviewnlp.analytics.recommendations import build_analytics

    analytics = [item for item in build_analytics(store.aspect_rows(hotel_id, days),
                                                  store.previous_period_rows(hotel_id, days))
                 if item["negative_count"] > 0][:3]
    if not analytics:
        return None
    return (f"Recent negative mentions at this hotel (last {days} days): "
            + ", ".join(f"{item['aspect']} {item['negative_count']}" for item in analytics))


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 2)


class TriagePipeline:
    def __init__(self, wrapper: SentimentModel, jev: JevClient | None = None, generator: ActionGenerator | None = None,
                 thresholds: RoutingThresholds | None = None, clock: Callable[[], float] = time.perf_counter):
        self.wrapper, self.jev, self.generator = wrapper, jev, generator
        self.thresholds, self._clock = thresholds or RoutingThresholds(), clock

    def _sentiment(self, text: str) -> SentimentResult:
        (distribution,) = self.wrapper.distribution_batch([text])
        if distribution:
            label = max(distribution, key=distribution.get)
            return SentimentResult(label=label, confidence=distribution[label], probabilities=dict(distribution),
                                   model_type=self.wrapper.model_type, model_path=self.wrapper.model_path)
        label, confidence = self.wrapper.predict(text)
        return SentimentResult(label=label, confidence=confidence, probabilities=None,
                               model_type=self.wrapper.model_type, model_path=self.wrapper.model_path)

    def _complaints(self, text: str) -> ComplaintsResult:
        base = {"questions_version": QUESTIONS_VERSION, "questions_sha256": QUESTIONS_SHA256}
        if self.jev is None or not self.jev.enabled:
            return ComplaintsResult(status="disabled", **base)
        route = self.jev.config.route
        try:
            result = self.jev.classify(text)
        except JevError as error:
            return ComplaintsResult(status="error", error=error.kind, route=route, **base)
        except Exception:  # noqa: BLE001 - whatever it was, the review must not travel with it
            log.error("jev stage failed unexpectedly")
            return ComplaintsResult(status="error", error="unexpected", route=route, **base)
        return ComplaintsResult(status="ok", topics=result.topics, other_complaint=result.other_complaint,
                                route=result.route, model=result.model, **base)

    def _actions(self, text: str, sentiment: SentimentResult, complaints: ComplaintsResult,
                 routing: RoutingDecision, hotel_context: str | None) -> ActionsResult:
        if self.generator is None:
            return ActionsResult(status="disabled")
        if not routing.qwen_triggered:
            return ActionsResult(status="not_triggered")
        flagged = [item.topic for item in [*complaints.topics, *([complaints.other_complaint]
                                                                 if complaints.other_complaint else [])]
                   if item.answer == "yes"]
        signals = ActionSignals(sentiment.label, sentiment.confidence, flagged, hotel_context)
        try:
            generated = self.generator.generate(text, signals)
        except Exception:  # noqa: BLE001
            log.error("action generation failed")
            return ActionsResult(status="error", error="generation_failed", model=self.generator.model_name)
        base = {"model": generated.model, "prompt_version": generated.prompt_version}
        parsed = parse_actions(generated.raw, text)
        if not parsed.json_valid:
            return ActionsResult(status="error", error="hit_token_budget" if generated.hit_token_budget
                                 else "invalid_output", **base)
        if not parsed.actions and parsed.dropped:
            return ActionsResult(status="no_grounded_actions", dropped=parsed.dropped, **base)
        return ActionsResult(status="ok", actions=parsed.actions, dropped=parsed.dropped, **base)

    def run(self, text: str, hotel_context: str | None = None) -> PipelineResult:
        start = self._clock()
        sentiment = self._sentiment(text)
        after_sentiment = self._clock()

        complaints = self._complaints(text)
        after_complaints = self._clock()

        routing = decide(sentiment, complaints, self.thresholds)
        after_routing = self._clock()

        actions = self._actions(text, sentiment, complaints, routing, hotel_context)
        end = self._clock()

        extra = {"error": "actions_failed", "no_grounded_actions": "actions_ungrounded"}.get(actions.status)
        if extra:
            routing = routing.model_copy(update={"needs_review": True, "review_reasons": [*routing.review_reasons, extra]})
        partial = complaints.status == "error" or actions.status in {"error", "no_grounded_actions"}
        timings = StageTimings(
            sentiment_ms=_ms(start, after_sentiment),
            complaints_ms=None if complaints.status == "disabled" else _ms(after_sentiment, after_complaints),
            routing_ms=_ms(after_complaints, after_routing),
            actions_ms=None if actions.status in {"disabled", "not_triggered"} else _ms(after_routing, end),
            total_ms=_ms(start, end))
        return PipelineResult("partial" if partial else "complete", sentiment, complaints, routing, actions, timings)
