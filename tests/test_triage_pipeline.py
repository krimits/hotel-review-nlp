"""The triage pipeline with fake stages: partial results, routing into the suggestion stage, and no review in any result."""

from __future__ import annotations

import json

import pytest
from triage_fakes import NEGATIVE, POSITIVE, REVIEW, FakeGenerator, FakeJev, FakeWrapper

from reviewnlp.triage import questions
from reviewnlp.triage.jev_client import JevError
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.schemas import OTHER


def run(wrapper=None, jev=None, generator=None, clock=None):
    pipeline = TriagePipeline(wrapper or FakeWrapper(POSITIVE), jev, generator, **({"clock": clock} if clock else {}))
    return pipeline.run(REVIEW)


# --- With the optional stages off -------------------------------------------------------------------------------------

def test_with_jev_and_qwen_off_only_the_sentiment_is_computed_and_the_result_says_so():
    result = run(FakeWrapper(NEGATIVE))
    assert result.status == "complete"
    assert (result.sentiment.label, result.sentiment.confidence, result.sentiment.probabilities) == (
        "negative", 0.95, NEGATIVE)
    assert (result.sentiment.model_type, result.sentiment.model_path) == ("fake", "models/fake")
    assert result.complaints.status == "disabled" and result.complaints.topics == []
    assert result.complaints.questions_version == questions.QUESTIONS_VERSION
    assert result.complaints.questions_sha256 == questions.QUESTIONS_SHA256
    assert result.routing.reasons == ["negative_sentiment"] and result.routing.thresholds_status == "provisional"
    assert result.actions.status == "disabled"
    assert (result.timings.complaints_ms, result.timings.actions_ms) == (None, None)


def test_a_switched_off_jev_client_is_not_called():
    jev = FakeJev(enabled=False)
    assert run(jev=jev).complaints.status == "disabled" and jev.calls == []


def test_a_model_without_a_distribution_falls_back_to_its_label():
    result = run(FakeWrapper(None, "negative", 0.8))
    assert (result.sentiment.label, result.sentiment.confidence, result.sentiment.probabilities) == ("negative", 0.8, None)


def test_a_failing_sentiment_model_is_not_hidden():
    with pytest.raises(RuntimeError, match="boom"):
        run(FakeWrapper(error=RuntimeError("boom")), FakeJev(), FakeGenerator())


# --- The whole workflow -------------------------------------------------------------------------------------------------

def test_a_positive_review_with_a_complaint_in_it_reaches_the_suggestion_stage():
    jev, generator = FakeJev(yes=["bathroom", "responsiveness"], other="yes"), FakeGenerator()
    result = run(FakeWrapper(POSITIVE), jev, generator)
    assert result.status == "complete" and jev.calls == [REVIEW]
    assert [t.topic for t in result.complaints.topics if t.answer == "yes"] == ["bathroom", "responsiveness"]
    assert result.complaints.status == "ok" and result.complaints.model == "typesafe/jev-test"
    assert result.routing.reasons == ["complaint_detected"] and result.routing.qwen_triggered
    (review, signals), = generator.calls
    assert review == REVIEW and signals.sentiment_label == "positive"
    assert signals.flagged_topics == ["bathroom", "responsiveness", OTHER]
    assert result.actions.status == "ok" and result.actions.model == "fake/qwen" and result.actions.prompt_version
    action = result.actions.actions[0]
    assert (action.department, action.to_confirm) == ("maintenance", ["room number"]) and result.actions.dropped == 0
    assert result.timings.complaints_ms is not None and result.timings.actions_ms is not None


def test_a_confident_positive_review_without_complaints_does_not_reach_the_suggestion_stage():
    generator = FakeGenerator()
    result = run(FakeWrapper(POSITIVE), FakeJev(), generator)
    assert result.actions.status == "not_triggered" and generator.calls == [] and result.status == "complete"
    assert result.routing.reasons == [] and not result.routing.needs_review


def test_the_hotel_context_is_given_to_the_generator():
    generator = FakeGenerator()
    TriagePipeline(FakeWrapper(NEGATIVE), None, generator).run(REVIEW, hotel_context="Recent negative mentions: noise 4")
    assert generator.calls[0][1].hotel_context == "Recent negative mentions: noise 4"


# --- Failures --------------------------------------------------------------------------------------------------------------

def test_when_jev_fails_a_negative_review_still_gets_its_suggestions_and_a_person_is_asked_to_look():
    result = run(FakeWrapper(NEGATIVE), FakeJev(error=JevError("timeout")), FakeGenerator())
    assert (result.complaints.status, result.complaints.error, result.complaints.route) == ("error", "timeout", "typesafe")
    assert result.actions.status == "ok" and result.status == "partial"
    assert result.routing.needs_review and result.routing.review_reasons == ["complaint_check_failed"]


def test_when_jev_fails_a_confident_positive_review_cannot_be_cleared():
    generator = FakeGenerator()
    result = run(FakeWrapper(POSITIVE), FakeJev(error=JevError("http_503")), generator)
    assert result.actions.status == "not_triggered" and generator.calls == []
    assert result.status == "partial" and result.routing.review_reasons == ["complaint_check_failed"]


def test_an_unexpected_jev_failure_is_named_without_its_message():
    result = run(FakeWrapper(NEGATIVE), FakeJev(error=ValueError("SECRETWORD in a message")))
    assert result.complaints.error == "unexpected" and "SECRETWORD" not in repr(result)


@pytest.mark.parametrize("generator, status, error, reason", [
    (FakeGenerator(error=RuntimeError("SECRETWORD in a message")), "error", "generation_failed", "actions_failed"),
    (FakeGenerator(raw="I cannot help."), "error", "invalid_output", "actions_failed"),
    (FakeGenerator(raw="{", hit_token_budget=True), "error", "hit_token_budget", "actions_failed"),
])
def test_a_failing_generator_gives_a_partial_result_that_needs_a_look(generator, status, error, reason):
    result = run(FakeWrapper(NEGATIVE), None, generator)
    assert (result.actions.status, result.actions.error, result.status) == (status, error, "partial")
    assert result.actions.actions == [] and reason in result.routing.review_reasons and result.routing.needs_review
    assert "SECRETWORD" not in repr(result)


def test_actions_that_are_not_in_the_review_are_dropped_and_a_person_is_asked_to_look():
    invented = json.dumps({"actions": [{"problem": "Broken lift", "excerpt": "the lift was broken",
                                        "measure": "Repair the lift", "department": "maintenance"}]})
    result = run(FakeWrapper(NEGATIVE), None, FakeGenerator(raw=invented))
    assert (result.actions.status, result.actions.dropped, result.actions.actions) == ("no_grounded_actions", 1, [])
    assert result.status == "partial" and "actions_ungrounded" in result.routing.review_reasons


def test_the_model_saying_there_is_nothing_to_fix_is_an_answer_not_a_failure():
    result = run(FakeWrapper(NEGATIVE), None, FakeGenerator(raw='{"actions": []}'))
    assert (result.actions.status, result.actions.actions, result.status) == ("ok", [], "complete")
    assert not result.routing.needs_review


# --- Timings --------------------------------------------------------------------------------------------------------------

def test_each_stage_is_timed_on_its_own():
    ticks = iter([0.0, 0.010, 0.050, 0.051, 0.251])  # start, after sentiment, complaints, routing, actions
    result = run(FakeWrapper(NEGATIVE), FakeJev(), FakeGenerator(), clock=lambda: next(ticks))
    assert result.timings.model_dump() == {"sentiment_ms": 10.0, "complaints_ms": 40.0, "routing_ms": 1.0,
                                           "actions_ms": 200.0, "total_ms": 251.0}
