"""The triage workflow's questions and routing rules: pure code, so every rule is tested without a model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from reviewnlp.triage import questions
from reviewnlp.triage.routing import REVIEW_REASONS, RoutingThresholds, decide, is_negative
from reviewnlp.triage.schemas import (
    OTHER,
    TOPICS,
    ComplaintsResult,
    ComplaintTopic,
    SentimentResult,
)

ROOT = Path(__file__).resolve().parents[1]


def _benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_jev_topics", ROOT / "scripts" / "benchmark_jev_topics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- The questions ---------------------------------------------------------------------------------------------

def test_the_topic_rules_are_the_benchmarks_and_the_framing_is_not():
    benchmark = _benchmark()
    for topic in TOPICS:
        assert questions.QUESTIONS[topic]["criteria"] == benchmark.QUESTIONS[topic]["criteria"], topic
        ours, theirs = questions.QUESTIONS[topic]["instructions"], benchmark.QUESTIONS[topic]["instructions"]
        assert "whole hotel guest review" in ours and "when Booking.com asked" not in ours
        assert "when Booking.com asked" in theirs and "whole hotel guest review" not in theirs


def test_there_is_one_question_per_topic_and_one_for_any_other_complaint():
    assert list(questions.QUESTIONS) == [*TOPICS, questions.OTHER_QUESTION]
    assert questions.TOPIC_OF_QUESTION[questions.OTHER_QUESTION] == OTHER
    for question in questions.QUESTIONS.values():
        assert question["type"] == "choice" and set(question["criteria"]) == set(questions.LABELS)
        assert "a complaint about this topic counts even when the rest of the review is positive" in (
            question["instructions"].lower())


def test_the_version_and_hash_identify_the_questions_and_differ_from_the_benchmarks():
    expected = hashlib.sha256(json.dumps({"version": questions.QUESTIONS_VERSION, "questions": questions.QUESTIONS},
                                         sort_keys=True).encode("utf-8")).hexdigest()
    assert questions.QUESTIONS_SHA256 == expected
    changed = json.loads(json.dumps(questions.QUESTIONS))
    changed["bathroom"]["instructions"] += " "
    assert hashlib.sha256(json.dumps({"version": questions.QUESTIONS_VERSION, "questions": changed},
                                     sort_keys=True).encode("utf-8")).hexdigest() != expected
    assert questions.QUESTIONS_SHA256 != _benchmark().QUESTIONS_SHA256
    assert questions.QUESTIONS_VERSION == "full-review-v1"


def test_the_request_body_is_the_review_the_model_and_every_question():
    body = questions.request_body("the shower was cold", "jev-latest")
    assert body == {"state": "the shower was cold", "model": "jev-latest", "questions": questions.QUESTIONS}


# --- Routing -----------------------------------------------------------------------------------------------------

def sentiment(label: str = "positive", confidence: float | None = 0.97) -> SentimentResult:
    return SentimentResult(label=label, confidence=confidence, probabilities=None, model_type="stub", model_path="")


def complaints(status: str = "ok", yes=(), unsure=(), doubtful: dict | None = None, other: str = "no",
               weak_yes: dict | None = None, other_probability: float = 0.9) -> ComplaintsResult:
    """Every topic answers no with probability 0, except the ones named. `weak_yes` maps a topic to the
    probability of a 'yes' that won without being sure."""
    doubtful, weak_yes = doubtful or {}, weak_yes or {}
    topics = []
    for topic in TOPICS:
        if topic in weak_yes:
            topics.append(ComplaintTopic(topic=topic, answer="yes", probability=weak_yes[topic]))
        elif topic in yes:
            topics.append(ComplaintTopic(topic=topic, answer="yes", probability=0.97))
        elif topic in unsure:
            topics.append(ComplaintTopic(topic=topic, answer="unsure", probability=0.4))
        else:
            topics.append(ComplaintTopic(topic=topic, answer="no", probability=doubtful.get(topic, 0.0)))
    extra = ComplaintTopic(topic=OTHER, answer=other, probability=other_probability if other == "yes" else 0.0)
    return ComplaintsResult(status=status, topics=topics if status == "ok" else [], other_complaint=extra if status == "ok" else None,
                            questions_version=questions.QUESTIONS_VERSION, questions_sha256=questions.QUESTIONS_SHA256)


@pytest.mark.parametrize("label, expected", [("negative", True), ("NEGATIVE", True), ("Neg", True),
                                             ("positive", False), ("LABEL_1", False), ("", False)])
def test_which_labels_are_negative(label, expected):
    assert is_negative(label) is expected


@pytest.mark.parametrize("case, senti, comp, qwen, reasons, review", [
    ("a confident positive review with nothing to complain about", sentiment(), complaints(), False, [], []),
    ("a negative review", sentiment("negative", 0.95), complaints(), True, ["negative_sentiment"], []),
    ("a positive review with a complaint in it", sentiment(), complaints(yes=["bathroom"]), True,
     ["complaint_detected"], []),
    ("a positive review with another complaint", sentiment(), complaints(other="yes"), True, ["complaint_detected"], []),
    ("a doubtful sentiment", sentiment("positive", 0.6), complaints(), True, ["uncertain_sentiment"],
     ["uncertain_sentiment"]),
    ("an unsure complaint answer", sentiment(), complaints(unsure=["pests"]), True, ["uncertain_complaint"],
     ["uncertain_complaint"]),
    ("a no with a doubtful probability", sentiment(), complaints(doubtful={"cleanliness": 0.3}), True,
     ["uncertain_complaint"], ["uncertain_complaint"]),
    ("a no with a low probability", sentiment(), complaints(doubtful={"cleanliness": 0.1}), False, [], []),
    ("the bottom edge of the doubtful band", sentiment(), complaints(doubtful={"cleanliness": 0.2}), True,
     ["uncertain_complaint"], ["uncertain_complaint"]),
    ("the top edge of the doubtful band", sentiment(), complaints(doubtful={"cleanliness": 0.5}), False, [], []),
    ("everything at once", sentiment("negative", 0.55), complaints(yes=["pests"], unsure=["bathroom"]), True,
     ["negative_sentiment", "uncertain_sentiment", "complaint_detected", "uncertain_complaint"],
     ["uncertain_sentiment", "uncertain_complaint"]),
    ("a model without a confidence", sentiment("positive", None), complaints(), False, [], []),
    ("the complaint stage failed on a positive review", sentiment(), complaints("error"), False, [],
     ["complaint_check_failed"]),
    ("the complaint stage failed on a negative review", sentiment("negative", 0.95), complaints("error"), True,
     ["negative_sentiment"], ["complaint_check_failed"]),
    ("the complaint stage is switched off", sentiment(), complaints("disabled"), False, [], []),
    ("a yes that won with 0.36 against 0.34 for no", sentiment(), complaints(weak_yes={"bathroom": 0.36}), True,
     ["complaint_detected", "uncertain_complaint"], ["uncertain_complaint"]),
    ("a yes just under the top edge", sentiment(), complaints(weak_yes={"bathroom": 0.49}), True,
     ["complaint_detected", "uncertain_complaint"], ["uncertain_complaint"]),
    ("a yes at the top edge", sentiment(), complaints(weak_yes={"bathroom": 0.5}), True, ["complaint_detected"], []),
    ("a yes that is clearly a yes", sentiment(), complaints(weak_yes={"bathroom": 0.75}), True,
     ["complaint_detected"], []),
    ("another complaint answered yes without being sure", sentiment(), complaints(other="yes", other_probability=0.4),
     True, ["complaint_detected", "uncertain_complaint"], ["uncertain_complaint"]),
    ("a sure yes next to an unsure one", sentiment(), complaints(yes=["pests"], unsure=["bathroom"]), True,
     ["complaint_detected", "uncertain_complaint"], ["uncertain_complaint"]),
])
def test_routing_rules(case, senti, comp, qwen, reasons, review):
    decision = decide(senti, comp)
    assert (decision.qwen_triggered, decision.reasons, decision.review_reasons) == (qwen, reasons, review), case
    assert decision.needs_review is bool(review)
    assert set(decision.review_reasons) <= set(REVIEW_REASONS)


def test_the_thresholds_are_applied_and_reported_as_provisional():
    thresholds = RoutingThresholds(sentiment_confidence_min=0.5, doubtful_band=(0.4, 0.6))
    decision = decide(sentiment("positive", 0.6), complaints(doubtful={"bathroom": 0.45, "pests": 0.3}), thresholds)
    assert decision.reasons == ["uncertain_complaint"]  # 0.6 is now confident enough; 0.45 is doubtful, 0.3 is not
    assert decision.thresholds == {"sentiment_confidence_min": 0.5, "doubtful_band": [0.4, 0.6]}
    assert decide(sentiment(), complaints()).thresholds == {"sentiment_confidence_min": 0.8, "doubtful_band": [0.2, 0.5]}
    assert decision.thresholds_status == "provisional"


def test_the_docs_name_every_reason_a_review_can_be_flagged_for():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "TRIAGE.md").read_text(encoding="utf-8")
    assert len(set(REVIEW_REASONS)) == len(REVIEW_REASONS) == 6
    for reason in REVIEW_REASONS:
        assert f"`{reason}`" in doc, reason
    assert "Without a `review_id` nothing is stored" in doc


def test_the_upper_edge_of_the_band_also_decides_when_a_yes_is_doubtful():
    thresholds = RoutingThresholds(doubtful_band=(0.4, 0.6))
    assert decide(sentiment(), complaints(weak_yes={"bathroom": 0.55}), thresholds).review_reasons == [
        "uncertain_complaint"]
    assert decide(sentiment(), complaints(weak_yes={"bathroom": 0.65}), thresholds).review_reasons == []
    # with the default band the same 0.55 is a clear enough yes
    assert decide(sentiment(), complaints(weak_yes={"bathroom": 0.55})).review_reasons == []


def test_a_yes_that_won_with_a_probability_above_the_edge_is_not_caught_because_only_its_probability_is_kept():
    # P(yes)=0.52 against P(no)=0.47 is a narrow win, but the result holds only the probability of 'yes'.
    assert decide(sentiment(), complaints(weak_yes={"bathroom": 0.52})).review_reasons == []
