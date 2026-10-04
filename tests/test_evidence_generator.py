"""Evidence policy checks preserve ambiguity without claiming semantic verification."""

from __future__ import annotations

import json

import pytest
from triage_fakes import POSITIVE, FakeGenerator, FakeWrapper

from reviewnlp.triage.evidence_generator import (
    VERIFY_CURRENT_STATE,
    EvidenceFirstGenerator,
    department_mapping,
    parse_evidence_issues,
)
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.qwen_generator import ActionSignals

TEXT = "The cupboard was dusty. Staff cleaned it and it was spotless afterwards."
REPORTED, RESOLVED = "The cupboard was dusty.", "Staff cleaned it and it was spotless afterwards."
ISSUE = {"problem": "Dusty cupboard", "excerpt": REPORTED, "category": "cleanliness", "status": "REAL_PENDING",
         "evidence": {"reported": REPORTED, "hypothetical": None, "resolved": None}}


def parse(issue, text=TEXT, **kwargs):
    return parse_evidence_issues(json.dumps({"issues": [issue]}), text, **kwargs)[0]


def generator(issue=ISSUE, measures=None):
    extractor = FakeGenerator(json.dumps({"issues": [issue]}))
    extractor._bundle = ("tokenizer", "weights")
    actioner = FakeGenerator(json.dumps({"actions": measures if measures is not None else [
        {"issue_id": 1, "measure": "Clean the cupboard surfaces.", "to_confirm": []}]}))
    actioner._bundle = None
    return EvidenceFirstGenerator(extractor, lambda pending: actioner), extractor, actioner


def test_pending_is_a_guest_report_and_current_condition_is_checked_before_acting():
    gen, _, actioner = generator()
    result = gen.generate(REPORTED, ActionSignals("positive", 0.97))
    assert json.loads(result.raw)["actions"][0]["to_confirm"] == [VERIFY_CURRENT_STATE]
    assert gen.issues[0]["department"] == "housekeeping"
    assert actioner._bundle is gen._bundle and len(actioner.calls) == 1
    assert result.prompt_version == "actions-v5-evidence"


@pytest.mark.parametrize("status,evidence", [
    ("REAL_RESOLVED", {"reported": REPORTED, "hypothetical": None, "resolved": None}),
    ("HYPOTHETICAL", {"reported": REPORTED, "hypothetical": None, "resolved": None}),
    ("REAL_PENDING", {"reported": None, "hypothetical": None, "resolved": None}),
    ("REAL_PENDING", {"reported": REPORTED, "hypothetical": None, "resolved": RESOLVED}),
])
def test_missing_or_conflicting_evidence_is_uncertain_and_no_measure_is_generated(status, evidence):
    gen, _, actioner = generator({**ISSUE, "status": status, "evidence": evidence})
    result = gen.generate(TEXT, ActionSignals("negative", 0.9))
    assert gen.issues[0]["model_status"] == status
    assert gen.issues[0]["status"] == "UNCERTAIN"
    assert "uncertain_issue" in gen.review_reasons
    assert json.loads(result.raw)["actions"] == [] and not actioner.calls


def test_explicit_resolution_is_retained_and_not_forwarded_for_measures():
    issue = {**ISSUE, "status": "REAL_RESOLVED", "evidence": {**ISSUE["evidence"], "resolved": RESOLVED}}
    gen, _, actioner = generator(issue)
    gen.generate(TEXT, ActionSignals("positive", 0.9))
    assert gen.issues[0]["status"] == "REAL_RESOLVED" and not actioner.calls


def test_workaround_quote_is_not_semantically_verified_by_the_code():
    # Quote presence cannot establish a successful fix. The human audit must catch this misclassification.
    text = "The door lock failed. Staff lent us a key for a different room."
    issue = {**ISSUE, "excerpt": "The door lock failed.", "status": "REAL_RESOLVED",
             "evidence": {"reported": "The door lock failed.", "hypothetical": None,
                          "resolved": "Staff lent us a key for a different room."}}
    assert parse(issue, text)["status"] == "REAL_RESOLVED"


@pytest.mark.parametrize("damage", [
    {"evidence": {**ISSUE["evidence"], "reported": "Invented exact quote."}},
    {"evidence": {"reported": REPORTED}}, {"category": "front_desk"}, {"department": "maintenance"},
    {"status": "HYPOTHECIAL"}, {"excerpt": "the cupboard was dusty."},
])
def test_invalid_fields_or_nonliteral_evidence_fail_the_workflow(damage):
    with pytest.raises(ValueError):
        parse({**ISSUE, **damage})


def test_department_mapping_is_configurable_but_uses_closed_categories_and_departments():
    assert parse(ISSUE, mapping={"cleanliness": "management"})["department"] == "management"
    for mapping in ({"unknown": "maintenance"}, {"cleanliness": "front_desk"}):
        with pytest.raises(ValueError):
            department_mapping(mapping)


def test_unknown_department_and_unaddressed_pending_issues_are_visible_for_review():
    gen, _, _ = generator({**ISSUE, "category": "other"}, measures=[])
    gen.generate(REPORTED, ActionSignals("negative", 0.9))
    assert set(gen.review_reasons) == {"department_needs_confirmation", "reported_issues_without_measures"}


def test_a_jev_complaint_without_supported_issue_is_flagged_not_silently_declared_clean():
    gen, _, _ = generator({**ISSUE, "status": "REAL_RESOLVED",
                           "evidence": {**ISSUE["evidence"], "resolved": RESOLVED}})
    gen.generate(TEXT, ActionSignals("positive", 0.97, ["cleanliness"]))
    assert "complaint_signal_without_supported_issue" in gen.review_reasons


def test_pipeline_preserves_uncertain_classification_and_demo_scans_even_confident_positive_reviews():
    gen, extractor, _ = generator({**ISSUE, "status": "UNCERTAIN"})
    pipeline = TriagePipeline(FakeWrapper(POSITIVE), generator=gen, evaluate_all_reviews=True)
    result = pipeline.run(REPORTED)
    assert len(extractor.calls) == 1
    assert result.routing.qwen_triggered and "all_reviews_diagnostic" in result.routing.reasons
    assert result.routing.needs_review and "uncertain_issue" in result.routing.review_reasons
    assert result.status == "complete" and result.actions.actions == []


def test_default_production_routing_remains_unchanged():
    gen, extractor, _ = generator()
    result = TriagePipeline(FakeWrapper(POSITIVE), generator=gen).run(REPORTED)
    assert not result.routing.qwen_triggered and not extractor.calls


def test_literal_excerpt_survives_the_final_action_parser_without_whitespace_changes():
    text = "The  cupboard\nwas dusty."
    gen, _, _ = generator({**ISSUE, "excerpt": text,
                           "evidence": {"reported": text, "hypothetical": None, "resolved": None}})
    result = TriagePipeline(FakeWrapper(POSITIVE), generator=gen, evaluate_all_reviews=True).run(text)
    assert result.actions.actions[0].excerpt == text
