"""Regression checks for source selection, literal evidence and failure visibility."""

from __future__ import annotations

import json

import pytest
from test_triage_jev_client import FakeTransport, body_for
from triage_fakes import POSITIVE, FakeGenerator, FakeWrapper

from reviewnlp.triage.demo_service import DemoService, make_generator, space_environment
from reviewnlp.triage.evidence_generator import parse_evidence_issues
from reviewnlp.triage.jev_client import JevConfig
from reviewnlp.triage.qwen_generator import ActionSignals
from reviewnlp.triage.span_evidence_generator import (
    VERSION,
    SourceSpanGenerator,
    messages_span_evidence,
    parse_span_issues,
    source_spans,
)

LONG_LAMP = "The reading lamp flickered all evening, although the bed was comfortable."
SHORT_LAMP = "The reading lamp flickered all evening"
ISSUE = {"problem": "Flickering reading lamp", "excerpt_span": 1,
         "evidence": {"reported": 1, "hypothetical": None, "resolved": None},
         "status": "REAL_PENDING", "category": "equipment_fault"}


def raw(issue=ISSUE):
    return json.dumps({"issues": [issue]})


def service(issue=ISSUE):
    extractor = FakeGenerator(raw(issue))
    extractor._bundle = ("tokenizer", "weights")
    actioner = FakeGenerator('{"actions":[{"issue_id":1,"measure":"Inspect and repair the lamp.","to_confirm":[]}]}')
    gen = SourceSpanGenerator(extractor, lambda pending: actioner)
    return DemoService(FakeWrapper(POSITIVE), gen), extractor, actioner


def test_the_old_parser_reproduces_both_screenshot_failure_codes():
    issue = {key: value for key, value in ISSUE.items() if key != "excerpt_span"}
    issue["excerpt"] = SHORT_LAMP + "."
    issue["evidence"] = {"reported": issue["excerpt"], "hypothetical": None, "resolved": None}
    with pytest.raises(ValueError, match="^issue_quote_missing$"):
        parse_evidence_issues(raw(issue), LONG_LAMP)
    issue["excerpt"] = SHORT_LAMP
    with pytest.raises(ValueError, match="^evidence_quote_missing_or_invalid$"):
        parse_evidence_issues(raw(issue), SHORT_LAMP)


@pytest.mark.parametrize("review", [LONG_LAMP, SHORT_LAMP, "The  lamp\nflickered all evening."])
def test_source_selection_reaches_measures_and_returns_original_evidence(review):
    demo, extractor, actioner = service()
    result = demo.analyze(review)
    assert result["status"] == "complete" and result["stage_failure"] is None
    assert len(extractor.calls) == len(actioner.calls) == 1
    assert actioner._bundle is extractor._bundle
    assert result["actions"]["actions"][0]["department"] == "maintenance"
    assert result["actions"]["actions"][0]["excerpt"] == source_spans(review)[0]["text"]
    assert result["issue_assessments"][0]["evidence"]["reported"] in review
    assert result["stored"] is False and result["api_cost"]["attempts"] == 0
    assert not demo.generator.last_stages and not demo.generator.issues


@pytest.mark.parametrize("bad", [0, -1, 2, True, 1.0, "1", "", "null", [], {}])
@pytest.mark.parametrize("field", ["excerpt_span", "reported"])
def test_nonexistent_or_coerced_ids_fail_visibly_and_never_reach_measures(field, bad):
    issue = ({**ISSUE, "excerpt_span": bad} if field == "excerpt_span" else
             {**ISSUE, "evidence": {**ISSUE["evidence"], "reported": bad}})
    demo, _, actioner = service(issue)
    result = demo.analyze(SHORT_LAMP)
    assert result["status"] == "partial"
    assert result["stage_failure"] == {"stage": "issues", "status": "error", "error": "invalid_evidence_span_id"}
    assert not result["actions"]["actions"] and not actioner.calls
    assert result["routing"]["needs_review"]


@pytest.mark.parametrize("status,evidence,expected", [
    ("REAL_RESOLVED", {"reported": 1, "hypothetical": None, "resolved": 2}, "REAL_RESOLVED"),
    ("REAL_PENDING", {"reported": 1, "hypothetical": None, "resolved": 2}, "UNCERTAIN"),
    ("REAL_PENDING", {"reported": None, "hypothetical": None, "resolved": None}, "UNCERTAIN"),
    ("HYPOTHETICAL", {"reported": None, "hypothetical": 2, "resolved": None}, "HYPOTHETICAL"),
])
def test_existing_evidence_policy_retains_excluded_and_uncertain_issues(status, evidence, expected):
    text = "The lamp failed. Staff replaced it and the new lamp worked perfectly."
    demo, _, actioner = service({**ISSUE, "status": status, "evidence": evidence})
    result = demo.analyze(text)
    assert result["issue_assessments"][0]["status"] == expected
    assert result["issue_assessments"][0]["evidence"] == {
        name: None if value is None else source_spans(text)[value - 1]["text"] for name, value in evidence.items()}
    assert not result["actions"]["actions"] and not actioner.calls


def test_distinct_issues_can_share_a_sentence_without_accepting_duplicate_issues():
    text = "The lamp flickered and the fan rattled."
    entries = [ISSUE, {**ISSUE, "problem": "Rattling fan"}]
    issues = parse_span_issues(json.dumps({"issues": entries}), text)
    assert [item["issue_id"] for item in issues] == [1, 2]
    assert all(item["excerpt"] == text for item in issues)
    with pytest.raises(ValueError, match="duplicate_span_issue"):
        parse_span_issues(json.dumps({"issues": [ISSUE, ISSUE]}), text)


@pytest.mark.parametrize("text", [LONG_LAMP, SHORT_LAMP, "No.\nThe lamp never worked.\nIt was NOT fixed.",
                                 "A" * 4000, ("The lamp never worked and was not fixed. " * 40).strip(),
                                 "The fan rattled; the heater failed. Staff did not repair either."])
def test_spans_are_bounded_literal_source_slices_and_preserve_all_nonwhitespace_characters(text):
    spans = source_spans(text)
    assert [item["id"] for item in spans] == list(range(1, len(spans) + 1))
    assert all(3 <= len(item["text"].strip()) <= 400 and item["text"] in text for item in spans)
    assert "".join("".join(item["text"].split()) for item in spans) == "".join(text.split())


def test_prompt_and_runtime_use_the_new_version_without_rewriting_historical_g():
    gen = make_generator()
    assert isinstance(gen, SourceSpanGenerator) and gen.prompt_version == VERSION
    assert space_environment()["candidate"] == "G-source-spans"
    messages = messages_span_evidence(SHORT_LAMP, ActionSignals("negative", 0.983, ["other"]))
    last = messages[-1]["content"]
    assert json.dumps(source_spans(SHORT_LAMP), ensure_ascii=False) in last
    assert '"topics": ["other"]' in last
    assert "Use JSON null" in messages[0]["content"]
    from reviewnlp.triage.evidence_generator import VERSION as historical_version
    assert historical_version == "actions-v5-evidence"


def test_jev_other_is_a_hint_and_does_not_replace_selected_source_evidence_or_department():
    demo, extractor, _ = service()
    demo.jev_config = JevConfig(enabled=True, route="openrouter", api_key="FAKE-KEY", max_attempts=1)
    payload = body_for(other_complaint="1")
    payload["usage"]["cost"] = 0.0001
    demo.transport = FakeTransport(payload)
    result = demo.analyze(SHORT_LAMP, True)
    assert extractor.calls[0][1].flagged_topics == ["other"]
    assert result["actions"]["actions"][0]["department"] == "maintenance"
    assert result["issue_assessments"][0]["evidence"]["reported"] == SHORT_LAMP
    assert result["api_cost"]["attempts"] == 1 and not result["stored"]
    assert "FAKE-KEY" not in json.dumps(result)
