"""Behavior checks of local primary topics; these are not a precision/recall pilot."""

from __future__ import annotations

import json

import pytest
from triage_fakes import POSITIVE, FakeGenerator, FakeJev, FakeWrapper

from reviewnlp.triage.demo_service import DemoService, space_environment
from reviewnlp.triage.jev_client import JevConfig, JevError
from reviewnlp.triage.lexicon import SHA256, VERSION, LexiconClassifier, enabled_from_env
from reviewnlp.triage.pipeline import TriagePipeline


@pytest.mark.parametrize("text,expected", [
    ("The shower was cold.", {"bathroom"}),
    ("The bathroom was dirty.", {"cleanliness"}),
    ("We found a dirty desk.", {"cleanliness"}),
    ("The air conditioner was broken.", {"air_conditioning"}),
    ("We saw cockroaches near the bed.", {"pests"}),
    ("The staff ignored our requests.", {"responsiveness"}),
    ("The shower was cold and the carpet was dusty.", {"bathroom", "cleanliness"}),
])
def test_explicit_topics_have_literal_evidence_and_no_invented_confidence(text, expected):
    result = LexiconClassifier().classify(text)
    assert {item.topic for item in result.topics if item.answer == "yes"} == expected
    assert result.source == "lexicon" and result.coverage_status == "limited"
    assert result.questions_version == VERSION and result.questions_sha256 == SHA256
    assert {item.topic for item in result.evidence} == expected
    assert all(item.excerpt == text[item.start:item.end] for item in result.evidence)
    assert all(item.probability is None for item in [*result.topics, result.other_complaint])
    assert all(item.answer == "unsure" for item in result.topics if item.topic not in expected)


@pytest.mark.parametrize("text", [
    "The air conditioner worked well but we heard traffic outside.",
    "We loved the welcome drink.",
    "The dirty streets outside were busy.",
    "The computer mouse worked well.",
    "The bathroom was clean and spotless.",
    "If the shower was cold it would be uncomfortable; it worked perfectly.",
    "The shower was not cold.",
    "The desk wasn't dirty.",
    "We saw no cockroaches.",
    "The shower was cold. Staff repaired it and the water was hot afterwards.",
    "The shower was cold. The attempted repair did not fix it.",
    "The carpet was dirty during our previous stay, but not this time.",
    "The lift failed and left guests waiting downstairs.",
])
def test_no_match_means_human_review_never_a_prediction_of_no_complaint(text):
    result = TriagePipeline(FakeWrapper(POSITIVE), lexicon=LexiconClassifier()).run(text)
    assert not result.complaints.evidence
    assert {item.answer for item in result.complaints.topics} == {"unsure"}
    assert result.complaints.other_complaint.answer == "unsure"
    assert result.routing.needs_review and "uncertain_complaint" in result.routing.review_reasons


def test_jev_can_add_hints_but_cannot_erase_the_primary_match():
    generator = FakeGenerator('{"actions":[]}')
    result = TriagePipeline(FakeWrapper(POSITIVE), FakeJev(yes=["pests"]), generator,
                           lexicon=LexiconClassifier()).run("The shower was cold.")
    assert result.complaints.topics[0].answer == "yes"
    assert result.complaints.enhancement.topics[0].answer == "no"
    assert generator.calls[0][1].flagged_topics == ["bathroom", "pests"]
    assert result.routing.needs_review


def test_jev_failure_keeps_local_topics_and_actions_but_is_explicitly_partial():
    result = TriagePipeline(FakeWrapper(POSITIVE), FakeJev(error=JevError("timeout")),
                           FakeGenerator('{"actions":[]}'), lexicon=LexiconClassifier()).run("The shower was cold.")
    assert result.status == "partial" and result.complaints.status == "ok"
    assert result.complaints.evidence and result.complaints.enhancement.error == "timeout"
    assert "complaint_check_failed" in result.routing.review_reasons
    assert result.actions.status == "ok"


@pytest.mark.parametrize("enabled,route", [(False, "typesafe"), (True, "typesafe"), (True, "openrouter")])
def test_space_key_or_enable_flag_never_overrides_per_request_opt_out(enabled, route):
    def forbidden(*args):
        raise AssertionError("external request")
    demo = DemoService(FakeWrapper(POSITIVE), FakeGenerator('{"actions":[]}'),
        jev_config=JevConfig(enabled=enabled, route=route, api_key="FAKE-SECRET"), transport=forbidden)
    result = demo.analyze("The shower was cold.", use_jev=False)
    assert result["complaints"]["source"] == "lexicon"
    assert result["complaints"]["enhancement"]["status"] == "disabled"
    assert result["api_cost"]["attempts"] == 0 and not result["stored"]
    assert "FAKE-SECRET" not in json.dumps(result)


def test_explicit_disable_restores_the_legacy_complaint_path(monkeypatch):
    monkeypatch.setenv("REVIEWNLP_TRIAGE_LEXICON_FALLBACK", "0")
    demo = DemoService(FakeWrapper(POSITIVE), FakeGenerator('{"actions":[]}'), jev_config=JevConfig())
    result = demo.analyze("The shower was cold.")
    assert result["complaints"]["source"] == "jev" and result["complaints"]["status"] == "disabled"
    assert result["complaints"]["enhancement"] is None
    assert space_environment()["complaint_primary"] == "jev"


def test_defaults_are_local_and_access_diagnostics_expose_no_key(monkeypatch):
    assert enabled_from_env({}) and enabled_from_env({"REVIEWNLP_TRIAGE_LEXICON_FALLBACK": "1"})
    assert not enabled_from_env({"REVIEWNLP_TRIAGE_LEXICON_FALLBACK": "0"})
    monkeypatch.setenv("REVIEWNLP_JEV_ENABLED", "1")
    monkeypatch.setenv("REVIEWNLP_JEV_ROUTE", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "FAKE-SECRET")
    assert space_environment()["jev_available"]
    assert "FAKE-SECRET" not in json.dumps(space_environment())
    monkeypatch.setenv("REVIEWNLP_JEV_ROUTE", "invalid")
    assert not space_environment()["jev_available"]
