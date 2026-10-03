"""Exercise both generations, linkage, ambiguous JSON, failures and actual pipeline flags offline."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reviewnlp.triage.generator_experiment import (
    prompt_fingerprint,
    run_candidate,
    structural_summary,
)
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.qwen_generator import ActionSignals, GenerationResult, parse_actions
from reviewnlp.triage.staged_generator import (
    TwoStageGenerator,
    assemble_actions,
    messages_measures,
    parse_issues,
)

TEXT = "The drawer remained jammed. The lamp was replaced immediately."
ISSUE = {"problem": "Jammed drawer", "excerpt": "The drawer remained jammed.",
         "status": "REAL_PENDING", "department": "maintenance"}
RESOLVED = {"problem": "Replaced lamp", "excerpt": "The lamp was replaced immediately.",
            "status": "REAL_RESOLVED", "department": "maintenance"}
MEASURE = {"issue_id": 1, "measure": "Inspect and repair the drawer runner.", "to_confirm": []}
SIGNALS = ActionSignals("negative", 0.9)


class Fake:
    model_name = "fixed/model"
    _bundle = ("tokenizer", "shared weights")

    def __init__(self, raw, budget=False):
        self.raw, self.budget, self.calls = raw, budget, 0

    def generate(self, review, signals):
        self.calls += 1
        if isinstance(self.raw, Exception):
            raise self.raw
        return GenerationResult(self.raw, self.budget, self.model_name)


def staged(issues=None, measures=None, *, first_budget=False, second_budget=False):
    extractor = Fake(json.dumps({"issues": [ISSUE, RESOLVED] if issues is None else issues}), first_budget)
    actioner = Fake(json.dumps({"actions": [MEASURE] if measures is None else measures}), second_budget)
    pending_seen = []

    def factory(pending):
        pending_seen.extend(pending)
        return actioner

    return TwoStageGenerator(extractor, factory), actioner, pending_seen


def test_second_call_only_gets_pending_issues_and_cannot_change_evidence_or_department():
    generator, actioner, pending = staged()
    result = generator.generate(TEXT, SIGNALS)
    action, = parse_actions(result.raw, TEXT).actions
    assert len(pending) == 1 and pending[0]["status"] == "REAL_PENDING"
    assert (action.problem, action.excerpt, action.department) == (ISSUE["problem"], ISSUE["excerpt"], "maintenance")
    assert actioner.calls == 1 and actioner._bundle is generator._bundle
    assert [s["stage"] for s in generator.last_stages] == ["issues", "measures"]
    assert all(s["raw"] and s["seconds"] >= 0 for s in generator.last_stages)
    assert RESOLVED["excerpt"] not in json.dumps(pending)
    assert prompt_fingerprint("F") != prompt_fingerprint("C")
    assert pending[0]["excerpt"] in messages_measures(TEXT, pending)[1]["content"]


@pytest.mark.parametrize("status", ["REAL_RESOLVED", "HYPOTHETICAL", "POSITIVE_COMMENT"])
def test_ineligible_items_do_not_call_the_second_generator(status):
    generator, actioner, pending = staged(issues=[{**ISSUE, "status": status}])
    result = generator.generate(TEXT, SIGNALS)
    assert json.loads(result.raw) == {"actions": []}
    assert actioner.calls == 0 and not pending and not generator.workflow_error


@pytest.mark.parametrize("damage", [{"excerpt": "An invented report."}, {"status": "pending"},
                                    {"department": "front_desk"}, {"extra": "unsupported"}])
def test_invalid_extraction_is_not_a_successful_empty_answer(damage):
    generator, actioner, _ = staged(issues=[{**ISSUE, **damage}])
    generator.generate(TEXT, SIGNALS)
    assert generator.workflow_error.startswith("issues:") and actioner.calls == 0


@pytest.mark.parametrize("damage", [{"issue_id": 2}, {"issue_id": True}, {"issue_id": "1"},
                                    {"department": "management"}, {"measure": ""}])
def test_measure_cannot_point_to_a_resolved_unknown_or_ambiguous_issue(damage):
    generator, _, _ = staged(measures=[{**MEASURE, **damage}])
    generator.generate(TEXT, SIGNALS)
    assert generator.workflow_error.startswith("measures:")


def test_both_token_budgets_are_failures_even_with_valid_complete_json():
    for stage in ("first", "second"):
        generator, actioner, _ = staged(first_budget=stage == "first", second_budget=stage == "second")
        result = generator.generate(TEXT, SIGNALS)
        assert result.hit_token_budget and generator.workflow_error.endswith("hit_token_budget")
        assert not parse_actions(result.raw, TEXT).json_valid
        assert actioner.calls == (stage == "second")


def test_raw_failure_is_retained_and_the_next_review_resets_diagnostics():
    generator, actioner, _ = staged()
    actioner.raw = RuntimeError("model failed")
    with pytest.raises(RuntimeError):
        generator.generate(TEXT, SIGNALS)
    assert generator.last_stages[-1]["error"] == "RuntimeError"
    actioner.raw = json.dumps({"actions": [MEASURE]})
    generator.generate(TEXT, SIGNALS)
    assert len(generator.last_stages) == 2 and not generator.last_stages[-1]["error"]


@pytest.mark.parametrize("raw", ['{"issues":[],"issues":[]}', '{"issues":[],"extra":1}',
                                 '{"issues":[],"x":NaN}'])
def test_extraction_requires_one_unambiguous_object(raw):
    with pytest.raises(ValueError):
        parse_issues(raw, TEXT)


@pytest.mark.parametrize("language", ["json", "", "JSON"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_a_single_whole_fence_preserves_the_schema_and_the_actual_model_response(language, newline):
    generator, actioner, _ = staged(issues=[ISSUE])
    generator.extractor.raw = "```" + language + newline + generator.extractor.raw + newline + "```"
    actioner.raw = "```" + language + newline + actioner.raw + newline + "```"
    parsed = parse_actions(generator.generate(TEXT, SIGNALS).raw, TEXT)
    assert parsed.json_valid and len(parsed.actions) == 1 and not generator.workflow_error
    assert parsed.actions[0].excerpt == ISSUE["excerpt"]
    assert generator.last_stages[0]["raw"] == generator.extractor.raw
    assert generator.last_stages[1]["raw"] == actioner.raw


@pytest.mark.parametrize("raw", [
    'Text before\n```json\n{"issues":[]}\n```',
    '```json\n{"issues":[]}\n```\nText after',
    '```json\n{"issues":[]}\n```\n```json\n{"issues":[]}\n```',
    '```json\n{"issues":[]}',
    '```python\n{"issues":[]}\n```',
    '```json\n{"issues":[]} {"issues":[]}\n```',
    '```json\n{"issues":[],"issues":[]}\n```',
    '```json\n{"issues":[],"extra":1}\n```',
    '```json\n{"issues":[],"x":NaN}\n```',
    '```json\n[]\n```',
])
def test_fences_do_not_allow_prose_fragments_multiple_objects_or_ambiguous_json(raw):
    with pytest.raises(ValueError):
        parse_issues(raw, TEXT)


def test_a_fence_does_not_bypass_the_measure_id_or_duplicate_key_guards():
    pending = parse_issues(json.dumps({"issues": [ISSUE]}), TEXT)
    for body in (json.dumps({"actions": [{**MEASURE, "issue_id": 99}]}),
                 '{"actions":[{"issue_id":1,"measure":"Repair it","measure":"Ignore it","to_confirm":[]}]}'):
        with pytest.raises(ValueError):
            assemble_actions("```json\n" + body + "\n```", pending)


def test_duplicate_issue_ids_and_duplicate_measure_keys_are_rejected():
    pending = parse_issues(json.dumps({"issues": [ISSUE]}), TEXT)
    with pytest.raises(ValueError):
        assemble_actions(json.dumps({"actions": [MEASURE, MEASURE]}), pending)
    with pytest.raises(ValueError):
        assemble_actions('{"actions":[{"issue_id":1,"measure":"Repair it",'
                         '"measure":"Ignore it","to_confirm":[]}]}', pending)


@pytest.mark.parametrize("tail", ["", "}", "\n```"])
def test_observed_duplicate_actions_never_become_a_valid_abstention_or_salvaged_array(tail):
    action = {"problem": ISSUE["problem"], "excerpt": ISSUE["excerpt"], "department": "maintenance",
              "measure": MEASURE["measure"], "to_confirm": []}
    ambiguous = '{"actions":' + json.dumps([action]) + ',"actions":[]}' + tail
    parsed = parse_actions(ambiguous, TEXT)
    assert not parsed.json_valid and not parsed.actions and parsed.error == "duplicate JSON keys"
    wrapper = SimpleNamespace(model_type="fake", model_path="fake", distribution_batch=lambda texts: [
        {"negative": 0.9, "positive": 0.1}])
    result = TriagePipeline(wrapper, generator=Fake(ambiguous)).run(TEXT)
    assert result.status == "partial" and result.actions.error == "invalid_output"
    assert "actions_failed" in result.routing.review_reasons
    assert "no_actions_suggested" not in result.routing.review_reasons


def test_duplicate_field_inside_an_action_does_not_get_silently_replaced():
    ambiguous = '{"actions":[{"problem":"Broken drawer","problem":"Fine drawer",'
    ambiguous += '"excerpt":"The drawer remained jammed.","measure":"Repair runner",'
    ambiguous += '"department":"maintenance","to_confirm":[]}]}'
    assert not parse_actions(ambiguous, TEXT).json_valid


def test_experiment_keeps_real_model_text_and_counts_stage_calls_separately():
    generator, _, _ = staged()
    cache = {"records": [{"sentiment": {"label": "negative", "confidence": 0.9},
                          "complaints": {"topics": [], "other_complaint": {"topic": "other", "answer": "no"}},
                          "routing": {"qwen_triggered": True}}]}
    records = run_candidate("F", [{"id": "dev-case", "text": TEXT}], cache, generator)
    assert records[0]["stages"][0]["raw"] == generator.extractor.raw
    assert len(records[0]["extracted_issues"]) == 2 and records[0]["full_json_valid"]
    summary = structural_summary(records)
    assert summary["issue_calls"] == summary["measure_calls"] == summary["reviews_with_accepted_actions"] == 1
    generator.extractor.raw = '{"issues":[],"issues":[]}'
    bad = run_candidate("F", [{"id": "dev-case", "text": TEXT}], cache, generator)
    assert not bad[0]["json_valid"] and bad[0]["workflow_error"]
    assert structural_summary(bad)["workflow_failures"] == 1


def test_a_schema_failure_is_also_an_error_if_the_adapter_is_given_to_the_actual_pipeline():
    generator, _, _ = staged(issues=[{**ISSUE, "excerpt": "Invented evidence."}])
    wrapper = SimpleNamespace(model_type="fake", model_path="fake", distribution_batch=lambda texts: [
        {"negative": 0.9, "positive": 0.1}])
    result = TriagePipeline(wrapper, generator=generator).run(TEXT)
    assert result.status == "partial" and result.actions.error == "invalid_output"
