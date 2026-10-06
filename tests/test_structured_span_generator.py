"""Real grammar checks; constrained formatting must not become a semantic acceptance shortcut."""

from __future__ import annotations

import json

import pytest
from lmformatenforcer import JsonSchemaParser

from reviewnlp.triage.span_evidence_generator import parse_span_issues
from reviewnlp.triage.structured_span_generator import issue_schema

REVIEW = "The shower leaked. Staff attempted a repair but it still leaked."
ISSUE = {"problem": "Leaking shower", "excerpt_span": 1,
         "evidence": {"reported": 1, "hypothetical": None, "resolved": None},
         "status": "REAL_PENDING", "category": "equipment_fault"}


def accepted(raw, review=REVIEW):
    parser = JsonSchemaParser(issue_schema(review))
    for char in raw:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


@pytest.mark.parametrize("entry", [
    {**ISSUE, "category": "leaks_or_mechanical_noise"},
    {**ISSUE, "category": "noise"},
    {**ISSUE, "category": "maintenance"},
    {**ISSUE, "excerpt_span": 0},
    {**ISSUE, "excerpt_span": 3},
    {**ISSUE, "excerpt_span": "1"},
    {**ISSUE, "excerpt_span": True},
    {**ISSUE, "excerpt_span": 1.0},
    {**ISSUE, "evidence": {**ISSUE["evidence"], "reported": 3}},
    {**ISSUE, "evidence": {**ISSUE["evidence"], "reported": "1"}},
    {**ISSUE, "status": "negative"},
    {**ISSUE, "extra": "ignored"},
    {name: value for name, value in ISSUE.items() if name != "status"},
])
def test_grammar_blocks_invented_categories_nonexistent_or_coerced_ids_and_missing_fields(entry):
    assert not accepted(json.dumps({"issues": [entry]}))


def test_empty_output_remains_allowed_and_no_issue_is_forced_into_praise():
    assert accepted('{"issues": []}', "The room was lovely.")
    assert not accepted('{"issues": []')
    assert not accepted('{"issues": [], "comment": "extra"}')
    assert not accepted(json.dumps({"issues": [ISSUE] * 6}))


def test_all_source_ids_and_null_evidence_remain_available_without_forcing_pending():
    issue = {**ISSUE, "excerpt_span": 2, "status": "UNCERTAIN",
             "evidence": dict.fromkeys(("reported", "hypothetical", "resolved"))}
    raw = json.dumps({"issues": [issue]})
    assert accepted(raw)
    (parsed,) = parse_span_issues(raw, REVIEW)
    assert parsed["status"] == "UNCERTAIN" and parsed["evidence"]["reported"] is None
    assert parsed["excerpt"] == "Staff attempted a repair but it still leaked."


def test_valid_format_does_not_bypass_evidence_consistency_or_duplicate_issue_rejection():
    contradictory = {**ISSUE, "evidence": {**ISSUE["evidence"], "hypothetical": 2}}
    raw = json.dumps({"issues": [contradictory]})
    assert accepted(raw)
    assert parse_span_issues(raw, REVIEW)[0]["status"] == "UNCERTAIN"
    duplicated = json.dumps({"issues": [ISSUE, ISSUE]})
    assert accepted(duplicated)
    with pytest.raises(ValueError, match="duplicate_span_issue"):
        parse_span_issues(duplicated, REVIEW)


def test_the_deployed_prompt_and_generator_are_unchanged():
    from reviewnlp.triage.demo_service import make_generator
    from reviewnlp.triage.span_evidence_generator import VERSION, SourceSpanGenerator

    generator = make_generator()
    assert type(generator) is SourceSpanGenerator
    assert generator.prompt_version == VERSION == "actions-v6-source-spans"
    assert type(generator.extractor).__name__ == "QwenActionGenerator"


def test_generate_filters_actual_token_ids_reuses_the_trie_and_resets_each_request(monkeypatch):
    import torch
    from lmformatenforcer.integrations import transformers as integration
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from reviewnlp.triage.qwen_generator import ActionSignals
    from reviewnlp.triage.structured_span_generator import StructuredSpanExtractor

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: i for i, char in enumerate(alphabet)}
    vocab["<eos>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", pad_token="<eos>")
    tokenizer.chat_template = "{{ messages[-1]['content'] }}"
    builds = []
    original = integration.build_token_enforcer_tokenizer_data

    def build(value):
        builds.append(value)
        return original(value)

    monkeypatch.setattr(integration, "build_token_enforcer_tokenizer_data", build)
    output = json.dumps({"issues": [ISSUE]})

    class Model:
        device = torch.device("cpu")
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 400
            prefix = kwargs["prefix_allowed_tokens_fn"]
            ids = kwargs["input_ids"][0].tolist()
            tokens = tokenizer.encode(output, add_special_tokens=False) + [tokenizer.eos_token_id]
            for token in tokens:
                assert token in prefix(0, torch.tensor(ids)), "format filter must be used in generate"
                ids.append(token)
            return torch.tensor([ids])

    model = Model()
    generator = StructuredSpanExtractor("fake", loader=lambda: (tokenizer, model))
    for _ in range(2):
        generated = generator.generate(REVIEW, ActionSignals("negative", 0.9))
        assert generated.raw == output and not generated.hit_token_budget
        assert parse_span_issues(generated.raw, REVIEW)[0]["status"] == "REAL_PENDING"
    assert model.calls == 2 and builds == [tokenizer]
