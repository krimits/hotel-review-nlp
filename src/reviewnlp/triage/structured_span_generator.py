"""Development-only constrained decoding of the existing source-span issue prompt.

The deployed workflow is unchanged. Formatting constraints do not establish semantic truth;
the unchanged source/evidence parser and human evaluation still apply.
"""

from __future__ import annotations

import hashlib
import json

from reviewnlp.triage.evidence_generator import CATEGORY_DEPARTMENTS, EVIDENCE_STATUSES
from reviewnlp.triage.qwen_generator import MAX_FIELD_CHARS, GenerationResult, QwenActionGenerator
from reviewnlp.triage.span_evidence_generator import messages_span_evidence, source_spans

VERSION = "actions-v6-source-spans-json-v1"
CANDIDATE = "G-source-spans+json-schema"
FORMAT_ENFORCER_VERSION = "0.11.3"


def issue_schema(review: str) -> dict:
    """Only restrict form and source membership; never force a reported/pending judgment."""
    span_id = {"type": "integer", "enum": [span["id"] for span in source_spans(review)]}
    evidence_id = {"anyOf": [span_id, {"type": "null"}]}
    evidence = {"type": "object", "additionalProperties": False,
                "properties": dict.fromkeys(("reported", "hypothetical", "resolved"), evidence_id),
                "required": ["reported", "hypothetical", "resolved"]}
    issue = {"type": "object", "additionalProperties": False,
             "properties": {"problem": {"type": "string", "minLength": 3, "maxLength": MAX_FIELD_CHARS},
                            "excerpt_span": span_id, "evidence": evidence,
                            "status": {"type": "string", "enum": list(EVIDENCE_STATUSES)},
                            "category": {"type": "string", "enum": list(CATEGORY_DEPARTMENTS)}},
             "required": ["problem", "excerpt_span", "evidence", "status", "category"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"issues": {"type": "array", "items": issue, "maxItems": 5}},
            "required": ["issues"]}


def schema_fingerprint(review: str) -> str:
    return hashlib.sha256(json.dumps(issue_schema(review), sort_keys=True).encode()).hexdigest()


class StructuredSpanExtractor(QwenActionGenerator):
    """One greedy bounded call. No repair, retry, new model, or weaker output parser."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, message_builder=messages_span_evidence,
                         prompt_version=VERSION + ":issues", **kwargs)
        self._tokenizer_data = None
        self._tokenizer_identity = None

    def generate(self, review, signals) -> GenerationResult:
        import torch
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
            build_transformers_prefix_allowed_tokens_fn,
        )

        tokenizer, model = self._models()
        prompt = tokenizer.apply_chat_template(self._message_builder(review, signals), tokenize=False,
                                               add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        inputs = {name: encoded[name].to(model.device) for name in ("input_ids", "attention_mask")}
        with self._run_lock, torch.inference_mode():
            if self._tokenizer_identity is not tokenizer:
                self._tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
                self._tokenizer_identity = tokenizer
            # Parser state is fresh per request; only the tokenizer trie is cached.
            prefix = build_transformers_prefix_allowed_tokens_fn(self._tokenizer_data,
                                                                  JsonSchemaParser(issue_schema(review)))
            generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        eos = tokenizer.eos_token_id
        return GenerationResult(tokenizer.decode(new_tokens, skip_special_tokens=True),
                                eos is None or not bool((new_tokens == eos).any()),
                                self.model_name, self.prompt_version)
