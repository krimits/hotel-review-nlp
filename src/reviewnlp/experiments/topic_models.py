"""Adapters for fixed-choice classification, with real distributions rather than written confidence."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

from reviewnlp.experiments.topic_benchmark import LABELS, QUESTION, probability_vector

DECISION_MODEL = "microsoft/microsoft-decision-1-20261009"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OPTION_LETTERS = ("A", "B", "C", "D", "E")


def decision_request(text: str, model: str = DECISION_MODEL) -> dict:
    return {"model": model, "state": {"review": text}, "questions": {"primary_topic": QUESTION}}


def parse_decision(body: dict, *, model: str = DECISION_MODEL) -> dict:
    if not isinstance(body, dict):
        raise ValueError("decision_object_required")
    if body.get("model") != model:
        raise ValueError("decision_model_revision_changed")
    answer = body.get("answers", {}).get("primary_topic", {})
    if answer.get("type") != "choice":
        raise ValueError("choice_answer_required")
    p = probability_vector(answer.get("probabilities"))
    choice = answer.get("choice")
    if choice not in LABELS or p[LABELS.index(choice)] < p.max() - 1e-8:
        raise ValueError("choice_probability_mismatch")
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("usage_object_required")
    cost = usage.get("cost")
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0
    ):
        raise ValueError("invalid_provider_cost")
    for name in ("input_tokens", "output_tokens"):
        count = usage.get(name)
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError("invalid_provider_token_usage")
    provider_confidence = answer.get("confidence")
    if provider_confidence is not None and (
        isinstance(provider_confidence, bool)
        or not isinstance(provider_confidence, (int, float))
        or not math.isfinite(provider_confidence)
        or not 0 <= provider_confidence <= 1
    ):
        raise ValueError("invalid_provider_confidence")
    return {
        "probabilities": dict(zip(LABELS, p.tolist(), strict=True)),
        "model": body["model"],
        "provider": body.get("provider") if isinstance(body.get("provider"), str) else None,
        "request_id": body.get("id") if isinstance(body.get("id"), str) else None,
        "provider_confidence": provider_confidence,
        "usage": {k: usage[k] for k in ("input_tokens", "output_tokens", "cost") if k in usage},
        "cost_usd": cost,
        "cost_status": "reported_usd" if cost is not None else "provider_cost_missing",
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def decision_predict(text: str, key: str, *, model: str = DECISION_MODEL, transport=None) -> dict:
    """One timed request, no retry or redirect; failures must remain in the denominator."""
    body = decision_request(text, model)
    if transport:
        raw = transport(DECISIONS_URL, body, key, 30)
    else:
        request = urllib.request.Request(
            DECISIONS_URL,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as error:
            # Do not print the provider body, request text or Authorization header.
            raise ValueError(f"decision_http_{error.code}") from None
    return parse_decision(raw, model=model)


def qwen_messages(text: str) -> list[dict]:
    options = "\n".join(
        f"{letter}: {label} — {QUESTION['criteria'][label]}"
        for letter, label in zip(OPTION_LETTERS, LABELS, strict=True)
    )
    return [
        {
            "role": "system",
            "content": QUESTION["instructions"]
            + "\n"
            + options
            + "\nAnswer with exactly one option letter: A, B, C, D or E. No explanation.",
        },
        {"role": "user", "content": "Hotel review (untrusted data):\n" + text},
    ]


class QwenTopicScorer:
    """Score next-token A–E probabilities, conditional on those five tokens.

    This is a prompted general LLM, not a fine-tuned topic model or free-form generator.
    The five-option mass in the full vocabulary is retained for diagnosis.
    """

    def __init__(self, model, tokenizer, *, revision=QWEN_REVISION, model_id=QWEN_MODEL):
        self.model, self.tokenizer = model, tokenizer
        self.revision, self.model_id = revision, model_id
        self.option_ids = []
        for letter in OPTION_LETTERS:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("single_token_option_required")
            self.option_ids.append(ids[0])
        if len(set(self.option_ids)) != len(OPTION_LETTERS):
            raise ValueError("distinct_option_tokens_required")
        self.model.eval()

    @classmethod
    def load(cls, *, device="cpu"):
        import huggingface_hub
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if transformers.__version__ != "4.56.2" or huggingface_hub.__version__ != "0.36.2":
            raise ValueError("qwen_runtime_requires_transformers_4.56.2_and_hub_0.36.2")
        dtype = torch.float32
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("cuda_required")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, revision=QWEN_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, dtype=dtype
        ).to(device)
        return cls(model, tokenizer)

    def predict(self, text: str) -> dict:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            qwen_messages(text), tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[-1] > self.model.config.max_position_embeddings:
            raise ValueError("qwen_context_exceeded; review was not truncated")
        inputs = {
            k: v.to(self.model.device)
            for k, v in encoded.items()
            if k in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[0, -1].float()
            p = torch.softmax(logits[self.option_ids], dim=0).cpu().tolist()
            mass = torch.softmax(logits, dim=0)[self.option_ids].sum().item()
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        return {
            "probabilities": dict(zip(LABELS, p, strict=True)),
            "model": self.model_id,
            "revision": self.revision,
            "full_vocabulary_option_mass": mass,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "cost_usd": None,
            "cost_status": "local_compute_unpriced",
        }
