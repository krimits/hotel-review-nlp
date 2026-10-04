"""Single-review experimental Space service with pinned models and no review database."""

from __future__ import annotations

import os
import threading
from dataclasses import replace

from reviewnlp.triage.evidence_generator import (
    VERSION,
    EvidenceFirstGenerator,
    messages_evidence,
    messages_evidence_measures,
)
from reviewnlp.triage.generator_experiment import RecordingTransport, cost_summary
from reviewnlp.triage.jev_client import JevClient, JevConfig, urllib_transport
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.qwen_generator import MAX_NEW_TOKENS, QwenActionGenerator

DISTILBERT_REPO = "krimits/distilbert-hotel-reviews"
DISTILBERT_REVISION = "7306aebcaaebc00d579f5d0a91001ae376f18158"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class PinnedSentiment:
    """CPU DistilBERT adapter; checks the label contract instead of guessing LABEL_0/1."""

    model_type, model_path = "encoder", DISTILBERT_REPO + "@" + DISTILBERT_REVISION

    def __init__(self):
        self._bundle = None

    def _models(self):
        if self._bundle is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_REPO, revision=DISTILBERT_REVISION)
            model = AutoModelForSequenceClassification.from_pretrained(
                DISTILBERT_REPO, revision=DISTILBERT_REVISION, dtype=torch.float32).to("cpu").eval()
            labels = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
            if len(labels) != 2 or set(labels.values()) != {"negative", "positive"} or set(labels) != {0, 1}:
                raise ValueError("sentiment_label_contract_mismatch")
            self._bundle = tokenizer, model, labels
        return self._bundle

    def distribution_batch(self, texts):
        import torch

        tokenizer, model, labels = self._models()
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.inference_mode():
            probabilities = model(**{name: encoded[name] for name in ("input_ids", "attention_mask")}).logits.softmax(dim=-1).tolist()
        return [{labels[index]: float(value) for index, value in enumerate(row)} for row in probabilities]

    def predict(self, text):
        (values,) = self.distribution_batch([text])
        label = max(values, key=values.get)
        return label, values[label]


def make_generator() -> EvidenceFirstGenerator:
    extractor = QwenActionGenerator(QWEN_REPO, revision=QWEN_REVISION, message_builder=messages_evidence,
                                    prompt_version=VERSION + ":issues")

    def action_factory(pending):
        return QwenActionGenerator(QWEN_REPO, revision=QWEN_REVISION,
            message_builder=lambda review, signals: messages_evidence_measures(review, pending),
            prompt_version=VERSION + ":measures")

    return EvidenceFirstGenerator(extractor, action_factory)


class DemoService:
    """All input reviews reach issue extraction; sentiment is context, not a gate.

    Models are loaded lazily and shared. Requests are serialized because issue assessments belong to the
    immediately preceding generation. No review database or raw-generation history is kept.
    """

    def __init__(self, wrapper=None, generator=None, *, jev_config=None, transport=urllib_transport):
        self.wrapper = wrapper if wrapper is not None else PinnedSentiment()
        self.generator = generator if generator is not None else make_generator()
        self.jev_config = jev_config if jev_config is not None else JevConfig.from_env()
        self.transport, self._lock = transport, threading.Lock()

    @property
    def jev_available(self) -> bool:
        return bool(self.jev_config.enabled and self.jev_config.api_key)

    def analyze(self, text: str, use_jev: bool = False) -> dict:
        if not isinstance(text, str) or len(text.strip()) < 3 or len(text) > 4000:
            raise ValueError("one_review_requires_3_to_4000_characters")
        if use_jev and not self.jev_available:
            raise ValueError("jev_not_configured")
        with self._lock:
            recorder = RecordingTransport(self.transport)
            client = JevClient(replace(self.jev_config, enabled=bool(use_jev)), transport=recorder)
            try:
                result = TriagePipeline(self.wrapper, client, self.generator, evaluate_all_reviews=True).run(text)
                payload = {"status": result.status, "validation_status": "unvalidated",
                    "sentiment": result.sentiment.model_dump(mode="json"),
                    "complaints": result.complaints.model_dump(mode="json"),
                    "routing": result.routing.model_dump(mode="json"),
                    "issue_assessments": list(getattr(self.generator, "issues", [])),
                    "actions": result.actions.model_dump(mode="json"),
                    "timings": result.timings.model_dump(mode="json"),
                    "api_cost": cost_summary(recorder.attempts),
                    "model_revisions": {"distilbert": DISTILBERT_REVISION, "qwen": QWEN_REVISION},
                    "generation_budget": {"max_calls_per_review": 2, "max_new_tokens_per_call": MAX_NEW_TOKENS},
                           "stored": False}
                return payload
            finally:
                # The experiment adapter records raw text for notebooks. The Space does not retain it.
                for name in ("issues", "last_stages", "review_reasons"):
                    if hasattr(self.generator, name):
                        setattr(self.generator, name, [])


def space_environment() -> dict:
    """Only public configuration; environment and keys must never be sent to the UI."""
    return {"prompt_version": VERSION, "distilbert_revision": DISTILBERT_REVISION,
            "qwen_revision": QWEN_REVISION, "validation_status": "unvalidated",
            "jev_enabled": os.environ.get("REVIEWNLP_JEV_ENABLED", "").strip().lower() in {"1", "true", "yes"}}
