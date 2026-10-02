"""Fake stages shared by the triage tests: a sentiment model, a Jev client and an action generator."""

from __future__ import annotations

import json
from types import SimpleNamespace

from reviewnlp.triage.jev_client import JevResult
from reviewnlp.triage.qwen_generator import GenerationResult
from reviewnlp.triage.schemas import OTHER, TOPICS, ComplaintTopic

REVIEW = "SECRETWORD the shower was cold and nobody came to fix it. The staff at breakfast were lovely though."
GOOD_ACTIONS = json.dumps({"actions": [{
    "problem": "Cold shower not repaired", "excerpt": "the shower was cold and nobody came to fix it",
    "measure": "Fix the boiler and tell the guest when it is done", "department": "maintenance",
    "to_confirm": ["room number"]}]})


class FakeWrapper:
    model_type, model_path = "fake", "models/fake"

    def __init__(self, distribution=None, label="positive", confidence=0.97, error: Exception | None = None):
        self.distribution, self.label, self.confidence, self.error = distribution, label, confidence, error
        self.calls = []

    def distribution_batch(self, texts):
        self.calls.append(texts)
        if self.error:
            raise self.error
        return [self.distribution]

    def predict(self, text):
        return self.label, self.confidence


class FakeJev:
    enabled = True
    config = SimpleNamespace(route="typesafe")

    def __init__(self, yes=(), unsure=(), other="no", error: Exception | None = None, enabled=True):
        self.yes, self.unsure, self.other, self.error, self.enabled, self.calls = yes, unsure, other, error, enabled, []

    def classify(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        topics = [ComplaintTopic(topic=t, answer="yes" if t in self.yes else "unsure" if t in self.unsure else "no",
                                 probability=0.97 if t in self.yes else 0.0) for t in TOPICS]
        other = ComplaintTopic(topic=OTHER, answer=self.other, probability=0.9 if self.other == "yes" else 0.0)
        return JevResult(topics=topics, other_complaint=other, model="typesafe/jev-test", route="typesafe")


class FakeGenerator:
    model_name = "fake/qwen"

    def __init__(self, raw=GOOD_ACTIONS, hit_token_budget=False, error: Exception | None = None):
        self.raw, self.hit_token_budget, self.error, self.calls = raw, hit_token_budget, error, []

    @property
    def signals(self):
        return self.calls

    def generate(self, review, signals):
        self.calls.append((review, signals))
        if self.error:
            raise self.error
        return GenerationResult(raw=self.raw, hit_token_budget=self.hit_token_budget, model="fake/qwen")


POSITIVE = {"negative": 0.03, "positive": 0.97}
NEGATIVE = {"negative": 0.95, "positive": 0.05}
