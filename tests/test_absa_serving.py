"""Offline tests for the ABSA serving path — no model download, no network.

The generation loop, the model manager and the two /absa endpoints had no
coverage at all. They could not be tested while the loop only existed inside a
function that loaded Qwen2.5-0.5B first; now that generation takes an
already-loaded tokenizer and model, a fake pair exercises the whole path.

The fake pair is deliberately literal about the two things that were wrong:
batches really are batched (one generate call per batch, not per review), and
a row that emitted EOS is distinguished from one that was cut off.
"""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from reviewnlp.absa.pipeline import generate_aspect_records
from reviewnlp.serving.absa_model_manager import AbsaModelManager
from reviewnlp.serving.absa_router import get_absa_model
from reviewnlp.serving.app import app
from reviewnlp.serving.model_wrapper import ModelWrapper

EOS_ID = 2
FILLER_ID = 7
RAW_BASE = 1000  # token ids at or above this encode "canned generation #i"

CLEAN = '[{"aspect": "staff", "sentiment": "positive", "quote": "kind staff"}]'
NEGATIVE = '[{"aspect": "noise", "sentiment": "negative", "quote": "very loud"}]'


class _FakeTokenizer:
    """Just enough tokenizer to drive the generation loop."""

    pad_token_id = EOS_ID
    eos_token_id = EOS_ID

    def __init__(self, canned: list[str]):
        self.canned = canned
        self.padded_batches: list[torch.Tensor] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[1]["content"]  # the user turn carries the review

    def encode(self, prompt, add_special_tokens=False):
        # Length varies with the review so batches genuinely need padding.
        return [5] * (len(prompt) % 3 + 1)

    def pad(self, features, padding=True, return_tensors="pt"):
        longest = max(len(f["input_ids"]) for f in features)
        input_ids, attention = [], []
        for feature in features:
            gap = longest - len(feature["input_ids"])
            # Left padding, as load_absa_model configures on the real tokenizer.
            input_ids.append([self.pad_token_id] * gap + feature["input_ids"])
            attention.append([0] * gap + feature["attention_mask"])
        batch = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
        }
        self.padded_batches.append(batch["attention_mask"])
        return batch

    def batch_decode(self, new_tokens, skip_special_tokens=True):
        decoded = []
        for row in new_tokens:
            marker = [int(t) for t in row if int(t) >= RAW_BASE]
            decoded.append(self.canned[marker[0] - RAW_BASE] if marker else "")
        return decoded


class _FakeModel:
    """Returns canned generations and records how many rows each call saw."""

    device = "cpu"

    def __init__(self, plan: list[tuple[int, bool]]):
        # plan: (index into canned outputs, whether the row emitted EOS)
        self.plan = list(plan)
        self.call_sizes: list[int] = []

    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        rows = input_ids.shape[0]
        self.call_sizes.append(rows)
        taken = [self.plan.pop(0) for _ in range(rows)]
        new_tokens = []
        for index, emits_eos in taken:
            tail = [EOS_ID, EOS_ID] if emits_eos else [FILLER_ID, FILLER_ID]
            new_tokens.append([RAW_BASE + index, *tail])
        return torch.cat([input_ids, torch.tensor(new_tokens)], dim=1)


def _pair(canned, plan):
    return _FakeTokenizer(canned), _FakeModel(plan)


# --------------------------------------------------------------------------
# generation loop
# --------------------------------------------------------------------------


def test_texts_are_generated_in_batches_not_one_at_a_time():
    tokenizer, model = _pair([CLEAN], [(0, True)] * 5)
    records = generate_aspect_records(tokenizer, model, ["a review"] * 5, batch_size=2)

    assert len(records) == 5
    # Five reviews at batch size two is three forward passes, not five.
    assert model.call_sizes == [2, 2, 1]


def test_records_come_back_in_the_order_the_texts_went_in():
    tokenizer, model = _pair([CLEAN, NEGATIVE], [(0, True), (1, True)])
    records = generate_aspect_records(
        tokenizer, model, ["staff were kind", "it was very loud"], batch_size=2
    )

    assert [r["text"] for r in records] == ["staff were kind", "it was very loud"]
    assert [r["aspects"][0]["aspect"] for r in records] == ["staff", "noise"]


def test_each_record_carries_the_generation_alongside_the_parse():
    tokenizer, model = _pair([CLEAN], [(0, True)])
    record = generate_aspect_records(tokenizer, model, ["kind staff"], batch_size=1)[0]

    assert record["raw_generation"] == CLEAN
    assert record["json_valid"] is True
    assert record["entries_kept"] == 1
    assert record["aspects"] == [
        {"aspect": "staff", "sentiment": "positive", "quote": "kind staff"}
    ]


def test_the_token_budget_flag_is_per_row_not_per_batch():
    # Generation runs until every row in the batch is done, so a chunk-wide
    # check marked rows that had emitted EOS long before as cut off too.
    tokenizer, model = _pair([CLEAN, NEGATIVE], [(0, True), (1, False)])
    records = generate_aspect_records(tokenizer, model, ["short one", "a longer one"], batch_size=2)

    assert records[0]["generation_hit_token_budget"] is False  # emitted EOS
    assert records[1]["generation_hit_token_budget"] is True  # never did


def test_a_batch_is_left_padded_to_its_longest_prompt():
    tokenizer, model = _pair([CLEAN], [(0, True)] * 2)
    generate_aspect_records(tokenizer, model, ["a", "a much longer review"], batch_size=2)

    mask = tokenizer.padded_batches[0]
    assert mask.shape[0] == 2
    # Padding on the left means any zeros in a row come before the ones.
    for row in mask.tolist():
        assert row == sorted(row)


def test_no_texts_means_no_generation_calls():
    tokenizer, model = _pair([CLEAN], [])
    assert generate_aspect_records(tokenizer, model, [], batch_size=4) == []
    assert model.call_sizes == []


# --------------------------------------------------------------------------
# model manager
# --------------------------------------------------------------------------


@pytest.fixture
def loaded_manager(monkeypatch):
    """A manager wired to the fake pair, counting how often it loads."""

    def make(canned, plan, **kwargs):
        tokenizer, model = _pair(canned, plan)
        loads = []

        def fake_load(adapter_dir=None, device=None):
            loads.append((adapter_dir, device))
            return tokenizer, model, "base"

        monkeypatch.setattr(
            "reviewnlp.serving.absa_model_manager.load_absa_model", fake_load
        )
        manager = AbsaModelManager(**kwargs)
        return manager, model, loads

    return make


def test_the_model_loads_once_however_many_requests_arrive(loaded_manager):
    manager, _model, loads = loaded_manager([CLEAN], [(0, True)] * 3)

    assert loads == []  # nothing loaded at construction
    manager.analyze("one")
    manager.analyze("two")
    manager.analyze("three")
    assert len(loads) == 1


def test_analyze_returns_the_single_record_from_the_batch_path(loaded_manager):
    manager, model, _loads = loaded_manager([NEGATIVE], [(0, True)])
    record = manager.analyze("it was very loud")

    assert record["aspects"][0]["aspect"] == "noise"
    assert model.call_sizes == [1]


def test_analyze_batch_respects_the_configured_batch_size(loaded_manager):
    manager, model, _loads = loaded_manager([CLEAN], [(0, True)] * 5, batch_size=2)
    records = manager.analyze_batch(["review"] * 5)

    assert len(records) == 5
    assert model.call_sizes == [2, 2, 1]


def test_the_batch_size_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ABSA_BATCH_SIZE", "3")
    assert AbsaModelManager().batch_size == 3


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_TYPE", "stub")
    import reviewnlp.serving.app as app_module

    app_module.wrapper = ModelWrapper("stub", "")
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _serve(canned, plan, **kwargs):
    """A manager already holding the fake pair, so no load is attempted."""
    tokenizer, model = _pair(canned, plan)
    manager = AbsaModelManager(**kwargs)
    manager.tokenizer, manager.model, manager._loaded = tokenizer, model, True
    return manager, model


def test_absa_endpoint_returns_aspects_and_a_measured_time(client):
    manager, _model = _serve([NEGATIVE], [(0, True)])
    app.dependency_overrides[get_absa_model] = lambda: manager

    response = client.post(
        "/absa", json={"hotel_id": "acme", "review_id": "r1", "text": "it was very loud"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["hotel_id"] == "acme"
    assert body["review_id"] == "r1"
    assert body["aspects"] == [
        {"aspect": "noise", "sentiment": "negative", "quote": "very loud", "confidence": None}
    ]
    assert body["overall_sentiment"] == "negative"
    assert body["json_valid"] is True
    # A single review is generated on its own, so its time is real.
    assert body["processing_time_ms"] >= 0


def test_batch_endpoint_generates_every_review_in_one_pass(client):
    manager, model = _serve([CLEAN, NEGATIVE], [(0, True), (1, True), (0, True)], batch_size=8)
    app.dependency_overrides[get_absa_model] = lambda: manager

    response = client.post(
        "/absa/batch",
        json={
            "hotel_id": "acme",
            "reviews": [
                {"hotel_id": "acme", "review_id": "r1", "text": "kind staff"},
                {"hotel_id": "acme", "review_id": "r2", "text": "it was very loud"},
                {"hotel_id": "acme", "review_id": "r3", "text": "kind staff again"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()

    # Three reviews, one forward pass - the whole point of the endpoint.
    assert model.call_sizes == [3]
    assert [r["review_id"] for r in body["results"]] == ["r1", "r2", "r3"]
    assert [r["aspects"][0]["aspect"] for r in body["results"]] == ["staff", "noise", "staff"]


def test_batch_results_report_no_per_review_time_and_one_real_total(client):
    manager, _model = _serve([CLEAN], [(0, True)] * 2)
    app.dependency_overrides[get_absa_model] = lambda: manager

    response = client.post(
        "/absa/batch",
        json={
            "hotel_id": "acme",
            "reviews": [
                {"hotel_id": "acme", "text": "kind staff"},
                {"hotel_id": "acme", "text": "kind staff too"},
            ],
        },
    )
    body = response.json()

    # This field used to be 0.0 with a "will be updated below" comment that
    # nothing acted on, which read as "took no time at all".
    assert [r["processing_time_ms"] for r in body["results"]] == [None, None]
    assert body["total_processing_time_ms"] >= 0


def test_batch_endpoint_still_enforces_its_limits(client):
    manager, _model = _serve([CLEAN], [])
    app.dependency_overrides[get_absa_model] = lambda: manager

    empty = client.post("/absa/batch", json={"hotel_id": "acme", "reviews": []})
    assert empty.status_code == 422  # min_length=1

    short = client.post("/absa", json={"hotel_id": "acme", "text": "no"})
    assert short.status_code == 422  # min_length=3
