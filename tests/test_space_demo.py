"""The hosted demo must not claim a complaint from rejected model output."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml
from reviewnlp.absa.extract import parse_absa_output

_PATH = Path(__file__).resolve().parents[1] / "spaces" / "hotel-ops-demo" / "logic.py"
_SPEC = importlib.util.spec_from_file_location("hotel_demo_logic", _PATH)
logic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(logic)
_INFERENCE_PATH = _PATH.with_name("inference.py")
_INFERENCE_SPEC = importlib.util.spec_from_file_location("hotel_demo_inference", _INFERENCE_PATH)
inference = importlib.util.module_from_spec(_INFERENCE_SPEC)
_INFERENCE_SPEC.loader.exec_module(inference)


def test_demo_counts_real_negative_evidence_only():
    quoted = {"aspect": "cleanliness", "sentiment": "negative", "quote": "bathroom was dirty"}
    valid = {"json_valid": True, "salvaged": False,
             "generation_hit_token_budget": False, "aspects": [quoted]}
    history = logic.accept_record([], "The bathroom was dirty.", valid)
    assert len(history) == 1
    _, reviews, complaints, evidence = logic.render(history, valid)
    assert reviews == [[1, "Αρνητικό", 1]]
    assert complaints[0][:4] == ["Καθαριότητα", 1, 1, 100]
    assert evidence == [[1, "Καθαριότητα", "bathroom was dirty"]]
    assert logic.render([])[2:] == ([], [])


def test_demo_discards_incomplete_or_malformed_generation():
    for override in ({"json_valid": False}, {"generation_hit_token_budget": True}):
        record = {"json_valid": True, "salvaged": False,
                  "generation_hit_token_budget": False,
                  "aspects": [{"aspect": "room", "sentiment": "negative", "quote": "bad room"}],
                  **override}
        assert logic.accept_record([], "bad room", record) == []


def test_demo_accepts_wrapped_json_only_when_quotes_are_in_the_review():
    review = "The staff were kind, but the bathroom was dirty and breakfast was cold."
    raw = (
        "Here are the aspects:\n```json\n"
        '[{"aspect":"staff","sentiment":"positive","quote":"staff were kind"},'
        '{"aspect":"cleanliness","sentiment":"negative","quote":"bathroom was dirty"},'
        '{"aspect":"food","sentiment":"negative","quote":"breakfast was cold"}]'
        "\n```"
    )
    record = {**parse_absa_output(raw, review=review), "generation_hit_token_budget": False}
    assert record["salvaged"] is True
    history = logic.accept_record([], review, record)
    aspects, reviews, complaints, evidence = logic.render(history, record)
    assert len(history) == 1
    assert len(aspects) == 3
    assert reviews == [[1, "Αρνητικό", 3]]
    assert {row[0] for row in complaints} == {"Καθαριότητα", "Φαγητό / πρωινό"}
    assert {row[2] for row in evidence} == {"bathroom was dirty", "breakfast was cold"}

    forged = parse_absa_output(
        'Result: [{"aspect":"food","sentiment":"negative","quote":"food was awful"}]',
        review=review,
    )
    assert logic.accept_record(history, review, forged) == history


def test_demo_retries_an_invalid_reply_once_with_grounded_output(monkeypatch):
    text = "The staff were kind, but the bathroom was dirty."
    monkeypatch.setattr(inference, "generate_aspect_records", lambda *_args, **_kw: [
        {"json_valid": False, "aspects": [], "generation_hit_token_budget": False}
    ])

    class Tokenizer:
        pad_token_id = eos_token_id = 2
        system_message = ""

        def apply_chat_template(self, messages, **_kwargs):
            self.system_message = messages[0]["content"]
            return "prompt"

        def encode(self, _prompt, **_kwargs):
            return [1, 3]

        def pad(self, _features, **_kwargs):
            return {"input_ids": torch.tensor([[1, 3]]),
                    "attention_mask": torch.tensor([[1, 1]])}

        def batch_decode(self, _tokens, **_kwargs):
            return ['[{"aspect":"cleanliness","sentiment":"negative",'
                    '"quote":"bathroom was dirty"}]']

    class Model:
        device = "cpu"
        calls = 0

        def generate(self, input_ids, **_kwargs):
            self.calls += 1
            return torch.cat([input_ids, torch.tensor([[9, 2]])], dim=1)

    tokenizer, model = Tokenizer(), Model()
    result = inference.analyze_review(tokenizer, model, text)
    assert model.calls == 1
    assert "Copy each quote word for word" in tokenizer.system_message
    assert result["retry_used"] is True
    assert result["generation_hit_token_budget"] is False
    assert result["json_valid"] is True
    assert logic.accept_record([], text, result)[0]["aspects"] == [
        {"aspect": "cleanliness", "sentiment": "negative", "quote": "bathroom was dirty"}
    ]


def test_demo_conflicting_aspect_votes_are_neutral_and_cannot_forge_quote():
    text = "The room was lovely but then the room was awful."
    record = {"json_valid": True, "salvaged": False,
              "generation_hit_token_budget": False,
              "aspects": [
                  {"aspect": "room", "sentiment": "positive", "quote": "room was lovely"},
                  {"aspect": "room", "sentiment": "negative", "quote": "room was awful"},
                  {"aspect": "staff", "sentiment": "negative", "quote": "rude staff"},
              ]}
    history = logic.accept_record([], text, record)
    assert history[0]["aspects"] == [
        {"aspect": "room", "sentiment": "neutral", "quote": "room was lovely"}
    ]
    assert logic.render(history)[2:] == ([], [])


def test_space_builder_and_requirements_agree_on_gradio_version():
    space_dir = _PATH.parent
    readme = (space_dir / "README.md").read_text(encoding="utf-8")
    metadata = yaml.safe_load(readme.split("---", 2)[1])
    requirements = (space_dir / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert metadata["sdk"] == "gradio"
    assert f"gradio=={metadata['sdk_version']}" in requirements
