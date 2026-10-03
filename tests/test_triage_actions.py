"""The suggested-actions stage: the prompt, the strict parse, and the checks that stand in for trust in a 0.5B model."""

from __future__ import annotations

import json

import pytest
import torch

from reviewnlp.serving.model_wrapper import DEVICE, ModelWrapper, _distribution_encoder
from reviewnlp.triage import qwen_generator as qg
from reviewnlp.triage.qwen_generator import (
    ActionSignals,
    QwenActionGenerator,
    build_messages,
    parse_actions,
)
from reviewnlp.triage.schemas import DEPARTMENTS

REVIEW = "The shower was cold and nobody came to fix it. The staff at breakfast were lovely though."


def entry(**fields) -> dict:
    return {"problem": "Cold shower", "excerpt": "the shower was cold", "measure": "Check the boiler",
            "department": "maintenance", "to_confirm": ["room number"], **fields}


def raw(*entries, prose: str = "") -> str:
    return prose + json.dumps({"actions": list(entries)})


# --- The prompt --------------------------------------------------------------------------------------------------

def test_the_prompt_carries_the_review_the_signals_and_the_rules():
    signals = ActionSignals("negative", 0.93, ["bathroom", "responsiveness"], "Recent negative mentions: noise 4")
    system, user = build_messages(REVIEW, signals)
    assert (system["role"], user["role"]) == ("system", "user")
    for department in DEPARTMENTS:
        assert department in system["content"]
    assert "copied character for character" in system["content"] and "Do not invent facts" in system["content"]
    assert REVIEW in user["content"] and "negative (0.93)" in user["content"]
    assert "bathroom, responsiveness" in user["content"] and "noise 4" in user["content"]
    assert "not evidence" in user["content"]


def test_the_prompt_copes_with_no_confidence_no_topics_and_a_long_review():
    _, user = build_messages("word " * 5000, ActionSignals("positive", None))
    assert "Overall sentiment: positive\n" in user["content"] and "flagged: none" in user["content"]
    assert len(user["content"]) < qg.MAX_REVIEW_CHARS + 600


# --- The parse ---------------------------------------------------------------------------------------------------

def test_a_grounded_action_is_kept_and_the_review_is_matched_loosely():
    parsed = parse_actions(raw(entry(excerpt="THE  shower was\ncold")), REVIEW)
    assert parsed.json_valid and parsed.dropped == 0 and len(parsed.actions) == 1
    action = parsed.actions[0]
    assert (action.department, action.to_confirm, action.measure) == ("maintenance", ["room number"], "Check the boiler")


def test_json_around_prose_or_in_a_fence_is_found_and_a_bare_list_is_accepted():
    assert parse_actions("Here you go:\n```json\n" + raw(entry()) + "\n```\nHope that helps.", REVIEW).actions
    assert parse_actions(json.dumps([entry()]), REVIEW).actions
    assert parse_actions("Not [1] of them. " + raw(entry()), REVIEW).actions  # a stray list does not hide the answer


@pytest.mark.parametrize("text", ["", "I cannot help with that.", "{broken", '{"actions": "none"}', "[1, 2]", '{"other": []}'])
def test_output_that_is_not_the_asked_for_json_is_invalid_not_repaired(text):
    parsed = parse_actions(text, REVIEW)
    assert (parsed.json_valid, parsed.actions, parsed.dropped) == (False, [], 0) and parsed.error


def test_an_empty_list_is_a_valid_answer_that_there_is_nothing_to_fix():
    parsed = parse_actions(raw(), REVIEW)
    assert parsed.json_valid and parsed.actions == [] and parsed.dropped == 0
    assert parse_actions("[]", REVIEW).json_valid


@pytest.mark.parametrize("damage", [
    {"excerpt": "the room smelled of smoke"},             # not in the review
    {"excerpt": "ab"},                                    # too short to show anything
    {"excerpt": ""},
    {"department": "front desk"},                         # off the closed list: dropped, not coerced to other
    {"department": None},
    {"problem": 5},
    {"measure": ""},
    {"problem": "x" * 401},
    {"to_confirm": "room number"},
    {"to_confirm": ["a", "b"]},
    {"to_confirm": ["one", "two", "three", "four", "five", "six"]},
    {"to_confirm": [3]},
])
def test_a_bad_entry_is_dropped_and_counted_and_the_good_ones_survive(damage):
    parsed = parse_actions(raw(entry(**damage), entry(problem="Slow repair", excerpt="nobody came to fix it",
                                                      department="management", to_confirm=[])), REVIEW)
    assert parsed.json_valid and parsed.dropped == 1
    assert [a.problem for a in parsed.actions] == ["Slow repair"]


def test_a_missing_field_and_a_non_object_entry_are_dropped():
    broken = entry()
    del broken["measure"]
    parsed = parse_actions(raw(broken, "text", None, entry()), REVIEW)
    assert parsed.dropped == 3 and len(parsed.actions) == 1


def test_to_confirm_may_be_missing():
    only = entry()
    del only["to_confirm"]
    assert parse_actions(raw(only), REVIEW).actions[0].to_confirm == []


def test_duplicates_are_dropped_and_no_more_than_five_are_kept():
    parsed = parse_actions(raw(entry(), entry()), REVIEW)
    assert len(parsed.actions) == 1 and parsed.dropped == 1
    many = [entry(problem=f"Problem number {i}") for i in range(8)]
    parsed = parse_actions(raw(*many), REVIEW)
    assert len(parsed.actions) == qg.MAX_ACTIONS and parsed.dropped == 3


def test_every_entry_ungrounded_leaves_nothing_and_says_how_many():
    parsed = parse_actions(raw(entry(excerpt="invented words"), entry(excerpt="more invented words")), REVIEW)
    assert parsed.json_valid and parsed.actions == [] and parsed.dropped == 2


# --- The generator -------------------------------------------------------------------------------------------------

EOS = 2


class FakeTokenizer:
    eos_token_id = EOS
    pad_token_id = EOS

    def __init__(self, answer: str):
        self.answer, self.prompts = answer, []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        self.prompts.append(messages)
        return "PROMPT:" + messages[1]["content"]

    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": torch.tensor([[5, 6, 7]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, tokens, skip_special_tokens=True):
        return self.answer if len(tokens) else ""


class FakeModel:
    device = "cpu"

    def __init__(self, new_tokens: list[int]):
        self.new_tokens, self.calls = new_tokens, []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[5, 6, 7, *self.new_tokens]])


def generator(answer: str, new_tokens=(9, 9, EOS)):
    tokenizer, model, loads = FakeTokenizer(answer), FakeModel(list(new_tokens)), []

    def loader():
        loads.append(1)
        return tokenizer, model

    return QwenActionGenerator(model_id="fake/qwen", loader=loader), tokenizer, model, loads


def test_the_generator_loads_once_greedily_and_returns_the_text_and_whether_it_was_cut_off():
    gen, tokenizer, model, loads = generator(raw(entry()))
    first = gen.generate(REVIEW, ActionSignals("negative", 0.9, ["bathroom"]))
    second = gen.generate(REVIEW, ActionSignals("negative", 0.9))
    assert loads == [1] and len(model.calls) == 2 and len(tokenizer.prompts) == 2
    call = model.calls[0]
    assert call["do_sample"] is False and call["max_new_tokens"] == qg.MAX_NEW_TOKENS
    assert set(call) == {"input_ids", "attention_mask", "max_new_tokens", "do_sample", "pad_token_id", "eos_token_id"}
    assert (first.model, first.prompt_version, first.hit_token_budget) == ("fake/qwen", "actions-v1", False)
    assert parse_actions(first.raw, REVIEW).actions and second.raw == first.raw


def test_a_tokenizer_that_returns_more_than_the_model_takes_does_not_break_the_call():
    gen, tokenizer, model, _ = generator(raw(entry()))
    plain = tokenizer.__class__.__call__
    tokenizer.__class__.__call__ = lambda self, prompt, return_tensors="pt": {
        **plain(self, prompt, return_tensors), "token_type_ids": torch.tensor([[0, 0, 0]])}
    try:
        gen.generate(REVIEW, ActionSignals("negative", 0.9))
    finally:
        tokenizer.__class__.__call__ = plain
    assert "token_type_ids" not in model.calls[0]


def test_a_generation_that_never_emitted_the_end_token_is_marked_as_cut_off():
    gen, *_ = generator("{", new_tokens=(9, 9, 9))
    assert gen.generate(REVIEW, ActionSignals("negative", 0.9)).hit_token_budget is True


def test_the_generator_is_off_unless_asked_for_and_names_its_model():
    assert QwenActionGenerator.from_env({}) is None
    assert QwenActionGenerator.from_env({"REVIEWNLP_TRIAGE_QWEN_ENABLED": "0"}) is None
    on = QwenActionGenerator.from_env({"REVIEWNLP_TRIAGE_QWEN_ENABLED": "1"})
    assert on.model_name == qg.BASE_MODEL == "Qwen/Qwen2.5-0.5B-Instruct" and on._bundle is None  # nothing loaded yet
    other = QwenActionGenerator.from_env({"REVIEWNLP_TRIAGE_QWEN_ENABLED": "yes", "REVIEWNLP_TRIAGE_QWEN_MODEL": "x/y"})
    assert other.model_name == "x/y"


# --- The wrapper's distribution ----------------------------------------------------------------------------------------

def test_the_stub_distribution_sums_to_one_and_agrees_with_the_stubs_label():
    wrapper = ModelWrapper("stub", "")
    for text in ["a lovely stay", "dirty room", "ok", "never again", "x" * 50]:
        (dist,) = wrapper.distribution_batch([text])
        assert sum(dist.values()) == pytest.approx(1.0) and set(dist) == {"negative", "positive"}
        assert max(dist, key=dist.get) == wrapper.predict(text)[0]


def test_a_classical_model_gives_its_class_probabilities():
    class Pipeline:
        classes_ = ["negative", "positive"]

        def predict_proba(self, texts):
            return [[0.7, 0.3] for _ in texts]

    wrapper = ModelWrapper("classical", "x")
    wrapper._obj, wrapper._loaded = Pipeline(), True
    assert wrapper.distribution_batch(["a", "b"]) == [{"negative": 0.7, "positive": 0.3}] * 2


def test_an_encoder_gives_a_softmax_over_its_labels_in_the_order_of_the_texts():
    class Config:
        id2label = {0: "negative", 1: "positive"}

    class Model:
        config = Config()

        def __call__(self, **batch):
            class Out:
                logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
            return Out()

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.zeros(len(texts), 3, dtype=torch.long)}

    bundle = {"tokenizer": Tokenizer(), "model": Model()}
    first, second = _distribution_encoder(bundle, ["bad", "good"])
    assert first["negative"] > 0.85 and second["positive"] > 0.95
    assert sum(first.values()) == pytest.approx(1.0) and DEVICE in {"cpu", "cuda"}
    wrapper = ModelWrapper("encoder", "x")
    wrapper._obj, wrapper._loaded = bundle, True
    assert wrapper.distribution_batch(["bad", "good"]) == [first, second]


def test_a_model_that_generates_a_word_has_no_distribution():
    wrapper = ModelWrapper("qwen_qlora", "x")
    wrapper._loaded = True
    assert wrapper.distribution_batch(["a", "b"]) == [None, None]


def test_the_model_is_loaded_in_bfloat16_only_on_a_gpu_that_has_it():
    from types import SimpleNamespace

    def fake_torch(capability):
        return SimpleNamespace(bfloat16="bf16", float32="fp32",
                               cuda=SimpleNamespace(get_device_capability=lambda device: capability))

    assert qg._dtype_for(fake_torch((8, 0)), "cuda") == "bf16"      # A100, L4, H100
    assert qg._dtype_for(fake_torch((9, 0)), "cuda:0") == "bf16"
    assert qg._dtype_for(fake_torch((7, 5)), "cuda") == "fp32"      # a Colab T4: bfloat16 would be emulated
    assert qg._dtype_for(fake_torch((8, 0)), "cpu") == "fp32"       # the capability is not even asked on a CPU

    def refuse(device):
        raise AssertionError("asked for a GPU's capability on a CPU")

    cpu = SimpleNamespace(bfloat16="bf16", float32="fp32", cuda=SimpleNamespace(get_device_capability=refuse))
    assert qg._dtype_for(cpu, "cpu") == "fp32"
