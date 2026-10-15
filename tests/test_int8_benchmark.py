"""The INT8 and batching benchmark, with fake encoders and a tiny local DistilBERT; no download."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "experiments" / "int8_and_batching" / "PROTOCOL.md"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


int8 = _load("benchmark_int8")

GOOD = ["great", "clean", "friendly", "lovely", "quiet", "helpful"]
BAD = ["dirty", "noisy", "rude", "broken", "cold", "small"]


class FakeEncoder:
    """Logits that depend on the text alone: its word count decides the label."""

    calls: list[list[str]]

    def __init__(self):
        self.calls = []

    def token_lengths(self, texts):
        return np.array([len(text.split()) for text in texts])

    def logits(self, texts):
        self.calls.append(list(texts))
        return np.array([[float(len(t.split()) % 2), float(1 - len(t.split()) % 2)] for t in texts])

    def __call__(self, texts):
        return self.logits(texts).argmax(axis=1)


def _texts(count: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(GOOD + BAD, size=rng.integers(1, 15))) for _ in range(count)]


def test_the_rule_is_the_protocols_word_for_word():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    rule = protocol.split("## 4. Decision rule", 1)[1].split("## 5. Limits", 1)[0]
    for condition, conclusion in int8.RULE:
        assert f"| `{condition}` | {conclusion} |" in rule
    assert int8.MODEL_REVISION in protocol and int8.WEIGHTS_SHA256 in protocol
    assert (int8.RESAMPLES, int8.BATCH, int8.SPEED_TIE) == (2000, 32, 0.9)


@pytest.mark.parametrize(("lo", "lat_int8", "lat_fp32", "branch"), [
    (-0.002, 30.0, 57.0, 0), (-0.002, 55.0, 57.0, 1), (-0.0099, 51.4, 57.0, 1), (-0.0099, 51.2, 57.0, 0),
    (-0.01, 30.0, 57.0, 2), (-0.05, 10.0, 57.0, 2)])
def test_each_outcome_lands_in_one_branch(lo, lat_int8, lat_fp32, branch):
    assert int8.outcome(lo, lat_int8, lat_fp32) == int8.RULE[branch][1]


def test_sorted_batching_is_faster_only_past_the_tie():
    assert int8.batching_verdict(8.9, 10.0) == "faster"
    assert int8.batching_verdict(9.0, 10.0) == "not faster"


def test_the_full_pass_returns_logits_in_input_order():
    texts = _texts(150)
    encoder = FakeEncoder()
    logits = int8.all_logits(encoder, texts, batch_size=16)
    assert np.array_equal(logits, FakeEncoder().logits(texts))
    sizes = [len(call) for call in encoder.calls]
    assert sizes[:-1] == [16] * (len(sizes) - 1)
    lengths = [max(len(t.split()) for t in call) for call in encoder.calls]
    assert lengths == sorted(lengths)  # batches really went shortest first


def test_sorted_timing_restores_the_order_before_checking_agreement():
    texts = _texts(80, seed=3)
    scored = FakeEncoder()(texts)
    report = int8.timed_sorted("fake", FakeEncoder(), texts, scored, batch_size=8)
    assert report["agrees"] == 80 and report["sorted_by"] == "token length"
    assert report["seconds"] >= report["sort_seconds"] >= 0
    with pytest.raises(int8.latency.BenchmarkError):  # labels in sorted order would not agree
        int8.timed_sorted("fake", FakeEncoder(), texts, scored[int8.length_order(FakeEncoder().token_lengths(texts))],
                          batch_size=8)


def _tiny_distilbert(directory: Path) -> None:
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    directory.mkdir(parents=True)
    (directory / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *GOOD, *BAD]) + "\n")
    DistilBertTokenizerFast(vocab_file=str(directory / "vocab.txt")).save_pretrained(directory)
    torch.manual_seed(0)
    DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=5 + len(GOOD) + len(BAD), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
        max_position_embeddings=64)).save_pretrained(directory)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    directory = tmp_path_factory.mktemp("int8") / "model"
    _tiny_distilbert(directory)
    texts = _texts(60, seed=5)
    test = pd.DataFrame({"text": texts, "label": np.where(np.arange(60) % 3, "positive", "negative")})
    fp32 = int8.Encoder(directory)
    return directory, test, int8.all_logits(fp32, texts)


def test_the_benchmark_measures_both_models_on_the_same_reviews(tiny):
    directory, test, committed = tiny
    threads = torch.get_num_threads()
    report, arrays = int8.benchmark(test, committed, directory, sample_size=40, resamples=50)
    assert torch.get_num_threads() == threads  # the one-thread timing does not leak
    assert set(arrays) == {"fp32_test_logits", "int8_test_logits", "test_labels"}  # never weights
    assert arrays["fp32_test_logits"].shape == arrays["int8_test_logits"].shape == (60, 2)
    assert np.array_equal(arrays["fp32_test_logits"].argmax(1), committed.argmax(1))
    assert report["integrity"] == {"fp32_agrees_with_committed": 60, "reviews": 60}
    for name in ("fp32", "int8"):
        cpu = report["models"][name]["cpu"]
        assert [cpu[key]["batch_size"] for key in ("batch_1", "batch_32", "batch_32_sorted")] == [1, 32, 32]
        assert all(timing["reviews"] == 40 and timing["agrees"] == 40 for timing in cpu.values())
        assert report["models"][name]["sorted_batching"] in ("faster", "not faster")
        assert set(report["models"][name]["test"]["recall"]) == {"negative", "positive"}
    primary = report["primary"]
    assert primary["paired_ci95"][0] <= primary["d_int8_minus_fp32"] <= primary["paired_ci95"][1]
    assert primary["conclusion"] in [text for _, text in int8.RULE]
    assert report["hardware"]["torch_threads"] == 1 and report["method"]["threads"] == 1
    assert report["selection_frozen"] is False and report["promoted"] is False


def test_a_checkpoint_that_was_not_scored_is_refused(tiny):
    directory, test, committed = tiny
    with pytest.raises(int8.latency.BenchmarkError, match="not the model that was scored"):
        int8.benchmark(test, committed[:, ::-1], directory, sample_size=40, resamples=50)
    with pytest.raises(int8.latency.BenchmarkError, match="cover 59 reviews"):
        int8.benchmark(test, committed[:59], directory, sample_size=40, resamples=50)


def test_the_size_is_the_serialized_state_dict(tiny):
    directory, _, _ = tiny
    fp32, quantized = int8.Encoder(directory), int8.Encoder(directory, int8=True)
    assert int8.state_dict_bytes(fp32.model) > 0 and int8.state_dict_bytes(quantized.model) > 0
    assert any("_packed_params" in key for key in quantized.model.state_dict())  # really quantized
    assert not any("_packed_params" in key for key in fp32.model.state_dict())
