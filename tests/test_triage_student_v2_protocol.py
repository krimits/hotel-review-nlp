"""Lock test for docs/experiments/triage_student_v2/PROTOCOL.md, version 1 (approved 3 November 2026).

The protocol is frozen once committed. This test pins its SHA-256 and the files it names (catalogue_v2 and the two
prompts written from it), and checks that the seeds, sample sizes, batches, models and decision rule it states are
the ones the code runs. A change is a version 2 beside this one and a new pin here, never an edit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs" / "experiments" / "triage_student_v2"
PROTOCOL = FOLDER / "PROTOCOL.md"
PROTOCOL_SHA256 = "fd72c554b1b16871eee2ed3acc84fe2363e347b372ee822f69d024a9f2f2a7e9"
RAW_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"
PINNED = {  # file -> SHA-256 at lock
    ROOT / "docs" / "experiments" / "triage_diagnosis" / "catalogue_v2.yaml":
        "5c592af14016cba5f97a0ee021af0a54e2196d1f50ad1f01b938be3a384f966d",
    FOLDER / "reference_prompt_v2.md": "16d07be08fe86584e16bfd74e8d98cd56d3d73a51c2adda1ab15400174c0a3c7",
    FOLDER / "teacher_prompt_v2.md": "f731026f001ea8262444edc7c3847b6729d0414588369a409de8f5717c98245a",
}
RULE = [
    "rec >= 0.90 and lo >= 0.80 and fa <= 5",
    "rec >= 0.90 and lo >= 0.80 and fa > 5",
    "rec >= 0.90 and lo < 0.80",
    "rec < 0.90",
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load("triage_student_v2")


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prose(protocol) -> str:
    return " ".join(protocol.split())


def test_version_1_is_locked_and_unchanged(protocol):
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256, (
        "PROTOCOL.md changed after lock: write a version 2 beside it instead of editing version 1")
    assert protocol.startswith("# Triage student, second attempt: protocol — version 1, locked\n")
    assert "approved by the owner on 3 November 2026" in protocol
    for heading in ("## 1. Question", "## 2. Data", "## 3. The student",
                    "## 4. One primary comparison and the decision rule", "## 5. Limits"):
        assert heading in protocol, f"missing section: {heading}"
    assert "draft" not in protocol.lower()


def test_the_question_is_agreement_with_the_reference_llm_under_catalogue_v2(prose):
    assert ("does it catch at least 90% of the urgent problems that a reference LLM (ChatGPT, OpenAI) labels with "
            "`catalogue_v2`?") in prose
    assert "agreement with one OpenAI model's labels, made through ChatGPT. Not agreement with the owner" in prose
    assert "**No human check.**" in prose


def test_the_data_the_catalogue_and_the_prompts_match_their_fingerprints(prose):
    assert RAW_SHA256 in prose and "238,154,765 bytes" in prose
    for path, digest in PINNED.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path.name
    assert PINNED[v2.CATALOGUE] in prose


def test_the_prompts_are_catalogue_v2s_word_for_word():
    assert v2.REFERENCE_PROMPT.read_text(encoding="utf-8") == v2.labels.reference_prompt(v2.CATALOGUE)
    assert v2.TEACHER_PROMPT.read_text(encoding="utf-8") == v2.labels.teacher_prompt(v2.CATALOGUE)


def test_seeds_sizes_and_batches_are_the_codes(prose):
    for _, _, seed, sizes in (v2.DEVELOPMENT, v2.TEST, v2.TEACHER):
        assert f"H {sizes['H']:,}, N {sizes['N']:,}, B {sizes['B']:,}, **seed {seed}**" in prose, seed
    assert "with `n_s` counting the old and new reviews together (H 180, N 360, B 60)" in prose
    assert "**27,000 for training** and **3,000 held out**" in prose
    assert "exactly when its pool number ends in 0 (pools 0, 10, …, 70)" in prose
    assert ("The 600 development reviews are items 1–600 and the 500 test reviews items 601–1,100, each set in a "
            "random order (seeds 20261103 and 20261102)") in prose
    batches = v2.BATCHES
    assert (f"**{sum(batches.values())} batches of {v2.BATCH_SIZE}**: {batches['development']} for the 600 "
            f"development reviews, {batches['test']} for the 500 test reviews, in item order") in prose


def test_the_reference_the_teacher_and_the_student_are_fixed(prose):
    assert 'the model the app shows as "GPT 6.1 Terra"' in prose
    assert f"`{v2.teacher.MODELS[v2.TEACHER_MODEL]}` through OpenRouter" in prose
    assert f"{v2.teacher.BATCH} reviews per call, at most two retries" in prose and v2.teacher.TRIES == 3
    assert "`distilbert/distilbert-base-uncased` at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`" in prose
    assert ("256 tokens, 4 epochs, batch 32, lr 2e-5, 6% warmup then linear decay, weight decay 0.01, fp16, seed 42"
            in prose)
    assert "`pos_weight` per type = min(negatives / positives in the training labels, 20)" in prose
    assert "The grid is 0.05, 0.10, …, 0.95; ties go to the smallest value." in prose
    assert "**Only the chosen variant is scored on the test set, once.**" in prose


def test_the_rule_is_the_first_protocols_and_the_scorers(prose):
    for condition in RULE:
        assert f"`{condition}`" in prose, condition
    score = _load("triage_diagnosis_score")
    assert [condition for condition, _ in score.RULE] == RULE
