"""Lock test for docs/experiments/triage_student_v2/PROTOCOL.md, version 1 (approved 3 November 2026).

The protocol is frozen once committed. This test pins its SHA-256 and the files it names (catalogue_v2 and the two
prompts written from it), and checks that the seeds, sample sizes, batches, models and decision rule it states are
the ones the code runs. A change is a version 2 beside this one and a new pin here, never an edit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
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
DRAWN = {  # step 2, drawn at 360e2fa: the keys are never redrawn or edited
    "key.csv": "14d4f955ffa0fecc482b4d2f31c48fce2d6a9db5413d6b8873b8d7b5fe42b134",
    "teacher_key.csv": "66d87a0f2c7acb851f234c53c3aace33cab47deffd2b512a5f9b3d3e99a82346",
    "manifest.json": "4b76c60095c285225d0bbcf26d1e74caa0914c5fae40af869b9689bf69d8215f",
    "reference_batches.json": "2dd52a3fc3de9b716569ae169f78483dc6206eb0eed91fe4fd6cbe2afce988f5",
}
RECORDED = {  # steps 3 and 4: labels are recorded once and never replaced or edited
    "reference_chatgpt_v2.csv": "60342b539a191c8f84804103ab97c2b9d19b21d135252b2e489531b517f71520",
    "reference_chatgpt_v2_record.json": "3707ff272c7c8cd9c55d9d0bfced9d7691d7b0d43f7984fb0753d2b709e3d5a2",
    "teacher_haiku_development.csv": "9de0b8547eecc38a324feba93dd95ef67318fa08505655b5ab4d3f1cf52f1846",
    "teacher_haiku_development_record.json": "1563dab924c84cdb42c7fce1eb9ed92a0db5710e30761d9bbae26518bb791c6e",
    "teacher_haiku_test.csv": "d68eac256f9109e88d1b00b0adb5cd7c5317b700942ee1774fe7e4b47639187a",
    "teacher_haiku_test_record.json": "b9c5f828a5aee322d59cb6fe44be719097441a7ff166c7b13fc18cd09ee6ba4c",
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


def test_the_drawn_samples_are_the_committed_ones_and_redrew_the_first_protocols():
    for name, digest in DRAWN.items():
        assert hashlib.sha256((v2.SAMPLES / name).read_bytes()).hexdigest() == digest, name
    manifest = json.loads((v2.SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_sha256"] == PROTOCOL_SHA256 and manifest["raw_csv_sha256"] == RAW_SHA256
    assert set(manifest["first_protocol_keys_redrawn_and_matched"]) == {"key.csv", "check_key.csv",
                                                                       "teacher_key.csv"}
    assert manifest["items"] == {"development": [1, 600], "test": [601, 1100], "teacher": [1, 30000]}
    batches = json.loads((v2.SAMPLES / "reference_batches.json").read_text(encoding="utf-8"))
    assert batches["prompt_sha256"] == PINNED[v2.REFERENCE_PROMPT] and len(batches["batches"]) == 22


def test_the_recorded_labels_are_the_committed_ones_under_this_protocol():
    for name, digest in RECORDED.items():
        assert hashlib.sha256((v2.LABELS / name).read_bytes()).hexdigest() == digest, name
    for name in (n for n in RECORDED if n.endswith("_record.json")):
        record = json.loads((v2.LABELS / name).read_text(encoding="utf-8"))
        assert record["protocol_sha256"] == PROTOCOL_SHA256, name
        assert "text" not in (v2.LABELS / name.replace("_record.json", ".csv")).read_text(
            encoding="utf-8").splitlines()[0].split(","), name
    reference = json.loads((v2.LABELS / "reference_chatgpt_v2_record.json").read_text(encoding="utf-8"))
    assert reference["catalogue_sha256"] == PINNED[v2.CATALOGUE]
    assert reference["reviews_by_set"] == {"development": 600, "test": 500}
    assert '"GPT 6.1 Terra", the same for all 22 batches; no reruns' in reference["source"]


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
