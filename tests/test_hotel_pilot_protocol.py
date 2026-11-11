"""Lock test for docs/experiments/hotel_pilot/PROTOCOL.md, version 1 (approved 11 November 2026).

The protocol is frozen once committed. This test pins its SHA-256 and the files the pilot's system is made of (the
prompt, the catalogue, the Greek labels), and checks the rule and the owner's fields it states. A change is a
version 2 beside this one and a new pin here, never an edit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"
PROTOCOL = EXPERIMENTS / "hotel_pilot" / "PROTOCOL.md"
PROTOCOL_SHA256 = "66f7c72734a01f3982e03138004cee5d960a47dfb6660ddf40bf5988710a40be"
SYSTEM = {  # what the pilot's system is made of, as the earlier lock tests pin them
    EXPERIMENTS / "triage_student_v2" / "teacher_prompt_v2.md":
        "f731026f001ea8262444edc7c3847b6729d0414588369a409de8f5717c98245a",
    EXPERIMENTS / "triage_diagnosis" / "catalogue_v2.yaml":
        "5c592af14016cba5f97a0ee021af0a54e2196d1f50ad1f01b938be3a384f966d",
    EXPERIMENTS / "hotelier_blind_eval" / "labels_el.yaml":
        "d0f992b9fe9fc8251da5afd20dbb233785992a36de79361467befcb5b3698a47",
}
RULE = ["acc >= 0.80 and worst >= 0.60 and miss <= 0.10", "acc < 0.60"]


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prose(protocol) -> str:
    return " ".join(protocol.split())


def test_version_1_is_locked_and_unchanged(protocol):
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256, (
        "PROTOCOL.md changed after lock: write a version 2 beside it instead of editing version 1")
    assert protocol.startswith("# Hotel pilot of the triage Space: protocol — version 1, locked\n")
    for heading in ("## 1. Question", "## 2. Data", "## 3. One primary comparison", "## 4. Decision rule",
                    "## 5. Limits"):
        assert heading in protocol, f"missing section: {heading}"
    assert "draft" not in protocol.lower() and "proposed" not in protocol.lower()


def test_the_selection_is_the_owners_decision_not_a_result(prose):
    assert "**This selection is the owner's decision at a human gate, not a result.**" in prose
    assert "it is not promoted" in prose and "No model is promoted by this protocol" in prose


def test_the_system_is_the_tested_teacher_unchanged(prose):
    assert "`anthropic/claude-haiku-5.5-20261007` through OpenRouter" in prose
    assert "the structured output of the triage student v2 protocol's teacher, unchanged" in prose
    for path, digest in SYSTEM.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path.name


def test_the_rule_and_the_owners_fields(prose):
    for condition in RULE:
        assert f"`{condition}`" in prose, condition
    assert "The run is decided only when **at least `H` hotels** qualify" in prose
    for row in ("| `H`, hotels that must qualify | 5 | owner (11 November 2026) |",
                "| `R`, rated reviews for a hotel to qualify | 30 | owner (11 November 2026) |",
                "| `G`, rated urgent flags for a hotel to qualify | 10 | owner (11 November 2026) |",
                "| Cap per hotel | 300 reviews analysed"):
        assert row in prose, row


def test_no_text_is_stored_and_the_records_stay_private(prose):
    assert "**What is never stored:** the review text, the hotel's name, the access code." in prose
    assert "**The records are not committed to this repository.**" in prose
    assert "each review's text is sent to OpenRouter and Anthropic for the analysis" in prose
