"""Shape of the triage diagnosis catalogue: the rules PROTOCOL.md relies on, checked before any label exists."""

from __future__ import annotations

from pathlib import Path

import yaml

from reviewnlp.triage.schemas import DEPARTMENTS, TOPICS

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = yaml.safe_load((ROOT / "docs/experiments/triage_diagnosis/catalogue_v1.yaml").read_text(encoding="utf-8"))
TYPES = {entry["id"]: entry for entry in CATALOGUE["types"]}


def test_types_are_unique_and_within_the_protocols_range():
    assert len(TYPES) == len(CATALOGUE["types"])
    assert 15 <= len(TYPES) <= 20


def test_every_type_has_one_known_department_and_priority():
    assert tuple(CATALOGUE["departments"]) == DEPARTMENTS
    for entry in TYPES.values():
        assert entry["department"] in DEPARTMENTS, entry["id"]
        assert entry["priority"] in {"urgent", "routine", "structural"}, entry["id"]
        assert entry["definition"] and entry["excludes"], entry["id"]


def test_the_owners_urgent_list_and_the_protocols_splits():
    urgent = {name for name, entry in TYPES.items() if entry["priority"] == "urgent"}
    assert urgent == {"cleanliness", "pests", "security_fault", "safety_hazard", "hvac_failure", "plumbing_major"}
    assert TYPES["plumbing_minor"]["priority"] == TYPES["noise_internal"]["priority"] == "routine"
    assert TYPES["plumbing_major"]["department"] == TYPES["plumbing_minor"]["department"] == "maintenance"
    assert (TYPES["noise_external"]["priority"], TYPES["noise_external"]["department"]) == ("structural", "management")


def test_operational_types_have_a_measure_and_structural_types_none():
    for entry in TYPES.values():
        if entry["priority"] == "structural":
            assert entry["measure"] is None and entry["deadline"] is None, entry["id"]
            assert entry["department"] == "management", entry["id"]
        else:
            assert entry["measure"] and entry["deadline"], entry["id"]
    assert all(TYPES[name]["deadline"] in {"today", "before the next arrival"}
               for name, entry in TYPES.items() if entry["priority"] == "urgent")


def test_lexicon_mapping_covers_every_lexicon_topic_with_a_known_type():
    assert set(CATALOGUE["lexicon_mapping"]) == set(TOPICS)
    assert set(CATALOGUE["lexicon_mapping"].values()) <= set(TYPES)


def test_every_exclusion_names_only_known_types():
    for entry in TYPES.values():
        for word in entry["excludes"].replace("(", " ").replace(")", " ").replace(";", " ").split():
            if "_" in word:
                assert word.strip(".,") in TYPES, (entry["id"], word)
