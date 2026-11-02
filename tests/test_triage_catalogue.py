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


# catalogue_v2: the owner's rulings of 2 November 2026 on the check-sample disagreements (draft).
V2 = yaml.safe_load((ROOT / "docs/experiments/triage_diagnosis/catalogue_v2.yaml").read_text(encoding="utf-8"))
V2_TYPES = {entry["id"]: entry for entry in V2["types"]}


def test_v2_keeps_v1s_shape():
    assert V2["version"] == "catalogue_v2" and tuple(V2["departments"]) == DEPARTMENTS
    assert len(V2_TYPES) == len(V2["types"]) == 21
    for entry in V2_TYPES.values():
        assert entry["department"] in DEPARTMENTS and entry["priority"] in {"urgent", "routine", "structural"}
        if entry["priority"] == "structural":
            assert entry["measure"] is None and entry["department"] == "management", entry["id"]
        else:
            assert entry["measure"] and entry["deadline"], entry["id"]
        for word in entry["excludes"].replace("(", " ").replace(")", " ").replace(";", " ").split():
            if "_" in word:
                assert word.strip(".,") in V2_TYPES, (entry["id"], word)
    assert set(V2["lexicon_mapping"]) == set(TOPICS) and set(V2["lexicon_mapping"].values()) <= set(V2_TYPES)


def test_v2_carries_the_owners_seven_rulings():
    urgent = {name: entry["department"] for name, entry in V2_TYPES.items() if entry["priority"] == "urgent"}
    assert urgent == {"cleanliness": "housekeeping", "pests": "housekeeping", "security_fault": "maintenance",
                      "safety_fault": "maintenance", "safety_incident": "management",
                      "hvac_failure": "maintenance", "plumbing_major": "maintenance"}
    assert "safety_hazard" not in V2_TYPES
    assert "whatever its cause" in V2_TYPES["cleanliness"]["definition"]
    assert "too weak" in V2_TYPES["hvac_failure"]["definition"]
    assert "keeps running" in V2_TYPES["plumbing_major"]["definition"]
    assert "gurgling" in V2_TYPES["plumbing_minor"]["definition"]
    assert "pipes" not in V2_TYPES["noise_internal"]["definition"]
    assert "smoke alarm going off" in V2_TYPES["safety_fault"]["definition"]
    assert "Key cards" in V2_TYPES["security_fault"]["excludes"] and "key cards" in V2_TYPES["checkin_billing"]["definition"]
    assert any("even a mild one" in rule for rule in V2["labelling_rules"])


def test_v1_is_unchanged_by_v2():
    import hashlib

    assert hashlib.sha256((ROOT / "docs/experiments/triage_diagnosis/catalogue_v1.yaml").read_bytes()).hexdigest() == (
        "2575996133afaa6e0178060df3bd5288d01ce9b7d842a6b3bf67d362d29445e9")
