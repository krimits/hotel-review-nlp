"""The hosted demo must not claim a complaint from rejected model output."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "spaces" / "hotel-ops-demo" / "logic.py"
_SPEC = importlib.util.spec_from_file_location("hotel_demo_logic", _PATH)
logic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(logic)


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
    for override in ({"json_valid": False}, {"salvaged": True},
                     {"generation_hit_token_budget": True}):
        record = {"json_valid": True, "salvaged": False,
                  "generation_hit_token_budget": False,
                  "aspects": [{"aspect": "room", "sentiment": "negative", "quote": "bad room"}],
                  **override}
        assert logic.accept_record([], "bad room", record) == []


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
