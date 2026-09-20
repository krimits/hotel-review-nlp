"""Offline tests for Greek data handling — no network, no torch."""

from __future__ import annotations

import pandas as pd

from reviewnlp.greek.data import LABEL_NAMES, clean_text, ordered_fingerprint


def test_clean_text_removes_urls_and_mentions_keeps_hashtags():
    raw = "Δωμάτιο ωραίο! http://t.co/abc123 @user_name τέλεια θέα #εκλογες2015"
    assert clean_text(raw) == "Δωμάτιο ωραίο! τέλεια θέα #εκλογες2015"


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\tc\nd") == "a b c d"


def test_clean_text_can_return_empty_string():
    assert clean_text("   @only_a_mention http://only.url   ") == ""


def test_ordered_fingerprint_is_deterministic_and_order_sensitive():
    frame_a = pd.DataFrame({"text": ["καλό", "κακό"], "label": [1, 0]})
    frame_b = pd.DataFrame({"text": ["κακό", "καλό"], "label": [0, 1]})
    assert ordered_fingerprint(frame_a) == ordered_fingerprint(frame_a.copy())
    assert ordered_fingerprint(frame_a)["sha256"] != ordered_fingerprint(frame_b)["sha256"]
    summary = ordered_fingerprint(frame_a)
    assert summary["rows"] == 2
    assert summary["positive"] == 1
    assert summary["negative"] == 1


def test_label_names_order_matches_fingerprint_convention():
    assert LABEL_NAMES == ("negative", "positive")
