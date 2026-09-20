"""Offline tests for Greek data handling — no network, no torch."""

from __future__ import annotations

import pandas as pd
import pytest

from reviewnlp.greek.config import DataConfig
from reviewnlp.greek.data import (
    LABEL_NAMES,
    apply_duplicate_mode,
    assign_splits,
    build_frame,
    clean_text,
    enforce_overlap_policy,
    group_key,
    ordered_fingerprint,
    split_overlap,
)


def _config(**overrides) -> DataConfig:
    base = {"dataset_name": "test/ds", "dataset_revision": "deadbeef"}
    return DataConfig(**{**base, **overrides})


def _corpus(pairs: int = 60) -> pd.DataFrame:
    """Balanced synthetic corpus with three variants of one positive text."""
    rows = []
    for index in range(pairs):
        rows.append({"text": f"καλό σχόλιο {index}", "label": 1})
        rows.append({"text": f"κακό σχόλιο {index}", "label": 0})
    rows.append({"text": "ΚΑΛΟ ΣΧΟΛΙΟ 7", "label": 1})
    rows.append({"text": "καλο   σχολιο 7 ", "label": 1})
    return pd.DataFrame(rows)


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


def test_group_key_ignores_case_spacing_and_unicode_form():
    assert group_key("Καλό  Σχόλιο") == group_key("καλό σχόλιο")
    assert group_key("καλό σχόλιο") != group_key("κακό σχόλιο")


def test_build_frame_honours_the_configured_column_names():
    raw = pd.DataFrame({"tweet": ["καλό", "@x http://y"], "sentiment": [1, 0]})
    frame = build_frame(raw, _config(text_column="tweet", label_column="sentiment"))
    # The all-mentions/URL row cleans down to empty and is dropped.
    assert list(frame.columns) == ["text", "label", "label_name", "group_key"]
    assert len(frame) == 1
    assert frame.loc[0, "label_name"] == "positive"


def test_build_frame_rejects_missing_columns_and_stray_labels():
    with pytest.raises(ValueError, match="missing column"):
        build_frame(pd.DataFrame({"text": ["a"]}), _config())
    with pytest.raises(ValueError, match="unexpected labels"):
        build_frame(pd.DataFrame({"text": ["a"], "label": [7]}), _config())


def test_duplicate_mode_drop_keeps_one_row_per_group():
    frame = build_frame(_corpus(), _config())
    dropped = apply_duplicate_mode(frame, "drop")
    assert len(dropped) == frame["group_key"].nunique() < len(frame)
    assert not dropped["group_key"].duplicated().any()


def test_assign_splits_matches_the_configured_proportions():
    config = _config()
    frame = apply_duplicate_mode(build_frame(_corpus(), config), config.duplicate_mode)
    frames = assign_splits(frame, config)

    assert sum(len(f) for f in frames.values()) == len(frame)  # nothing lost
    total = len(frame)
    assert len(frames["test"]) / total == pytest.approx(config.test_size, abs=0.03)
    assert len(frames["validation"]) / total == pytest.approx(config.val_size, abs=0.03)


def test_assign_splits_is_deterministic_for_a_seed_and_varies_across_seeds():
    frame = build_frame(_corpus(), _config())
    first = assign_splits(frame, _config(seed=42))
    assert all(first[name].equals(assign_splits(frame, _config(seed=42))[name]) for name in first)
    other = assign_splits(frame, _config(seed=7))
    assert set(first["test"]["text"]) != set(other["test"]["text"])


def test_grouped_duplicates_never_straddle_a_split():
    """The whole point of duplicate_mode: group — the three casing variants of
    'καλό σχόλιο 7' must land in one split, or the test set leaks into train."""
    config = _config()
    frame = build_frame(_corpus(), config)
    variants = frame[frame["group_key"] == group_key("καλό σχόλιο 7")]
    assert len(variants) == 3  # sanity: the fixture really does duplicate

    frames = assign_splits(frame, config)
    homes = {name for name, f in frames.items() if group_key("καλό σχόλιο 7") in set(f["group_key"])}
    assert len(homes) == 1
    assert split_overlap(frames) == {
        "train_validation": 0, "train_test": 0, "validation_test": 0
    }


def test_duplicate_mode_keep_leaks_where_group_does_not():
    """Same corpus, same seed, one setting apart — this is the A/B that shows
    grouping is doing real work. Seed 1 is pinned because it puts the
    duplicate group across train and test under `keep`."""
    frame = build_frame(_corpus(), _config())

    leaky = assign_splits(frame, _config(duplicate_mode="keep", seed=1))
    assert split_overlap(leaky)["train_test"] == 1

    grouped = assign_splits(frame, _config(duplicate_mode="group", seed=1))
    assert sum(split_overlap(grouped).values()) == 0


def test_overlap_policy_raise_warn_and_ignore():
    frames = assign_splits(build_frame(_corpus(), _config()), _config(duplicate_mode="keep", seed=1))
    assert sum(split_overlap(frames).values()) == 1  # fixture precondition

    with pytest.raises(ValueError, match="cross-split text overlap"):
        enforce_overlap_policy(frames, "raise")
    with pytest.warns(UserWarning, match="cross-split text overlap"):
        enforce_overlap_policy(frames, "warn")
    assert sum(enforce_overlap_policy(frames, "ignore").values()) == 1


def test_clean_splits_pass_every_policy():
    config = _config()
    frames = assign_splits(build_frame(_corpus(), config), config)
    for policy in ("raise", "warn", "ignore"):
        assert sum(enforce_overlap_policy(frames, policy).values()) == 0


def test_assign_splits_refuses_a_corpus_too_small_to_divide():
    config = _config()
    tiny = build_frame(pd.DataFrame({"text": ["καλό", "κακό"], "label": [1, 0]}), config)
    with pytest.raises(ValueError, match="cannot fill"):
        assign_splits(tiny, config)
