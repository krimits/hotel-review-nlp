"""Offline tests for the ABSA prototype — pure Python, no model and no network.

Everything the model itself produces is fed in here as a literal string, so the
parser, the taxonomy and the vote are pinned without a download. The generation
loop in absa/pipeline.py is deliberately not covered: it cannot run without
fetching Qwen2.5-0.5B-Instruct, which CI has no business doing.

Several cases below are the v9 behaviour changes stated in the module
docstrings, written as known-answer tests so a regression to v8 semantics
(substring aspect fallback, duplicate entries flipping the vote) fails here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from reviewnlp.absa.aspects import (
    ASPECT_ALIASES,
    ASPECTS,
    SENTIMENTS,
    normalize_aspect,
    normalize_sentiment,
)
from reviewnlp.absa.extract import (
    format_absa_messages,
    overall_from_aspects,
    parse_absa_output,
    review_key,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    """scripts/run_absa.py is a script, not a package module — load it by path."""
    spec = importlib.util.spec_from_file_location("run_absa", ROOT / "scripts" / "run_absa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# taxonomy
# --------------------------------------------------------------------------


def test_every_alias_resolves_to_a_declared_aspect():
    # The taxonomy is the contract that makes reports comparable across runs:
    # an alias pointing at a name outside ASPECTS would silently widen it.
    assert set(ASPECT_ALIASES.values()) == set(ASPECTS)
    for aspect in ASPECTS:
        assert ASPECT_ALIASES[aspect] == aspect


def test_normalize_aspect_accepts_case_and_separator_noise():
    assert normalize_aspect("Cleanliness") == "cleanliness"
    assert normalize_aspect("  WIFI  ") == "facilities"
    assert normalize_aspect("breakfast") == "food"
    assert normalize_aspect("front-desk") is None  # hyphen handled, term still unknown


def test_unknown_aspect_returns_none_rather_than_a_substring_guess():
    # v8 mapped these onto whichever ASPECTS entry appeared first; v9 drops them
    # and counts them. Coercion here is worse than a dropped entry: it invents
    # evidence for an aspect the review never got scored on.
    assert normalize_aspect("food and staff") is None
    assert normalize_aspect("weather") is None
    assert normalize_aspect("") is None


def test_normalize_sentiment_is_closed_over_the_declared_labels():
    assert normalize_sentiment("POSITIVE ") == "positive"
    assert [normalize_sentiment(s) for s in SENTIMENTS] == list(SENTIMENTS)
    assert normalize_sentiment("mixed") is None
    assert normalize_sentiment("4/5") is None


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------


def test_messages_are_chat_role_dicts_with_collapsed_whitespace():
    messages = format_absa_messages("Nice   room.\n\n  Bad\tbreakfast.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "JSON array only" in messages[0]["content"]
    assert "Nice room. Bad breakfast." in messages[1]["content"]


def test_long_reviews_are_truncated_to_the_prompt_budget():
    messages = format_absa_messages("a" * 5000)
    body = messages[1]["content"]
    assert "a" * 4000 in body
    assert "a" * 4001 not in body


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_clean_array_parses_with_the_quote_checked_against_the_review():
    review = "The sheets were dirty but the staff were lovely."
    raw = (
        '[{"aspect": "cleanliness", "sentiment": "negative", "quote": "sheets were dirty"},'
        ' {"aspect": "staff", "sentiment": "positive", "quote": "staff were lovely"}]'
    )
    out = parse_absa_output(raw, review=review)

    assert out["json_valid"] is True
    assert out["salvaged"] is False
    assert (out["entries_total"], out["entries_kept"], out["entries_dropped"]) == (2, 2, 0)
    assert out["empty_valid"] is False
    assert out["quote_absent"] == out["quote_not_in_review"] == out["quote_truncated"] == 0
    assert out["aspects"] == [
        {"aspect": "cleanliness", "sentiment": "negative", "quote": "sheets were dirty"},
        {"aspect": "staff", "sentiment": "positive", "quote": "staff were lovely"},
    ]


def test_array_wrapped_in_prose_is_salvaged_and_flagged():
    raw = 'Sure! Here is the JSON:\n[{"aspect": "noise", "sentiment": "negative", "quote": "loud"}]\nHope this helps.'
    out = parse_absa_output(raw, review="It was loud all night.")

    assert out["json_valid"] is True
    assert out["salvaged"] is True  # parseable, but the model ignored "JSON only"
    assert out["entries_kept"] == 1


@pytest.mark.parametrize(
    ("raw", "expected_error"),
    [
        ("", "empty generation"),
        ("   ", "empty generation"),
        # The fine-tuned adapter's failure mode: the trained single-word loop,
        # with no JSON anywhere. This is what a parse rate of 0.0 looks like.
        ("positive\npositive\npositive\n", "no JSON array found"),
        ('{"aspect": "staff", "sentiment": "positive"}', "no JSON array found"),
        ('[{"aspect": "staff",}]', "invalid JSON"),
    ],
)
def test_unparseable_generations_report_why_and_keep_nothing(raw, expected_error):
    out = parse_absa_output(raw, review="anything")
    assert out["json_valid"] is False
    assert out["aspects"] == []
    assert out["entries_kept"] == 0
    assert out["empty_valid"] is False  # no array at all is not an empty array
    assert out["error"].startswith(expected_error)


def test_invalid_entries_are_dropped_and_counted_not_folded_into_the_parse_flag():
    raw = (
        '[{"aspect": "cleanliness", "sentiment": "negative", "quote": "dirty"},'
        ' {"aspect": "weather", "sentiment": "negative", "quote": "rain"},'
        ' {"aspect": "staff", "sentiment": "mixed", "quote": "fine"},'
        ' "staff was good"]'
    )
    out = parse_absa_output(raw, review="It was dirty, it rained, staff were fine.")

    # Valid JSON that yields one usable entry is a different outcome from
    # unparseable JSON, and the record has to be able to say which it was.
    assert out["json_valid"] is True
    assert (out["entries_total"], out["entries_kept"], out["entries_dropped"]) == (4, 1, 3)
    assert [a["aspect"] for a in out["aspects"]] == ["cleanliness"]


def test_empty_array_is_marked_valid_and_empty():
    # The model saying "nothing to extract" is a real answer, not a failure.
    out = parse_absa_output("[]", review="Stayed one night.")

    assert out["json_valid"] is True
    assert out["empty_valid"] is True
    assert (out["entries_total"], out["entries_kept"], out["entries_dropped"]) == (0, 0, 0)
    assert out["aspects"] == []
    assert out["error"] is None


def test_empty_valid_separates_a_silent_model_from_a_rejected_one():
    # Both records end with aspects == [] and json_valid True. Without
    # empty_valid the summary cannot tell "the model found nothing" from
    # "the model produced only garbage" — one is a quiet corpus, the other is
    # extraction failing, and they need opposite responses.
    silent = parse_absa_output("[]", review="Fine.")
    rejected = parse_absa_output(
        '[{"aspect": "weather", "sentiment": "negative", "quote": "rain"},'
        ' {"aspect": "staff", "sentiment": "mixed", "quote": "ok"}]',
        review="It rained and the staff were ok.",
    )

    assert silent["aspects"] == rejected["aspects"] == []
    assert silent["json_valid"] == rejected["json_valid"] is True
    assert silent["empty_valid"] is True
    assert rejected["empty_valid"] is False
    assert (rejected["entries_total"], rejected["entries_dropped"]) == (2, 2)


def test_every_return_path_carries_the_documented_keys():
    # empty_valid was documented for a release before it was implemented.
    # Callers read these keys unguarded, so a missing one is a KeyError in a
    # batch run, not a missing number.
    documented = {
        "json_valid", "salvaged", "entries_total", "entries_kept",
        "entries_dropped", "quote_absent", "quote_not_in_review",
        "quote_truncated", "empty_valid", "aspects", "error",
    }
    for raw in ("", "no json here", "[oops", "[]", '[{"aspect": "staff", "sentiment": "positive"}]'):
        assert set(parse_absa_output(raw, review="staff")) == documented, raw


def test_quote_problems_are_counted_without_rejecting_the_entry():
    review = "The room was fine."
    raw = (
        '[{"aspect": "room", "sentiment": "neutral", "quote": ""},'
        ' {"aspect": "food", "sentiment": "positive", "quote": "None"},'
        ' {"aspect": "staff", "sentiment": "positive", "quote": "best concierge in Athens"}]'
    )
    out = parse_absa_output(raw, review=review)

    assert out["entries_kept"] == 3  # sentiment is still usable evidence
    assert out["quote_absent"] == 2  # "" and the literal "None"
    assert out["quote_not_in_review"] == 1  # hallucinated span


def test_overlong_quotes_are_truncated_before_the_review_check():
    span = "b" * 250
    out = parse_absa_output(
        f'[{{"aspect": "room", "sentiment": "positive", "quote": "{span}"}}]',
        review=f"The room: {span} — lovely.",
    )
    assert out["quote_truncated"] == 1
    assert len(out["aspects"][0]["quote"]) == 200
    # Truncation must not manufacture a hallucination: the 200-char prefix is
    # still a real span of the review.
    assert out["quote_not_in_review"] == 0


def test_quote_validation_is_skipped_when_no_review_is_supplied():
    out = parse_absa_output(
        '[{"aspect": "room", "sentiment": "positive", "quote": "never said this"}]'
    )
    assert out["entries_kept"] == 1
    assert out["quote_not_in_review"] == 0


def test_review_key_matches_the_normalization_quotes_are_compared_under():
    assert review_key("  Great   STAFF\n") == "great staff"


# --------------------------------------------------------------------------
# overall vote
# --------------------------------------------------------------------------


def _aspects(*pairs) -> list[dict]:
    return [{"aspect": a, "sentiment": s, "quote": ""} for a, s in pairs]


def test_majority_over_distinct_aspects_decides_the_vote():
    assert overall_from_aspects(
        _aspects(("staff", "positive"), ("room", "positive"), ("food", "negative"))
    ) == "positive"
    assert overall_from_aspects(
        _aspects(("staff", "negative"), ("room", "negative"), ("food", "positive"))
    ) == "negative"


def test_duplicate_entries_for_one_aspect_cannot_outvote_another_aspect():
    # Three cleanliness entries and one staff entry is 1 vs 1, not 3 vs 1 —
    # the model repeating itself is not extra evidence.
    votes = _aspects(
        ("cleanliness", "positive"),
        ("cleanliness", "positive"),
        ("cleanliness", "positive"),
        ("staff", "negative"),
    )
    assert overall_from_aspects(votes) is None


@pytest.mark.parametrize(
    "aspects",
    [
        pytest.param([], id="nothing-extracted"),
        pytest.param(_aspects(("room", "positive"), ("food", "negative")), id="mixed-review"),
        pytest.param(_aspects(("room", "neutral"), ("food", "neutral")), id="all-neutral"),
        pytest.param(_aspects(("staff", "positive"), ("staff", "negative")), id="split-aspect"),
    ],
)
def test_the_vote_abstains_instead_of_breaking_a_tie(aspects):
    # This abstention is the declared limitation behind the headline number:
    # a genuinely mixed review gets no overall label rather than a coin flip.
    assert overall_from_aspects(aspects) is None


# --------------------------------------------------------------------------
# runner helpers
# --------------------------------------------------------------------------


def test_gold_labels_map_strictly_and_fail_loudly():
    runner = _load_runner()
    assert runner.map_gold(1) == runner.map_gold(1.0) == runner.map_gold("1") == "positive"
    assert runner.map_gold(0) == runner.map_gold("0.0") == runner.map_gold("negative") == "negative"
    for bad in ("maybe", 2.0, -1):
        with pytest.raises(ValueError):
            runner.map_gold(bad)


def test_summary_counts_and_rates_use_the_record_count_as_denominator():
    runner = _load_runner()
    records = [
        {"json_valid": True, "salvaged": True, "entries_dropped": 0},
        {"json_valid": True, "salvaged": False, "entries_dropped": 2},
        {"json_valid": False, "salvaged": False, "entries_dropped": 0},
        {"json_valid": True, "salvaged": False, "entries_dropped": 1},
    ]
    fields = runner.parse_summary_fields(
        records, ("json_valid", "salvaged", "entries_dropped"), len(records)
    )
    assert fields["json_valid"] == 3
    assert fields["salvaged"] == 1
    assert fields["entries_dropped"] == 2  # records with drops, not entries dropped
    assert fields["json_valid_rate"] == 0.75  # over all records, not over parsed ones
    assert fields["salvaged_rate"] == 0.25


def test_code_sha256_changes_with_the_code_it_stamps(tmp_path, monkeypatch):
    runner = _load_runner()
    code_dir = tmp_path / "absa"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "ABSA_CODE_DIR", code_dir)

    first = runner.code_sha256()
    assert first == runner.code_sha256()  # deterministic, or results cannot be matched

    (code_dir / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert runner.code_sha256() != first

    (code_dir / "a.py").rename(code_dir / "b.py")  # path is hashed too
    renamed = runner.code_sha256()
    (code_dir / "b.py").rename(code_dir / "a.py")
    assert renamed != runner.code_sha256()
