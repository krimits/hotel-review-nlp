"""Set C scoring in scripts/eval_space_triage.py, on hand-made findings instead of the model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_space_triage", ROOT / "scripts" / "eval_space_triage.py")
evaluation = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = evaluation
_spec.loader.exec_module(evaluation)


def mention(topic: str, sentiment: str, quote: str = "q", flags: tuple[str, ...] = ()) -> dict:
    return {"aspect": topic.split(".")[0], "topic": topic, "sentiment": sentiment, "quote": quote,
            "flags": list(flags)}


LABELS = {
    "a": [("staff.people", "positive"), ("staff.response", "positive"), ("room.bed", "negative")],
    "b": [("noise.general", "negative")],
}
FOUND = {
    "a": [mention("staff.people", "positive", "kind host"), mention("staff.people", "positive", "helpful"),
          mention("room.climate", "negative", "too hot", flags=("check",))],
    "b": [mention("noise.general", "negative"), mention("value.price", "negative")],
}


def test_topic_level_counts_each_topic_and_sentiment_once_per_review():
    scores = evaluation.score_topics(LABELS, FOUND)
    assert scores["topic"]["all"] == [2, 4, 4]  # right, reported, labelled
    assert scores["topic"]["complaints"] == [1, 3, 2]
    assert scores["topic"]["praise"] == [1, 1, 2]


def test_category_level_accepts_another_topic_of_the_same_category():
    scores = evaluation.score_topics(LABELS, FOUND)
    assert scores["category"]["complaints"] == [2, 3, 2]  # the room complaint is right, the price one is not
    assert scores["category"]["praise"] == [1, 1, 1]


def test_tables_and_unsure_complaints():
    scores = evaluation.score_topics(LABELS, FOUND)
    assert scores["per_topic"][("room.bed", "negative")] == [0, 0, 1]
    assert scores["per_topic"][("room.climate", "negative")] == [0, 1, 0]
    assert scores["per_category"][("room", "negative")] == [1, 1, 1]
    assert scores["check"] == [0, 1]


def test_a_version_without_topics_is_scored_by_category_only():
    old = {uid: [{key: value for key, value in m.items() if key not in ("topic", "flags")} for m in mentions]
           for uid, mentions in FOUND.items()}
    scores = evaluation.score_topics(LABELS, old)
    assert "topic" not in scores
    assert scores["category"]["all"] == [3, 4, 3]


def test_reviews_labelled_with_every_topic():
    labels = {"x": [("a.one", "positive"), ("b.two", "positive")], "y": [("a.one", "negative")]}
    assert evaluation.labelled_with_every_topic(labels, ["a.one", "b.two"]) == ["x"]


def test_dump_marks_right_extra_and_missed_findings():
    texts = {"a": ("kind host, helpful", "too hot"), "b": ("", "noisy and pricey")}
    record = evaluation.dump_records(texts, LABELS, FOUND)[0]
    assert record["liked"] == "kind host, helpful"
    people, climate = record["findings"]
    assert people == {"topic": "staff.people", "sentiment": "positive", "quotes": ["kind host", "helpful"],
                      "flags": [], "right": True, "category_right": True}
    assert (climate["right"], climate["category_right"], climate["flags"]) == (False, True, ["check"])
    assert record["missed"] == [{"topic": "staff.response", "sentiment": "positive", "category_found": True},
                                {"topic": "room.bed", "sentiment": "negative", "category_found": True}]
