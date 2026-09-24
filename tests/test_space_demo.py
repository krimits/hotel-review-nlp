"""The hosted demo must quote real review text and never hide a complaint."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from reviewnlp.analytics.recommendations import RECOMMENDATION_TEXT

SPACE = Path(__file__).resolve().parents[1] / "spaces" / "hotel-ops-demo"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"hotel_demo_{name}", SPACE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


triage = _load("triage")
logic = _load("logic")

COMPLAINT_WORDS = ("cold", "dirty", "disconnect", "rude", "tiny", "noisy", "awful", "expensive")


def keyword_scorer(pairs):
    """Stand-in for the model: negative when the clause holds a complaint word."""
    return ["negative" if any(word in clause.lower() for word in COMPLAINT_WORDS) else "positive"
            for clause, _ in pairs]


def test_clauses_are_verbatim_and_keep_short_fragments():
    text = ("Quite a spectacular hotel from the outside.Room was clean and tidy ;-)Hotel staff were "
            "always helpfulMost impressive is the location. Hudson Cafeteria... closed. Horrible.")
    clauses = triage.split_clauses(text)
    assert clauses == [
        "Quite a spectacular hotel from the outside.",
        "Room was clean and tidy ;-)",
        "Hotel staff were always helpful",
        "Most impressive is the location.",
        "Hudson Cafeteria... closed. Horrible.",
    ]
    assert all(clause in triage.normalize(text) for clause in clauses)
    assert triage.split_clauses("Tom &amp; Jerry loved the pool.") == ["Tom & Jerry loved the pool."]


def test_demo_examples_name_the_right_aspects():
    found = [(clause, [aspect for aspect, _ in triage.lexicon_matches(clause)])
             for clause in triage.split_clauses(
                 "The room was spotless and the staff were kind, but breakfast was cold.")]
    assert found == [("The room was spotless", ["cleanliness"]),
                     ("the staff were kind", ["staff"]),
                     ("breakfast was cold.", ["food"])]
    assert [triage.lexicon_matches(c) for c in triage.split_clauses(
        "The Wi-Fi kept disconnecting. We loved the sea view, but the bathroom was dirty.")] == [
        [("facilities", ["Wi-Fi"])], [("room", ["view"])], [("cleanliness", ["dirty"])]]


@pytest.mark.parametrize(("clause", "aspects"), [
    ("However, after requesting a late check-out we found ourselves locked out of our room.",
     ["staff"]),
    ("The location was fabulous, with central park and loads of restaurants.", ["location"]),
    ("the noise from the bars below was really bad", ["noise"]),
    ("i had a chance to check out the spa which i loved", ["facilities"]),
    ("Staineless steel washing hand basin in bathroom is a bit tacky", ["room"]),
    ("The room was very clean, not much of a view.", ["cleanliness", "room"]),
])
def test_words_that_only_name_a_place_are_not_findings(clause, aspects):
    assert [aspect for aspect, _ in triage.lexicon_matches(clause)] == aspects


def test_one_clause_can_praise_and_criticise_the_same_aspect():
    def scorer(pairs):
        return ["negative" if term == "view" else "positive" for _, term in pairs]

    [findings] = triage.analyze(["The room was very clean, not much of a view."], scorer)
    assert {(f["aspect"], f["sentiment"], f["term"]) for f in findings} == {
        ("cleanliness", "positive", "clean"),
        ("room", "positive", "room"),
        ("room", "negative", "view"),
    }
    assert {f["quote"] for f in findings} == {"The room was very clean, not much of a view."}


def test_neutral_answers_and_empty_reviews_produce_no_findings():
    assert triage.analyze(["The staff were there.", ""], lambda pairs: [None] * len(pairs)) == [[], []]
    assert triage.analyze([], keyword_scorer) == []


def test_looks_english_rejects_other_languages():
    assert triage.looks_english("The room was spotless and the staff were kind.")
    assert triage.looks_english("Great location")
    assert not triage.looks_english("Το δωμάτιο ήταν πεντακάθαρο.")
    assert not triage.looks_english("Das Zimmer war sauber und das Personal sehr freundlich.")


def test_summary_counts_reviews_and_keeps_mixed_aspects_as_complaints():
    def scorer(pairs):  # judges each word, like the real model
        return ["negative" if "tiny" in clause or term in ("rude", "receptionist") else "positive"
                for clause, term in pairs]

    findings = triage.analyze([
        "The room was tiny but the bed was great.",
        "Our room was tiny. The room was tiny.",
        "Lovely room and a rude receptionist.",
    ], scorer)
    summary = logic.summarize(findings)
    fix_first = logic.fix_first_rows(summary, analysed=3)
    assert fix_first[0][:3] == ["Δωμάτιο", 2, "67%"]  # one guest complaining twice counts once
    assert fix_first[0][4] == logic.RECOMMENDATIONS["room"]
    strengths = logic.strength_rows(summary, analysed=3)
    assert ["Δωμάτιο", 2, "67%"] == strengths[0][:3]  # review 1 praises and criticises the room
    rows = logic.finding_rows(findings)
    assert rows[0][2] == "Αρνητικό"  # complaints first
    assert "Συχνότερο παράπονο: **Δωμάτιο** σε 2 από 3 κριτικές." in logic.summary_markdown(
        3, set(), summary)


def test_reviews_are_read_from_paste_csv_and_txt(tmp_path):
    assert logic.split_pasted("First review here.\n\n  \nSecond\nreview.\n") == [
        "First review here.", "Second\nreview."]
    named = tmp_path / "export.csv"
    with named.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([["Hotel", "Review"], ["A", "Clean, quiet room."],
                                      ["A", ""], ["B", 'Staff said "hi", breakfast was cold.']])
    assert logic.read_upload(named) == ["Clean, quiet room.", 'Staff said "hi", breakfast was cold.']
    unnamed = tmp_path / "plain.csv"
    unnamed.write_text("5,Loved the pool and the helpful staff\n2,Dirty bathroom\n", encoding="utf-8")
    assert logic.read_upload(unnamed) == ["Loved the pool and the helpful staff", "Dirty bathroom"]
    text = tmp_path / "reviews.txt"
    text.write_text("One.\n\nTwo.", encoding="utf-8")
    assert logic.collect_reviews("Zero.", text) == ["Zero.", "One.", "Two."]


def test_input_limits_are_explained_to_the_owner():
    with pytest.raises(logic.InputError, match="τουλάχιστον μία κριτική"):
        logic.collect_reviews("  \n\n ", None)
    with pytest.raises(logic.InputError, match=f"Έως {logic.MAX_REVIEWS} κριτικές"):
        logic.collect_reviews("\n\n".join(["Nice."] * (logic.MAX_REVIEWS + 1)), None)


def test_csv_export_holds_every_finding():
    reviews = ["The room was spotless and the staff were kind, but breakfast was cold."]
    findings = triage.analyze(reviews, keyword_scorer)
    with open(logic.write_csv(reviews, findings), encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["review", "aspect", "sentiment", "word", "quote", "review_text"]
    assert [row[1:5] for row in rows[1:]] == [
        ["Καθαριότητα", "Θετικό", "spotless", "The room was spotless"],
        ["Προσωπικό", "Θετικό", "staff", "the staff were kind"],
        ["Φαγητό / πρωινό", "Αρνητικό", "breakfast", "breakfast was cold."],
    ]


def test_space_texts_match_the_package():
    assert logic.RECOMMENDATIONS == RECOMMENDATION_TEXT
    assert set(logic.ASPECT_NAMES) == set(triage.ASPECTS) == set(RECOMMENDATION_TEXT)


def test_space_card_requirements_and_model_agree():
    metadata = yaml.safe_load((SPACE / "README.md").read_text(encoding="utf-8").split("---", 2)[1])
    requirements = (SPACE / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert metadata["sdk"] == "gradio"
    assert f"gradio=={metadata['sdk_version']}" in requirements
    assert metadata["models"] == [triage.MODEL_ID]
    assert not any(line.startswith("git+") for line in requirements)


def test_app_runs_end_to_end_with_a_stub_model(monkeypatch):
    gr = pytest.importorskip("gradio")
    monkeypatch.syspath_prepend(str(SPACE))
    for name in ("app", "logic", "triage"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "score_pairs", lambda pairs, progress=None: keyword_scorer(pairs))

    summary, fix_first, strengths, findings, per_review, csv_path = app.analyze(
        app.SAMPLE_BATCH + "\n\nΤο δωμάτιο ήταν πεντακάθαρο.", None)
    assert "Αναλύθηκαν 6 από 7 κριτικές." in summary
    assert per_review[-1][3] == "Δεν αναλύθηκε: δεν φαίνεται αγγλική"
    assert fix_first and strengths and Path(csv_path).is_file()
    assert all(row[4] in triage.normalize(app.SAMPLE_BATCH) for row in findings)
    with pytest.raises(gr.Error):
        app.analyze("", None)
