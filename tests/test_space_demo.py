"""The hosted demo must quote real review text, name the right topic and never hide a complaint."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from reviewnlp.analytics.recommendations import RECOMMENDATION_TEXT

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "spaces" / "hotel-ops-demo"
USER_REVIEWS = json.loads((ROOT / "data" / "eval" / "space_triage_user6.json").read_text(encoding="utf-8"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"hotel_demo_{name}", SPACE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


triage = _load("triage")
logic = _load("logic")

COMPLAINT_WORDS = ("cold", "dirty", "disconnect", "rude", "tiny", "noisy", "awful", "expensive", "confusing",
                   "nobody", "no lift", "not help", "no hangers", "smell", "small")


def keyword_scorer(pairs):
    """Stand-in for the model: negative when the clause holds a complaint word."""
    return [("negative" if any(word in clause.lower() for word in COMPLAINT_WORDS) else "positive", 0.9)
            for clause, _ in pairs]


def topics(clause: str) -> list[str]:
    return [topic for topic, _ in triage.clause_topics(clause)]


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
    # A line break ends a sentence even without a full stop.
    assert triage.split_clauses("Parking is 7 min walk also fine\nThe biggest issue was the check-in") == [
        "Parking is 7 min walk also fine", "The biggest issue was the check-in"]


def test_a_clause_that_names_nothing_stays_with_what_it_is_about():
    assert triage.split_clauses("Airconds provided but all not cold.") == ["Airconds provided but all not cold."]
    assert triage.split_clauses("The staff were lovely, but the breakfast was cold.") == [
        "The staff were lovely", "the breakfast was cold."]


def test_demo_examples_name_the_right_topics():
    found = [(clause, topics(clause)) for clause in triage.split_clauses(
        "The room was spotless and the staff were kind, but breakfast was cold.")]
    assert found == [("The room was spotless", ["cleanliness.general"]),
                     ("the staff were kind", ["staff.people"]),
                     ("breakfast was cold.", ["food.breakfast"])]
    assert [triage.clause_topics(c) for c in triage.split_clauses(
        "The Wi-Fi kept disconnecting. We loved the sea view, but the bathroom was dirty.")] == [
        [("facilities.wifi", ["Wi-Fi"])], [("room.view", ["view"])], [("cleanliness.general", ["dirty"])]]


@pytest.mark.parametrize(("clause", "expected"), [
    # Places, not opinions.
    ("However, after requesting a late check-out we found ourselves locked out of our room.", ["staff.checkin"]),
    ("I got the room at 3rd floor without lift with baggages.", ["facilities.access"]),
    ("there was not a single place to put your cloths, neither in the rooms nor in the bathroom.",
     ["room.storage"]),
    ("The location was fabulous, with central park and loads of restaurants.", ["location.general"]),
    ("i had a chance to check out the spa which i loved", ["facilities.leisure"]),
    ("Staineless steel washing hand basin in bathroom is a bit tacky", ["room.bathroom"]),
    ("Since it’s located in an old building, there’s a strong and unpleasant odor as soon as you open the door",
     ["cleanliness.odour"]),
    # What an opinion word describes.
    ("Airconds should be well service before you sell your rooms.", ["room.climate"]),
    ("The apartment itself was clean, quiet, comfortable, and very welcoming.",
     ["cleanliness.general", "room.comfort", "noise.general"]),
    ("Very helpful host with all of the instructions and recommendations.", ["staff.people"]),
    ("the sounds of walking or moving furniture were extremely loud", ["noise.general"]),
    ("the noise from the bars below was really bad", ["noise.general"]),
    ("For me this is just very poor and cheap, normal you send your passport", ["staff.checkin"]),
    ("The bed was too small, we had a baby, there was no cod", ["room.bed", "facilities.family"]),
    ("140cm bed meaning even without a baby it’s actually small for two adults", ["room.bed"]),
    ("Great for families with small kids.", ["facilities.family"]),
    ("the shower area was very small.", ["room.bathroom"]),
    ("Below you see the hair in my towel", ["cleanliness.linen"]),
    ("The room was very clean, not much of a view.", ["cleanliness.general", "room.view"]),
    ("the self check-in process was straightforward, although we never met the host in person.",
     ["staff.checkin"]),
    ("Parking was expensive.", ["value.charges", "facilities.parking"]),
    ("A small pool and a tiny breakfast room", ["food.breakfast", "facilities.leisure"]),
])
def test_each_word_counts_for_the_topic_it_is_about(clause, expected):
    assert topics(clause) == expected


@pytest.mark.parametrize("number", range(1, 7))
def test_the_owners_reviews_name_every_topic_they_complain_or_praise(number):
    review = USER_REVIEWS["reviews"][number - 1]
    text = triage.normalize(review["text"])
    found = {topic for start, end in triage.clause_spans(text) for topic in topics(text[start:end])}
    found |= {mention["topic"] for mention in triage.resolution_mentions(text)}
    assert {topic for topic, _ in review["must"]} <= found


def test_suggestions_are_complaints_and_advice_to_guests_is_not():
    always_positive = lambda pairs: [("positive", 0.95)] * len(pairs)  # noqa: E731
    [[mention]] = triage.analyze(["Airconds should be well service before you sell your rooms."], always_positive)
    assert (mention["topic"], mention["sentiment"], mention["flags"]) == ("room.climate", "negative", ["suggestion"])
    [mentions] = triage.analyze(["You should definitely stay here, the location is great."], always_positive)
    assert [(m["topic"], m["sentiment"], m["flags"]) for m in mentions] == [("location.general", "positive", [])]


def test_complaints_the_model_reads_the_wrong_way_round_are_corrected():
    always_positive = lambda pairs: [("positive", 0.95)] * len(pairs)  # noqa: E731

    def found(text):
        [mentions] = triage.analyze([text], always_positive)
        return {(m["topic"], m["sentiment"]) for m in mentions}

    assert found("Airconds provided but all not cold.") == {("room.climate", "negative")}
    assert found("The shower was lukewarm at best.") == {("room.bathroom", "negative")}
    assert found("the host is saving on hangers.") == {("staff.people", "negative"), ("room.storage", "negative")}
    assert found("I woke up with bed bug bites.") == {("cleanliness.pests", "negative")}
    assert found("No bed bugs, no stains.") == {("cleanliness.pests", "positive"), ("cleanliness.general", "positive")}
    # Something missing that a guest expects to find is a complaint; "no problem with" is not.
    assert found("There was no kettle and no hangers.") == {("room.equipment", "negative"),
                                                           ("room.storage", "negative")}
    assert found("Breakfast was pastries only, no eggs or freshly prepared food.") >= {("food.dining", "negative")}
    assert found("No problems at all with the wifi.") == {("facilities.wifi", "positive")}


def test_the_model_reads_the_thing_not_the_opinion_word():
    seen = []

    def scorer(pairs):
        seen.extend(pairs)
        return [("positive", 0.9)] * len(pairs)

    triage.analyze(["Small fridge in room was nice.", "The apartment was comfortable."], scorer)
    assert [term for _, term in seen] == ["fridge", "apartment", "comfortable"]


def test_a_reported_problem_that_was_not_solved_is_a_finding_of_its_own():
    text = ("Even after I reported the issue, and Francesco kindly came with tools to clean, "
            "it didn’t make any real difference.")
    [mention] = triage.resolution_mentions(text)
    assert (mention["topic"], mention["sentiment"], mention["flags"]) == ("staff.resolution", "negative",
                                                                           ["unresolved"])
    assert mention["quote"] == text
    [solved] = triage.resolution_mentions("Whenever I had an issue, he responded quickly and took care of it.")
    assert (solved["topic"], solved["sentiment"]) == ("staff.resolution", "positive")
    assert triage.resolution_mentions("The AC didn't work, so we asked for a fan.") == []


def test_one_clause_can_praise_one_topic_and_criticise_another():
    def scorer(pairs):
        return [("negative" if term == "view" else "positive", 0.9) for _, term in pairs]

    [mentions] = triage.analyze(["The room was very clean, not much of a view."], scorer)
    assert {(m["topic"], m["sentiment"], m["term"]) for m in mentions} == {
        ("cleanliness.general", "positive", "clean"), ("room.view", "negative", "view")}
    assert {m["quote"] for m in mentions} == {"The room was very clean, not much of a view."}


def test_neutral_answers_and_empty_reviews_produce_no_findings():
    assert triage.analyze(["The staff were there.", ""], lambda pairs: [(None, 0.9)] * len(pairs)) == [[], []]
    assert triage.analyze([], keyword_scorer) == []


def test_unsure_answers_and_typos_are_marked_for_checking():
    [[mention]] = triage.analyze(["The breakfast was fine."], lambda pairs: [("negative", 0.55)] * len(pairs))
    assert mention["flags"] == ["check"]
    [mentions] = triage.analyze(["We had a baby and there was no cod."], keyword_scorer)
    cot = next(m for m in mentions if m["sentiment"] == "negative")
    assert (cot["topic"], cot["term"], cot["flags"]) == ("facilities.family", "cod", ["check", "typo"])


def test_looks_english_rejects_other_languages():
    assert triage.looks_english("The room was spotless and the staff were kind.")
    assert triage.looks_english("Great location")
    assert not triage.looks_english("Το δωμάτιο ήταν πεντακάθαρο.")
    assert not triage.looks_english("Das Zimmer war sauber und das Personal sehr freundlich.")


def test_a_review_counts_once_per_topic_with_every_quote_kept():
    reviews = [
        "The bed was too small. The bed was small for two adults. Great location.",
        "The bed was comfortable but tiny. We reported the smell and the owner came, but it did not help.",
        "Lovely location.",
    ]
    findings = logic.group(triage.analyze(reviews, keyword_scorer))
    beds = [f for f in findings if f["topic"] == "room.bed"]
    assert [(f["review"], f["sentiment"], len(f["quotes"])) for f in beds] == [(1, "negative", 2), (2, "negative", 1)]
    summary = logic.summarize(findings)
    fix_first = logic.fix_first_rows(summary, analysed=3)
    assert fix_first[0][:4] == ["Δωμάτιο › Κρεβάτι", 2, "67%", 3]  # two reviews, three quotes
    assert fix_first[0][6] == logic.recommendation("room.bed")
    resolution = next(row for row in fix_first if row[0] == "Εξυπηρέτηση › Επίλυση προβλημάτων")
    assert "ανεπίλυτο: 1" in resolution[4]
    strengths = logic.strength_rows(summary, analysed=3)
    assert strengths[0][:3] == ["Τοποθεσία", 2, "67%"]
    rows = logic.finding_rows(findings)
    assert rows[0][2] == "Αρνητικό"  # complaints first
    text = logic.summary_markdown(3, set(), summary)
    assert "Συχνότερο παράπονο: **Δωμάτιο › Κρεβάτι** σε 2 από 3 κριτικές." in text
    assert "**Πρόβλημα που αναφέρθηκε και δεν λύθηκε:** κριτική #2." in text
    assert "κάθε κριτική μετρά μία φορά ανά θέμα" in text


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


def test_csv_export_has_one_row_per_review_and_topic():
    reviews = ["The room was spotless and the staff were kind, but breakfast was cold. Breakfast was cold again."]
    findings = logic.group(triage.analyze(reviews, keyword_scorer))
    with open(logic.write_csv(reviews, findings), encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["review", "category", "topic", "sentiment", "mentions", "notes", "quotes", "review_text"]
    assert [row[1:7] for row in rows[1:]] == [
        ["Καθαριότητα", "Καθαριότητα › Γενική", "Θετικό", "1", "", "The room was spotless"],
        ["Εξυπηρέτηση", "Εξυπηρέτηση › Προσωπικό / οικοδεσπότης", "Θετικό", "1", "", "the staff were kind"],
        ["Φαγητό", "Φαγητό › Πρωινό", "Αρνητικό", "2", "", "breakfast was cold. | Breakfast was cold again."],
    ]
    assert rows[1][7] == reviews[0]


def test_space_texts_match_the_package_and_cover_every_topic():
    assert logic.RECOMMENDATIONS == RECOMMENDATION_TEXT
    assert set(logic.ASPECT_NAMES) == set(triage.ASPECTS) == set(RECOMMENDATION_TEXT)
    assert list(logic.TOPIC_NAMES) == list(triage.TOPICS)
    assert {topic.split(".")[0] for topic in triage.TOPICS} == set(triage.ASPECTS)
    assert set(triage.NOUNS) | set(triage.OPINIONS) | {"staff.resolution"} == set(triage.TOPICS)
    assert all(logic.recommendation(topic) for topic in triage.TOPICS)


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
    assert "**Πρόβλημα που αναφέρθηκε και δεν λύθηκε:** κριτική #5." in summary
    assert per_review[-1][3] == "Δεν αναλύθηκε: δεν φαίνεται αγγλική"
    assert fix_first and strengths and Path(csv_path).is_file()
    climate = next(row for row in findings if row[1] == "Δωμάτιο › Κλιματισμός & αερισμός" and row[0] == 5)
    assert climate[2:5] == ["Αρνητικό", 1, "πρόταση βελτίωσης"]
    with pytest.raises(gr.Error):
        app.analyze("", None)
