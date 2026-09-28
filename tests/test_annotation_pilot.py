"""The pilot's sampling and scoring (docs/annotation/pilot_protocol.md), on data whose answers are known."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sample = _load("annotation_sample")
score = _load("score_annotations")
trends = sample.trends

# One lexicon topic each for the first five, then another topic, then two texts that are not complaints.
TEXTS = ["the shower was broken and the bathroom tiny", "the room was dirty and dusty",
         "the air conditioning did not work", "we found a cockroach in the room", "nobody replied to our emails",
         "the bed was too soft", "No Negative", "Nothing"]
MONTHS = {"base": [f"2016-{m:02d}" for m in range(2, 8)], "recent": [f"2017-{m:02d}" for m in range(2, 8)]}
SMALL = {"r_size": 20, "m_sizes": dict.fromkeys(sample.TOPICS, 3), "b_from_r": 4, "b_from_m": 4}


def _review(hotel: str, address: str, date: str, text: str, score_value: float) -> dict:
    return {"Hotel_Name": hotel, "Hotel_Address": address, "Review_Date": date, "Reviewer_Nationality": " Greece ",
            "Reviewer_Score": score_value, "Negative_Review": text, "Positive_Review": "friendly staff",
            "Tags": "[' Leisure trip ', ' Couple ']"}


def _write_reviews(path: Path) -> Path:
    """Hotels A-C have 48 reviews in each period; hotel D has 10 and is not compared."""
    rows = []
    for hotel, address, count in (("Hotel A", "A street 1 Paris France", 48),
                                  ("Hotel B", "B street 2 London United Kingdom", 48),
                                  ("Hotel C", "C street 3 Vienna Austria", 48),
                                  ("Hotel D", "D street 4 Milan Italy", 10)):
        for months in MONTHS.values():
            for index in range(count):
                year, month = months[index % 6].split("-")
                date = f"{int(month)}/{index // 6 + 1}/{year}"
                rows.append(_review(hotel, address, date, TEXTS[index % len(TEXTS)], 5.0 + index % 5))
    # Close July 2017 as a full month, and add a review between the periods.
    rows.append(_review("Hotel D", "D street 4 Milan Italy", "7/31/2017", "the bed was too soft", 7.0))
    rows.append(_review("Hotel A", "A street 1 Paris France", "10/15/2016", "the bed was too soft", 7.0))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def reviews(tmp_path_factory) -> Path:
    return _write_reviews(tmp_path_factory.mktemp("pilot") / "reviews.csv")


@pytest.fixture(scope="module")
def prepared(reviews):
    frame, _ = trends.load_reviews(reviews)
    params = trends.analysis_windows(frame["review_date"], 30, sample.SINCE)
    period = sample.period_of(frame, params)
    hotels = sample.compared_hotels(frame, period, 30)
    return frame, params, period, hotels, sample.population(frame, period, hotels)


# --- Sampling -------------------------------------------------------------------------------------------------

def test_windows_are_the_second_run_months(prepared):
    _, params, *_ = prepared
    assert (params["base_start"], params["base_end"]) == ("2016-02", "2016-07")
    assert (params["recent_start"], params["recent_end"]) == ("2017-02", "2017-07")


def test_compared_hotels_match_query_03(prepared):
    frame, params, _, hotels, _ = prepared
    connection = trends.build_database(frame, trends.tag_complaints(frame), params)
    counts = pd.read_sql_query((trends.SQL_DIR / "03_hotel_counts.sql").read_text(), connection)
    addresses = dict(connection.execute("SELECT hotel_id, address FROM hotels").fetchall())
    assert hotels == sorted({addresses[hotel_id] for hotel_id in counts["hotel_id"]})
    assert len(hotels) == 3 and "D street 4 Milan Italy" not in hotels


def test_population_and_blind_spot(prepared):
    frame, _, period, hotels, pop = prepared
    for name in sample.PERIODS:
        texts = pop[pop["period"] == name]
        assert len(texts) == 3 * 36                              # six complaint texts in each cycle of eight
        assert texts[[f"lex_{c}" for c in sample.TOPICS]].sum().tolist() == [18] * 5
    assert sample.blind_spot(frame, period, hotels) == {
        name: {"placeholder_or_empty": 18, "excluded_by_filter": 18} for name in sample.PERIODS}


def test_draw_follows_the_protocol(prepared):
    *_, pop = prepared
    key, sizes = sample.draw(pop, seed=0, **SMALL)
    for name in sample.PERIODS:
        rows = key[key["period"] == name]
        in_m = rows["drawn_for"] != ""
        assert rows["in_R"].sum() == 20
        assert not (in_m & (rows["in_R"] == 1)).any()            # M is drawn outside R
        for column in sample.TOPICS:
            drawn = rows[rows["drawn_for"].str.split(";").map(lambda topics, c=column: c in topics)]
            assert len(drawn) == 3 and (drawn[f"lex_{column}"] == 1).all()
        assert rows.loc[rows["in_R"] == 1, "in_B"].sum() == 4 and rows.loc[in_m, "in_B"].sum() == 4
        assert sizes[name]["M_shortfall"] == dict.fromkeys(sample.TOPICS, 0)
    assert key["item"].tolist() == list(range(1, len(key) + 1))
    assert key["review_id"].is_unique
    again, _ = sample.draw(pop, seed=0, **SMALL)
    pd.testing.assert_frame_equal(key, again)
    other, _ = sample.draw(pop, seed=1, **SMALL)
    assert set(other["review_id"]) != set(key["review_id"])


def test_a_short_stratum_is_recorded_not_padded(prepared):
    *_, pop = prepared
    key, sizes = sample.draw(pop, seed=0, r_size=20, m_sizes={**SMALL["m_sizes"], "pests": 1000},
                             b_from_r=4, b_from_m=4)
    for name in sample.PERIODS:
        available = int(pop.loc[pop["period"] == name, "lex_pests"].sum()
                        - key[(key["period"] == name) & (key["in_R"] == 1)]["lex_pests"].sum())
        assert sizes[name]["M"]["pests"] == available
        assert sizes[name]["M_shortfall"]["pests"] == 1000 - available


def test_sheets_hide_the_answer(prepared):
    *_, pop = prepared
    key, _ = sample.draw(pop, seed=0, **SMALL)
    sheet_a, sheet_b = sample.sheets(key, pop)
    assert list(sheet_a.columns) == sample.SHEET_COLUMNS == list(sheet_b.columns)
    assert (sheet_a[sample.SHEET_COLUMNS[2:]] == "").all().all()
    assert sheet_a["text"].tolist() == pop.loc[key["review_id"], "text"].tolist()
    assert sheet_b["item"].tolist() == key.loc[key["in_B"] == 1, "item"].tolist()
    assert "text" not in key.columns


def test_build_sample_writes_the_files_and_refuses_to_redraw(reviews, tmp_path):
    manifest = sample.build_sample(reviews, tmp_path, expect_hotels=3, **SMALL)
    assert {path.name for path in tmp_path.iterdir()} == {"sheet_A.csv", "sheet_B.csv", "key.csv", "manifest.json"}
    assert manifest["protocol_sha256"] == hashlib.sha256(sample.PROTOCOL.read_bytes()).hexdigest()
    assert manifest["hotels_compared"] == 3 and manifest["items"] == len(pd.read_csv(tmp_path / "key.csv"))
    assert manifest["periods"]["base"]["excluded_by_filter"] == 18
    with pytest.raises(SystemExit, match="exists"):
        sample.build_sample(reviews, tmp_path, **SMALL)
    with pytest.raises(SystemExit, match="expected 863"):
        sample.build_sample(reviews, tmp_path / "other", expect_hotels=863, **SMALL)


# --- Scoring --------------------------------------------------------------------------------------------------

def _sheet(items, labels: dict[str, list[str]] | None = None, done: str = "yes") -> pd.DataFrame:
    frame = pd.DataFrame({"item": [str(item) for item in items], "text": "a review"})
    for topic in score.TOPICS:
        frame[topic] = (labels or {}).get(topic, ["0"] * len(items))
    frame["done"], frame["note"] = done, ""
    return frame


@pytest.mark.parametrize(("change", "message"), [
    (lambda s: s.assign(bathroom=["", *s["bathroom"][1:]]), "item 1: bathroom is ''"),
    (lambda s: s.assign(pests=["yes", *s["pests"][1:]]), "item 1: pests is 'yes'"),
    (lambda s: s.assign(done=["yes", "", "yes"]), "item 2: done is ''"),
    (lambda s: pd.concat([s, s.iloc[[0]]]), "item 1 appears twice"),
    (lambda s: s.assign(item=["1", "2", "9"]), "item 9 is not in this sheet's sample"),
    (lambda s: s.iloc[:2], "item 3 is missing"),
    (lambda s: s.drop(columns="done"), "missing columns ['done']"),
])
def test_an_invalid_submission_is_refused_whole(change, message):
    with pytest.raises(score.SubmissionError, match=message.replace("[", r"\[").replace("]", r"\]")):
        score.check_sheet(change(_sheet([1, 2, 3])), {1, 2, 3}, "sheet A")


def test_labels_are_read_case_and_space_insensitive():
    labels = score.check_sheet(_sheet([1, 2], {"bathroom": [" 1 ", "Unsure"]}, done=" YES"), {1, 2}, "sheet A")
    assert labels.loc[1, "bathroom"] == 1 and labels.loc[2, "bathroom"] == "unsure"


def test_kappa_matches_sklearn_and_follows_the_protocol():
    rng = np.random.default_rng(3)
    a = pd.Series(rng.integers(0, 2, 80))
    b = pd.Series(np.where(rng.random(80) < 0.8, a, 1 - a))
    report = score.kappa_report(a, b, seed=0, resamples=500)
    assert report["status"] == "ok"
    assert report["kappa"] == pytest.approx(cohen_kappa_score(a, b), abs=1e-4)
    assert report["bootstrap95"][0] < report["kappa"] < report["bootstrap95"][1]
    unsure = a.astype(object).copy()
    unsure[:5] = "unsure"
    assert score.kappa_report(unsure, b, resamples=50)["left_out_unsure"] == 5
    assert score.kappa_report(pd.Series([0] * 80), b)["status"] == "undefined"
    few = pd.Series([1, 1, 1] + [0] * 77)
    assert score.kappa_report(few, few)["status"] == "too few positives"


def _key(rows: list[dict]) -> pd.DataFrame:
    """A parsed key: one dict per text with period, in_R, drawn_for and the lexicon's bathroom verdict."""
    key = pd.DataFrame(rows)
    key["item"] = range(1, len(key) + 1)
    for topic in score.TOPICS:
        key[f"lex_{topic}"] = key.get(f"lex_{topic}", 0)
    key["in_B"] = 0
    return key.set_index("item", drop=False)


def _example():
    """Bathroom: base precision 12/15 and recall 9/12; recent precision 9/10 and recall 5/7."""
    rows, labels = [], []

    def add(count, period, in_r, lexicon, label, drawn_for=()):
        for _ in range(count):
            rows.append({"period": period, "in_R": in_r, "lex_bathroom": lexicon, "drawn_for": set(drawn_for)})
            labels.append(label)

    add(9, "base", 1, 1, 1)
    add(2, "base", 1, 1, 0)
    add(1, "base", 1, 1, "unsure")
    add(3, "base", 1, 0, 1)                                   # complaints the lexicon missed
    add(20, "base", 1, 0, 0)
    add(3, "base", 0, 1, 1, ["bathroom"])
    add(1, "base", 0, 1, 0, ["bathroom"])
    add(1, "base", 0, 1, 0, ["cleanliness"])                  # drawn for another topic: not in bathroom's sample
    add(5, "recent", 1, 1, 1)
    add(2, "recent", 1, 0, 1)
    add(20, "recent", 1, 0, 0)
    add(4, "recent", 0, 1, 1, ["bathroom"])
    add(1, "recent", 0, 1, 0, ["bathroom"])
    key = _key(rows)
    final = pd.DataFrame({topic: 0 for topic in score.TOPICS}, index=key.index).astype(object)
    final["bathroom"] = labels
    return key, final


def test_precision_uses_its_own_sample_and_reports_unsure_bounds():
    key, final = _example()
    result = score.precision(key, final, "bathroom")
    assert (result["base"]["numerator"], result["base"]["denominator"]) == (12, 15)
    assert result["base"]["unsure"] == 1
    assert (result["base"]["if_unsure_all_1"], result["base"]["if_unsure_all_0"]) == (round(13 / 16, 4), 0.75)
    assert result["recent"]["estimate"] == 0.9
    low, high = score.newcombe_interval(12, 15, 9, 10)
    assert result["change"]["newcombe95"] == [round(low, 4), round(high, 4)]
    assert score.precision(key, final, "pests")["base"]["status"] == "insufficient sample"


def test_recall_pools_the_periods_when_one_falls_short():
    key, final = _example()
    result = score.recall(key, final, "bathroom")
    assert result["base"]["estimate"] == 0.75
    assert result["recent"]["status"] == "insufficient sample" and result["recent"]["denominator"] == 7
    assert (result["pooled"]["numerator"], result["pooled"]["denominator"]) == (14, 19)
    assert result["change"]["status"].startswith("not reported")


def test_scoring_end_to_end_writes_no_text(reviews, tmp_path):
    drawn = tmp_path / "drawn"
    sample.build_sample(reviews, drawn, **SMALL)
    key = pd.read_csv(drawn / "key.csv")
    # Annotator A copies the lexicon; B agrees except on the first text; the adjudication keeps A.
    labels = {topic: key[f"lex_{topic}"].astype(str).tolist() for topic in score.TOPICS}
    _sheet(key["item"], labels).to_csv(drawn / "labelled_A.csv", index=False)
    in_b = key[key["in_B"] == 1]
    labels_b = {topic: in_b[f"lex_{topic}"].astype(str).tolist() for topic in score.TOPICS}
    labels_b["bathroom"][0] = "unsure"
    _sheet(in_b["item"], labels_b).to_csv(drawn / "labelled_B.csv", index=False)
    _sheet(in_b["item"], {t: in_b[f"lex_{t}"].astype(str).tolist() for t in score.TOPICS}).to_csv(
        drawn / "adjudicated.csv", index=False)
    out = tmp_path / "results"
    common = {"key": drawn / "key.csv", "sheet_a": drawn / "labelled_A.csv", "out": out, "seed": 0}
    score.run_agreement(argparse.Namespace(sheet_b=drawn / "labelled_B.csv", **common))
    score.run_metrics(argparse.Namespace(adjudicated=drawn / "adjudicated.csv", **common))

    agreement = json.loads((out / "agreement.json").read_text())
    assert agreement["topics"]["bathroom"]["left_out_unsure"] == 1
    assert agreement["sheets"]["B"]["sha256"] == hashlib.sha256((drawn / "labelled_B.csv").read_bytes()).hexdigest()
    result = json.loads((out / "metrics.json").read_text())["topics"]["bathroom"]
    assert result["precision"]["base"]["numerator"] == result["precision"]["base"]["denominator"]
    assert result["recall"]["base"]["numerator"] == result["recall"]["base"]["denominator"]
    for path in out.iterdir():
        content = path.read_text()
        assert "a review" not in content and not any(text in content for text in TEXTS[:6])
        if path.suffix == ".csv":
            assert {"text", "note"}.isdisjoint(pd.read_csv(path).columns)


def test_a_key_with_text_is_refused(tmp_path):
    path = tmp_path / "key.csv"
    pd.DataFrame({"item": [1], "text": ["a review"]}).to_csv(path, index=False)
    with pytest.raises(score.SubmissionError, match="must not hold review text"):
        score.read_key(path)

