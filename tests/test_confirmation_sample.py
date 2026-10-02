"""The confirmation sample (docs/annotation/confirmation_protocol.md), on data whose answers are known."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, binomtest

from reviewnlp.evaluation.metrics import wilson_interval

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


conf = _load("confirmation_sample")
pilot, score = conf.pilot, conf.score
trends = pilot.trends

# One lexicon topic each for the first five, then another topic, then two texts that are not complaints.
TEXTS = ["the shower was broken and the bathroom tiny", "the room was dirty and dusty",
         "the air conditioning did not work", "we found a cockroach in the room", "nobody replied to our emails",
         "the bed was too soft", "No Negative", "Nothing"]
MONTHS = {"base": [f"2016-{m:02d}" for m in range(2, 8)], "recent": [f"2017-{m:02d}" for m in range(2, 8)]}
SIZE = 20


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
    rows.append(_review("Hotel D", "D street 4 Milan Italy", "7/31/2017", "the bed was too soft", 7.0))
    rows.append(_review("Hotel A", "A street 1 Paris France", "10/15/2016", "the bed was too soft", 7.0))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def reviews(tmp_path_factory) -> Path:
    return _write_reviews(tmp_path_factory.mktemp("confirmation") / "reviews.csv")


@pytest.fixture(scope="module")
def prepared(reviews):
    frame, _ = trends.load_reviews(reviews)
    params = trends.analysis_windows(frame["review_date"], 30, pilot.SINCE)
    period = pilot.period_of(frame, params)
    hotels = pilot.compared_hotels(frame, period, 30)
    return frame, period, hotels, conf.eligible(frame, period, hotels)


@pytest.fixture(scope="module")
def pilot_key(prepared, tmp_path_factory) -> Path:
    """A stand-in for the pilot's key: 10 texts of each period, which must not be drawn again."""
    *_, pop = prepared
    rows = [row for name in pilot.PERIODS for row in pop.loc[pop["period"] == name, "raw_row"].tolist()[:10]]
    path = tmp_path_factory.mktemp("pilot_key") / "key.csv"
    pd.DataFrame({"item": range(1, len(rows) + 1), "raw_row": rows}).to_csv(path, index=False)
    return path


def _fill(sheet: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    filled = sheet.copy()
    filled["responsiveness"] = labels
    filled["done"] = "yes"
    return filled


# --- Drawing ---------------------------------------------------------------------------------------------------

def test_the_pilot_texts_are_the_300_raw_rows_of_its_committed_key():
    rows = conf.excluded_rows(conf.PILOT_KEY)
    assert len(rows) == 300
    assert rows == set(pd.read_csv(conf.PILOT_KEY)["raw_row"])


def test_draw_leaves_out_the_pilot_texts_and_follows_the_design(prepared, pilot_key):
    *_, pop = prepared
    excluded = conf.excluded_rows(pilot_key)
    key, sizes = conf.draw(pop, excluded, seed=1, size=SIZE)
    assert len(key) == 2 * SIZE and key["review_id"].is_unique
    assert key["item"].tolist() == list(range(1, len(key) + 1))
    assert key["period"].value_counts().to_dict() == {"base": SIZE, "recent": SIZE}
    assert not set(pop.loc[key["review_id"], "raw_row"]) & excluded
    for name in pilot.PERIODS:
        assert sizes[name] == {"population": 108, "excluded_pilot_texts": 10, "drawn": SIZE}
    again, _ = conf.draw(pop, excluded, seed=1, size=SIZE)
    pd.testing.assert_frame_equal(key, again)
    other, _ = conf.draw(pop, excluded, seed=0, size=SIZE)
    assert set(other["review_id"]) != set(key["review_id"])
    assert key["period"].tolist() != sorted(key["period"])  # items come in a random order, not period by period


def test_draw_refuses_a_period_with_too_few_texts_left(prepared, pilot_key):
    *_, pop = prepared
    with pytest.raises(ValueError, match="fewer than 1000"):
        conf.draw(pop, conf.excluded_rows(pilot_key), size=1000)


def test_the_key_has_the_pilots_format_and_the_lexicons_verdicts(prepared, pilot_key):
    frame, _, _, pop = prepared
    key, _ = conf.draw(pop, conf.excluded_rows(pilot_key), seed=1, size=SIZE)
    key = conf.with_lexicon(key, frame, pop)
    assert key.columns.tolist() == conf.KEY_COLUMNS
    assert (key["in_R"] == 1).all() and (key["drawn_for"] == "").all() and (key["in_B"] == 0).all()
    for _, row in key.iterrows():
        text = pop.loc[row["review_id"], "text"]
        named = {topic for topic, _ in trends.complaint_topics(text)}
        for column, topic in pilot.TOPICS.items():
            assert row[f"lex_{column}"] == int(topic in named), (text, column)
        assert row["text_sha256"] == conf.text_hash(text)
        assert row["raw_row"] == pop.loc[row["review_id"], "raw_row"]
    assert key[[f"lex_{column}" for column in pilot.TOPICS]].sum().sum() > 0


def test_the_sheet_hides_the_answer(prepared, pilot_key):
    frame, _, _, pop = prepared
    key, _ = conf.draw(pop, conf.excluded_rows(pilot_key), seed=1, size=SIZE)
    table = conf.sheet(key, pop)
    assert table.columns.tolist() == ["item", "text", "responsiveness", "done", "note"]
    assert (table[["responsiveness", "done", "note"]] == "").all().all()
    assert table["item"].tolist() == key["item"].tolist()
    assert table["text"].tolist() == pop.loc[key["review_id"], "text"].tolist()


def test_text_hash_ignores_white_space_only():
    assert conf.text_hash("a  b\r\nc") == conf.text_hash("a b c") == conf.text_hash(" a b\nc ")
    assert conf.text_hash("a b c") != conf.text_hash("a b d")


# --- The files ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def drawn(reviews, pilot_key, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("drawn")
    conf.build_sample(reviews, out, seed=1, expect_hotels=3, exclude_key=pilot_key, size=SIZE)
    return out


def test_the_manifest_records_what_the_draw_rests_on(drawn, reviews, pilot_key):
    manifest = json.loads((drawn / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "docs/annotation/confirmation_protocol.md"
    assert manifest["protocol_sha256"] == hashlib.sha256(conf.PROTOCOL.read_bytes()).hexdigest()
    assert manifest["guideline_sha256"] == hashlib.sha256(conf.GUIDELINE.read_bytes()).hexdigest()
    assert manifest["raw_csv_sha256"] == hashlib.sha256(reviews.read_bytes()).hexdigest()
    assert (manifest["seed"], manifest["hotels_compared"], manifest["items"]) == (1, 3, 2 * SIZE)
    assert manifest["design"]["topic"] == "responsiveness" and manifest["design"]["texts_per_period"] == SIZE
    left_out = manifest["design"]["left_out"]
    assert left_out["texts"] == 20
    assert left_out["raw_rows_sha256"] == hashlib.sha256(
        ",".join(map(str, sorted(conf.excluded_rows(pilot_key)))).encode()).hexdigest()
    for name in pilot.PERIODS:
        assert manifest["periods"][name] == {"population": 108, "excluded_pilot_texts": 10, "drawn": SIZE,
                                             "placeholder_or_empty": 18, "excluded_by_filter": 18}
    assert manifest["windows"]["base_start"] == "2016-02" and manifest["windows"]["recent_end"] == "2017-07"
    assert manifest["sheet_columns"] == conf.SHEET_COLUMNS


def test_a_drawn_sample_is_not_redrawn_by_accident(drawn, reviews, pilot_key):
    with pytest.raises(SystemExit, match="never redrawn by accident"):
        conf.build_sample(reviews, drawn, seed=1, exclude_key=pilot_key, size=SIZE)
    again = conf.build_sample(reviews, drawn, seed=1, exclude_key=pilot_key, size=SIZE, overwrite=True)
    assert again["items"] == 2 * SIZE


def test_a_wrong_number_of_hotels_stops_the_draw(reviews, pilot_key, tmp_path):
    with pytest.raises(SystemExit, match="3 hotels compared, expected 863"):
        conf.build_sample(reviews, tmp_path, expect_hotels=863, exclude_key=pilot_key, size=SIZE)
    assert not (tmp_path / "sheet.csv").exists()


def test_a_pilot_text_missing_from_the_population_stops_the_draw(reviews, pilot_key, tmp_path):
    rows = pd.read_csv(pilot_key)
    rows.loc[0, "raw_row"] = 10**9  # not a review of this data
    other = tmp_path / "other_key.csv"
    rows.to_csv(other, index=False)
    with pytest.raises(SystemExit, match="only 19 of the 20 pilot texts are in the population"):
        conf.build_sample(reviews, tmp_path / "out", exclude_key=other, size=SIZE)
    assert not (tmp_path / "out" / "sheet.csv").exists()


def test_a_key_that_holds_text_is_refused_as_the_exclusion(tmp_path):
    path = tmp_path / "key.csv"
    pd.DataFrame({"item": [1], "raw_row": [5], "text": ["a review"]}).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="must not hold review text"):
        conf.excluded_rows(path)


# --- The sheet that comes back -----------------------------------------------------------------------------------

def _returned(drawn: Path, tmp_path: Path, labels: list[str] | None = None) -> tuple[Path, pd.DataFrame]:
    sheet = pd.read_csv(drawn / "sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rng = np.random.default_rng(0)
    labels = labels or rng.choice(["1", "0", "0", "0", "unsure"], size=len(sheet)).tolist()
    filled = _fill(sheet, labels).assign(note="private")
    path = tmp_path / "filled.csv"
    filled.to_csv(path, index=False, encoding="utf-8-sig")
    return path, filled


def test_finish_records_a_valid_sheet_without_text_or_notes(drawn, tmp_path):
    path, filled = _returned(drawn, tmp_path)
    out = tmp_path / "committed"
    record = conf.finish(path, drawn / "key.csv", drawn / "manifest.json", out)
    assert sorted(file.name for file in out.iterdir()) == ["key.csv", "labels.csv", "sample_manifest.json",
                                                           "sheet_record.json"]
    labels = pd.read_csv(out / "labels.csv", dtype=str, keep_default_na=False)
    assert labels.columns.tolist() == ["item", "annotator", "responsiveness"]
    assert (labels["annotator"] == "final").all()
    assert labels["responsiveness"].tolist() == filled["responsiveness"].tolist()
    counts = filled["responsiveness"].value_counts().to_dict()
    assert record["labels"] == {value: counts.get(value, 0) for value in ("1", "0", "unsure")}
    assert record["sheet_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["items"] == 2 * SIZE and record["topic"] == "responsiveness"
    manifest = json.loads((drawn / "manifest.json").read_text(encoding="utf-8"))
    assert (record["protocol_sha256"], record["sample_code_commit"]) == (manifest["protocol_sha256"],
                                                                         manifest["code_commit"])
    assert (out / "key.csv").read_bytes() == (drawn / "key.csv").read_bytes()
    sheet = pd.read_csv(drawn / "sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    committed = "\n".join(file.read_text(encoding="utf-8") for file in out.iterdir())
    assert "private" not in committed
    assert not any(text in committed for text in set(sheet["text"]))


@pytest.mark.parametrize("damage, message", [
    ("blank", "responsiveness is ''"),
    ("unknown", "responsiveness is 'maybe'"),
    ("not done", "done is ''"),
    ("repeated", "appears twice"),
    ("unknown item", "is not in this sheet's sample"),
    ("missing", "is missing"),
    ("changed text", "is not the one drawn"),
    ("no text column", "missing column 'text'"),
])
def test_finish_refuses_a_sheet_that_is_not_valid_and_writes_nothing(drawn, tmp_path, damage, message):
    _, filled = _returned(drawn, tmp_path, labels=["1"] * (2 * SIZE))
    if damage == "blank":
        filled.loc[3, "responsiveness"] = ""
    elif damage == "unknown":
        filled.loc[3, "responsiveness"] = "maybe"
    elif damage == "not done":
        filled.loc[3, "done"] = ""
    elif damage == "repeated":
        filled = pd.concat([filled, filled.iloc[[0]]])
    elif damage == "unknown item":
        filled.loc[3, "item"] = "999"
    elif damage == "missing":
        filled = filled.iloc[1:]
    elif damage == "changed text":
        filled.loc[3, "text"] = filled.loc[3, "text"] + " and more"
    elif damage == "no text column":
        filled = filled.drop(columns="text")
    path = tmp_path / "damaged.csv"
    filled.to_csv(path, index=False, encoding="utf-8-sig")
    out = tmp_path / "committed"
    with pytest.raises(score.SubmissionError, match=message):
        conf.finish(path, drawn / "key.csv", drawn / "manifest.json", out)
    assert not out.exists()


def test_finish_accepts_a_spreadsheets_line_ends_and_spacing(drawn, tmp_path):
    _, filled = _returned(drawn, tmp_path, labels=[" 1 ", "0", "Unsure", *["0"] * (2 * SIZE - 3)])
    filled["text"] = filled["text"].map(lambda text: text.replace(" ", "  ", 1) + "\r\n")
    path = tmp_path / "spreadsheet.csv"
    filled.to_csv(path, index=False, encoding="utf-8-sig")
    record = conf.finish(path, drawn / "key.csv", drawn / "manifest.json", tmp_path / "committed")
    assert record["labels"]["unsure"] == 1 and record["labels"]["1"] == 1


# --- The protocol's figures ---------------------------------------------------------------------------------------

@functools.cache
def _jev_finds_more(b: int, c: int) -> bool:
    return b > c and binomtest(min(b, c), b + c, 0.5).pvalue < 0.05


def _recall_passes(recall_jev: float, recall_lexicon: float = 0.125, rate: float = 0.08, texts: int = 400) -> float:
    """Chance that the recall part of the rule passes, if Jev and the lexicon find complaints independently."""
    only_jev, only_lexicon = recall_jev * (1 - recall_lexicon), recall_lexicon * (1 - recall_jev)
    discordant, share = only_jev + only_lexicon, only_jev / (only_jev + only_lexicon)
    discordant_counts = np.arange(1, 61)
    # chance that the discordant texts split so that Jev finds more, for each number of discordant texts
    passing = np.array([sum(binom.pmf(b, m, share) for b in range(m + 1) if _jev_finds_more(b, m - b))
                        for m in discordant_counts])
    complaints = np.arange(0, 101)
    return float(sum(weight * binom.pmf(discordant_counts, n, discordant) @ passing
                     for n, weight in zip(complaints, binom.pmf(complaints, texts, rate), strict=True)))


def test_the_figures_the_protocol_quotes():
    protocol = (ROOT / "docs" / "annotation" / "confirmation_protocol.md").read_text(encoding="utf-8")
    low, high = wilson_interval(16, 200)
    assert (round(400 * low), round(400 * high)) == (20, 50) and "between 20 and 50 are plausible" in protocol
    assert [round(x, 2) for x in wilson_interval(26, 32)] == [0.65, 0.91]
    assert binomtest(0, 6, 0.5).pvalue < 0.05 <= binomtest(0, 5, 0.5).pvalue  # with c = 0, b of 6 is needed
    assert _recall_passes(0.65) >= 0.98 and _recall_passes(0.8) >= 0.99
    assert 0.84 <= _recall_passes(0.5) <= 0.86 and 0.58 <= _recall_passes(0.4) <= 0.62
    # The guard: 46 flags, and the estimate must be 0.5 or more, so 23 right or more.
    assert round(float(binom.cdf(22, 46, 0.57)), 2) == 0.13 and round(float(binom.cdf(22, 46, 0.50)), 2) == 0.44
