"""The triage diagnosis draw (docs/experiments/triage_diagnosis/PROTOCOL.md), on a tiny synthetic CSV."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sample = _load("triage_diagnosis_sample")
SMALL = (("development", range(80, 90), 20261021, {"H": 3, "N": 4, "B": 2}),
         ("test", range(90, 100), 20261022, {"H": 5, "N": 6, "B": 2}))


def raw_csv(path: Path, n: int = 3000) -> Path:
    negatives = ["The room was dirty {i}", "Staff were slow at desk {i}", "No Negative"]
    positives = ["Great breakfast {i}", "No Positive", "Lovely view {i}"]
    rows = [{"Hotel_Address": f"{i % 7} Street London United Kingdom", "Review_Date": "8/3/2017",
             "Reviewer_Nationality": " UK ", "Negative_Review": negatives[i % 3].format(i=i),
             "Positive_Review": positives[i % 3].format(i=i), "Reviewer_Score": 7.5,
             "Hotel_Name": f"Hotel {i % 7}", "Tags": "[' Leisure trip ', ' Couple ']"} for i in range(n)]
    rows.append({**rows[0], "Hotel_Address": "Another 1 London United Kingdom"})  # same text as row 0
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_constants_are_the_protocols():
    prose = " ".join((ROOT / "docs/experiments/triage_diagnosis/PROTOCOL.md").read_text(encoding="utf-8").split())
    # Version 2 kept version 1's samples; only the wording changed.
    assert "Development:** 300 reviews, H 90, N 180, B 30, seed 20261021" in prose
    assert "Test:** 500 reviews, H 150, N 300, B 50, seed 20261022" in prose
    assert "0–79 teacher, 80–89 development, 90–99 test" in prose
    assert [(name, pools.start, pools.stop, seed, sizes) for name, pools, seed, sizes in sample.SAMPLES] == [
        ("development", 80, 90, 20261021, {"H": 90, "N": 180, "B": 30}),
        ("test", 90, 100, 20261022, {"H": 150, "N": 300, "B": 50})]
    strata = ", ".join(f"{name} {count:,}" for name, count in sample.EXPECTED_STRATA.items())
    assert f"**Strata:** {strata}" in prose
    assert sample.trigger_regex().search("the carpet was STAINED") and not sample.trigger_regex().search("slow")
    assert len(sample.catalogue_types()) == 20


def test_the_whole_review_leaves_out_placeholders_and_collapses_white_space():
    assert sample.whole_review("  too   noisy\n", "nice  staff") == "Disliked: too noisy\n\nLiked: nice staff"
    assert sample.whole_review("No Negative", "nice") == "Liked: nice"
    assert sample.whole_review("dirty", "No Positive") == "Disliked: dirty"
    assert sample.whole_review("", "") == ""
    assert sample.pool_of("Disliked: Dirty") == sample.pool_of("disliked:   dirty")


@pytest.fixture
def frame(tmp_path):
    csv = raw_csv(tmp_path / "raw.csv")
    loaded, _ = sample.trends.load_reviews(csv)
    positive = pd.read_csv(csv)["Positive_Review"].fillna("").astype(str)
    return loaded, positive


def test_population_strata_exclusions_and_duplicates(frame):
    loaded, positive = frame
    pop, counts = sample.population(loaded, positive, sample.trigger_regex(), excluded={1, 2})
    assert counts["strata"] == {"H": 1001, "N": 1000, "B": 1000}
    assert counts["excluded_owner_labelled"] == 2 and counts["excluded_duplicate_text"] == 1
    assert counts["eligible"] == 3001 - 3 == len(pop)
    assert 0 in set(pop["raw_row"]) and 3000 not in set(pop["raw_row"])  # the earliest copy is kept
    assert not {1, 2} & set(pop["raw_row"])
    row = pop.set_index("raw_row").loc[0]
    assert (row["stratum"], row["text"]) == ("H", "Disliked: The room was dirty 0\n\nLiked: Great breakfast 0")
    assert pop.set_index("raw_row").loc[5, "stratum"] == "B"


def test_draw_weights_sizes_and_reproducibility(frame, monkeypatch):
    monkeypatch.setattr(sample, "SAMPLES", SMALL)
    loaded, positive = frame
    pop, _ = sample.population(loaded, positive, sample.trigger_regex(), excluded=set())
    drawn, sizes = sample.draw(pop)
    again, _ = sample.draw(pop)
    assert drawn.equals(again) and drawn["raw_row"].is_unique
    for name, pools, _, wanted in SMALL:
        part = drawn[drawn["set"] == name]
        assert part["pool"].isin(pools).all()
        for stratum, n in wanted.items():
            in_pool = ((pop["pool"].isin(pools)) & (pop["stratum"] == stratum)).sum()
            rows = part[part["stratum"] == stratum]
            assert len(rows) == n and (rows["weight"] == in_pool / n).all()
            assert sizes[name]["strata"][stratum] == {"N_s": in_pool, "n_s": n, "weight": in_pool / n}
    monkeypatch.setattr(sample, "SAMPLES", (("test", range(90, 100), 1, {"H": 10**6, "N": 1, "B": 1}),))
    with pytest.raises(ValueError, match="fewer than"):
        sample.draw(pop)


def test_build_writes_a_blind_sheet_and_a_key_without_text(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "SAMPLES", SMALL)
    keys = tmp_path / "seen.csv"
    pd.DataFrame({"item": [1, 2], "raw_row": [1, 2]}).to_csv(keys, index=False)
    monkeypatch.setattr(sample, "SEEN_KEYS", (keys,))
    csv = raw_csv(tmp_path / "raw.csv")
    out, record = tmp_path / "runs", tmp_path / "record"
    manifest = sample.build(csv, out, record, check_population=False)

    sheet = pd.read_csv(out / "sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(sheet.columns) == ["item", "text", *sample.catalogue_types(), "none", "done", "note"]
    assert (sheet.drop(columns=["item", "text"]) == "").all().all()
    assert len(sheet) == manifest["items"] == 22
    key = pd.read_csv(record / "key.csv")
    assert list(key.columns) == ["item", "set", "raw_row", "stratum", "weight", "pool", "text_sha256"]
    assert key["item"].tolist() == list(range(1, 23)) and key["set"].tolist() != sorted(key["set"])
    assert [sample.text_hash(text) for text in sheet["text"]] == key["text_sha256"].tolist()
    stored = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    assert stored["sheet_sha256"] == sample.trends._sha256(out / "sheet.csv")
    assert stored["excluded_owner_labelled"] == 2 and stored["excluded_duplicate_text"] == 1
    assert stored["samples"]["test"]["seed"] == 20261022
    with pytest.raises(SystemExit, match="never redrawn"):
        sample.build(csv, out, record, check_population=False)


def test_build_refuses_data_that_is_not_the_protocols(tmp_path):
    with pytest.raises(SystemExit, match="not the Kaggle file"):
        sample.build(raw_csv(tmp_path / "raw.csv", n=30), tmp_path / "o", tmp_path / "r")


def test_files_are_written_with_lf_on_every_system(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "SAMPLES", SMALL)
    monkeypatch.setattr(sample, "SEEN_KEYS", ())
    out, record = tmp_path / "runs", tmp_path / "record"
    sample.build(raw_csv(tmp_path / "raw.csv"), out, record, check_population=False)
    for path in (out / "sheet.csv", record / "key.csv", record / "manifest.json"):
        assert b"\r\n" not in path.read_bytes(), path


SMALL_CHECK = ("check", range(90, 100), 20261024, {"H": 3, "N": 4, "B": 2})


def test_the_check_sample_constants_are_version_2s():
    prose = " ".join((ROOT / "docs/experiments/triage_diagnosis/PROTOCOL.md").read_text(encoding="utf-8").split())
    name, pools, seed, sizes = sample.CHECK
    assert (name, pools.start, pools.stop, seed, sizes) == ("check", 90, 100, 20261024, {"H": 30, "N": 60, "B": 10})
    assert "100 reviews from the test pool, none of them among the 500 test reviews: H 30, N 60, B 10" in prose
    assert "**seed 20261024**" in prose


def test_the_check_draw_leaves_out_the_drawn_reviews_and_reweights(frame, monkeypatch):
    monkeypatch.setattr(sample, "SAMPLES", SMALL)
    monkeypatch.setattr(sample, "CHECK", SMALL_CHECK)
    loaded, positive = frame
    pop, _ = sample.population(loaded, positive, sample.trigger_regex(), excluded=set())
    drawn, _ = sample.draw(pop)
    taken = set(drawn["raw_row"])
    check, sizes = sample.draw_check(pop, taken)
    again, _ = sample.draw_check(pop, taken)
    assert check.equals(again) and not taken & set(check["raw_row"])
    assert (check["set"] == "check").all() and check["pool"].isin(range(90, 100)).all()
    assert sizes["left_out_already_drawn"] == int((drawn["set"] == "test").sum())
    for stratum, n in SMALL_CHECK[3].items():
        left = int((pop["pool"].isin(range(90, 100)) & ~pop["raw_row"].isin(taken) & (pop["stratum"] == stratum)).sum())
        rows = check[check["stratum"] == stratum]
        assert len(rows) == n and (rows["weight"] == left / n).all()


def test_build_check_writes_a_blind_sheet_only_after_the_samples_redraw_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "SAMPLES", SMALL)
    monkeypatch.setattr(sample, "CHECK", SMALL_CHECK)
    monkeypatch.setattr(sample, "SEEN_KEYS", ())
    csv = raw_csv(tmp_path / "raw.csv")
    out, record = tmp_path / "runs", tmp_path / "record"
    sample.build(csv, out, record, check_population=False)
    manifest = sample.build_check(csv, out, record, check_population=False)
    sheet = pd.read_csv(out / "check_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(sheet.columns) == ["item", "text", *sample.catalogue_types(), "none", "done", "note"]
    assert (sheet.drop(columns=["item", "text"]) == "").all().all() and len(sheet) == manifest["items"] == 9
    check_key = pd.read_csv(record / "check_key.csv")
    assert "text" not in check_key.columns and (check_key["set"] == "check").all()
    assert not set(check_key["raw_row"]) & set(pd.read_csv(record / "key.csv")["raw_row"])
    assert [sample.text_hash(t) for t in sheet["text"]] == check_key["text_sha256"].tolist()
    for path in (out / "check_sheet.csv", record / "check_key.csv", record / "check_manifest.json"):
        assert b"\r\n" not in path.read_bytes(), path
    with pytest.raises(SystemExit, match="never redrawn"):
        sample.build_check(csv, out, record, check_population=False)
    (record / "key.csv").write_text((record / "key.csv").read_text(encoding="utf-8").replace(",test,", ",dev,", 1),
                                    encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit, match="byte for byte"):
        sample.build_check(csv, out, record, overwrite=True, check_population=False)
