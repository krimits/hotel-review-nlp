"""Offline human-label integrity and known-count scoring, using invented reviews only."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from scripts import triage_lexicon_pilot as pilot

TEXTS = ["The shower was cold.", "The desk was dirty.", "We saw cockroaches.",
         "The bed was comfortable.", "Water barely dripped from the tap.", "The lift stopped upstairs."]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def pinned(arguments, **kwargs):
        if arguments[1] == "rev-parse":
            return "a" * 40 if kwargs.get("text") else b"a" * 40
        return (pilot.ROOT / arguments[-1].split(":", 1)[1]).read_bytes()
    monkeypatch.setattr(pilot.subprocess, "check_output", pinned)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args: pytest.fail("provider call"))
    source = tmp_path / "reviews.csv"
    pilot.write_csv(source, [{"review_id": f"r-{i}", "text": text, "language": "en"}
                             for i, text in enumerate(TEXTS)], ("review_id", "text", "language"))
    folder = tmp_path / "pilot"
    manifest = pilot.prepare(source, folder, size=6, seed=42, population="Invented wiring-test reviews")
    return folder, manifest, source


def fill(run: Path, output: Path, *, conflict=False, uncertain=False):
    _, rows = pilot.read_csv(run / "sheet_A.csv")
    for row in rows:
        row.update(dict.fromkeys(pilot.LABEL_TOPICS, "0"))
        row["done"] = "yes"
        if row["text"] in {TEXTS[0], TEXTS[4]}:
            row["bathroom"] = "1"
        if row["text"] == TEXTS[1]:
            row["cleanliness"] = "1"
        if row["text"] == TEXTS[2]:
            row["pests"] = "1"
        if row["text"] == TEXTS[5]:
            row["other"] = "1"
        if conflict and row["text"] == TEXTS[0]:
            row["bathroom"] = "0"
        if uncertain and row["text"] == TEXTS[3]:
            row["bathroom"] = "unsure"
    pilot.write_csv(output, rows)
    return rows


def test_uniform_sample_is_frozen_private_and_blind_before_any_labels(prepared):
    folder, manifest, _ = prepared
    assert manifest["sampling"] == "uniform_whole_review_without_model_or_sentiment_filter"
    assert manifest["sample_reviews"] == 6 and manifest["provider_requests"] == 0
    assert not manifest["labels_collected"] and not manifest["quality_evaluated"]
    _, a = pilot.read_csv(folder / "sheet_A.csv")
    _, b = pilot.read_csv(folder / "sheet_B.csv")
    assert {row["text"] for row in a} == set(TEXTS)
    assert {row["item"] for row in a} == {row["item"] for row in b}
    assert all(row[topic] == "" for row in [*a, *b] for topic in (*pilot.LABEL_TOPICS, "done"))
    assert all("text" not in row for row in json.loads((folder / "key.json").read_text()))
    assert "predicted" not in (folder / "sheet_A.csv").read_text()
    pilot.verify(folder)


@pytest.mark.parametrize("change", ["partial", "text", "duplicate", "count", "value"])
def test_incomplete_or_changed_human_sheets_are_rejected(prepared, tmp_path, change):
    folder, _, _ = prepared
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    rows = fill(folder, a)
    fill(folder, b)
    if change == "partial":
        rows[0]["pests"] = ""
    elif change == "text":
        rows[0]["text"] += " changed"
    elif change == "duplicate":
        rows[0]["item"] = rows[1]["item"]
    elif change == "count":
        rows.pop()
    else:
        rows[0]["bathroom"] = "0.5"
    pilot.write_csv(a, rows)
    with pytest.raises(ValueError):
        pilot.agreement(folder, a, b)


def test_scoring_counts_abstained_true_complaints_as_missed_and_unknown_topics_separately(prepared, tmp_path):
    folder, _, _ = prepared
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    gold, record = tmp_path / "gold.csv", tmp_path / "agreement.json"
    fill(folder, a)
    fill(folder, b)
    fill(folder, gold)
    agreement = pilot.agreement(folder, a, b)
    assert all(row["agreement"] == 1 for row in agreement["per_topic"].values())
    assert agreement["reviewer_independence_verified"] is False
    pilot.write_json(record, agreement)
    report = pilot.score(folder, a, b, record, gold)
    bathroom = report["per_topic"]["bathroom"]
    assert (bathroom["tp"], bathroom["fn"], bathroom["precision"], bathroom["recall"]) == (1, 1, 1, .5)
    assert report["per_topic"]["other"]["fn"] == 1
    assert report["per_topic"]["other"]["precision"] is None
    assert report["per_topic"]["air_conditioning"]["recall"] is None
    assert report["supported_topics_micro"]["recall"] == .75
    assert report["human_referral_reviews"] == 6 and report["reviews_with_supported_match"] == 3
    assert report["promotion"] is report["production_reliability_established"] is False
    assert all(text not in json.dumps(report) for text in TEXTS)


def test_agreement_keeps_disagreements_and_scoring_requires_original_receipt(prepared, tmp_path):
    folder, _, _ = prepared
    a, b, gold = (tmp_path / name for name in ("a.csv", "b.csv", "gold.csv"))
    fill(folder, a)
    fill(folder, b, conflict=True)
    fill(folder, gold)
    result = pilot.agreement(folder, a, b)
    assert len(result["disagreements"]) == 1 and result["per_topic"]["bathroom"]["agreement"] == 5/6
    receipt = tmp_path / "agreement.json"
    pilot.write_json(receipt, result)
    pilot.score(folder, a, b, receipt, gold)
    fill(folder, b)
    with pytest.raises(ValueError, match="original_agreement"):
        pilot.score(folder, a, b, receipt, gold)


def test_human_unsure_keeps_an_explicit_denominator_and_no_fake_precision(prepared, tmp_path):
    folder, _, _ = prepared
    a, b, gold = (tmp_path / name for name in ("a.csv", "b.csv", "gold.csv"))
    for path in (a, b, gold):
        fill(folder, path, uncertain=True)
    receipt = tmp_path / "agreement.json"
    pilot.write_json(receipt, pilot.agreement(folder, a, b))
    report = pilot.score(folder, a, b, receipt, gold)
    assert report["per_topic"]["bathroom"]["known_gold_reviews"] == 5
    assert report["per_topic"]["bathroom"]["uncertain_gold_reviews"] == 1
    assert report["per_topic"]["responsiveness"]["precision_wilson_95"] is None


def test_a_false_match_is_counted_in_precision_instead_of_hidden_by_referral(prepared, tmp_path):
    folder, _, _ = prepared
    a, b, gold = (tmp_path / name for name in ("a.csv", "b.csv", "gold.csv"))
    for path in (a, b, gold):
        rows = fill(folder, path)
        next(row for row in rows if row["text"] == TEXTS[1])["cleanliness"] = "0"
        pilot.write_csv(path, rows)
    receipt = tmp_path / "agreement.json"
    pilot.write_json(receipt, pilot.agreement(folder, a, b))
    result = pilot.score(folder, a, b, receipt, gold)
    assert result["per_topic"]["cleanliness"]["fp"] == 1
    assert result["per_topic"]["cleanliness"]["precision"] == 0
    assert result["supported_topics_micro"]["precision"] == 2/3


def test_prediction_tampering_fails_even_if_someone_updates_the_manifest_hash(prepared):
    folder, _, _ = prepared
    path = folder / "predictions.json"
    values = json.loads(path.read_text())
    values[next(iter(values))]["bathroom"] = "fabricated"
    path.write_text(json.dumps(values))
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["files"]["predictions.json"] = pilot.digest(path)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen_prediction"):
        pilot.verify(folder)


@pytest.mark.parametrize("rows", [
    [{"review_id": "x", "text": "The shower was cold.", "language": "el"}],
    [{"review_id": "x", "text": "The shower was cold.", "language": "en"}] * 2,
    [{"review_id": "x", "text": "The shower was cold.", "language": "en"},
     {"review_id": "y", "text": " THE SHOWER WAS COLD. ", "language": "en"}],
])
def test_ineligible_or_duplicate_reviews_never_create_a_sample(prepared, tmp_path, rows):
    source = tmp_path / "bad.csv"
    pilot.write_csv(source, rows, ("review_id", "text", "language"))
    output = tmp_path / "bad-pilot"
    with pytest.raises(ValueError):
        pilot.prepare(source, output, size=1, seed=0, population="test")
    assert not output.exists()
