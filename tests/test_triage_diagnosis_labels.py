"""Returned labelling sheets and the relabel draw of the triage diagnosis protocol, on tiny synthetic sheets."""

from __future__ import annotations

import datetime as dt
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


labels = _load("triage_diagnosis_labels")
TYPES = labels.sample.catalogue_types()
DAY = dt.date(2026, 10, 27)


def drawn(tmp_path: Path, n: int = 12) -> tuple[Path, Path, pd.DataFrame]:
    """A key and a blank sheet for n reviews: half development, half test."""
    texts = [f"Disliked: problem {i}\n\nLiked: nice {i}" for i in range(n)]
    key = pd.DataFrame({"item": range(1, n + 1), "set": ["development", "test"] * (n // 2),
                        "raw_row": [100 + 7 * i for i in range(n)], "stratum": "N", "weight": 2.0, "pool": 91,
                        "text_sha256": [labels.sample.text_hash(t) for t in texts]})
    samples, runs = tmp_path / "samples", tmp_path / "runs"
    samples.mkdir()
    runs.mkdir()
    key.to_csv(samples / "key.csv", index=False)
    blank = pd.DataFrame({"item": key["item"], "text": texts})
    for column in labels.sheet_columns(TYPES)[2:]:
        blank[column] = ""
    blank.to_csv(runs / "sheet.csv", index=False, encoding="utf-8-sig")
    return samples, runs, blank


def filled(blank: pd.DataFrame, path: Path, change=None) -> Path:
    """The blank sheet labelled: odd items get cleanliness and wifi, even items none; every row done."""
    table = blank.astype(str)
    for i, item in enumerate(table["item"]):
        if int(item) % 2:
            table.loc[i, ["cleanliness", "wifi"]] = "1"
        else:
            table.loc[i, "none"] = "1"
    table["done"] = "1"
    table.loc[0, "note"] = "my own note"
    if change:
        change(table)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_a_complete_sheet_becomes_labels_without_text_or_notes(tmp_path):
    samples, _, blank = drawn(tmp_path)
    sheet = filled(blank, tmp_path / "back.csv")
    record = labels.finish(sheet, out=tmp_path / "labels", samples=samples, received_on=DAY)
    table = pd.read_csv(tmp_path / "labels" / "owner_initial.csv")
    assert list(table.columns) == ["raw_row", "set", *TYPES, "none"]
    assert table["raw_row"].is_monotonic_increasing and len(table) == 12
    odd = table[table["raw_row"].isin([100 + 7 * i for i in range(0, 12, 2)])]  # items 1, 3, ...
    assert (odd["cleanliness"] == 1).all() and (odd["wifi"] == 1).all() and (odd["none"] == 0).all()
    assert record["type_labels"] == 12 and record["reviews_with_none"] == 6
    assert record["reviews_by_set"] == {"development": 6, "test": 6}
    assert record["notes_left_out"] == 1 and record["relabel_not_before"] == "2026-11-03"
    written = (tmp_path / "labels" / "owner_initial.csv").read_text(encoding="utf-8")
    assert "Disliked" not in written and "my own note" not in written
    stored = json.loads((tmp_path / "labels" / "owner_initial_record.json").read_text(encoding="utf-8"))
    assert stored["sheet_sha256"] == labels.sample.trends._sha256(sheet)


def test_recorded_labels_are_never_replaced(tmp_path):
    samples, _, blank = drawn(tmp_path)
    sheet = filled(blank, tmp_path / "back.csv")
    labels.finish(sheet, out=tmp_path / "labels", samples=samples, received_on=DAY)
    with pytest.raises(labels.SheetError, match="never replaced"):
        labels.finish(sheet, out=tmp_path / "labels", samples=samples, received_on=DAY)


@pytest.mark.parametrize(("change", "message"), [
    (lambda t: t.__setitem__("text", t["text"].str.replace("problem 3", "problem 33")), "not the one drawn"),
    (lambda t: t.loc.__setitem__((2, "done"), ""), "`done` is missing"),
    (lambda t: t.loc.__setitem__((0, "none"), "1"), "not both"),  # item 1 has types
    (lambda t: t.loc.__setitem__((0, TYPES), ""), "not both"),
    (lambda t: t.loc.__setitem__((4, "pests"), "x"), "only be empty, 0 or 1"),
    (lambda t: t.loc.__setitem__((5, "item"), "4"), "repeated items"),
    (lambda t: t.loc.__setitem__((5, "item"), "99"), "unknown"),
    (lambda t: t.drop(index=3, inplace=True), "missing"),
    (lambda t: t.insert(2, "model_guess", ""), "columns"),
])
def test_a_sheet_with_any_fault_is_refused_as_a_whole(tmp_path, change, message):
    samples, _, blank = drawn(tmp_path)
    sheet = filled(blank, tmp_path / "back.csv", change)
    with pytest.raises(labels.SheetError, match=message):
        labels.finish(sheet, out=tmp_path / "labels", samples=samples, received_on=DAY)
    assert not (tmp_path / "labels").exists()


def test_line_ends_and_zeros_from_a_spreadsheet_are_accepted(tmp_path):
    samples, _, blank = drawn(tmp_path)

    def spreadsheet(table):
        table["text"] = table["text"].str.replace("\n", "\r\n")
        table.loc[table["cleanliness"] == "", "cleanliness"] = "0"
    sheet = filled(blank, tmp_path / "back.csv", spreadsheet)
    record = labels.finish(sheet, out=tmp_path / "labels", samples=samples, received_on=DAY)
    assert record["per_type"]["cleanliness"] == 6


def test_the_relabel_draw_reads_the_key_only_and_is_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "RELABEL_SIZES", {"development": 2, "test": 3})
    samples, _, _ = drawn(tmp_path)
    key = pd.read_csv(samples / "key.csv", dtype=str, keep_default_na=False)
    first, again = labels.draw_relabel(key), labels.draw_relabel(key)
    assert first.equals(again)
    assert first["set"].value_counts().to_dict() == {"test": 3, "development": 2}
    assert first["relabel_item"].tolist() == [1, 2, 3, 4, 5] and first["item"].is_unique


def test_the_relabel_sheet_waits_seven_days_and_hides_the_original_items(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "RELABEL_SIZES", {"development": 2, "test": 3})
    samples, runs, blank = drawn(tmp_path)
    out = tmp_path / "labels"
    with pytest.raises(labels.SheetError, match="not recorded yet"):
        labels.relabel(DAY, samples, out, runs)
    labels.finish(filled(blank, tmp_path / "back.csv"), out=out, samples=samples, received_on=DAY)
    with pytest.raises(labels.SheetError, match="from 2026-11-03, 7 day"):
        labels.relabel(DAY, samples, out, runs)
    with pytest.raises(labels.SheetError, match="1 day"):
        labels.relabel(dt.date(2026, 11, 2), samples, out, runs)

    manifest = labels.relabel(dt.date(2026, 11, 3), samples, out, runs)
    sheet = pd.read_csv(runs / "relabel_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(sheet.columns) == labels.sheet_columns(TYPES) and len(sheet) == manifest["items"] == 5
    assert (sheet.drop(columns=["item", "text"]) == "").all().all()
    rkey = pd.read_csv(samples / "relabel_key.csv", dtype=str)
    assert "text" not in rkey.columns and rkey["item"].tolist() == [str(i) for i in range(1, 6)]
    assert [labels.sample.text_hash(t) for t in sheet["text"]] == rkey["text_sha256"].tolist()
    with pytest.raises(labels.SheetError, match="exists"):
        labels.relabel(dt.date(2026, 11, 3), samples, out, runs)

    # The relabel comes back through the same check, into its own file.
    back = sheet.copy()
    back["none"], back["done"] = "1", "1"
    back.to_csv(tmp_path / "relabel_back.csv", index=False, encoding="utf-8-sig")
    record = labels.finish(tmp_path / "relabel_back.csv", "relabel", out, samples, dt.date(2026, 11, 4))
    assert record["labels"] == "owner_relabel" and record["reviews"] == 5 and "relabel_not_before" not in record
    assert (out / "owner_initial.csv").exists() and (out / "owner_relabel.csv").exists()


def test_the_reference_prompt_carries_every_definition_and_rule_word_for_word():
    import yaml

    data = yaml.safe_load(labels.sample.CATALOGUE.read_text(encoding="utf-8"))
    prompt = labels.reference_prompt()
    for entry in data["types"]:
        assert f"- `{entry['id']}`: {entry['definition']}" in prompt
        if entry["excludes"] != "none":
            assert f"  Not this type: {entry['excludes']}" in prompt
    assert all(f"- {rule}" in prompt for rule in data["labelling_rules"])
    assert "Do not write or run code" in prompt and "(Catalogue catalogue_v1, 20 types.)" in prompt
    committed = labels.REFERENCE_PROMPT
    if committed.exists():
        assert committed.read_text(encoding="utf-8") == prompt


def test_batches_split_the_blank_sheet_in_item_order_and_refuse_a_used_one(tmp_path):
    samples, runs, blank = drawn(tmp_path, n=12)
    prompt = tmp_path / "prompt.md"
    prompt.write_text(labels.reference_prompt(), encoding="utf-8", newline="\n")
    record = labels.write_batches(runs, samples, prompt, size=5)
    assert list(record["batches"]) == ["batch_01.csv", "batch_02.csv", "batch_03.csv"]
    assert [b["items"] for b in record["batches"].values()] == [[1, 5], [6, 10], [11, 12]]
    parts = [pd.read_csv(runs / "reference" / name, dtype=str, keep_default_na=False, encoding="utf-8-sig")
             for name in record["batches"]]
    assert pd.concat(parts)["text"].tolist() == blank["text"].tolist()
    assert json.loads((samples / "reference_batches.json").read_text(encoding="utf-8"))["batch_size"] == 5

    prompt.write_text("an older prompt", encoding="utf-8")
    with pytest.raises(labels.SheetError, match="does not match the catalogue"):
        labels.write_batches(runs, samples, prompt, size=5)
    prompt.write_text(labels.reference_prompt(), encoding="utf-8", newline="\n")
    filled(blank, runs / "sheet.csv")
    with pytest.raises(labels.SheetError, match="not blank"):
        labels.write_batches(runs, samples, prompt, size=5)


def test_the_second_rater_gets_the_references_batches_and_separate_check_batches(tmp_path):
    samples, runs, blank = drawn(tmp_path, n=12)
    check_texts = [f"Disliked: check problem {i}" for i in range(7)]
    pd.DataFrame({"item": range(1, 8), "set": "check", "raw_row": [9000 + i for i in range(7)], "stratum": "N",
                  "weight": 3.0, "pool": 95, "text_sha256": [labels.sample.text_hash(t) for t in check_texts]}
                 ).to_csv(samples / "check_key.csv", index=False)
    check = pd.DataFrame({"item": range(1, 8), "text": check_texts})
    for column in labels.sheet_columns(TYPES)[2:]:
        check[column] = ""
    check.to_csv(runs / "check_sheet.csv", index=False, encoding="utf-8-sig")
    prompt = tmp_path / "prompt.md"
    prompt.write_text(labels.reference_prompt(), encoding="utf-8", newline="\n")
    labels.write_batches(runs, samples, prompt, size=5)
    record = labels.write_second_rater_batches(runs, samples, prompt, size=5)
    assert list(record["main"]["batches"]) == ["batch_01.csv", "batch_02.csv", "batch_03.csv"]
    assert list(record["check"]["batches"]) == ["check_batch_01.csv", "check_batch_02.csv"]
    for name in record["main"]["batches"]:
        assert (runs / "second_rater" / name).read_bytes() == (runs / "reference" / name).read_bytes()

    # Back in one folder: the main kind reads batch_*, the check kind check_batch_*.
    back = tmp_path / "back"
    back.mkdir()
    main = pd.read_csv(filled(blank, tmp_path / "main.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for i, start in enumerate(range(0, 12, 5), start=1):
        main.iloc[start:start + 5].to_csv(back / f"batch_{i:02d}_filled.csv", index=False, encoding="utf-8-sig")
    done = check.astype(str).assign(none="1", done="1")
    done.iloc[:5].to_csv(back / "check_batch_01_filled.csv", index=False, encoding="utf-8-sig")
    done.iloc[5:].to_csv(back / "check_batch_02_filled.csv", index=False, encoding="utf-8-sig")
    out = tmp_path / "labels"
    first = labels.finish(back, "second_rater", out, samples, DAY, "Gemini web")
    second = labels.finish(back, "second_rater_check", out, samples, DAY, "Gemini web")
    assert (first["labels"], first["reviews"], len(first["sheet_sha256"])) == ("second_rater", 12, 3)
    assert (second["labels"], second["reviews"], len(second["sheet_sha256"])) == ("second_rater_check", 7, 2)

    # A record of the reference's batches that no longer matches stops the second rater's.
    reference = json.loads((samples / "reference_batches.json").read_text(encoding="utf-8"))
    reference["batches"]["batch_02.csv"]["sha256"] = "0" * 64
    (samples / "reference_batches.json").write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(labels.SheetError, match="differ from the reference's"):
        labels.write_second_rater_batches(runs, samples, prompt, size=5)


def test_returned_batches_are_checked_as_one_sheet(tmp_path):
    samples, _, blank = drawn(tmp_path, n=12)
    back = tmp_path / "back"
    back.mkdir()
    whole = pd.read_csv(filled(blank, tmp_path / "whole.csv"), dtype=str, keep_default_na=False,
                        encoding="utf-8-sig")
    for i, start in enumerate(range(0, 12, 5), start=1):
        whole.iloc[start:start + 5].to_csv(back / f"batch_{i:02d}_filled.csv", index=False, encoding="utf-8-sig")
    (back / "batch_02_filled.csv").rename(back / "batch_02_missing.csv")
    with pytest.raises(labels.SheetError, match="items missing"):
        labels.finish(back, "reference", tmp_path / "labels", samples, DAY, "ChatGPT")
    (back / "batch_02_missing.csv").rename(back / "batch_02_filled.csv")
    record = labels.finish(back, "reference", tmp_path / "labels", samples, DAY, "ChatGPT, model X")
    assert record["labels"] == "reference_chatgpt" and record["reviews"] == 12
    assert record["source"] == "ChatGPT, model X" and sorted(record["sheet_sha256"]) == [
        "batch_01_filled.csv", "batch_02_filled.csv", "batch_03_filled.csv"]
    assert "relabel_not_before" not in record


def spreadsheet_wrapped(path: Path, target: Path) -> Path:
    """What a spreadsheet with a `;` list separator saves after opening a comma CSV as one column per line."""
    lines = path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").split("\n")
    cells = ['"' + line.replace('"', '""') + '"' if ("," in line or '"' in line) else line for line in lines]
    target.write_bytes("\r\n".join(cells).encode("utf-8"))
    return target


def test_a_spreadsheet_wrapped_sheet_is_recorded_only_with_unwrap_and_gives_the_same_labels(tmp_path):
    samples, _, blank = drawn(tmp_path)
    clean = filled(blank, tmp_path / "clean.csv")
    wrapped = spreadsheet_wrapped(clean, tmp_path / "wrapped.csv")
    with pytest.raises(labels.SheetError):
        labels.finish(wrapped, out=tmp_path / "plain", samples=samples, received_on=DAY)
    record = labels.finish(wrapped, out=tmp_path / "unwrapped", samples=samples, received_on=DAY, unwrapped=True)
    labels.finish(clean, out=tmp_path / "clean", samples=samples, received_on=DAY)
    assert (tmp_path / "unwrapped" / "owner_initial.csv").read_bytes() == (tmp_path / "clean" / "owner_initial.csv").read_bytes()
    assert record["sheet_sha256"] == labels.sample.trends._sha256(wrapped)
    assert record["unwrapped_text_sha256"] and record["unwrapped_text_sha256"] != record["sheet_sha256"]


def test_unwrap_refuses_anything_but_whole_wrapped_lines(tmp_path):
    assert labels.unwrap('"1,""a"""\r\nplain line\r\n') == '1,"a"\nplain line\n'
    with pytest.raises(labels.SheetError, match="not one wrapped cell"):
        labels.unwrap('"1,a\nrest')
    samples, _, _ = drawn(tmp_path)
    with pytest.raises(labels.SheetError, match="not a folder"):
        labels.finish(tmp_path, "reference", tmp_path / "labels", samples, DAY, "ChatGPT", unwrapped=True)


def test_the_teacher_prompt_shares_the_reference_prompts_rules_and_types_word_for_word():
    reference, teacher = labels.reference_prompt(), labels.teacher_prompt()
    for heading, end in (("## How to work", "## What to"), ("## Labelling rules", None)):
        part = reference[reference.index(heading):reference.index(end) if end else None]
        assert part in teacher, heading
    assert "## What to put in the file" not in teacher and "CSV" not in teacher
    assert '{"labels": [{"item": 1, "types": ["cleanliness"]}]}' in teacher and "up to 25 reviews" in teacher
    committed = labels.TEACHER_PROMPT
    if committed.exists():
        assert committed.read_text(encoding="utf-8") == teacher
