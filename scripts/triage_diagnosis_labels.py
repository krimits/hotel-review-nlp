"""Check the owner's returned labelling sheets for the triage diagnosis protocol, and draw the relabel sample.

PROTOCOL.md (version 1, section 2, "Labels") fixes the order: the owner labels the 800 drawn reviews
(`owner_initial`), and at least 7 days later relabels 80 of them blind (30 development, 50 test) before seeing
any agreement figure (`owner_relabel`). This script turns each returned sheet into labels that may be
committed, and writes the relabel sheet. The second rater and the adjudication are later steps.

    python scripts/triage_diagnosis_labels.py finish --sheet filled_sheet.csv
    python scripts/triage_diagnosis_labels.py relabel
    python scripts/triage_diagnosis_labels.py finish --kind relabel --sheet filled_relabel_sheet.csv

`finish` refuses a sheet as a whole if a column, an item or a text is not the one drawn, if a label is not
empty, 0 or 1, if `done` is missing on any row, or if a row has both a type and `none`, or neither. It writes
into --out (default docs/experiments/triage_diagnosis/labels), with no text and no notes:

    owner_initial.csv / owner_relabel.csv     raw_row, set, one 0/1 column per catalogue type, none
    owner_initial_record.json / ...           hashes of the sheet, the key and the catalogue, counts, the date

and never replaces labels already written: `owner_initial` is never edited.

`relabel` draws the 80 reviews from the key alone, never from the labels, and refuses before 7 days have
passed since `owner_initial` was recorded. The sheet goes to runs/triage_diagnosis (git-ignored); its key and
manifest, with no text, go next to the main sample's.

No model, provider call or review text leaves the machine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import triage_diagnosis_sample as sample  # noqa: E402

FOLDER = sample.FOLDER
SAMPLES = FOLDER / "samples"
LABELS = FOLDER / "labels"
RUNS = ROOT / "runs" / "triage_diagnosis"
# PROTOCOL.md: "relabels 80 reviews (30 development, 50 test) blind, at least 7 days later".
RELABEL_SIZES = {"development": 30, "test": 50}
RELABEL_SEED = 20261023
RELABEL_WAIT = dt.timedelta(days=7)
KINDS = {  # kind -> (key, labels name)
    "initial": ("key.csv", "owner_initial"),
    "relabel": ("relabel_key.csv", "owner_relabel"),
}


class SheetError(ValueError):
    """A returned sheet that cannot be recorded; nothing is written."""


def sheet_columns(types: list[str]) -> list[str]:
    return ["item", "text", *types, "none", "done", "note"]


def check_sheet(path: Path, key: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """The labels of a filled sheet as 0/1 integers, indexed by item, or SheetError for the whole sheet."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(raw.columns) != sheet_columns(types):
        raise SheetError(f"the columns are not the drawn sheet's: {list(raw.columns)}")
    items = raw["item"].str.strip()
    if items.duplicated().any():
        raise SheetError(f"repeated items: {sorted(set(items[items.duplicated()]))[:20]}")
    drawn = dict(zip(key["item"].astype(str), key["text_sha256"], strict=True))
    if set(items) != set(drawn):
        missing, unknown = sorted(set(drawn) - set(items)), sorted(set(items) - set(drawn))
        raise SheetError(f"items missing {missing[:20]}, unknown {unknown[:20]}")
    changed = [item for item, text in zip(items, raw["text"], strict=True) if sample.text_hash(text) != drawn[item]]
    if changed:
        raise SheetError(f"the text of {len(changed)} item(s) is not the one drawn: {changed[:20]}")
    marks = raw[[*types, "none", "done"]].apply(lambda column: column.str.strip())
    bad = sorted({value for column in marks for value in marks[column]} - {"", "0", "1"})
    if bad:
        raise SheetError(f"labels may only be empty, 0 or 1; found {bad[:10]}")
    marks = (marks == "1").astype(int)
    marks.index = items
    if (marks["done"] == 0).any():
        raise SheetError(f"`done` is missing on {int((marks['done'] == 0).sum())} row(s): "
                         f"{list(marks.index[marks['done'] == 0][:20])}")
    typed = marks[types].sum(axis=1) > 0
    both, neither = typed & (marks["none"] == 1), ~typed & (marks["none"] == 0)
    if both.any() or neither.any():
        raise SheetError(f"each row needs a type or `none`, not both: both on {list(marks.index[both][:20])}, "
                         f"neither on {list(marks.index[neither][:20])}")
    return marks.drop(columns="done")


def finish(sheet: Path, kind: str = "initial", out: Path = LABELS, samples: Path = SAMPLES,
           received_on: dt.date | None = None) -> dict:
    """Check a returned sheet and write its labels and record; refuse to replace labels already written."""
    key_name, name = KINDS[kind]
    if (out / f"{name}.csv").exists():
        raise SheetError(f"{out / f'{name}.csv'} exists; recorded labels are never replaced")
    key = pd.read_csv(samples / key_name, dtype=str, keep_default_na=False)
    if "text" in key.columns:
        raise SheetError(f"{samples / key_name} must not hold review text")
    types = sample.catalogue_types()
    marks = check_sheet(sheet, key, types)
    notes = pd.read_csv(sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")["note"].str.strip()
    by_item = key.set_index(key["item"].astype(str))
    labels = marks.join(by_item[["raw_row", "set"]])
    labels["raw_row"] = labels["raw_row"].astype(int)
    labels = labels[["raw_row", "set", *types, "none"]].sort_values("raw_row")
    received_on = received_on or dt.date.today()
    record = {
        "labels": name,
        "protocol": sample.PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": sample.trends._sha256(sample.PROTOCOL),
        "catalogue_sha256": sample.trends._sha256(sample.CATALOGUE),
        "key": key_name,
        "key_sha256": sample.trends._sha256(samples / key_name),
        "sheet_sha256": sample.trends._sha256(sheet),
        "code_commit": sample.pilot.git_commit(),
        "received_on": received_on.isoformat(),
        "reviews": len(labels),
        "reviews_by_set": {s: int((labels["set"] == s).sum()) for s in sorted(labels["set"].unique())},
        "reviews_with_a_type": int((labels["none"] == 0).sum()),
        "reviews_with_none": int(labels["none"].sum()),
        "type_labels": int(labels[types].to_numpy().sum()),
        "per_type": {t: int(labels[t].sum()) for t in types},
        "notes_left_out": int((notes != "").sum()),
    }
    if kind == "initial":
        record["relabel_not_before"] = (received_on + RELABEL_WAIT).isoformat()
    out.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    (out / f"{name}_record.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return record


def draw_relabel(key: pd.DataFrame, seed: int = RELABEL_SEED) -> pd.DataFrame:
    """The relabel sample from the key alone: per set a simple random draw, then new item numbers in a random order."""
    picked = []
    for name, n in RELABEL_SIZES.items():
        rows = key[key["set"] == name].sort_values("raw_row", key=lambda column: column.astype(int))
        if len(rows) < n:
            raise ValueError(f"{name}: {len(rows)} reviews in the key, fewer than {n}")
        picked.append(rows.sample(n=n, random_state=seed))
    chosen = pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)
    chosen.insert(0, "relabel_item", range(1, len(chosen) + 1))
    return chosen


def relabel(today: dt.date | None = None, samples: Path = SAMPLES, labels: Path = LABELS, runs: Path = RUNS,
            overwrite: bool = False) -> dict:
    """Write the relabel sheet, its key and its manifest, once the 7 days have passed."""
    record_path = labels / "owner_initial_record.json"
    if not record_path.exists():
        raise SheetError("owner_initial is not recorded yet; the relabel follows it")
    not_before = dt.date.fromisoformat(json.loads(record_path.read_text(encoding="utf-8"))["relabel_not_before"])
    today = today or dt.date.today()
    if today < not_before:
        raise SheetError(f"the relabel sheet may be written from {not_before.isoformat()}, "
                         f"{(not_before - today).days} day(s) from now")
    if (runs / "relabel_sheet.csv").exists() and not overwrite:
        raise SheetError(f"{runs / 'relabel_sheet.csv'} exists; pass --overwrite to replace it")
    key = pd.read_csv(samples / "key.csv", dtype=str, keep_default_na=False)
    main = pd.read_csv(runs / "sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    text_of = dict(zip(main["item"], main["text"], strict=True))
    chosen = draw_relabel(key)
    texts = [text_of[item] for item in chosen["item"]]
    if [sample.text_hash(text) for text in texts] != chosen["text_sha256"].tolist():
        raise SheetError(f"{runs / 'sheet.csv'} is not the drawn sheet; nothing written")
    table = pd.DataFrame({"item": chosen["relabel_item"], "text": texts})
    for column in sheet_columns(sample.catalogue_types())[2:]:
        table[column] = ""
    runs.mkdir(parents=True, exist_ok=True)
    table.to_csv(runs / "relabel_sheet.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    relabel_key = chosen.rename(columns={"item": "drawn_item"}).rename(columns={"relabel_item": "item"})
    relabel_key = relabel_key[["item", "drawn_item", "set", "raw_row", "stratum", "weight", "pool", "text_sha256"]]
    relabel_key.to_csv(samples / "relabel_key.csv", index=False, lineterminator="\n")
    manifest = {
        "protocol_sha256": sample.trends._sha256(sample.PROTOCOL),
        "code_commit": sample.pilot.git_commit(),
        "drawn_from": "samples/key.csv only; no label was read",
        "sizes": RELABEL_SIZES,
        "seed": RELABEL_SEED,
        "written_on": today.isoformat(),
        "not_before": not_before.isoformat(),
        "items": len(table),
        "sheet_sha256": sample.trends._sha256(runs / "relabel_sheet.csv"),
    }
    (samples / "relabel_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                                                   newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("finish", help="check a returned sheet and write its labels")
    first.add_argument("--sheet", type=Path, required=True, help="the filled sheet, as it came back")
    first.add_argument("--kind", choices=sorted(KINDS), default="initial")
    first.add_argument("--out", type=Path, default=LABELS)
    second = steps.add_parser("relabel", help="write the blind relabel sheet (7 days after owner_initial)")
    second.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.step == "finish":
            record = finish(args.sheet, args.kind, args.out)
            print(f"{record['labels']}: {record['reviews']} reviews {record['reviews_by_set']}, "
                  f"{record['type_labels']} type labels, {record['reviews_with_none']} with none -> {args.out}")
            if "relabel_not_before" in record:
                print(f"the relabel sheet can be written from {record['relabel_not_before']}")
        else:
            manifest = relabel(overwrite=args.overwrite)
            print(f"{manifest['items']} reviews in {RUNS / 'relabel_sheet.csv'}; key in {SAMPLES}")
    except SheetError as error:
        raise SystemExit(f"nothing recorded: {error}") from None


if __name__ == "__main__":
    main()
