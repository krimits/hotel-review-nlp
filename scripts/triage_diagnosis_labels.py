"""Check the owner's returned labelling sheets for the triage diagnosis protocol, and draw the relabel sample.

PROTOCOL.md (version 1, section 2, "Labels") fixes the order: the owner labels the 800 drawn reviews
(`owner_initial`), and at least 7 days later relabels 80 of them blind (30 development, 50 test) before seeing
any agreement figure (`owner_relabel`). This script turns each returned sheet into labels that may be
committed, and writes the relabel sheet. The second rater and the adjudication are later steps.

    python scripts/triage_diagnosis_labels.py finish --sheet filled_sheet.csv
    python scripts/triage_diagnosis_labels.py relabel
    python scripts/triage_diagnosis_labels.py finish --kind relabel --sheet filled_relabel_sheet.csv

For reference labels made by an LLM through a chat interface (PROTOCOL.md version 2):

    python scripts/triage_diagnosis_labels.py prompt
    python scripts/triage_diagnosis_labels.py batches
    python scripts/triage_diagnosis_labels.py finish --kind reference --sheet runs/triage_diagnosis/reference \\
        --source "ChatGPT, model as shown to the owner"

`prompt` writes reference_prompt.md from the catalogue, so its definitions reach the labeller word for word.
`batches` splits the blank sheet into batches of 50 rows in item order (runs/, git-ignored) and records each
blank batch's SHA-256. `finish` accepts the folder of returned `batch_NN_filled.csv` files and checks them as
one sheet.

A spreadsheet that opens a comma-separated file as one column per line (Excel with a `;` list separator) saves
every line back as one quoted cell, with inner quotes doubled. `finish --unwrap` undoes exactly that before the
checks, and records the SHA-256 of the file as returned and of the unwrapped text.

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
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

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
    "reference": ("key.csv", "reference_chatgpt"),
    "check": ("check_key.csv", "owner_check"),
}
REFERENCE_PROMPT = FOLDER / "reference_prompt.md"
BATCH_SIZE = 50


class SheetError(ValueError):
    """A returned sheet that cannot be recorded; nothing is written."""


def sheet_columns(types: list[str]) -> list[str]:
    return ["item", "text", *types, "none", "done", "note"]


def unwrap(text: str) -> str:
    """Undo a save that put every line in one quoted cell. No line of a drawn sheet starts with a quote (a record
    starts with its item number, a continuation of a multi-line text with text), so a line that does was wrapped:
    drop the outer quotes and undouble the inner ones. Any other line is kept as it is."""
    lines = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith('"'):
            if len(line) < 2 or not line.endswith('"'):
                raise SheetError(f"a line starts with a quote but is not one wrapped cell: {line[:60]!r}")
            line = line[1:-1].replace('""', '"')
        lines.append(line)
    return "\n".join(lines)


def read_unwrapped(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """A returned sheet whose lines a spreadsheet wrapped, read after `unwrap`, with both SHA-256."""
    text = unwrap(path.read_bytes().decode("utf-8-sig"))
    table = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    return table, {path.name: sample.trends._sha256(path),
                   "unwrapped_text": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def read_returned(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """A returned sheet, or the folder of returned batch files read as one sheet, and each file's SHA-256."""
    files = sorted(path.glob("batch_*_filled.csv")) if path.is_dir() else [path]
    if not files:
        raise SheetError(f"no batch_*_filled.csv in {path}")
    tables = [pd.read_csv(file, dtype=str, keep_default_na=False, encoding="utf-8-sig") for file in files]
    for file, table in zip(files, tables, strict=True):
        if list(table.columns) != list(tables[0].columns):
            raise SheetError(f"{file.name} has other columns than {files[0].name}")
    return pd.concat(tables, ignore_index=True), {file.name: sample.trends._sha256(file) for file in files}


def check_sheet(raw: pd.DataFrame, key: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """The labels of a filled sheet as 0/1 integers, indexed by item, or SheetError for the whole sheet."""
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
           received_on: dt.date | None = None, source: str | None = None, unwrapped: bool = False) -> dict:
    """Check a returned sheet and write its labels and record; refuse to replace labels already written."""
    key_name, name = KINDS[kind]
    if (out / f"{name}.csv").exists():
        raise SheetError(f"{out / f'{name}.csv'} exists; recorded labels are never replaced")
    key = pd.read_csv(samples / key_name, dtype=str, keep_default_na=False)
    if "text" in key.columns:
        raise SheetError(f"{samples / key_name} must not hold review text")
    types = sample.catalogue_types()
    if unwrapped and sheet.is_dir():
        raise SheetError("--unwrap takes one returned file, not a folder")
    raw, files = read_unwrapped(sheet) if unwrapped else read_returned(sheet)
    marks = check_sheet(raw, key, types)
    notes = raw["note"].str.strip()
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
        "sheet_sha256": files if sheet.is_dir() else files[sheet.name],
        "unwrapped_text_sha256": files.get("unwrapped_text"),
        "source": source,
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


def reference_prompt(catalogue: Path = sample.CATALOGUE) -> str:
    """The instructions an LLM labeller is given, with the catalogue's definitions and rules word for word."""
    data = yaml.safe_load(catalogue.read_text(encoding="utf-8"))
    lines = [
        "# Labelling hotel guest reviews",
        "",
        "You label hotel guest reviews for the problems the guest reports. You will receive the reviews as CSV",
        "files of 50 rows, one batch at a time. Label each batch and return it before the next one.",
        "",
        "## How to work",
        "",
        "- Read every review yourself, in full: both the `Disliked:` and the `Liked:` part. A complaint can be in",
        "  either part.",
        "- Decide every row from what the guest wrote. Do not write or run code, keyword rules or scripts to assign",
        '  labels, and do not label from single words: "room" alone is not a problem, and "clean" is usually',
        "  praise.",
        "- Use only the problem types below, as they are defined here, including what each one excludes.",
        "- Do not skip rows. If a review is unclear, decide as well as you can from its text.",
        "",
        "## What to put in the file",
        "",
        "- Keep every column and every row, in the same order. Do not change `item` or `text`.",
        "- Put `1` in the column of each problem type the review reports. Leave the other type columns empty.",
        "- If the review reports no problem, put `1` in `none` and leave every type column empty.",
        "- Put `1` in `done` on every row. Leave `note` empty.",
        "- Return the completed batch as a downloadable UTF-8 CSV file named like the batch with `_filled`, for",
        "  example `batch_01_filled.csv`.",
        "",
        "## Labelling rules",
        "",
        *[f"- {rule}" for rule in data["labelling_rules"]],
        "",
        "## Problem types",
        "",
    ]
    for entry in data["types"]:
        lines += [f"- `{entry['id']}`: {entry['definition']}"]
        if entry["excludes"] != "none":
            lines += [f"  Not this type: {entry['excludes']}"]
    lines += ["", f"(Catalogue {data['version']}, {len(data['types'])} types.)", ""]
    return "\n".join(lines)


def write_batches(runs: Path = RUNS, samples: Path = SAMPLES, prompt: Path = REFERENCE_PROMPT,
                  size: int = BATCH_SIZE) -> dict:
    """Split the blank sheet into batches in item order, after checking it is the drawn, blank sheet."""
    blank = pd.read_csv(runs / "sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(samples / "key.csv", dtype=str, keep_default_na=False)
    drawn = dict(zip(key["item"], key["text_sha256"], strict=True))
    if [sample.text_hash(t) for t in blank["text"]] != [drawn[i] for i in blank["item"]]:
        raise SheetError(f"{runs / 'sheet.csv'} is not the drawn sheet")
    if (blank.drop(columns=["item", "text"]) != "").any().any():
        raise SheetError(f"{runs / 'sheet.csv'} is not blank")
    if prompt.read_text(encoding="utf-8") != reference_prompt():
        raise SheetError(f"{prompt} does not match the catalogue; run `prompt` first")
    blank = blank.sort_values("item", key=lambda column: column.astype(int)).reset_index(drop=True)
    folder = runs / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    batches = {}
    for start in range(0, len(blank), size):
        name = f"batch_{start // size + 1:02d}.csv"
        part = blank.iloc[start:start + size]
        part.to_csv(folder / name, index=False, encoding="utf-8-sig", lineterminator="\n")
        batches[name] = {"items": [int(part["item"].iloc[0]), int(part["item"].iloc[-1])],
                         "sha256": sample.trends._sha256(folder / name)}
    record = {"prompt": prompt.name, "prompt_sha256": sample.trends._sha256(prompt),
              "blank_sheet_sha256": sample.trends._sha256(runs / "sheet.csv"), "batch_size": size, "batches": batches}
    (samples / "reference_batches.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8",
                                                    newline="\n")
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
    first.add_argument("--source", help="who or what filled the sheet, as recorded (for --kind reference)")
    first.add_argument("--unwrap", action="store_true",
                       help="undo a spreadsheet save that wrapped every line in one quoted cell")
    second = steps.add_parser("relabel", help="write the blind relabel sheet (7 days after owner_initial)")
    second.add_argument("--overwrite", action="store_true")
    steps.add_parser("prompt", help="write reference_prompt.md from the catalogue")
    steps.add_parser("batches", help="split the blank sheet into batches for an LLM labeller")
    args = parser.parse_args()
    try:
        if args.step == "prompt":
            REFERENCE_PROMPT.write_text(reference_prompt(), encoding="utf-8", newline="\n")
            print(f"{REFERENCE_PROMPT} ({sample.trends._sha256(REFERENCE_PROMPT)})")
        elif args.step == "batches":
            record = write_batches()
            print(f"{len(record['batches'])} batches in {RUNS / 'reference'}; record in {SAMPLES}")
        elif args.step == "finish":
            if args.kind == "reference" and not args.source:
                raise SheetError("--source is required for reference labels")
            record = finish(args.sheet, args.kind, args.out, source=args.source, unwrapped=args.unwrap)
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
