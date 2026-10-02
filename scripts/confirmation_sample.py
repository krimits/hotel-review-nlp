"""Draw the confirmation sample for Jev's responsiveness result, and check the sheet that comes back.

The design is fixed in docs/annotation/confirmation_protocol.md, and this script follows it. The
texts, hotels and periods are those of the pilot (scripts/annotation_sample.py). In each period it
draws a simple random sample of 200 texts that pass has_complaint, outside the pilot's 300 texts.

    python scripts/confirmation_sample.py draw --expect-hotels 863
    python scripts/confirmation_sample.py finish --sheet filled_sheet.csv

`draw` writes into --out (default runs/confirmation, git-ignored because the sheet holds review
text):

    sheet.csv       item, text and empty responsiveness, done and note columns
    key.csv         item -> raw_row, period, the lexicon's verdicts and a hash of each text;
                    no text
    manifest.json   population sizes, the filter's blind spot, seed, code commit, and the SHA-256
                    of the protocol, of the guideline and of the pilot texts left out

`finish` checks the filled sheet and writes into --out (default
docs/experiments/jev_topic_benchmark/confirmation) what may be committed, with no text and no
notes: labels.csv, key.csv, sample_manifest.json and sheet_record.json.

A sheet is refused as a whole if any label is blank or unknown, if `done` is missing, if an item
is missing, repeated or unknown, or if a text is not the one that was drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import annotation_sample as pilot  # noqa: E402
import score_annotations as score  # noqa: E402

trends = pilot.trends
PROTOCOL = ROOT / "docs" / "annotation" / "confirmation_protocol.md"
GUIDELINE = ROOT / "docs" / "annotation" / "complaint_topics_guideline.md"
PILOT_KEY = ROOT / "docs" / "case_study" / "results" / "annotation" / "key.csv"
COMMITTED = ROOT / "docs" / "experiments" / "jev_topic_benchmark" / "confirmation"
TOPIC = "responsiveness"
SEED = 1
TEXTS_PER_PERIOD = 200
SHEET_COLUMNS = ["item", "text", TOPIC, "done", "note"]
KEY_COLUMNS = ["item", "raw_row", "review_id", "period", "in_R", "drawn_for",
               *(f"lex_{column}" for column in pilot.TOPICS), "in_B", "text_sha256"]


def text_hash(text: str) -> str:
    """SHA-256 of a text with its white space collapsed, so a spreadsheet's line ends do not matter."""
    return hashlib.sha256(" ".join(str(text).split()).encode("utf-8")).hexdigest()


def eligible(frame: pd.DataFrame, period: pd.Series, hotels: list[str]) -> pd.DataFrame:
    """P_p: the texts that pass has_complaint in the compared hotels and periods, indexed by review_id."""
    rows = frame[(period != "") & frame["address"].isin(hotels) & frame["has_complaint_text"]]
    return pd.DataFrame({"raw_row": rows["raw_row"].to_numpy(), "period": period[rows.index].to_numpy(),
                         "text": rows["negative"].to_numpy()}, index=pd.Index(rows.index, name="review_id"))


def excluded_rows(path: Path) -> set[int]:
    """The raw rows of the pilot's texts, which are a test set and are not drawn again."""
    key = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" in key.columns:
        raise SystemExit(f"{path} must not hold review text")
    return {int(row) for row in key["raw_row"]}


def draw(pop: pd.DataFrame, excluded: set[int], seed: int = SEED,
         size: int = TEXTS_PER_PERIOD) -> tuple[pd.DataFrame, dict]:
    """One row per drawn text, numbered in a random order, and the sizes behind each period."""
    rng = np.random.default_rng(seed)
    chosen, sizes = [], {}
    for name in pilot.PERIODS:
        texts = pop[pop["period"] == name]
        left = texts[~texts["raw_row"].isin(excluded)]
        if len(left) < size:
            raise ValueError(f"{name}: {len(left)} texts left after the exclusion, fewer than {size}")
        picked = rng.choice(np.sort(left.index.to_numpy()), size=size, replace=False)
        chosen += [(int(review_id), name) for review_id in np.sort(picked)]
        sizes[name] = {"population": len(texts), "excluded_pilot_texts": len(texts) - len(left), "drawn": size}
    key = pd.DataFrame(chosen, columns=["review_id", "period"])
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    key.insert(0, "item", np.arange(1, len(key) + 1))
    return key, sizes


def with_lexicon(key: pd.DataFrame, frame: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """The key in the pilot's format: the lexicon's verdict on each drawn text, and a hash of the text."""
    rows = frame.loc[key["review_id"]]
    found = pd.DataFrame(trends.tag_complaints(rows), columns=["review_id", "topic", "quote"])
    lexicon = pd.DataFrame({f"lex_{column}": key["review_id"].isin(found.loc[found["topic"] == topic, "review_id"])
                            .astype(int) for column, topic in pilot.TOPICS.items()})
    key = pd.concat([key, lexicon], axis=1).join(pop["raw_row"], on="review_id")
    key["in_R"], key["drawn_for"], key["in_B"] = 1, "", 0
    key["text_sha256"] = [text_hash(text) for text in pop.loc[key["review_id"], "text"]]
    return key[KEY_COLUMNS]


def sheet(key: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """The sheet to label: item and text, with empty label columns and no hint of the answer."""
    table = pd.DataFrame({"item": key["item"].to_numpy(), "text": pop.loc[key["review_id"], "text"].to_numpy()})
    for column in SHEET_COLUMNS[2:]:
        table[column] = ""
    return table


def source_of(path: Path) -> str:
    """A path as the manifest records it: relative to the repository when it is inside."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(path)


def build_sample(csv: Path, out: Path, seed: int = SEED, expect_hotels: int | None = None, overwrite: bool = False,
                 exclude_key: Path = PILOT_KEY, size: int = TEXTS_PER_PERIOD,
                 min_hotel_reviews: int = pilot.MIN_HOTEL_REVIEWS) -> dict:
    """Draw the sample, write the sheet, the key and the manifest, and return the manifest."""
    if (out / "sheet.csv").exists() and not overwrite:
        raise SystemExit(f"{out / 'sheet.csv'} exists; a drawn sample is never redrawn by accident "
                         "(pass --overwrite to replace it)")
    frame, source = trends.load_reviews(csv)
    params = trends.analysis_windows(frame["review_date"], min_hotel_reviews, pilot.SINCE)
    period = pilot.period_of(frame, params)
    hotels = pilot.compared_hotels(frame, period, min_hotel_reviews)
    if expect_hotels is not None and len(hotels) != expect_hotels:
        raise SystemExit(f"{len(hotels)} hotels compared, expected {expect_hotels}")
    pop = eligible(frame, period, hotels)
    excluded = excluded_rows(exclude_key)
    key, sizes = draw(pop, excluded, seed, size)
    found = sum(numbers["excluded_pilot_texts"] for numbers in sizes.values())
    if found != len(excluded):
        raise SystemExit(f"only {found} of the {len(excluded)} pilot texts are in the population; this is not "
                         "the data the pilot was drawn from, so nothing was drawn")
    key = with_lexicon(key, frame, pop)
    table = sheet(key, pop)
    blind = pilot.blind_spot(frame, period, hotels)

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "sheet.csv", index=False, encoding="utf-8-sig")
    key.to_csv(out / "key.csv", index=False)
    manifest = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": trends._sha256(PROTOCOL),
        "guideline_sha256": trends._sha256(GUIDELINE),
        "code_commit": pilot.git_commit(),
        "raw_csv_sha256": trends._sha256(csv),
        "reviews": source["reviews"],
        "seed": seed,
        "windows": {name: params[name] for name in ("base_start", "base_end", "recent_start", "recent_end",
                                                    "min_hotel_reviews")},
        "hotels_compared": len(hotels),
        "design": {"topic": TOPIC, "texts_per_period": size,
                   "left_out": {"source": source_of(exclude_key), "texts": len(excluded),
                                "raw_rows_sha256": hashlib.sha256(
                                    ",".join(map(str, sorted(excluded))).encode()).hexdigest()}},
        "periods": {name: {**sizes[name], **blind[name]} for name in pilot.PERIODS},
        "items": len(key),
        "sheet_columns": SHEET_COLUMNS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def finish(filled: Path, key_path: Path, manifest_path: Path, out: Path) -> dict:
    """Check the filled sheet and write what may be committed: no text, no notes."""
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    if {"text", "note"} & set(key.columns):
        raise SystemExit(f"{key_path} must not hold review text")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(filled, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "text" not in raw.columns:
        raise score.SubmissionError("the confirmation sheet: missing column 'text'")
    labels = score.check_sheet(raw, {int(item) for item in key["item"]}, "the confirmation sheet", (TOPIC,))
    drawn = dict(zip(key["item"].astype(int), key["text_sha256"], strict=True))
    changed = [int(item) for item, text in zip(raw["item"], raw["text"], strict=True)
               if text_hash(text) != drawn[int(item)]]
    if changed:
        listed = ", ".join(map(str, changed[:20])) + (" ..." if len(changed) > 20 else "")
        raise score.SubmissionError(f"the confirmation sheet: the text of {len(changed)} item(s) is not the one "
                                    f"drawn (items {listed}); nothing was recorded")
    out.mkdir(parents=True, exist_ok=True)
    table = labels.assign(annotator="final").rename_axis("item").reset_index()[["item", "annotator", TOPIC]]
    table.to_csv(out / "labels.csv", index=False)
    shutil.copyfile(key_path, out / "key.csv")
    shutil.copyfile(manifest_path, out / "sample_manifest.json")
    record = {"protocol": manifest["protocol"], "protocol_sha256": manifest["protocol_sha256"],
              "sample_code_commit": manifest["code_commit"], "topic": TOPIC, "items": len(labels),
              "labels": {value: int((labels[TOPIC].astype(str) == value).sum()) for value in ("1", "0", "unsure")},
              "sheet_sha256": score._sha256(filled)}
    (out / "sheet_record.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("draw", help="draw the texts to label")
    first.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    first.add_argument("--out", type=Path, default=ROOT / "runs" / "confirmation")
    first.add_argument("--seed", type=int, default=SEED, help="the protocol fixes 1")
    first.add_argument("--expect-hotels", type=int, help="stop unless this many hotels are compared (863)")
    first.add_argument("--exclude-key", type=Path, default=PILOT_KEY, help="the pilot's key: its texts are left out")
    first.add_argument("--overwrite", action="store_true", help="replace a sample already drawn in --out")
    second = steps.add_parser("finish", help="check the filled sheet and write what may be committed")
    second.add_argument("--sheet", type=Path, required=True, help="the filled sheet, as it came back")
    second.add_argument("--key", type=Path, default=ROOT / "runs" / "confirmation" / "key.csv")
    second.add_argument("--manifest", type=Path, default=ROOT / "runs" / "confirmation" / "manifest.json")
    second.add_argument("--out", type=Path, default=COMMITTED)
    args = parser.parse_args()
    try:
        if args.step == "draw":
            manifest = build_sample(args.csv, args.out, args.seed, args.expect_hotels, args.overwrite,
                                    args.exclude_key)
            for name in pilot.PERIODS:
                numbers = manifest["periods"][name]
                print(f"{name}: {numbers['population']:,} texts, {numbers['excluded_pilot_texts']} of the pilot's "
                      f"left out, {numbers['drawn']} drawn; set aside by the filter {numbers['excluded_by_filter']:,}")
            print(f"{manifest['items']} texts in sheet.csv -> {args.out}")
        else:
            record = finish(args.sheet, args.key, args.manifest, args.out)
            print(f"{record['items']} items, labels {record['labels']} -> {args.out}")
    except score.SubmissionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
