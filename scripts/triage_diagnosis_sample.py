"""Draw the development and test samples of the triage diagnosis protocol, and write the blind sheet.

The design is fixed in docs/experiments/triage_diagnosis/PROTOCOL.md (version 1, section 2), and this
script follows it: whole reviews from the raw Booking CSV, the owner's 700 labelled texts and duplicate
whole-review texts left out, hash pools 80-89 (development) and 90-99 (test), and within each pool a
stratified draw (H, N, B) with the protocol's sizes and seeds. The trigger-word regex is read from
planning/population.py, the copy the lock test pins.

    python scripts/triage_diagnosis_sample.py draw
    python scripts/triage_diagnosis_sample.py draw-check

It writes into --out (default runs/triage_diagnosis, git-ignored because the sheet holds review text):

    sheet.csv       the 800 reviews in a random order: item, text, one empty column per catalogue type,
                    `none`, `done` and `note`; no set, stratum, weight or model output

and into --record (default docs/experiments/triage_diagnosis/samples), what may be committed, with no text:

    key.csv         item -> set, raw_row, stratum, weight, pool and a hash of the text
    manifest.json   population and stratum sizes per pool, the exclusions, the seeds, the code commit, and
                    the SHA-256 of the protocol, the catalogue, the raw CSV and the blank sheet

`draw-check` draws version 2's check sample: 100 test-pool reviews outside the 500 test reviews, stratified the
same way, for the owner's own labels. It first redraws both samples and stops unless they match the committed
key, so the 500 it leaves out are exactly the recorded ones. It writes check_sheet.csv into --out, and
check_key.csv and check_manifest.json into --record.

No model, provider call or label is involved. A drawn sample is never redrawn by accident.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import annotation_sample as pilot  # noqa: E402

from reviewnlp.data.integrity import canonical_text  # noqa: E402

trends = pilot.trends
FOLDER = ROOT / "docs" / "experiments" / "triage_diagnosis"
PROTOCOL = FOLDER / "PROTOCOL.md"
CATALOGUE = FOLDER / "catalogue_v1.yaml"
POPULATION = FOLDER / "planning" / "population.py"
RAW_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"
SEEN_KEYS = (ROOT / "docs" / "case_study" / "results" / "annotation" / "key.csv",
             ROOT / "docs" / "experiments" / "jev_topic_benchmark" / "confirmation" / "key.csv")
STRATA = ("H", "N", "B")
# (set, pools, seed, sizes per stratum): PROTOCOL.md, "Pools" and "Samples".
SAMPLES = (("development", range(80, 90), 20261021, {"H": 90, "N": 180, "B": 30}),
           ("test", range(90, 100), 20261022, {"H": 150, "N": 300, "B": 50}))
# PROTOCOL.md version 2, "Samples": the check sample, from the test pool, outside the 500 test reviews.
CHECK = ("check", range(90, 100), 20261024, {"H": 30, "N": 60, "B": 10})
# The strata of the whole population, before any exclusion: PROTOCOL.md, "Strata".
EXPECTED_STRATA = {"H": 63325, "N": 285436, "B": 166412}
NEGATIVE_PLACEHOLDER, POSITIVE_PLACEHOLDER = "No Negative", "No Positive"


def trigger_regex(path: Path = POPULATION) -> re.Pattern[str]:
    """The stratifier's regex, read from planning/population.py without running that script."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    (call,) = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(getattr(target, "id", None) == "TRIGGER" for target in node.targets)]
    assert isinstance(call, ast.Call)
    return re.compile(ast.literal_eval(call.args[0]), re.I)


def catalogue_types(path: Path = CATALOGUE) -> list[str]:
    return [entry["id"] for entry in yaml.safe_load(path.read_text(encoding="utf-8"))["types"]]


def whole_review(negative: str, positive: str) -> str:
    """The string the annotator and the model read: placeholders and empty fields are left out."""
    parts = []
    for prefix, field, placeholder in (("Disliked", negative, NEGATIVE_PLACEHOLDER),
                                       ("Liked", positive, POSITIVE_PLACEHOLDER)):
        field = " ".join(str(field).split())
        if field and field != placeholder:
            parts.append(f"{prefix}: {field}")
    return "\n\n".join(parts)


def pool_of(text: str) -> int:
    """0-99 from the canonical text, so no text can be in two pools."""
    return int(hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()[:8], 16) % 100


def text_hash(text: str) -> str:
    """SHA-256 of a text with its white space collapsed, so a spreadsheet's line ends do not matter."""
    return hashlib.sha256(" ".join(str(text).split()).encode("utf-8")).hexdigest()


def population(frame: pd.DataFrame, positive: pd.Series, trigger: re.Pattern[str],
               excluded: set[int]) -> tuple[pd.DataFrame, dict]:
    """One row per eligible review with its text, stratum and pool, and the counts behind it.

    `frame` is complaint_trends.load_reviews' output; `positive` is the raw positive field by raw_row.
    """
    complaint = frame["has_complaint_text"]
    hit = complaint & frame["negative"].map(lambda text: bool(trigger.search(text)))
    stratum = pd.Series("B", index=frame.index).mask(complaint, "N").mask(hit, "H")
    pop = pd.DataFrame({"raw_row": frame["raw_row"].astype(int), "stratum": stratum,
                        "text": [whole_review(neg, pos) for neg, pos in
                                 zip(frame["negative"], positive.loc[frame["raw_row"]].to_numpy(), strict=True)]})
    counts = {"reviews": len(pop), "strata": {name: int((pop["stratum"] == name).sum()) for name in STRATA}}
    seen = pop["raw_row"].isin(excluded)
    counts["excluded_owner_labelled"] = int(seen.sum())
    pop = pop[~seen].sort_values("raw_row")
    duplicate = pop["text"].map(canonical_text).duplicated(keep="first")
    counts["excluded_duplicate_text"] = int(duplicate.sum())
    pop = pop[~duplicate].reset_index(drop=True)
    pop["pool"] = pop["text"].map(pool_of)
    counts["eligible"] = len(pop)
    return pop, counts


def stratified(in_pool: pd.DataFrame, name: str, pools: range, seed: int,
               wanted: dict[str, int]) -> tuple[list[pd.DataFrame], dict]:
    """One stratified draw: per stratum, sorted by raw_row, `DataFrame.sample(n, random_state=seed)`."""
    drawn = []
    sizes = {"pools": [pools.start, pools.stop - 1], "seed": seed, "reviews_in_pool": len(in_pool), "strata": {}}
    for stratum in STRATA:
        group = in_pool[in_pool["stratum"] == stratum].sort_values("raw_row")
        n = wanted[stratum]
        if len(group) < n:
            raise ValueError(f"{name} {stratum}: {len(group)} reviews in the pool, fewer than {n}")
        picked = group.sample(n=n, random_state=seed)
        drawn.append(picked.assign(set=name, weight=len(group) / n))
        sizes["strata"][stratum] = {"N_s": len(group), "n_s": n, "weight": len(group) / n}
    return drawn, sizes


def draw(pop: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The development and test samples, each review with its stratum weight N_s / n_s within its pool."""
    drawn, sizes = [], {}
    for name, pools, seed, wanted in SAMPLES:
        parts, sizes[name] = stratified(pop[pop["pool"].isin(pools)], name, pools, seed, wanted)
        drawn += parts
    return pd.concat(drawn, ignore_index=True), sizes


def draw_check(pop: pd.DataFrame, taken: set[int]) -> tuple[pd.DataFrame, dict]:
    """The check sample: the test pool without the reviews already drawn, N_s counted after their removal."""
    name, pools, seed, wanted = CHECK
    in_pool = pop[pop["pool"].isin(pools) & ~pop["raw_row"].isin(taken)]
    parts, sizes = stratified(in_pool, name, pools, seed, wanted)
    sizes["left_out_already_drawn"] = int((pop["pool"].isin(pools) & pop["raw_row"].isin(taken)).sum())
    return pd.concat(parts, ignore_index=True), sizes


def number(sample: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Items 1..n in a random order, so neither the set nor the stratum can be read from the position."""
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "item", range(1, len(sample) + 1))
    return sample


def sheet(sample: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """Item and text, one empty column per type, `none` for a review with no problem, `done` and `note`."""
    table = pd.DataFrame({"item": sample["item"], "text": sample["text"]})
    for column in [*types, "none", "done", "note"]:
        table[column] = ""
    return table


def key(sample: pd.DataFrame) -> pd.DataFrame:
    table = sample[["item", "set", "raw_row", "stratum", "weight", "pool"]].copy()
    table["text_sha256"] = sample["text"].map(text_hash)
    return table


def build(csv: Path, out: Path, record: Path, overwrite: bool = False, check_population: bool = True) -> dict:
    """Draw both samples, write the sheet, the key and the manifest, and return the manifest."""
    if (out / "sheet.csv").exists() and not overwrite:
        raise SystemExit(f"{out / 'sheet.csv'} exists; a drawn sample is never redrawn by accident "
                         "(pass --overwrite to replace it)")
    raw_sha = trends._sha256(csv)
    if check_population and raw_sha != RAW_SHA256:
        raise SystemExit(f"{csv} is not the Kaggle file the protocol fixes (SHA-256 {raw_sha})")
    frame, source = trends.load_reviews(csv)
    positive = pd.read_csv(csv, usecols=["Positive_Review"])["Positive_Review"].fillna("").astype(str)
    excluded: set[int] = set()
    for path in SEEN_KEYS:
        excluded |= pilot_rows(path)
    pop, counts = population(frame, positive, trigger_regex(), excluded)
    if check_population:
        if counts["strata"] != EXPECTED_STRATA:
            raise SystemExit(f"strata {counts['strata']}, the protocol records {EXPECTED_STRATA}; nothing drawn")
        if counts["excluded_owner_labelled"] != len(excluded):
            raise SystemExit(f"only {counts['excluded_owner_labelled']} of the {len(excluded)} owner-labelled "
                             "texts are in the data; nothing drawn")
    sample, sizes = draw(pop)
    sample = number(sample, SAMPLES[-1][2])
    types = catalogue_types()
    table = sheet(sample, types)

    out.mkdir(parents=True, exist_ok=True)
    record.mkdir(parents=True, exist_ok=True)
    # LF everywhere, so the recorded sheet hash does not depend on the system that drew it.
    table.to_csv(out / "sheet.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    key(sample).to_csv(record / "key.csv", index=False, lineterminator="\n")
    manifest = {
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": trends._sha256(PROTOCOL),
        "catalogue_sha256": trends._sha256(CATALOGUE),
        "population_script_sha256": trends._sha256(POPULATION),
        "code_commit": pilot.git_commit(),
        "raw_csv_sha256": raw_sha,
        "duplicate_rows_dropped": source["duplicate_rows_dropped"],
        **counts,
        "excluded_owner_labelled_raw_rows_sha256": hashlib.sha256(
            ",".join(map(str, sorted(excluded))).encode()).hexdigest(),
        "exclusion_order": "owner-labelled raw_rows first, then duplicate whole-review texts (earliest raw_row kept)",
        "samples": sizes,
        "item_order_seed": SAMPLES[-1][2],
        "items": len(sample),
        "text_hash": "SHA-256 of the whole-review text with its white space collapsed",
        "sheet_columns": list(table.columns),
        "sheet_sha256": trends._sha256(out / "sheet.csv"),
    }
    (record / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def build_check(csv: Path, out: Path, record: Path, overwrite: bool = False, check_population: bool = True) -> dict:
    """Redraw both samples, stop unless they match the committed key, then draw and write the check sample."""
    if (out / "check_sheet.csv").exists() and not overwrite:
        raise SystemExit(f"{out / 'check_sheet.csv'} exists; a drawn sample is never redrawn by accident "
                         "(pass --overwrite to replace it)")
    raw_sha = trends._sha256(csv)
    if check_population and raw_sha != RAW_SHA256:
        raise SystemExit(f"{csv} is not the Kaggle file the protocol fixes (SHA-256 {raw_sha})")
    frame, _ = trends.load_reviews(csv)
    positive = pd.read_csv(csv, usecols=["Positive_Review"])["Positive_Review"].fillna("").astype(str)
    excluded: set[int] = set()
    for path in SEEN_KEYS:
        excluded |= pilot_rows(path)
    pop, counts = population(frame, positive, trigger_regex(), excluded)
    redrawn = key(number(draw(pop)[0], SAMPLES[-1][2])).to_csv(index=False, lineterminator="\n")
    if redrawn != (record / "key.csv").read_text(encoding="utf-8"):
        raise SystemExit(f"redrawing the samples does not give {record / 'key.csv'} byte for byte; nothing drawn")
    committed = pd.read_csv(record / "key.csv")
    taken = set(committed["raw_row"].astype(int))
    check, sizes = draw_check(pop, taken)
    check = number(check, CHECK[2])
    table = sheet(check, catalogue_types())
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "check_sheet.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    key(check).to_csv(record / "check_key.csv", index=False, lineterminator="\n")
    manifest = {
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": trends._sha256(PROTOCOL),
        "catalogue_sha256": trends._sha256(CATALOGUE),
        "code_commit": pilot.git_commit(),
        "raw_csv_sha256": raw_sha,
        "eligible": counts["eligible"],
        "samples_redrawn_and_matched": "samples/key.csv",
        "check": sizes,
        "item_order_seed": CHECK[2],
        "items": len(check),
        "text_hash": "SHA-256 of the whole-review text with its white space collapsed",
        "sheet_columns": list(table.columns),
        "sheet_sha256": trends._sha256(out / "check_sheet.csv"),
    }
    (record / "check_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                                                newline="\n")
    return manifest


def pilot_rows(path: Path) -> set[int]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" in table.columns:
        raise SystemExit(f"{path} must not hold review text")
    return {int(row) for row in table["raw_row"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("draw", help="draw both samples and write the blind sheet")
    first.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    first.add_argument("--out", type=Path, default=ROOT / "runs" / "triage_diagnosis")
    first.add_argument("--record", type=Path, default=FOLDER / "samples")
    first.add_argument("--overwrite", action="store_true", help="replace a sample already drawn in --out")
    check = steps.add_parser("draw-check", help="draw version 2's check sample and write its blind sheet")
    check.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    check.add_argument("--out", type=Path, default=ROOT / "runs" / "triage_diagnosis")
    check.add_argument("--record", type=Path, default=FOLDER / "samples")
    check.add_argument("--overwrite", action="store_true", help="replace a check sample already drawn in --out")
    args = parser.parse_args()
    if args.step == "draw-check":
        manifest = build_check(args.csv, args.out, args.record, args.overwrite)
        numbers = manifest["check"]
        strata = ", ".join(f"{s} {v['n_s']} of {v['N_s']:,}" for s, v in numbers["strata"].items())
        print(f"check: test pool without {numbers['left_out_already_drawn']} drawn reviews, "
              f"{numbers['reviews_in_pool']:,} left; drawn {strata}")
        print(f"{manifest['items']} reviews in {args.out / 'check_sheet.csv'}; key and manifest in {args.record}")
        return
    manifest = build(args.csv, args.out, args.record, args.overwrite)
    for name, numbers in manifest["samples"].items():
        strata = ", ".join(f"{s} {v['n_s']} of {v['N_s']:,}" for s, v in numbers["strata"].items())
        print(f"{name}: pools {numbers['pools']}, {numbers['reviews_in_pool']:,} reviews; drawn {strata}")
    print(f"{manifest['items']} reviews in {args.out / 'sheet.csv'}; key and manifest in {args.record}")


if __name__ == "__main__":
    main()
