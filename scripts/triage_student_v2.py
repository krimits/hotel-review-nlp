"""The data steps of the triage student v2 protocol (docs/experiments/triage_student_v2/PROTOCOL.md, version 1).

    python scripts/triage_student_v2.py prompts
    python scripts/triage_student_v2.py draw --csv data/raw/booking_reviews_515k.csv
    python scripts/triage_student_v2.py batches
    set OPENROUTER_API_KEY=...       the owner's key, with a spending limit set on OpenRouter
    python scripts/triage_student_v2.py teacher --subset development --max-usd 1
    python scripts/triage_student_v2.py teacher --subset test --max-usd 1
    python scripts/triage_student_v2.py teacher --subset teacher --max-usd 4
    python scripts/triage_student_v2.py reference --source "ChatGPT, model shown as ..., reruns: ..."

`prompts` writes reference_prompt_v2.md and teacher_prompt_v2.md from catalogue_v2 with the first protocol's
code, so the definitions reach both labellers word for word.

`draw` redraws the first protocol's development, test, check and teacher samples and stops unless each matches
its committed key byte for byte. Then, as section 2 fixes them:

- development: the first protocol's 300 and 300 new from pools 80-89 outside them (H 90, N 180, B 30, seed
  20261103). Each of the 600 weighs N_s / n_s, with N_s the stratum's size in the pool and n_s counting old and
  new together (H 180, N 360, B 60). Items 1-600, in a random order (seed 20261103).
- test: 500 new from pools 90-99, outside the first protocol's 500 test and 100 check reviews (H 150, N 300,
  B 50, seed 20261102), N_s counted after their removal. Items 601-1,100, in a random order (seed 20261102).
- teacher: 30,000 from pools 0-79 (H 12,000, N 15,000, B 3,000, seed 20261104), items 1-30,000 in a random
  order (same seed). A review is held out when its pool number ends in 0, which is `int(sha256(text)[:8], 16) %
  10 == 0` on the canonical text the pools hash.

It writes the texts only under runs/triage_student_v2 (git-ignored): sheet.csv (the blank sheet of the 1,100) and
teacher_sample.csv. Under samples/ it writes key.csv, teacher_key.csv and manifest.json, with no text.

`batches` splits the blank sheet into the 22 ChatGPT batches of 50 in item order (12 development, 10 test) and
copies the prompt beside them; samples/reference_batches.json records each batch's items and SHA-256.

`teacher` runs the first protocol's teacher code (anthropic/claude-haiku-5.5-20261007, 25 reviews per call, two
retries, every response hashed) with this protocol's catalogue, prompt and samples. The API key is read from the
environment, sent only to OpenRouter, and never written or printed.

`reference` records the 22 returned ChatGPT batches (runs/triage_student_v2/reference/batch_*_filled.csv) as
labels/reference_chatgpt_v2.csv, by raw_row and with no text, through the first protocol's checks with catalogue_v2:
every drawn item once with its drawn text, labels only 0 or 1, a type or `none` on every row, `done` on every row.
A file the checks refuse is not recorded; its batch is rerun. `--source` records the model as ChatGPT showed it and
any reruns, as the owner reports them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import triage_diagnosis_labels as labels  # noqa: E402
import triage_diagnosis_teacher as teacher  # noqa: E402

sample = labels.sample
FOLDER = ROOT / "docs" / "experiments" / "triage_student_v2"
PROTOCOL = FOLDER / "PROTOCOL.md"
CATALOGUE = sample.FOLDER / "catalogue_v2.yaml"
REFERENCE_PROMPT = FOLDER / "reference_prompt_v2.md"
TEACHER_PROMPT = FOLDER / "teacher_prompt_v2.md"
SAMPLES = FOLDER / "samples"
LABELS = FOLDER / "labels"
RUNS = ROOT / "runs" / "triage_student_v2"
FIRST = sample.FOLDER / "samples"  # the first protocol's committed keys
# (set, pools, seed, sizes per stratum): PROTOCOL.md section 2, "Samples".
DEVELOPMENT = ("development", range(80, 90), 20261103, {"H": 90, "N": 180, "B": 30})
TEST = ("test", range(90, 100), 20261102, {"H": 150, "N": 300, "B": 50})
TEACHER = ("teacher", range(0, 80), 20261104, {"H": 12000, "N": 15000, "B": 3000})
BATCH_SIZE = 50
BATCHES = {"development": 12, "test": 10}
TEACHER_MODEL = "haiku"  # the first protocol's teacher, anthropic/claude-haiku-5.5-20261007; no new choice
SUBSETS = {  # subset -> (sheet under runs/, key under samples/, set to keep or None for all)
    "development": ("sheet.csv", "key.csv", "development"),
    "test": ("sheet.csv", "key.csv", "test"),
    "teacher": ("teacher_sample.csv", "teacher_key.csv", None),
}
SETUP = teacher.Setup(CATALOGUE, TEACHER_PROMPT, PROTOCOL, SUBSETS)
KINDS = {"reference": ("key.csv", "reference_chatgpt_v2")}  # kind -> (key, labels name)


def write_prompts() -> dict[str, str]:
    """Both prompts from catalogue_v2, by the code that wrote the first protocol's."""
    written = {}
    for path, text in ((REFERENCE_PROMPT, labels.reference_prompt(CATALOGUE)),
                       (TEACHER_PROMPT, labels.teacher_prompt(CATALOGUE))):
        path.write_text(text, encoding="utf-8", newline="\n")
        written[path.name] = sample.trends._sha256(path)
    return written


def held_out(text: str) -> bool:
    """The teacher sample's held-out tenth: the pool hash, so a review is held out when its pool ends in 0."""
    return sample.pool_of(text) % 10 == 0


def first_keys_match(pop: pd.DataFrame, first: Path = FIRST) -> dict[str, str]:
    """Redraw the first protocol's check and teacher samples and stop unless both match their committed keys.

    `sample.verified_population` has already matched its development and test key.
    """
    committed = pd.read_csv(first / "key.csv")
    check, _ = sample.draw_check(pop, set(committed["raw_row"].astype(int)))
    name, pools, seed, wanted = sample.TEACHER
    parts, _ = sample.stratified(pop[pop["pool"].isin(pools)], name, pools, seed, wanted)
    redrawn = {"check_key.csv": sample.key(sample.number(check, sample.CHECK[2])),
               "teacher_key.csv": sample.key(sample.number(pd.concat(parts, ignore_index=True), seed))}
    for file, table in redrawn.items():
        if table.to_csv(index=False, lineterminator="\n") != (first / file).read_text(encoding="utf-8"):
            raise SystemExit(f"redrawing the first protocol's samples does not give {first / file}; nothing drawn")
    return {"key.csv": sample.trends._sha256(first / "key.csv"),
            **{file: sample.trends._sha256(first / file) for file in redrawn}}


def draw_samples(pop: pd.DataFrame, first_key: pd.DataFrame, first_check: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The 1,100 (development then test, items 1-1,100) and the 30,000, with their sizes."""
    sizes: dict = {}
    name, pools, seed, wanted = DEVELOPMENT
    dev_pool = pop[pop["pool"].isin(pools)]
    old_rows = set(first_key.loc[first_key["set"] == "development", "raw_row"].astype(int))
    old = dev_pool[dev_pool["raw_row"].isin(old_rows)]
    if len(old) != len(old_rows):
        raise SystemExit(f"only {len(old)} of the first protocol's {len(old_rows)} development reviews are in the pool")
    parts, sizes["development_new"] = sample.stratified(dev_pool[~dev_pool["raw_row"].isin(old_rows)], name, pools,
                                                        seed, wanted)
    development = pd.concat([old.assign(set=name, first_protocol=1),
                             *[part.assign(first_protocol=0) for part in parts]], ignore_index=True)
    sizes["development"] = {"pools": [pools.start, pools.stop - 1], "reviews_in_pool": len(dev_pool), "strata": {}}
    for stratum in sample.STRATA:
        in_stratum = development["stratum"] == stratum
        big_n, small_n = int((dev_pool["stratum"] == stratum).sum()), int(in_stratum.sum())
        if small_n != 2 * wanted[stratum]:
            raise SystemExit(f"development {stratum}: {small_n} reviews, expected {2 * wanted[stratum]}")
        development.loc[in_stratum, "weight"] = big_n / small_n
        sizes["development"]["strata"][stratum] = {"N_s": big_n, "n_s": small_n, "weight": big_n / small_n}
    development = sample.number(development, seed)

    name, pools, seed, wanted = TEST
    taken = (set(first_key.loc[first_key["set"] == "test", "raw_row"].astype(int))
             | set(first_check["raw_row"].astype(int)))
    test_pool = pop[pop["pool"].isin(pools)]
    parts, sizes["test"] = sample.stratified(test_pool[~test_pool["raw_row"].isin(taken)], name, pools, seed, wanted)
    sizes["test"]["left_out_already_drawn"] = int(test_pool["raw_row"].isin(taken).sum())
    test = sample.number(pd.concat(parts, ignore_index=True).assign(first_protocol=0), seed)
    test["item"] += len(development)
    both = pd.concat([development, test], ignore_index=True)

    name, pools, seed, wanted = TEACHER
    parts, sizes["teacher"] = sample.stratified(pop[pop["pool"].isin(pools)], name, pools, seed, wanted)
    drawn = sample.number(pd.concat(parts, ignore_index=True), seed)
    drawn["held_out"] = drawn["text"].map(held_out).astype(int)
    return both, drawn, sizes


def build(csv: Path, runs: Path = RUNS, record: Path = SAMPLES, first: Path = FIRST, overwrite: bool = False,
          check_population: bool = True) -> dict:
    """Draw the samples and write the sheet, the teacher sample, the keys and the manifest."""
    if (runs / "sheet.csv").exists() and not overwrite:
        raise SystemExit(f"{runs / 'sheet.csv'} exists; a drawn sample is never redrawn by accident "
                         "(pass --overwrite to replace it)")
    pop, counts, raw_sha = sample.verified_population(csv, first, check_population)
    matched = first_keys_match(pop, first)
    both, drawn, sizes = draw_samples(pop, pd.read_csv(first / "key.csv"), pd.read_csv(first / "check_key.csv"))
    types = sample.catalogue_types(CATALOGUE)
    table = sample.sheet(both, types)
    runs.mkdir(parents=True, exist_ok=True)
    record.mkdir(parents=True, exist_ok=True)
    table.to_csv(runs / "sheet.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    drawn[["item", "text"]].to_csv(runs / "teacher_sample.csv", index=False, encoding="utf-8-sig",
                                   lineterminator="\n")
    sample.key(both).assign(first_protocol=both["first_protocol"].astype(int)).to_csv(
        record / "key.csv", index=False, lineterminator="\n")
    sample.key(drawn).assign(held_out=drawn["held_out"]).to_csv(record / "teacher_key.csv", index=False,
                                                                lineterminator="\n")
    manifest = {
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": sample.trends._sha256(PROTOCOL),
        "catalogue_sha256": sample.trends._sha256(CATALOGUE),
        "population_script_sha256": sample.trends._sha256(sample.POPULATION),
        "code_commit": sample.pilot.git_commit(),
        "raw_csv_sha256": raw_sha,
        "eligible": counts["eligible"],
        "first_protocol_keys_redrawn_and_matched": matched,
        "samples": sizes,
        "items": {"development": [1, 600], "test": [601, 1100], "teacher": [1, len(drawn)]},
        "item_order_seeds": {"development": DEVELOPMENT[2], "test": TEST[2], "teacher": TEACHER[2]},
        "teacher_held_out": int(drawn["held_out"].sum()),
        "teacher_held_out_rule": "pool number ends in 0: int(sha256(canonical text)[:8], 16) % 10 == 0",
        "text_hash": "SHA-256 of the whole-review text with its white space collapsed",
        "sheet_columns": list(table.columns),
        "sheet_sha256": sample.trends._sha256(runs / "sheet.csv"),
        "teacher_sample_sha256": sample.trends._sha256(runs / "teacher_sample.csv"),
    }
    (record / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def write_batches(runs: Path = RUNS, samples: Path = SAMPLES, prompt: Path = REFERENCE_PROMPT) -> dict:
    """The 22 ChatGPT batches of 50, in item order, with the prompt beside them."""
    if prompt.read_text(encoding="utf-8") != labels.reference_prompt(CATALOGUE):
        raise SystemExit(f"{prompt} does not match catalogue_v2; run `prompts` first")
    folder = runs / "reference"
    batches = labels.split_sheet(runs / "sheet.csv", samples / "key.csv", folder, "batch_", BATCH_SIZE)
    ends = [batch["items"][1] for batch in batches.values()]
    last_development = BATCHES["development"] * BATCH_SIZE
    if len(ends) != sum(BATCHES.values()) or ends[BATCHES["development"] - 1] != last_development:
        raise SystemExit(f"expected {sum(BATCHES.values())} batches, the first {BATCHES['development']} ending at "
                         f"item {last_development}")
    shutil.copyfile(prompt, folder / prompt.name)
    record = {"prompt": prompt.name, "prompt_sha256": sample.trends._sha256(prompt),
              "blank_sheet_sha256": sample.trends._sha256(runs / "sheet.csv"), "batch_size": BATCH_SIZE,
              "development_batches": [1, BATCHES["development"]],
              "test_batches": [BATCHES["development"] + 1, sum(BATCHES.values())], "batches": batches}
    (samples / "reference_batches.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8",
                                                    newline="\n")
    return record


def run_teacher(subset: str, max_usd: float, **kwargs) -> dict:
    """The fixed teacher on one of this protocol's subsets."""
    return teacher.label(TEACHER_MODEL, subset, max_usd, runs=RUNS, samples=SAMPLES, labels_dir=LABELS,
                         setup=SETUP, **kwargs)


def record_reference(source: str, returned: Path | None = None, out: Path | None = None,
                     samples: Path | None = None, received_on: dt.date | None = None) -> dict:
    """`reference_chatgpt_v2` from the 22 returned batches, through the first protocol's checks with catalogue_v2."""
    if not source.strip():
        raise SystemExit("--source is required: the model as ChatGPT showed it, and any reruns")
    returned = returned or RUNS / "reference"
    files = sorted(returned.glob("batch_*_filled.csv"))
    if len(files) != sum(BATCHES.values()):
        raise SystemExit(f"{len(files)} returned batches in {returned}; the protocol has {sum(BATCHES.values())}")
    try:
        return labels.finish(returned, "reference", out or LABELS, samples or SAMPLES, received_on, source,
                             catalogue=CATALOGUE, protocol=PROTOCOL, kinds=KINDS)
    except labels.SheetError as error:
        raise SystemExit(str(error)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("prompts", help="write reference_prompt_v2.md and teacher_prompt_v2.md from catalogue_v2")
    first = steps.add_parser("draw", help="draw the development, test and teacher samples")
    first.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    first.add_argument("--overwrite", action="store_true")
    steps.add_parser("batches", help="split the blank sheet into the 22 ChatGPT batches")
    second = steps.add_parser("teacher", help="label a subset with the fixed teacher through OpenRouter")
    second.add_argument("--subset", choices=sorted(SUBSETS), required=True)
    second.add_argument("--max-usd", type=float, required=True, help="stop before a call once this much is spent")
    third = steps.add_parser("reference", help="record the 22 returned ChatGPT batches as reference_chatgpt_v2")
    third.add_argument("--source", required=True, help="the model as ChatGPT showed it, and any reruns")
    args = parser.parse_args()
    if args.step == "prompts":
        for name, digest in write_prompts().items():
            print(f"{name} ({digest})")
    elif args.step == "draw":
        manifest = build(args.csv, overwrite=args.overwrite)
        print(f"drawn: items {manifest['items']}, held out {manifest['teacher_held_out']} of the teacher sample")
    elif args.step == "batches":
        record = write_batches()
        print(f"{len(record['batches'])} batches in {RUNS / 'reference'}")
    elif args.step == "reference":
        record = record_reference(args.source)
        print(f"{record['labels']}: {record['reviews']} reviews {record['reviews_by_set']}, "
              f"{record['reviews_with_a_type']} with a type, {record['reviews_with_none']} none")
    else:
        try:
            record = run_teacher(args.subset, args.max_usd)
        except teacher.Stop as error:
            raise SystemExit(str(error)) from None
        print(f"{record['labels']}: {record['reviews']} reviews, {record['failed_reviews']} failed, "
              f"{record['calls']} calls ({record['retries']} retries), ${record['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
