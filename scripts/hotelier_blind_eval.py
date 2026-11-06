"""The data steps of the hoteliers' blind evaluation (docs/experiments/hotelier_blind_eval/PROTOCOL.md, version 1).

    python scripts/hotelier_blind_eval.py draw
    python scripts/hotelier_blind_eval.py bundle --csv data/raw/booking_reviews_515k.csv

`draw` reads, from the triage student v2 protocol's committed files, both systems' outputs on its 500 test reviews:
the teacher's labels (labels/teacher_haiku_test.csv) and variant B's emissions (student/logits_B_test.csv at the
thresholds of student/thresholds_locked.json). It splits the reviews by their urgent outputs into D (they differ),
C (the same, not empty) and Z (neither emits an urgent type), keeps all of D and draws 11 of C (H 8, N 3; per
stratum sorted by raw_row, `DataFrame.sample(n, random_state=20261106)`), and numbers the 60 in a random order
(same seed). It writes samples/rated_key.csv (no text), samples/outputs.json (each item's types per system) and
samples/manifest.json (inputs by SHA-256, group sizes, the protocol's hash, the code commit).

`bundle` rebuilds the 60 texts from the raw CSV by raw_row, stops unless each matches its key's SHA-256, and writes
the Space's data file runs/hotelier_blind_eval/space/items.json (git-ignored: review text is never committed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import triage_student_v2_score as v2score  # noqa: E402

from reviewnlp.triage import diagnosis_metrics as metrics  # noqa: E402

v2 = v2score.v2
sample = v2.sample
FOLDER = ROOT / "docs" / "experiments" / "hotelier_blind_eval"
PROTOCOL = FOLDER / "PROTOCOL.md"
SAMPLES = FOLDER / "samples"
LABELS_EL = FOLDER / "labels_el.yaml"
RUNS = ROOT / "runs" / "hotelier_blind_eval"
V2_SAMPLES, V2_STUDENT, V2_LABELS = v2.SAMPLES, v2.FOLDER / "student", v2.LABELS
SEED = 20261106  # PROTOCOL.md §2: the draw of the 11 and the item order
C_SIZES = {"H": 8, "N": 3}  # PROTOCOL.md §2: 11 of C
VARIANT = "B"  # the v2 protocol's chosen variant


def test_reviews(samples: Path = V2_SAMPLES) -> pd.DataFrame:
    key = pd.read_csv(samples / "key.csv")
    return key[key["set"] == "test"].set_index("raw_row")


def outputs(samples: Path = V2_SAMPLES, student: Path = V2_STUDENT, labels: Path = V2_LABELS) -> pd.DataFrame:
    """Each test review's emitted types, per system, as tuples of type ids in catalogue order."""
    data = v2score.catalogue()
    types = [entry["id"] for entry in data["types"]]
    test = test_reviews(samples)
    record = json.loads((student / "thresholds_locked.json").read_text(encoding="utf-8"))
    thresholds = v2score.variant_thresholds(record, VARIANT, data)
    emitted = v2score.emit_types(v2score.first.load_indexed(student / f"logits_{VARIANT}_test.csv").loc[test.index],
                                 thresholds, types)
    teacher = v2score.first.load_indexed(labels / "teacher_haiku_test.csv").loc[test.index]
    if int(teacher["failed"].sum()):
        raise SystemExit("the teacher failed on a test review; the protocol assumes it did not")

    def as_tuples(table: pd.DataFrame) -> pd.Series:
        return table[types].apply(lambda row: tuple(t for t in types if row[t]), axis=1)
    return pd.DataFrame({"teacher": as_tuples(teacher), "student": as_tuples(emitted)}, index=test.index)


def groups(out: pd.DataFrame) -> pd.Series:
    """D where the urgent outputs differ, C where they are the same and not empty, Z where neither is urgent."""
    urgent = set(metrics.urgent_departments(v2score.catalogue()["types"]))
    teacher = out["teacher"].map(lambda ts: frozenset(t for t in ts if t in urgent))
    student = out["student"].map(lambda ts: frozenset(t for t in ts if t in urgent))
    group = pd.Series("Z", index=out.index)
    group[(teacher == student) & (teacher.map(len) > 0)] = "C"
    group[teacher != student] = "D"
    return group


def draw_rated(test: pd.DataFrame, group: pd.Series, seed: int = SEED) -> pd.DataFrame:
    """All of D and C_SIZES of C, numbered 1..n in a random order; each with its test and sampling weights."""
    frame = test.drop(columns=["item"], errors="ignore").assign(group=group).reset_index()  # v2's item numbers
    rated = [frame[frame["group"] == "D"].assign(sample_weight=1.0)]
    for stratum, n in C_SIZES.items():
        pool = frame[(frame["group"] == "C") & (frame["stratum"] == stratum)].sort_values("raw_row")
        if len(pool) < n:
            raise SystemExit(f"C {stratum}: {len(pool)} reviews, fewer than {n}")
        rated.append(pool.sample(n=n, random_state=seed).assign(sample_weight=len(pool) / n))
    table = pd.concat(rated, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    table.insert(0, "item", range(1, len(table) + 1))
    return table.rename(columns={"weight": "test_weight"})[
        ["item", "raw_row", "stratum", "group", "test_weight", "sample_weight", "text_sha256"]]


def inputs() -> dict[str, str]:
    paths = [V2_SAMPLES / "key.csv", V2_LABELS / "teacher_haiku_test.csv", V2_STUDENT / f"logits_{VARIANT}_test.csv",
             V2_STUDENT / "thresholds_locked.json", v2.CATALOGUE]
    return {p.relative_to(ROOT).as_posix(): sample.trends._sha256(p) for p in paths}


def build(out_dir: Path = SAMPLES, overwrite: bool = False) -> dict:
    """Draw the rated set and write its key, both outputs per item and the manifest; never redraw by accident."""
    if (out_dir / "rated_key.csv").exists() and not overwrite:
        raise SystemExit(f"{out_dir / 'rated_key.csv'} exists; a drawn set is never redrawn by accident")
    test = test_reviews()
    out = outputs()
    group = groups(out)
    rated = draw_rated(test, group)
    out_dir.mkdir(parents=True, exist_ok=True)
    rated.to_csv(out_dir / "rated_key.csv", index=False, lineterminator="\n")
    per_item = {str(row.item): {"teacher": list(out.loc[row.raw_row, "teacher"]),
                                "student": list(out.loc[row.raw_row, "student"])} for row in rated.itertuples()}
    (out_dir / "outputs.json").write_text(json.dumps(per_item, indent=1) + "\n", encoding="utf-8", newline="\n")
    sizes = {g: {"reviews": int((group == g).sum()),
                 "strata": {s: int(((group == g) & (test["stratum"] == s)).sum()) for s in sample.STRATA}}
             for g in ("D", "C", "Z")}
    manifest = {"protocol": PROTOCOL.relative_to(ROOT).as_posix(), "protocol_sha256": sample.trends._sha256(PROTOCOL),
                "code_commit": sample.pilot.git_commit(), "inputs": inputs(), "variant": VARIANT, "groups": sizes,
                "rated": {"D": int((rated["group"] == "D").sum()), "C": int((rated["group"] == "C").sum()),
                          "C_by_stratum": C_SIZES, "seed": SEED, "items": len(rated)},
                "rated_key_sha256": sample.trends._sha256(out_dir / "rated_key.csv"),
                "outputs_sha256": sample.trends._sha256(out_dir / "outputs.json")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def bundle(csv: Path, samples: Path = SAMPLES, runs: Path = RUNS) -> Path:
    """The Space's items.json: each item's text, rebuilt from the raw CSV and checked, and both outputs."""
    rated = pd.read_csv(samples / "rated_key.csv")
    per_item = json.loads((samples / "outputs.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(csv, usecols=["Negative_Review", "Positive_Review"]).fillna("").astype(str)
    rows = raw.loc[rated["raw_row"]]
    texts = [sample.whole_review(n, p) for n, p in zip(rows["Negative_Review"], rows["Positive_Review"], strict=True)]
    wrong = [int(item) for item, text, digest in zip(rated["item"], texts, rated["text_sha256"], strict=True)
             if sample.text_hash(text) != digest]
    if wrong:
        raise SystemExit(f"rebuilt texts differ from the key for items {wrong[:5]}; nothing written")
    items = [{"item": int(item), "text": text, **per_item[str(int(item))]}
             for item, text in zip(rated["item"], texts, strict=True)]
    folder = runs / "space"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "items.json"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("draw", help="draw the rated set and write its key, outputs and manifest")
    first.add_argument("--overwrite", action="store_true")
    second = steps.add_parser("bundle", help="write the Space's items.json with the texts (git-ignored)")
    second.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    args = parser.parse_args()
    if args.step == "draw":
        manifest = build(overwrite=args.overwrite)
        print("groups:", {g: v["reviews"] for g, v in manifest["groups"].items()}, "| rated:", manifest["rated"])
    else:
        if sample.trends._sha256(args.csv) != sample.RAW_SHA256:
            raise SystemExit(f"{args.csv} is not the Kaggle file the protocol fixes")
        print(bundle(args.csv))


if __name__ == "__main__":
    main()
