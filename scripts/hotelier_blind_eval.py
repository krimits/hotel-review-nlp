"""The data steps of the hoteliers' blind evaluation (docs/experiments/hotelier_blind_eval/PROTOCOL.md, version 1).

    python scripts/hotelier_blind_eval.py draw
    python scripts/hotelier_blind_eval.py bundle --csv data/raw/booking_reviews_515k.csv
    hf download krimits/hotelier-blind-eval-ratings --repo-type dataset \
        --local-dir runs/hotelier_blind_eval/ratings_export          # the owner, when the collection is closed
    python scripts/hotelier_blind_eval.py record --source "who rated, as the owner reports it"
    git add ... && git commit ...                                    # the record is committed before the rule
    python scripts/hotelier_blind_eval.py analyse                    # once

`draw` reads, from the triage student v2 protocol's committed files, both systems' outputs on its 500 test reviews:
the teacher's labels (labels/teacher_haiku_test.csv) and variant B's emissions (student/logits_B_test.csv at the
thresholds of student/thresholds_locked.json). It splits the reviews by their urgent outputs into D (they differ),
C (the same, not empty) and Z (neither emits an urgent type), keeps all of D and draws 11 of C (H 8, N 3; per
stratum sorted by raw_row, `DataFrame.sample(n, random_state=20261106)`), and numbers the 60 in a random order
(same seed). It writes samples/rated_key.csv (no text), samples/outputs.json (each item's types per system) and
samples/manifest.json (inputs by SHA-256, group sizes, the protocol's hash, the code commit).

`bundle` rebuilds the 60 texts from the raw CSV by raw_row, stops unless each matches its key's SHA-256, and writes
the Space's folder runs/hotelier_blind_eval/space/ (git-ignored: review text is never committed): the app files
from spaces/hotelier-blind-eval/, items.json (each item's text and both outputs), display.json (catalogue_v2 through
labels_el.yaml) and meta.json (the protocol's hash, the code commit, and the hashes of what it was built from).
scripts/deploy_hotelier_space.py publishes that folder.

`record` checks the owner's export of the ratings dataset: every file must be one the Space wrote for that hotelier
and item under this protocol (the sides the Space shows them, complete and consistent answers, one bundle); one
problem refuses the whole export. It writes ratings/record.json: who rated as the owner reports it, counts, times
and one SHA-256 over every rating file. The rating files themselves are not committed (the owner's decision of 11
November 2026: the hoteliers were told their answers are kept in a private dataset); they stay in that dataset and
in the owner's export. One digest, not one hash per file: a rating file is small enough for its own hash to be
searched for its answers.

`analyse` applies §4 once to the exported ratings (one JSON file per hotelier and review, the systems named x and y
as in the Space), after the record is committed and only if the export still has the record's digest: `d` over U
with its bootstrap interval, `p_teacher`, the outcome, and the reported items. It writes results/result.json once
and never again; it holds aggregates, no single answer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

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


def display_table(labels_path: Path = LABELS_EL) -> dict:
    """What the Space shows for each type: catalogue_v2's priority, department and deadline, in the Greek table's
    words. Stops unless the table covers every type, department and deadline the catalogue uses."""
    data = v2score.catalogue()
    labels = yaml.safe_load(labels_path.read_text(encoding="utf-8"))
    if labels["catalogue"] != data["version"]:
        raise SystemExit(f"{labels_path.name} is for {labels['catalogue']}, not {data['version']}")
    table = {}
    for entry in data["types"]:
        t = entry["id"]
        words = labels["types"].get(t)
        if words is None or (entry["priority"] != "structural" and not words.get("measure")):
            raise SystemExit(f"{labels_path.name} has no label or measure for {t}")
        table[t] = {"label": words["label"], "priority": entry["priority"],
                    "priority_el": labels["priorities"][entry["priority"]],
                    "department_el": labels["departments"][entry["department"]],
                    "measure": words.get("measure") or labels["structural_measure"],
                    "deadline_el": labels["deadlines"][entry["deadline"]] if entry.get("deadline") else None}
    return {"order": [entry["id"] for entry in data["types"]], "types": table}


SPACE_FILES = ("README.md", "app.py", "logic.py", "requirements.txt")
# The Space names the systems only x and y, so its public files cannot unblind a hotelier; the analysis maps them back.
NEUTRAL = {"x": "teacher", "y": "student"}
SPACE_SOURCE = ROOT / "spaces" / "hotelier-blind-eval"


def bundle(csv: Path, samples: Path = SAMPLES, runs: Path = RUNS) -> Path:
    """The Space's folder: the app, each item's text (rebuilt from the raw CSV and checked) with both outputs, the
    Greek display table, and meta.json naming what it was built from."""
    rated = pd.read_csv(samples / "rated_key.csv")
    per_item = json.loads((samples / "outputs.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(csv, usecols=["Negative_Review", "Positive_Review"]).fillna("").astype(str)
    rows = raw.loc[rated["raw_row"]]
    texts = [sample.whole_review(n, p) for n, p in zip(rows["Negative_Review"], rows["Positive_Review"], strict=True)]
    wrong = [int(item) for item, text, digest in zip(rated["item"], texts, rated["text_sha256"], strict=True)
             if sample.text_hash(text) != digest]
    if wrong:
        raise SystemExit(f"rebuilt texts differ from the key for items {wrong[:5]}; nothing written")
    items = [{"item": int(item), "text": text,
              **{neutral: per_item[str(int(item))][system] for neutral, system in NEUTRAL.items()}}
             for item, text in zip(rated["item"], texts, strict=True)]
    folder = runs / "space"
    folder.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copyfile(SPACE_SOURCE / name, folder / name)
    (folder / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                       newline="\n")
    (folder / "display.json").write_text(json.dumps(display_table(), ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    data_hash = hashlib.sha256((folder / "items.json").read_bytes() + (folder / "display.json").read_bytes())
    meta = {"protocol_sha256": sample.trends._sha256(PROTOCOL), "code_commit": sample.pilot.git_commit(),
            "bundle": data_hash.hexdigest()[:12], "rated_key_sha256": sample.trends._sha256(samples / "rated_key.csv"),
            "outputs_sha256": sample.trends._sha256(samples / "outputs.json"),
            "labels_el_sha256": sample.trends._sha256(LABELS_EL), "items": len(items)}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    return folder


RATINGS = FOLDER / "ratings"
RESULT = FOLDER / "results" / "result.json"
EXPORT = RUNS / "ratings_export"
RECORD = RATINGS / "record.json"
DATASET = "krimits/hotelier-blind-eval-ratings"
DIGEST = "SHA-256 of the lines `<path> <SHA-256 of the file>`, one per rating file, sorted by path"
RATINGS_KEPT = ("in the private dataset and the owner's export, not in this repository: the hoteliers were told their "
                "answers are kept in a private dataset (owner's decision, 11 November 2026)")
MARGIN, FLOOR, MIN_COMPLETE = -0.10, 0.80, 2  # PROTOCOL.md §4, the owner's fields of 6 November 2026
BOOTSTRAP, BOOTSTRAP_SEED = 2000, 0
RULE = [  # PROTOCOL.md §4, in order
    ("p_teacher < 0.80", "Neither ready"),
    ("lo_d >= -0.10", "Student non-inferior"),
    ("hi_d < -0.10", "Student inferior"),
    ("otherwise", "Not established"),
]


def outcome(p_teacher: float, lo_d: float, hi_d: float) -> str:
    if p_teacher < FLOOR:
        return RULE[0][1]
    if lo_d >= MARGIN:
        return RULE[1][1]
    if hi_d < MARGIN:
        return RULE[2][1]
    return RULE[3][1]


def read_ratings(folder: Path, protocol_sha256: str) -> pd.DataFrame:
    """One row per (hotelier, item, system), with the systems named again: x the teacher, y the student."""
    rows = []
    for path in sorted(folder.glob("ratings/*/item_*.json")):
        rating = json.loads(path.read_text(encoding="utf-8"))
        if rating["protocol_sha256"] != protocol_sha256:
            raise SystemExit(f"{path} was given under another version of the protocol")
        for neutral, answers in rating["by_system"].items():
            rows.append({"hotelier": rating["hotelier"], "item": int(rating["item"]), "system": NEUTRAL[neutral],
                         "right": bool(answers["q1_urgent_right"]), "wrong": answers["q2_what_is_wrong"],
                         "others": answers["q3_others"], "useful": int(answers["q4_useful"]),
                         "preferred": NEUTRAL.get(rating["preferred_system"], "same")})
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit(f"no ratings under {folder}")
    if table.duplicated(["hotelier", "item", "system"]).any():
        raise SystemExit("a (hotelier, item) was rated twice")
    return table


def fleiss_kappa(table: pd.DataFrame) -> float | None:
    """Fleiss' kappa of the yes/no Q1 over every (item, system) the complete hoteliers rated."""
    counts = table.groupby(["item", "system"])["right"].agg(["sum", "count"])
    n = counts["count"].iloc[0]
    if n < 2 or (counts["count"] != n).any():
        return None
    yes, no = counts["sum"], n - counts["sum"]
    agreement = ((yes * (yes - 1) + no * (no - 1)) / (n * (n - 1))).mean()
    p_yes = yes.sum() / (n * len(counts))
    expected = p_yes ** 2 + (1 - p_yes) ** 2
    return None if expected == 1 else round(float((agreement - expected) / (1 - expected)), 4)


def primary(diff: pd.Series, union: pd.DataFrame) -> float:
    """d: the weighted mean over U of the difference, 0 for every review outside D."""
    weights = union["weight"]
    return float((weights * diff.reindex(union.index).fillna(0.0)).sum() / weights.sum())


def bootstrap(diff: pd.Series, union: pd.DataFrame, reps: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED
              ) -> tuple[float, float]:
    """2.5th and 97.5th percentiles of d over resamples of U's reviews within strata."""
    import numpy as np

    rng = np.random.default_rng(seed)
    values = diff.reindex(union.index).fillna(0.0).to_numpy()
    weights = union["weight"].to_numpy()
    positions = [np.flatnonzero(union["stratum"].to_numpy() == s) for s in sorted(union["stratum"].unique())]
    draws = []
    for _ in range(reps):
        picked = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in positions])
        draws.append((weights[picked] * values[picked]).sum() / weights[picked].sum())
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def analyse(ratings: Path = EXPORT, samples: Path = SAMPLES, out: Path = RESULT, require_commits: bool = True,
            record: Path | None = RECORD) -> dict:
    """The rule of §4, applied once to the exported ratings, with §4's reported items.

    The ratings are read from the owner's export, never from the repository; with `record`, they must be the ones
    recorded: the same number of files and the same digest."""
    if out.exists():
        raise SystemExit(f"{out} exists; the rule is applied once")
    needed = [samples / "rated_key.csv", samples / "outputs.json", PROTOCOL, *([record] if record else [])]
    if require_commits:
        missing = [p.name for p in needed if not v2score.first.committed(p)]
        if missing:
            raise SystemExit(f"not committed: {missing[:5]}; the record is committed before the rule is applied")
    digest, count = ratings_digest(ratings)
    if record:
        recorded = json.loads(record.read_text(encoding="utf-8"))
        if (recorded["ratings_sha256"], recorded["files"]) != (digest, count):
            raise SystemExit(f"{ratings} does not hold the recorded ratings; nothing analysed")
    protocol_sha = sample.trends._sha256(PROTOCOL)
    rated = pd.read_csv(samples / "rated_key.csv").set_index("item")
    table = read_ratings(ratings, protocol_sha)
    per_hotelier = table.groupby("hotelier")["item"].nunique()
    complete = sorted(per_hotelier[per_hotelier == len(rated)].index)
    partial = sorted(per_hotelier[per_hotelier < len(rated)].index)
    result: dict = {"protocol_sha256": protocol_sha, "code_commit": sample.pilot.git_commit(),
                    "hoteliers_complete": complete, "hoteliers_partial": partial,
                    "rule": [condition for condition, _ in RULE],
                    "ratings": {"files": count, "sha256": digest, "kept": RATINGS_KEPT},
                    "inputs": {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix():
                               sample.trends._sha256(p) for p in needed}}
    if len(complete) < MIN_COMPLETE:
        result["decision"] = {"outcome": f"not decided: {len(complete)} complete hotelier(s), fewer than {MIN_COMPLETE}"}
    else:
        rows = table[table["hotelier"].isin(complete)]
        right = rows.pivot_table(index="item", columns="system", values="right", aggfunc="mean")
        right.index = rated.loc[right.index, "raw_row"].to_numpy()
        test = test_reviews()
        group = groups(outputs())
        union = test[group.isin(["D", "C"])].assign(group=group)
        d_rows = union.index[union["group"] == "D"]
        diff = (right["student"] - right["teacher"]).reindex(d_rows)
        if diff.isna().any():
            raise SystemExit("a review of D has no rating from the complete hoteliers")
        d = primary(diff, union)
        lo_d, hi_d = bootstrap(diff, union)
        rated_c = rated[rated["group"] == "C"].set_index("raw_row")
        weight_d = union.loc[d_rows, "weight"]
        weight_c = rated_c["test_weight"] * rated_c["sample_weight"]
        p_teacher = float(((weight_d * right.loc[d_rows, "teacher"]).sum()
                           + (weight_c * right.loc[rated_c.index, "teacher"]).sum())
                          / (weight_d.sum() + weight_c.sum()))
        result["decision"] = {"d": round(d, 6), "lo_d": round(lo_d, 6), "hi_d": round(hi_d, 6),
                              "p_teacher": round(p_teacher, 6), "outcome": outcome(p_teacher, lo_d, hi_d),
                              "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED},
                              "u_reviews": len(union), "d_reviews": len(d_rows)}
        result["reported"] = reported(rows, rated, right, complete)
        result["reported"]["hoteliers_vs_chatgpt"] = versus_reference(rows, rated, samples)
        others = table[table["hotelier"].isin(partial)]
        result["reported"]["partial_raters"] = {
            "ratings": int(len(others) // 2),
            "right": {s: round(float(others[others["system"] == s]["right"].mean()), 4) if len(others) else None
                      for s in ("teacher", "student")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return result


def versus_reference(rows: pd.DataFrame, rated: pd.DataFrame, samples: Path = SAMPLES) -> dict:
    """How often a hotelier's Q1 agrees with whether that output's urgent departments are the ChatGPT reference's."""
    data = v2score.catalogue()
    urgent = metrics.urgent_departments(data["types"])
    per_item = json.loads((samples / "outputs.json").read_text(encoding="utf-8"))
    reference = v2score.first.load_indexed(V2_LABELS / "reference_chatgpt_v2.csv")

    def departments(types) -> frozenset:
        return frozenset(urgent[t] for t in types if t in urgent)
    agree = []
    for row in rows.itertuples():
        raw_row = int(rated.loc[row.item, "raw_row"])
        gold = departments([t for t in urgent if reference.loc[raw_row, t]])
        agree.append(row.right == (departments(per_item[str(row.item)][row.system]) == gold))
    return {"ratings": len(agree), "agreement": round(sum(agree) / len(agree), 4) if agree else None}


def reported(rows: pd.DataFrame, rated: pd.DataFrame, right: pd.DataFrame, complete: list[str]) -> dict:
    """§4, reported, not deciding."""
    group_of = rated["group"]
    rows = rows.assign(group=rows["item"].map(group_of))
    by_group = {g: {s: round(float(rows[(rows["group"] == g) & (rows["system"] == s)]["right"].mean()), 4)
                    for s in ("teacher", "student")} for g in ("D", "C")}
    reasons = {s: pd.Series([r for wrong in rows[rows["system"] == s]["wrong"] for r in wrong]).value_counts().to_dict()
               for s in ("teacher", "student")}
    pref = rows.drop_duplicates(["hotelier", "item"])["preferred"].value_counts().to_dict()
    c_items = rows[rows["group"] == "C"].pivot_table(index=["hotelier", "item"], columns="system", values="right")
    leave_one_out = {}
    for h in complete:
        kept = rows[rows["hotelier"] != h]
        if kept.empty:
            continue
        sub = kept.pivot_table(index="item", columns="system", values="right", aggfunc="mean")
        d_items = rated.index[rated["group"] == "D"]
        leave_one_out[h] = round(float((sub.loc[d_items, "student"] - sub.loc[d_items, "teacher"]).mean()), 4)
    return {"right_by_group": by_group, "what_is_wrong": reasons,
            "others": {s: rows[rows["system"] == s]["others"].value_counts().to_dict() for s in ("teacher", "student")},
            "useful_mean": {s: round(float(rows[rows["system"] == s]["useful"].mean()), 3)
                            for s in ("teacher", "student")},
            "preference": pref, "fleiss_kappa_q1": fleiss_kappa(rows),
            "identical_outputs_judged_differently": int((c_items["teacher"] != c_items["student"]).sum()),
            "identical_outputs_rated": len(c_items),
            "unweighted_mean_difference_on_d_without_each_hotelier": leave_one_out}


HOTELIER_FOLDER = re.compile(r"H\d{2}")
RATING_FILE = re.compile(r"item_(\d{2})\.json")


def space_logic():
    """The Space's own logic module, so a rating is checked against the sides the Space showed."""
    spec = importlib.util.spec_from_file_location("hotelier_space_logic", SPACE_SOURCE / "logic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ratings_digest(folder: Path) -> tuple[str, int]:
    """One SHA-256 over every rating file of an export, and how many there are (DIGEST says how)."""
    lines = [f"{path.relative_to(folder).as_posix()} {sample.trends._sha256(path)}\n"
             for path in sorted(folder.glob("ratings/*/item_*.json"), key=lambda p: p.relative_to(folder).as_posix())]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(), len(lines)


def check_rating(rating: dict, hotelier: str, item: int, items: set[int], protocol_sha256: str, logic) -> list[str]:
    """What is wrong with one exported rating; empty when it is one the Space wrote for this hotelier and item."""
    found = []
    if rating.get("hotelier") != hotelier or rating.get("item") != item:
        found.append("the hotelier or item inside differs from its path")
    if item not in items:
        found.append("not an item of the rated set")
    if rating.get("protocol_sha256") != protocol_sha256:
        found.append("given under another version of the protocol")
    first = logic.system_1(hotelier, item)
    second = logic.SYSTEMS[1 - logic.SYSTEMS.index(first)]
    if (rating.get("system_1"), rating.get("system_2")) != (first, second):
        found.append("the sides are not the ones the Space shows this hotelier for this item")
    shown = rating.get("shown") or {}
    if set(shown) != {"1", "2"} or rating.get("by_system") != {first: shown.get("1"), second: shown.get("2")}:
        found.append("the answers by system are not the answers by side")
    for side in ("1", "2"):
        answers = shown.get(side) or {}
        q1, q2, q4 = answers.get("q1_urgent_right"), answers.get("q2_what_is_wrong"), answers.get("q4_useful")
        if not (isinstance(q1, bool) and isinstance(q2, list) and set(q2) <= set(logic.Q2.values())
                and bool(q2) == (not q1)):
            found.append(f"side {side}: Q1 and Q2 are not a complete, consistent answer")
        if answers.get("q3_others") not in logic.Q3.values():
            found.append(f"side {side}: Q3 is not one of its answers")
        if isinstance(q4, bool) or not isinstance(q4, int) or not 1 <= q4 <= 5:
            found.append(f"side {side}: Q4 is not 1 to 5")
    preference = rating.get("preference")
    if preference not in logic.PREFERENCE.values() or rating.get("preferred_system") != {
            "1": first, "2": second, "same": "same"}.get(preference):
        found.append("the preference is missing or does not match its system")
    try:
        dt.datetime.fromisoformat(rating.get("submitted_at", ""))
    except (TypeError, ValueError):
        found.append("no submission time")
    return found


def record_ratings(source: str, export: Path = EXPORT, out: Path = RATINGS, samples: Path = SAMPLES,
                   received_on: dt.date | None = None) -> dict:
    """Check an export of the ratings dataset and write its record; the rating files are not copied.

    Every file must be one the Space wrote: its hotelier and item as its path says, this protocol's hash, the sides
    the Space shows that hotelier for that item, complete and consistent answers, one bundle. One problem refuses
    the whole export. The record holds who rated as the owner reports it, counts, times and one digest over every
    file; it summarises no answer, so nothing about the outcome is read before the rule is applied."""
    if not source.strip():
        raise SystemExit("--source is required: who gave the ratings, as the owner reports it")
    if (out / "record.json").exists():
        raise SystemExit(f"{out / 'record.json'} exists; recorded ratings are never replaced")
    if not (export / "ratings").is_dir():
        raise SystemExit(f"no ratings/ folder in {export}; download the dataset there first")
    logic = space_logic()
    protocol_sha = sample.trends._sha256(PROTOCOL)
    items = set(pd.read_csv(samples / "rated_key.csv")["item"].astype(int))
    problems, files, bundles, times = [], 0, set(), {}
    for folder in sorted((export / "ratings").iterdir()):
        if not folder.is_dir() or not HOTELIER_FOLDER.fullmatch(folder.name):
            problems.append(f"ratings/{folder.name}: not a hotelier's folder")
            continue
        for path in sorted(folder.iterdir()):
            name = f"ratings/{folder.name}/{path.name}"
            match = RATING_FILE.fullmatch(path.name)
            try:
                rating = json.loads(path.read_text(encoding="utf-8")) if match else None
            except ValueError:
                rating = None
            if not isinstance(rating, dict):
                problems.append(f"{name}: not a rating file")
                continue
            found = check_rating(rating, folder.name, int(match.group(1)), items, protocol_sha, logic)
            problems += [f"{name}: {problem}" for problem in found]
            files += 1
            bundles.add(rating.get("bundle"))
            times.setdefault(folder.name, []).append(rating.get("submitted_at") or "")
    if not files:
        problems.append("no rating files")
    if len(bundles) > 1:
        problems.append(f"ratings from more than one bundle: {sorted(map(str, bundles))}")
    if problems:
        raise SystemExit(f"the export is refused ({len(problems)} problem(s)); nothing recorded:\n"
                         + "\n".join(problems[:20]))
    digest, counted = ratings_digest(export)
    assert counted == files, (counted, files)
    record = {
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(), "protocol_sha256": protocol_sha,
        "code_commit": sample.pilot.git_commit(), "dataset": DATASET, "source": source,
        "received_on": (received_on or dt.date.today()).isoformat(), "bundle": bundles.pop(),
        "items_in_the_rated_set": len(items),
        "hoteliers": {h: {"ratings": len(stamps), "complete": len(stamps) == len(items),
                          "first_submission": min(stamps), "last_submission": max(stamps)}
                      for h, stamps in sorted(times.items())},
        "files": files, "ratings_sha256": digest, "digest": DIGEST, "rating_files_kept": RATINGS_KEPT,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                     newline="\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("draw", help="draw the rated set and write its key, outputs and manifest")
    first.add_argument("--overwrite", action="store_true")
    second = steps.add_parser("bundle", help="write the Space's items.json with the texts (git-ignored)")
    second.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    third = steps.add_parser("analyse", help="apply the rule once to the committed ratings")
    third.add_argument("--ratings", type=Path, default=EXPORT)
    fourth = steps.add_parser("record", help="check the exported ratings and write their record")
    fourth.add_argument("--source", required=True, help="who gave the ratings, as the owner reports it")
    fourth.add_argument("--export", type=Path, default=EXPORT)
    args = parser.parse_args()
    if args.step == "draw":
        manifest = build(overwrite=args.overwrite)
        print("groups:", {g: v["reviews"] for g, v in manifest["groups"].items()}, "| rated:", manifest["rated"])
    elif args.step == "record":
        record = record_ratings(args.source, args.export)
        print({h: (v["ratings"], "complete" if v["complete"] else "partial") for h, v in record["hoteliers"].items()},
              "| bundle", record["bundle"], "|", record["files"], "files | digest", record["ratings_sha256"][:12])
    elif args.step == "analyse":
        decision = analyse(args.ratings)["decision"]
        print(decision)
    else:
        if sample.trends._sha256(args.csv) != sample.RAW_SHA256:
            raise SystemExit(f"{args.csv} is not the Kaggle file the protocol fixes")
        print(bundle(args.csv))


if __name__ == "__main__":
    main()
