"""The steps of the hotel pilot around its Space (docs/experiments/hotel_pilot/PROTOCOL.md, version 1).

    python scripts/hotel_pilot.py bundle                             # writes runs/hotel_pilot/space/ (git-ignored)
    python scripts/deploy_hotel_pilot_space.py publish               # the owner, with the owner's own login
    hf download krimits/hotel-triage-pilot-records --repo-type dataset \
        --local-dir runs/hotel_pilot/records_export                   # the owner, when the collection is closed
    python scripts/hotel_pilot.py record --source "who took part, as the owner reports it"
    git add ... && git commit ...                                    # the record is committed before the rule
    python scripts/hotel_pilot.py analyse                            # once

`bundle` writes the Space's folder: the app files from spaces/hotel-pilot/, prompt.md (teacher_prompt_v2.md, byte for
byte), display.json (catalogue_v2 through the Greek labels) and meta.json (the model and the hashes of the protocol
and the prompt, which every record then carries).

`record` checks the owner's export of the private dataset: every record must be one the Space wrote under this
protocol (its hotel and key as its path says, this protocol's and this prompt's hash, a verdict for exactly the
reported types that were in the hotel's urgent set, complete answers); one problem refuses the whole export. It
writes records/record.json: who took part as the owner reports it, counts per hotel, and one SHA-256 over every
exported file. The records themselves are not committed, as in the blind evaluation.

`analyse` applies §4 once, after the record is committed and only if the export still has the record's digest:
`acc`, `worst` and `miss` over the qualifying hotels, the outcome, and the reported items. It writes
results/result.json once and never again; it holds aggregates, no single judgement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import hotelier_blind_eval as ev  # noqa: E402

sample = ev.sample
FOLDER = ROOT / "docs" / "experiments" / "hotel_pilot"
PROTOCOL = FOLDER / "PROTOCOL.md"
PROMPT = ev.v2.TEACHER_PROMPT
RECORD = FOLDER / "records" / "record.json"
RESULT = FOLDER / "results" / "result.json"
RUNS = ROOT / "runs" / "hotel_pilot"
EXPORT = RUNS / "records_export"
SPACE_SOURCE = ROOT / "spaces" / "hotel-pilot"
SPACE_FILES = ("README.md", "app.py", "logic.py", "requirements.txt")
DATASET = "krimits/hotel-triage-pilot-records"
# PROTOCOL.md §4 and §7, the owner's fields of 11 November 2026.
H, R, G = 5, 30, 10
ACC, WORST, MISS, NOT_ACCEPTED_BELOW = 0.80, 0.60, 0.10, 0.60
RULE = [  # PROTOCOL.md §4, in order
    ("acc >= 0.80 and worst >= 0.60 and miss <= 0.10", "Accepted"),
    ("acc < 0.60", "Not accepted"),
    ("otherwise", "Partly accepted"),
]
KEPT = ("in the private dataset and the owner's export, not in this repository, as in the blind evaluation")
DIGEST = "SHA-256 of the lines `<path> <SHA-256 of the file>`, one per exported file, sorted by path"
HOTEL_FOLDER = re.compile(r"P\d{2}")
KINDS = ("records", "analysed", "settings")


def space_logic():
    """The Space's own logic module, so the repository checks records with the Space's own rules."""
    spec = importlib.util.spec_from_file_location("hotel_pilot_logic", SPACE_SOURCE / "logic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def display_table() -> dict:
    """The blind evaluation's Greek display of catalogue_v2, with the word for a problem that is not urgent."""
    import yaml

    labels = yaml.safe_load(ev.LABELS_EL.read_text(encoding="utf-8"))
    return {**ev.display_table(), "routine_el": labels["priorities"]["routine"]}


def meta() -> dict:
    return {"model": space_logic().MODEL, "protocol_sha256": sample.trends._sha256(PROTOCOL),
            "prompt_sha256": sample.trends._sha256(PROMPT), "labels_el_sha256": sample.trends._sha256(ev.LABELS_EL),
            "catalogue_sha256": sample.trends._sha256(ev.v2.CATALOGUE)}


def bundle(runs: Path = RUNS) -> Path:
    """The Space's folder: the app, the prompt as tested, the Greek display and meta.json."""
    folder = runs / "space"
    folder.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copyfile(SPACE_SOURCE / name, folder / name)
    shutil.copyfile(PROMPT, folder / "prompt.md")
    (folder / "display.json").write_text(json.dumps(display_table(), ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    data_hash = hashlib.sha256((folder / "prompt.md").read_bytes() + (folder / "display.json").read_bytes())
    content = {**meta(), "code_commit": sample.pilot.git_commit(), "bundle": data_hash.hexdigest()[:12]}
    (folder / "meta.json").write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8", newline="\n")
    return folder


def export_digest(folder: Path) -> tuple[str, int]:
    """One SHA-256 over every exported file under records/, analysed/ and settings/, and how many there are."""
    paths = sorted((p for kind in KINDS for p in (folder / kind).rglob("*.json")),
                   key=lambda p: p.relative_to(folder).as_posix())
    lines = [f"{p.relative_to(folder).as_posix()} {sample.trends._sha256(p)}\n" for p in paths]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(), len(lines)


def check_record(rating: dict, hotel: str, key: str, expected: dict, logic, display: dict) -> list[str]:
    """What is wrong with one exported record; empty when it is one the Space wrote for this hotel and review."""
    found = []
    if rating.get("hotel") != hotel or rating.get("key") != key:
        found.append("the hotel or key inside differs from its path")
    for name in ("protocol_sha256", "prompt_sha256", "model"):
        if rating.get(name) != expected[name]:
            found.append(f"another {name} than this protocol's")
    types, urgent_set = rating.get("types"), rating.get("urgent_set")
    urgent = logic.urgent_types(display)
    if not isinstance(types, list) or not set(types) <= set(display["order"]):
        return [*found, "the reported types are not the catalogue's"]
    if not isinstance(urgent_set, list) or not set(urgent_set) <= set(urgent):
        return [*found, "the urgent set is not made of the catalogue's urgent types"]
    flags = rating.get("flags")
    if not isinstance(flags, dict) or list(flags) != logic.flags(types, urgent_set, display) \
            or not set(flags.values()) <= set(logic.VERDICTS.values()):
        found.append("the verdicts are not one per urgent flag")
    missing, missing_type = rating.get("missing_urgent"), rating.get("missing_type")
    if not isinstance(missing, bool) or (missing and missing_type not in {*urgent, "other"}) \
            or (not missing and missing_type is not None):
        found.append("the missing-urgent answer is not complete")
    if rating.get("others") not in logic.OTHERS.values():
        found.append("the other-problems answer is not one of its answers")
    useful = rating.get("useful")
    if isinstance(useful, bool) or not isinstance(useful, int) or not 1 <= useful <= 5:
        found.append("the usefulness is not 1 to 5")
    if rating.get("script") not in ("greek", "latin", "other"):
        found.append("no script")
    if "text" in rating:
        found.append("it holds a review text")
    return found


def read_export(export: Path) -> tuple[list[dict], dict[str, set[str]], list[str]]:
    """Every record, the analysed keys per hotel, and the problems found. Nothing is written."""
    logic, display, expected = space_logic(), display_table(), meta()
    records, analysed, problems = [], {}, []
    if not (export / "records").is_dir():
        return [], {}, [f"no records/ folder in {export}; download the dataset there first"]
    for kind in ("records", "analysed"):
        for folder in sorted((export / kind).iterdir()) if (export / kind).is_dir() else []:
            if not folder.is_dir() or not HOTEL_FOLDER.fullmatch(folder.name):
                problems.append(f"{kind}/{folder.name}: not a hotel's folder")
                continue
            for path in sorted(folder.glob("*")):
                name = f"{kind}/{folder.name}/{path.name}"
                try:
                    content = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
                except ValueError:
                    content = None
                if not isinstance(content, dict):
                    problems.append(f"{name}: not a JSON record")
                elif kind == "analysed":
                    analysed.setdefault(folder.name, set()).add(path.stem)
                else:
                    found = check_record(content, folder.name, path.stem, expected, logic, display)
                    problems += [f"{name}: {problem}" for problem in found]
                    records.append(content)
    return records, analysed, problems


def record_export(source: str, export: Path = EXPORT, out: Path = RECORD,
                  received_on: dt.date | None = None) -> dict:
    """Check an export of the pilot's dataset and write its record; the records are not copied."""
    if not source.strip():
        raise SystemExit("--source is required: who took part, as the owner reports it")
    if out.exists():
        raise SystemExit(f"{out} exists; a recorded export is never replaced")
    records, analysed, problems = read_export(export)
    if not records and not problems:
        problems.append("no records")
    if problems:
        raise SystemExit(f"the export is refused ({len(problems)} problem(s)); nothing recorded:\n"
                         + "\n".join(problems[:20]))
    digest, count = export_digest(export)
    per_hotel = Counter(r["hotel"] for r in records)
    content = {"protocol": PROTOCOL.relative_to(ROOT).as_posix(), **meta(),
               "code_commit": sample.pilot.git_commit(), "dataset": DATASET, "source": source,
               "received_on": (received_on or dt.date.today()).isoformat(),
               "hotels": {h: {"rated_reviews": n, "rated_urgent_flags": sum(len(r["flags"]) for r in records
                                                                             if r["hotel"] == h),
                              "analysed_reviews": len(analysed.get(h, set()))} for h, n in sorted(per_hotel.items())},
               "files": count, "export_sha256": digest, "digest": DIGEST, "records_kept": KEPT}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(content, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return content


def outcome(acc: float, worst: float, miss: float) -> str:
    if acc >= ACC and worst >= WORST and miss <= MISS:
        return RULE[0][1]
    if acc < NOT_ACCEPTED_BELOW:
        return RULE[1][1]
    return RULE[2][1]


def per_hotel(records: list[dict], analysed: dict[str, set[str]]) -> dict[str, dict]:
    """Each hotel's counts and shares, and whether it qualifies (§3)."""
    out = {}
    for hotel in sorted({r["hotel"] for r in records}):
        rows = [r for r in records if r["hotel"] == hotel]
        verdicts = [v for r in rows for v in r["flags"].values()]
        accepted = sum(v == "accepted" for v in verdicts)
        out[hotel] = {
            "rated_reviews": len(rows), "rated_urgent_flags": len(verdicts), "accepted_urgent_flags": accepted,
            "reviews_with_a_missing_urgent_problem": sum(r["missing_urgent"] for r in rows),
            "qualifies": len(rows) >= R and len(verdicts) >= G,
            "acc": round(accepted / len(verdicts), 6) if verdicts else None,
            "miss": round(sum(r["missing_urgent"] for r in rows) / len(rows), 6),
            "others": dict(Counter(r["others"] for r in rows)),
            "useful_mean": round(statistics.fmean(r["useful"] for r in rows), 3),
            "analysed_never_rated": len(analysed.get(hotel, set()) - {r["key"] for r in rows}),
            "urgent_set_at_last_rating": max(rows, key=lambda r: r["submitted_at"])["urgent_set"],
        }
    return out


def reported(records: list[dict], hotels: dict[str, dict], display: dict, logic) -> dict:
    """§4, reported, not deciding. Pooled figures use the qualifying hotels' records."""
    qualifying = {h for h, stats in hotels.items() if stats["qualifies"]}
    rows = [r for r in records if r["hotel"] in qualifying]
    urgent = logic.urgent_types(display)
    by_type = {t: dict(Counter(r["flags"][t] for r in rows if t in r["flags"])) for t in urgent}
    by_script = {}
    for script in ("greek", "latin", "other"):
        verdicts = [v for r in rows if r["script"] == script for v in r["flags"].values()]
        by_script[script] = {"rated_urgent_flags": len(verdicts),
                             "acc": round(sum(v == "accepted" for v in verdicts) / len(verdicts), 4)
                             if verdicts else None}
    calls = [r["call"] for r in rows if r.get("call")]
    costs = [c["cost_usd"] / c["reviews"] for c in calls if c.get("cost_usd") is not None and c.get("reviews")]
    return {
        "hotels": hotels,
        "verdicts_by_urgent_type": {t: v for t, v in by_type.items() if v},
        "hotels_keeping_each_urgent_type_on": {t: sum(t in hotels[h]["urgent_set_at_last_rating"] for h in qualifying)
                                               for t in urgent},
        "acc_by_script": by_script,
        "missing_types": dict(Counter(r["missing_type"] for r in rows if r["missing_urgent"])),
        "hotels_not_qualifying": sorted(set(hotels) - qualifying),
        "cost_usd_per_1000_reviews": round(1000 * statistics.fmean(costs), 4) if costs else None,
        "median_seconds_per_call": round(statistics.median(c["seconds"] for c in calls), 2) if calls else None,
    }


def analyse(export: Path = EXPORT, out: Path = RESULT, require_commits: bool = True,
            record: Path | None = RECORD) -> dict:
    """The rule of §4, applied once to the exported records, with §4's reported items."""
    if out.exists():
        raise SystemExit(f"{out} exists; the rule is applied once")
    needed = [PROTOCOL, *([record] if record else [])]
    if require_commits:
        missing = [p.name for p in needed if not ev.v2score.first.committed(p)]
        if missing:
            raise SystemExit(f"not committed: {missing}; the record is committed before the rule is applied")
    digest, count = export_digest(export)
    if record:
        recorded = json.loads(record.read_text(encoding="utf-8"))
        if (recorded["export_sha256"], recorded["files"]) != (digest, count):
            raise SystemExit(f"{export} does not hold the recorded export; nothing analysed")
    records, analysed, problems = read_export(export)
    if problems:
        raise SystemExit("the export is refused; nothing analysed:\n" + "\n".join(problems[:20]))
    hotels = per_hotel(records, analysed)
    qualifying = [h for h, stats in hotels.items() if stats["qualifies"]]
    result: dict = {"protocol_sha256": sample.trends._sha256(PROTOCOL), "code_commit": sample.pilot.git_commit(),
                    "rule": [condition for condition, _ in RULE], "fields": {"H": H, "R": R, "G": G},
                    "export": {"files": count, "sha256": digest, "kept": KEPT},
                    "hotels_qualifying": qualifying}
    if len(qualifying) < H:
        result["decision"] = {"outcome": f"not decided: {len(qualifying)} qualifying hotel(s), fewer than {H}"}
    else:
        # From the counts, as exact fractions: a mean that is exactly 0.80 must not miss the threshold by a
        # rounding of the shares shown per hotel or of floating-point sums.
        accs = [Fraction(hotels[h]["accepted_urgent_flags"], hotels[h]["rated_urgent_flags"]) for h in qualifying]
        acc, worst = float(sum(accs) / len(accs)), float(min(accs))
        miss = float(sum(Fraction(hotels[h]["reviews_with_a_missing_urgent_problem"], hotels[h]["rated_reviews"])
                         for h in qualifying) / len(qualifying))
        result["decision"] = {"acc": round(acc, 6), "worst": round(worst, 6), "miss": round(miss, 6),
                              "outcome": outcome(acc, worst, miss)}
    result["reported"] = reported(records, hotels, display_table(), space_logic())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("bundle", help="write the Space's folder (git-ignored)")
    second = steps.add_parser("record", help="check the exported records and write their record")
    second.add_argument("--source", required=True, help="who took part, as the owner reports it")
    second.add_argument("--export", type=Path, default=EXPORT)
    third = steps.add_parser("analyse", help="apply the rule once to the recorded export")
    third.add_argument("--export", type=Path, default=EXPORT)
    args = parser.parse_args()
    if args.step == "bundle":
        print(bundle())
    elif args.step == "record":
        content = record_export(args.source, args.export)
        print(content["hotels"], "|", content["files"], "files | digest", content["export_sha256"][:12])
    else:
        print(analyse(args.export)["decision"])


if __name__ == "__main__":
    main()
