"""Reconcile two external coverage sheets for a captured development snapshot; never promote it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reviewnlp.triage.generator_experiment import RATINGS  # noqa: E402
from reviewnlp.triage.workflow_review import COUNTS, score_coverage  # noqa: E402


def reconcile(run: Path, ratings_a: Path, ratings_b: Path) -> dict:
    info = json.loads((run / "run.json").read_bytes())
    for name, expected in info["files"].items():
        if Path(name).name != name or hashlib.sha256((run / name).read_bytes()).hexdigest() != expected:
            raise ValueError("captured artifact changed: " + name)
    reports = [score_coverage(run, path) for path in (ratings_a, ratings_b)]
    def rows(path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {row["blind_id"]: row for row in csv.DictReader(handle)}
    a, b = rows(ratings_a), rows(ratings_b)
    for sheet in (a, b):
        for row in sheet.values():
            actions = json.loads(row["accepted_actions"])
            useful = int(row["useful_action_count"])
            if (useful + int(row["wrong_department_count"]) > len(actions)
                    or useful + int(row["unnecessary_action_count"]) > len(actions)
                    or int(row["missed_problem_count"]) > int(row["unaddressed_problem_count"])):
                raise ValueError("inconsistent action usefulness or missed/unaddressed counts")
            for field in RATINGS:
                allowed = {"0", "1"} if actions or field == "useful" else {"0", "1", "na"}
                if row.get(field) not in allowed:
                    raise ValueError("complete all four quality dimensions with 1/0/na")
            if row["execution_issue"] and row["useful"] == "1":
                raise ValueError("a failed workflow cannot be rated fully useful")
            if row["useful"] == "1" and (any(int(row[field]) for field in
                    ("missed_problem_count", "unaddressed_problem_count", "invented_problem_count",
                     "unnecessary_action_count", "wrong_department_count"))
                    or (actions and any(row[field] != "1" for field in RATINGS))):
                raise ValueError("fully useful judgment contradicts recorded shortcomings")
            if not actions and int(row["unaddressed_problem_count"]) != int(row["actual_problem_count"]):
                raise ValueError("without accepted actions, all actual problems remain unaddressed")
    differences = [{"blind_id": key, "field": field, "A": int(a[key][field]), "B": int(b[key][field])}
                   for key in sorted(a) for field in COUNTS if int(a[key][field]) != int(b[key][field])]
    differences += [{"blind_id": key, "field": field, "A": a[key][field], "B": b[key][field]}
                    for key in sorted(a) for field in RATINGS if a[key][field] != b[key][field]]
    quality = {name: {field: {value: sum(row[field] == value for row in sheet.values())
                for value in ("0", "1", "na")} for field in RATINGS} for name, sheet in (("A", a), ("B", b))}
    return {"scope": "two completed coverage sheets for an authored development capture; no production accuracy",
            "source_run_sha256": reports[0]["source_run_sha256"],
            "ratings_A_sha256": reports[0]["ratings_sha256"], "ratings_B_sha256": reports[1]["ratings_sha256"],
            "reviewer_summaries": {"A": reports[0]["summary"], "B": reports[1]["summary"]},
            "quality_dimensions": quality, "machine_failures": info["failures"],
            "disagreements": differences, "judgments_reconciled": not differences,
            "agreed_summary": reports[0]["summary"] if not differences else None,
            "reviewer_independence_verified": False, "selection_frozen": False, "promoted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ratings-a", type=Path, required=True)
    parser.add_argument("--ratings-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reconcile(args.run, args.ratings_a, args.ratings_b)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Judgments reconciled:", report["judgments_reconciled"], "· no selection or promotion.")
    raise SystemExit(0 if report["judgments_reconciled"] else 1)


if __name__ == "__main__":
    main()
