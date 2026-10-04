"""Paired human coverage review; predictions and review text are immutable input columns."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reviewnlp.triage.generator_experiment import digest_bytes

COUNTS = ("actual_problem_count", "missed_problem_count", "false_exclusion_count",
          "unaddressed_problem_count", "invented_problem_count", "unnecessary_action_count",
          "useful_action_count", "wrong_department_count")
FIXED = ("blind_id", "review", "issue_assessments", "accepted_actions", "execution_issue")


def export_coverage_review(output: Path, rows: list[dict], records: list[dict]) -> None:
    key = json.loads((output / "annotation_key.json").read_text(encoding="utf-8"))
    texts = {row["id"]: row["text"] for row in rows}
    lookup = {(item["candidate"], item["id"]): item for item in records}
    with (output / "coverage_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FIXED, *COUNTS, "notes"])
        writer.writeheader()
        for item in key:
            record = lookup[item["candidate"], item["review_id"]]
            writer.writerow({"blind_id": item["blind_id"], "review": texts[item["review_id"]],
                             "issue_assessments": json.dumps(record.get("extracted_issues", []), ensure_ascii=False),
                             "accepted_actions": json.dumps(record["actions"], ensure_ascii=False),
                             "execution_issue": record["error"] or record.get("workflow_error")
                             or ("hit_token_budget" if record["hit_token_budget"] else ""),
                             **dict.fromkeys(COUNTS, ""), "notes": ""})


def score_coverage(run_dir: Path, ratings: Path) -> dict:
    info_bytes = (run_dir / "run.json").read_bytes()
    info = json.loads(info_bytes)
    if info.get("split") != "dev" or info.get("execution_complete") is not True:
        raise ValueError("coverage review requires a completed dev run")
    for name in ("coverage_review.csv", "annotation_key.json", "results.jsonl"):
        if digest_bytes((run_dir / name).read_bytes()) != info.get("files", {}).get(name):
            raise ValueError("source artifact differs: " + name)
    with (run_dir / "coverage_review.csv").open(newline="", encoding="utf-8") as handle:
        source = {row["blind_id"]: row for row in csv.DictReader(handle)}
    with ratings.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(source) or {row.get("blind_id") for row in rows} != set(source):
        raise ValueError("coverage ratings missing, duplicated or unexpected")
    key = {item["blind_id"]: item for item in json.loads((run_dir / "annotation_key.json").read_text())}
    actual_by_review, summaries = {}, {}
    for row in rows:
        if any(row.get(name) != source[row["blind_id"]][name] for name in FIXED):
            raise ValueError("review or model output changed")
        if any(not str(row.get(name, "")).isdigit() for name in COUNTS):
            raise ValueError("complete every coverage count with a nonnegative integer")
        values = {name: int(row[name]) for name in COUNTS}
        actions = len(json.loads(row["accepted_actions"]))
        if (values["missed_problem_count"] > values["actual_problem_count"]
                or values["false_exclusion_count"] > values["missed_problem_count"]
                or values["unaddressed_problem_count"] > values["actual_problem_count"]
                or any(values[name] > actions for name in
                       ("unnecessary_action_count", "useful_action_count", "wrong_department_count"))):
            raise ValueError("inconsistent coverage counts")
        identity = key[row["blind_id"]]
        previous = actual_by_review.setdefault(identity["review_id"], values["actual_problem_count"])
        if previous != values["actual_problem_count"]:
            raise ValueError("paired candidates must have the same human ground truth")
        total = summaries.setdefault(identity["candidate"], {"reviews": 0, "reviews_with_actual_problems": 0,
                                                             "accepted_actions": 0, **dict.fromkeys(COUNTS, 0)})
        total["reviews"] += 1
        total["reviews_with_actual_problems"] += values["actual_problem_count"] > 0
        total["accepted_actions"] += actions
        for name, value in values.items():
            total[name] += value
    for total in summaries.values():
        actual, actions = total["actual_problem_count"], total["accepted_actions"]
        total["human_issue_coverage"] = (actual - total["missed_problem_count"]) / actual if actual else None
        total["human_action_coverage"] = (actual - total["unaddressed_problem_count"]) / actual if actual else None
        total["fully_useful_action_fraction"] = total["useful_action_count"] / actions if actions else None
    return {"source_run_sha256": digest_bytes(info_bytes), "ratings_sha256": digest_bytes(ratings.read_bytes()),
            "split": "dev", "population": "synthetic-authored development cases; not production accuracy",
            "summary": summaries, "selection_frozen": False}
