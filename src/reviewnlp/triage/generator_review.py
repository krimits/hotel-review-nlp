"""Audit external human ratings against an intact run; never infer a winner or resolve partial judgments."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from reviewnlp.triage.generator_experiment import RATINGS, digest_bytes, json_diagnostics

TOKENS = {"yes": "1", "no": "0", "1": "1", "0": "0", "n/a": "na", "na": "na"}


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_ratings(run_dir: Path, ratings_path: Path) -> tuple[dict, list[dict]]:
    """Return an audit and a separate adjudication sheet. Source files remain evidence."""
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    required = {"human_review.csv", "annotation_key.json", "results.jsonl", "upstream.json"}
    if not required <= set(info.get("files", {})) or info.get("split") != "dev":
        raise ValueError("an intact development archive is required")
    for name, expected in info["files"].items():
        if Path(name).name != name or digest_bytes((run_dir / name).read_bytes()) != expected:
            raise ValueError("run artifact fingerprint differs: " + name)
    original, ratings = csv_rows(run_dir / "human_review.csv"), csv_rows(ratings_path)
    key = json.loads((run_dir / "annotation_key.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_ids = {row["blind_id"] for row in key}
    for rows in (original, ratings, key):
        if len(rows) != len(expected_ids) or {row.get("blind_id") for row in rows} != expected_ids:
            raise ValueError("missing, duplicated or unexpected blind IDs")
    if len(records) != len(key) or len({(row["candidate"], row["id"]) for row in records}) != len(records):
        raise ValueError("missing or duplicated results")
    originals, keys = {row["blind_id"]: row for row in original}, {row["blind_id"]: row for row in key}
    results = {(row["candidate"], row["id"]): row for row in records}
    changed_machine, pending, ambiguous, sheet = [], [], [], []
    for row in ratings:
        blind_id = row["blind_id"]
        source, identity = originals[blind_id], keys[blind_id]
        record = results.get((identity["candidate"], identity["review_id"]))
        if record is None or json.loads(source["accepted_actions"]) != record["actions"]:
            raise ValueError("annotation key or exported actions differ from results")
        if (row.get("review") != source["review"]
                or json.loads(row.get("accepted_actions", "null")) != record["actions"]):
            raise ValueError("review or machine actions were edited: " + blind_id)
        if row.get("execution_issue") != source["execution_issue"]:
            changed_machine.append(blind_id)
        normalized = {**source, "notes": row.get("notes", "")}
        for field in RATINGS:
            value = row.get(field, "").strip().lower()
            allowed = {"0", "1"} if field == "useful" or record["actions"] else {"0", "1", "na"}
            converted = TOKENS.get(value)
            if converted not in allowed:
                pending.append({"blind_id": blind_id, "field": field, "original_value": row.get(field, "")})
                normalized[field] = row.get(field, "")
            else:
                normalized[field] = converted
        failed = record.get("error") or record.get("workflow_error") or record.get("hit_token_budget")
        if failed and normalized["useful"] == "1":
            pending.append({"blind_id": blind_id, "field": "useful", "reason": "failed_generation_rated_useful"})
        diagnostics = json_diagnostics(record.get("raw"))
        if diagnostics["duplicate_json_keys"]:
            ambiguous.append(blind_id)
        sheet.append(normalized)
    counts = {}
    for candidate in info["candidates"]:
        chosen = [row for row in sheet if keys[row["blind_id"]]["candidate"] == candidate]
        with_actions = [row for row in chosen if json.loads(row["accepted_actions"])]
        counts[candidate] = {"outputs": len(chosen), "with_actions": len(with_actions),
                             "fully_correct_nonempty": sum(all(row[field] == "1" for field in RATINGS)
                                                           for row in with_actions),
                             "empty_rated_missed": sum(not json.loads(row["accepted_actions"])
                                                       and row["useful"] == "0" for row in chosen),
                             "judgments": {field: dict(Counter(row[field] for row in chosen)) for field in RATINGS}}
    report = {"run_sha256": digest_bytes((run_dir / "run.json").read_bytes()),
              "source_ratings_sha256": digest_bytes(ratings_path.read_bytes()),
              "original_sheet_sha256": info["files"]["human_review.csv"],
              "outputs": len(sheet), "counts_by_candidate": counts,
              "execution_issue_overwrites": changed_machine, "duplicate_key_outputs": ambiguous,
              "pending_judgments": pending,
              "ready_for_freeze": not pending and info.get("execution_complete") is True,
              "scope": "provided ratings of historical accepted actions; no semantic regrading or winner",
              "normalization": "yes/no/n/a -> 1/0/na; partial and incomplete judgments require adjudication"}
    return report, sheet


def write_audit(output: Path, report: dict, sheet: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "review_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    name = "human_review_normalized.csv" if report["ready_for_freeze"] else "adjudication.csv"
    with (output / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
