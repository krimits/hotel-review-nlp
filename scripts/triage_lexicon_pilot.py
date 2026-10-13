"""Offline preparation, independent agreement and scoring of whole-review topic labels.

Private CSV input: review_id,text,language (en). No sentiment gate, lexicon-based
sampling, Qwen/Jev call, model download or Space upload. Outputs belong in runs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter
from pathlib import Path

from reviewnlp.triage.lexicon import SHA256, VERSION, LexiconClassifier
from reviewnlp.triage.schemas import TOPICS

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/experiments/triage_lexicon_pilot/PROTOCOL.md"
LABEL_TOPICS = (*TOPICS, "other")
LABELS = {"0", "1", "unsure"}
COLUMNS = ("item", "text", "text_sha256", *LABEL_TOPICS, "done", "note")
SOURCES = ("src/reviewnlp/triage/lexicon.py", "src/reviewnlp/triage/schemas.py",
           "scripts/triage_lexicon_pilot.py", "docs/experiments/triage_lexicon_pilot/PROTOCOL.md")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    if path.exists():
        raise ValueError("output_already_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns=COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate_csv_column")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("invalid_csv_row")
    return fields, rows


def prepare(source: Path, output: Path, *, size: int, seed: int, population: str) -> dict:
    fields, rows = read_csv(source)
    if set(fields) != {"review_id", "text", "language"} or not rows or not population.strip():
        raise ValueError("whole_review_csv_and_population_required")
    if size <= 0 or size > len(rows):
        raise ValueError("sample_size_out_of_range")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in SOURCES:
        try:
            content = subprocess.check_output(["git", "show", source_commit + ":" + name], cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise ValueError("commit_pilot_sources_before_sampling") from None
        if content != (ROOT / name).read_bytes():
            raise ValueError("commit_pilot_sources_before_sampling")
    ids, texts = set(), set()
    for row in rows:
        identity, text = row["review_id"].strip(), row["text"]
        normalized = " ".join(text.casefold().split())
        if not identity or identity in ids or normalized in texts:
            raise ValueError("duplicate_or_empty_review_id_or_text")
        if row["language"].strip().lower() != "en" or not 3 <= len(text.strip()) <= 4000 or len(text) > 4000:
            raise ValueError("declared_english_whole_review_required")
        ids.add(identity)
        texts.add(normalized)
    chosen = random.Random(seed).sample(rows, size)
    key, sheets, predictions = [], [], {}
    classifier = LexiconClassifier()
    for index, row in enumerate(chosen, 1):
        item = f"{index:04d}"
        key.append({"item": item, "review_id": row["review_id"].strip(), "text_sha256": text_digest(row["text"])})
        sheets.append({"item": item, "text": row["text"], "text_sha256": text_digest(row["text"]),
                       **dict.fromkeys(LABEL_TOPICS, ""), "done": "", "note": ""})
        prediction = classifier.classify(row["text"])
        predictions[item] = {entry.topic: entry.answer for entry in [*prediction.topics, prediction.other_complaint]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "key.json", key)
    write_json(output / "predictions.json", predictions)
    write_csv(output / "sheet_A.csv", sheets)
    shuffled = list(sheets)
    random.Random(seed + 1).shuffle(shuffled)
    write_csv(output / "sheet_B.csv", shuffled)
    manifest = {"kind": "whole_review_lexicon_pilot", "source_commit": source_commit,
        "source_hashes": {name: digest(ROOT / name) for name in SOURCES},
        "input_sha256": digest(source), "population_description": population.strip(),
        "sampling": "uniform_whole_review_without_model_or_sentiment_filter", "population_reviews": len(rows),
        "sample_reviews": size, "seed": seed, "declared_language": "en", "language_independently_verified": False,
        "lexicon_version": VERSION, "lexicon_sha256": SHA256, "protocol_sha256": digest(PROTOCOL),
        "topics": list(LABEL_TOPICS), "labels_collected": False, "quality_evaluated": False,
        "provider_requests": 0, "selection_status": "experimental_not_selected_not_promoted",
        "files": {name: digest(output / name) for name in ("key.json", "predictions.json", "sheet_A.csv", "sheet_B.csv")}}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify(run: Path) -> tuple[dict, dict[str, dict], dict]:
    manifest = json.loads((run / "manifest.json").read_bytes())
    if (manifest.get("kind") != "whole_review_lexicon_pilot" or manifest.get("lexicon_sha256") != SHA256
            or manifest.get("protocol_sha256") != digest(PROTOCOL)
            or set(manifest.get("files", {})) != {"key.json", "predictions.json", "sheet_A.csv", "sheet_B.csv"}):
        raise ValueError("pilot_definition_mismatch")
    for name, expected in manifest["files"].items():
        if digest(run / name) != expected:
            raise ValueError("pilot_artifact_changed")
    if set(manifest.get("source_hashes", {})) != set(SOURCES) or any(
            digest(ROOT / name) != value for name, value in manifest["source_hashes"].items()):
        raise ValueError("pilot_source_changed")
    _, rows = read_csv(run / "sheet_A.csv")
    original = {row["item"]: row for row in rows}
    if len(original) != manifest["sample_reviews"]:
        raise ValueError("sample_size_mismatch")
    predicted = json.loads((run / "predictions.json").read_bytes())
    if set(predicted) != set(original):
        raise ValueError("prediction_items_mismatch")
    classifier = LexiconClassifier()
    for item, row in original.items():
        result = classifier.classify(row["text"])
        expected = {entry.topic: entry.answer for entry in [*result.topics, result.other_complaint]}
        if row["text_sha256"] != text_digest(row["text"]) or predicted[item] != expected:
            raise ValueError("frozen_prediction_or_text_changed")
    return manifest, original, predicted


def completed(path: Path, original: dict[str, dict]) -> dict[str, dict]:
    fields, rows = read_csv(path)
    if set(fields) != set(COLUMNS) or len(rows) != len(original):
        raise ValueError("complete_original_sheet_required")
    filled = {}
    for row in rows:
        item = row["item"]
        if item not in original or item in filled:
            raise ValueError("sheet_items_mismatch")
        if row["text"] != original[item]["text"] or row["text_sha256"] != original[item]["text_sha256"]:
            raise ValueError("review_text_changed")
        if row["done"].strip().lower() != "yes" or any(row[topic].strip() not in LABELS for topic in LABEL_TOPICS):
            raise ValueError("complete_every_topic_and_done")
        filled[item] = {**row, **{topic: row[topic].strip() for topic in LABEL_TOPICS}}
    return filled


def agreement(run: Path, ratings_a: Path, ratings_b: Path) -> dict:
    manifest, original, _ = verify(run)
    a, b = completed(ratings_a, original), completed(ratings_b, original)
    results, differences = {}, []
    count = len(original)
    for topic in LABEL_TOPICS:
        ca = Counter(row[topic] for row in a.values())
        cb = Counter(row[topic] for row in b.values())
        observed = sum(a[item][topic] == b[item][topic] for item in original) / count
        expected = sum(ca[label] * cb[label] for label in LABELS) / count**2
        results[topic] = {"reviews": count, "agreement": observed,
            "cohens_kappa": (observed - expected) / (1 - expected) if expected < 1 else None}
        differences += [{"item": item, "topic": topic, "A": a[item][topic], "B": b[item][topic]}
                        for item in original if a[item][topic] != b[item][topic]]
    return {"kind": "whole_review_topic_agreement", "manifest_sha256": digest(run / "manifest.json"),
        "source_commit": manifest["source_commit"], "ratings_a_sha256": digest(ratings_a),
        "ratings_b_sha256": digest(ratings_b), "reviewer_independence_verified": False,
        "per_topic": results, "disagreements": differences, "quality_evaluated": False}


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p, denominator = successes / total, 1 + z*z / total
    middle = (p + z*z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z*z / (4 * total**2)) / denominator
    return [max(0.0, middle - spread), min(1.0, middle + spread)]


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "precision_wilson_95": wilson(tp, tp + fp), "recall_wilson_95": wilson(tp, tp + fn)}


def score(run: Path, ratings_a: Path, ratings_b: Path, agreement_file: Path, gold: Path) -> dict:
    recorded = json.loads(agreement_file.read_bytes())
    if recorded != agreement(run, ratings_a, ratings_b):
        raise ValueError("original_agreement_and_sheets_required")
    manifest, original, predictions = verify(run)
    a, b, labels = completed(ratings_a, original), completed(ratings_b, original), completed(gold, original)
    if any(a[item][topic] == b[item][topic] and labels[item][topic] != a[item][topic]
           and not labels[item]["note"].strip() for item in original for topic in LABEL_TOPICS):
        raise ValueError("explain_changes_to_agreed_labels")
    per_topic = {}
    totals = [0, 0, 0, 0]
    for topic in LABEL_TOPICS:
        counts = [0, 0, 0, 0]  # TP, FP, FN, TN; abstention is a nondetection, not excluded.
        uncertain_gold, abstentions = 0, 0
        for item in original:
            guess, truth = predictions[item][topic], labels[item][topic]
            abstentions += guess == "unsure"
            if truth == "unsure":
                uncertain_gold += 1
                continue
            positive = guess == "yes"
            index = 0 if positive and truth == "1" else 1 if positive else 2 if truth == "1" else 3
            counts[index] += 1
        per_topic[topic] = {**binary_metrics(*counts), "known_gold_reviews": sum(counts),
            "uncertain_gold_reviews": uncertain_gold, "abstentions_all_reviews": abstentions}
        if topic in TOPICS:
            totals = [a + b for a, b in zip(totals, counts, strict=True)]
    return {"kind": "whole_review_lexicon_development_pilot", "source_commit": manifest["source_commit"],
        "manifest_sha256": digest(run / "manifest.json"), "agreement_sha256": digest(agreement_file),
        "gold_sha256": digest(gold), "sample_reviews": len(original), "lexicon_sha256": SHA256,
        "quality_evaluated": True, "evaluation_scope": "topic_detection_development_pilot_only",
        "reviewer_independence_verified": False, "per_topic": per_topic,
        "supported_topics_micro": binary_metrics(*totals),
        "reviews_with_supported_match": sum(any(row[topic] == "yes" for topic in TOPICS) for row in predictions.values()),
        "human_referral_reviews": sum(any(value == "unsure" for value in row.values()) for row in predictions.values()),
        "interval_assumption": "Wilson per-topic binomial intervals; hotel/review dependence not modelled",
        "provider_requests": 0, "promotion": False, "production_reliability_established": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draw = commands.add_parser("prepare")
    draw.add_argument("--reviews", type=Path, required=True)
    draw.add_argument("--size", type=int, required=True)
    draw.add_argument("--seed", type=int, default=20261013)
    draw.add_argument("--population", required=True)
    for name in ("agreement", "score"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--ratings-a", type=Path, required=True)
        command.add_argument("--ratings-b", type=Path, required=True)
        if name == "score":
            command.add_argument("--agreement", type=Path, required=True)
            command.add_argument("--gold", type=Path, required=True)
    for command in (draw, *[commands.choices[name] for name in ("agreement", "score")]):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.output.resolve().is_relative_to((ROOT / "runs").resolve()):
            parser.error("--output must be a new private directory under runs/")
        result = prepare(args.reviews, args.output, size=args.size, seed=args.seed, population=args.population)
    else:
        result = (agreement(args.run, args.ratings_a, args.ratings_b) if args.command == "agreement" else
                  score(args.run, args.ratings_a, args.ratings_b, args.agreement, args.gold))
        write_json(args.output, result)
    print(args.command, "completed; provider requests: 0; no production promotion.")


if __name__ == "__main__":
    main()
