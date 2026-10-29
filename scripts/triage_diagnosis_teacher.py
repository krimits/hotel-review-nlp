"""Run the teacher of the triage diagnosis protocol through OpenRouter, and choose it on the development set.

PROTOCOL.md (version 4, "The teacher") fixes the steps, in this order:

    set OPENROUTER_API_KEY=...       the owner's key, with a spending limit set on OpenRouter
    python scripts/triage_diagnosis_teacher.py label --model haiku --subset development --max-usd 1
    python scripts/triage_diagnosis_teacher.py label --model sonnet --subset development --max-usd 2
    python scripts/triage_diagnosis_teacher.py choose
    python scripts/triage_diagnosis_teacher.py label --model chosen --subset teacher --max-usd 10
    python scripts/triage_diagnosis_teacher.py label --model chosen --subset test --max-usd 2
    python scripts/triage_diagnosis_teacher.py label --model chosen --subset check --max-usd 1

`label` sends the subset's reviews, 25 per call in item order, with teacher_prompt.md as the system message and
structured output (the JSON schema of the catalogue's types); nothing else is set, as these models refuse
sampling parameters. A call whose output does not parse, misses or repeats a review, or names an unknown type is
retried at most twice; a review still unlabelled is recorded as failed. Rate limits and server errors are waited
out and do not count as tries; any other HTTP error stops the run. Every raw response is saved under
runs/triage_diagnosis/teacher (git-ignored) and hashed. A run stops before a call once --max-usd is spent and
resumes where it stopped: finished batches are never sent again.

When every batch is finished it writes, with no text, labels/teacher_<model>_<subset>.csv (raw_row, set, 0/1 per
type, none, failed) and its record (the model, the prompt's SHA-256, every response's SHA-256, tokens and cost).

`choose` applies the rule: the teacher is claude-sonnet-5.5 if its development recall against reference_chatgpt
leads claude-haiku-5.5's by at least 0.05, otherwise claude-haiku-5.5. It writes labels/teacher_choice.json.

The API key is read from the environment, sent only to OpenRouter, and never written or printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import triage_diagnosis_labels as labels  # noqa: E402

from reviewnlp.triage import diagnosis_metrics as metrics  # noqa: E402
from reviewnlp.triage.jev_client import retry_delay, urllib_transport  # noqa: E402

sample = labels.sample
URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {"haiku": "anthropic/claude-haiku-5.5-20261007", "sonnet": "anthropic/claude-sonnet-5.5-20260928"}
# USD per million tokens (input, output), OpenRouter on 29 October 2026; used only when a response reports no cost.
PRICES = {"haiku": (0.10, 0.50), "sonnet": (2.00, 10.00)}
BATCH = 25  # PROTOCOL.md version 4: 25 reviews per call
TRIES = 3  # the first call and at most two retries
TRANSIENT = {408, 429, 500, 502, 503, 504}
TRANSIENT_WAITS = 5
MARGIN = 0.05  # PROTOCOL.md version 4: Sonnet only if it leads by at least 0.05
SUBSETS = {  # subset -> (sheet under runs/, key under samples/, set to keep or None for all)
    "development": ("sheet.csv", "key.csv", "development"),
    "test": ("sheet.csv", "key.csv", "test"),
    "check": ("check_sheet.csv", "check_key.csv", None),
    "teacher": ("teacher_sample.csv", "teacher_key.csv", None),
}
Transport = Callable[[str, dict, str, float], tuple[int, dict, bytes]]


class OutputError(ValueError):
    """A response whose labels cannot be used; the call is tried again."""


class Stop(RuntimeError):
    """The run stops here; finished batches are kept and a later run resumes after them."""


def catalogue() -> dict:
    return yaml.safe_load(sample.CATALOGUE.read_text(encoding="utf-8"))


def load_subset(subset: str, runs: Path = labels.RUNS, samples: Path = labels.SAMPLES) -> pd.DataFrame:
    """The subset's reviews in item order, each text checked against the key's hash."""
    sheet_name, key_name, keep = SUBSETS[subset]
    texts = pd.read_csv(runs / sheet_name, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(samples / key_name, dtype=str, keep_default_na=False)
    if keep:
        key = key[key["set"] == keep]
    table = key.merge(texts[["item", "text"]], on="item", how="left", validate="one_to_one")
    if table["text"].isna().any() or [sample.text_hash(t) for t in table["text"]] != table["text_sha256"].tolist():
        raise Stop(f"{runs / sheet_name} does not hold the texts {key_name} records")
    table["item"] = table["item"].astype(int)
    return table.sort_values("item").reset_index(drop=True)


def schema(types: list[str]) -> dict:
    entry = {"type": "object", "additionalProperties": False, "required": ["item", "types"],
             "properties": {"item": {"type": "integer"},
                            "types": {"type": "array", "items": {"type": "string", "enum": types}}}}
    return {"type": "object", "additionalProperties": False, "required": ["labels"],
            "properties": {"labels": {"type": "array", "items": entry}}}


def request_body(model: str, prompt: str, reviews: list[dict], types: list[str]) -> dict:
    """One call: the prompt as system message, the reviews as a JSON list, the catalogue's schema; nothing else."""
    return {"model": model,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(reviews, ensure_ascii=False)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "labels", "strict": True, "schema": schema(types)}},
            "usage": {"include": True}}


def parse(raw: bytes, items: list[int], types: list[str]) -> dict[int, list[str]]:
    """The labels of one response, or OutputError if they do not cover exactly the items sent."""
    try:
        content = json.loads(raw)["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        entries = json.loads(content)["labels"]
        pairs = [(int(entry["item"]), list(entry["types"])) for entry in entries]
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise OutputError(f"unparsable output: {type(error).__name__}") from None
    got = [item for item, _ in pairs]
    if sorted(got) != sorted(items) or len(set(got)) != len(got):
        raise OutputError(f"items {sorted(set(items) ^ set(got)) or 'repeated'} differ from those sent")
    unknown = sorted({t for _, ts in pairs for t in ts} - set(types))
    if unknown:
        raise OutputError(f"unknown types {unknown[:5]}")
    return {item: sorted(set(ts)) for item, ts in pairs}


def usage(raw: bytes, model_key: str) -> dict:
    """Tokens and cost of a response; the cost estimated from PRICES when the response reports none."""
    try:
        reported = json.loads(raw).get("usage") or {}
    except ValueError:
        reported = {}
    prompt, completion = int(reported.get("prompt_tokens") or 0), int(reported.get("completion_tokens") or 0)
    cost = reported.get("cost")
    if cost is None:
        price_in, price_out = PRICES[model_key]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "cost_usd": (prompt * price_in + completion * price_out) / 1e6, "cost_source": "estimated"}
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cost_usd": float(cost),
            "cost_source": "reported"}


def send(transport: Transport, body: dict, api_key: str, sleep: Callable[[float], None]) -> bytes:
    """One call, waiting out rate limits and server errors; any other non-200 status stops the run."""
    for wait in range(1, TRANSIENT_WAITS + 2):
        status, headers, raw = transport(URL, body, api_key, 120.0)
        if status == 200:
            return raw
        if status not in TRANSIENT or wait > TRANSIENT_WAITS:
            raise Stop(f"OpenRouter answered {status}: {raw[:300].decode('utf-8', 'replace')}")
        sleep(retry_delay(headers, wait))
    raise AssertionError("unreachable")


def model_key_of(model: str, labels_dir: Path) -> str:
    if model != "chosen":
        return model
    choice = labels_dir / "teacher_choice.json"
    if not choice.exists():
        raise Stop("no teacher_choice.json yet: run `choose` after both candidates label the development set")
    return json.loads(choice.read_text(encoding="utf-8"))["teacher"]


def label(model: str, subset: str, max_usd: float, transport: Transport | None = None, api_key: str | None = None,
          runs: Path = labels.RUNS, samples: Path = labels.SAMPLES, labels_dir: Path = labels.LABELS,
          sleep: Callable[[float], None] = time.sleep) -> dict:
    """Label a subset in batches, resuming after finished ones; write the labels when every batch is finished."""
    model_key = model_key_of(model, labels_dir)
    name = f"teacher_{model_key}_{subset}"
    if (labels_dir / f"{name}.csv").exists():
        raise Stop(f"{labels_dir / f'{name}.csv'} exists; recorded labels are never replaced")
    api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise Stop("set OPENROUTER_API_KEY first; nothing was sent")
    transport = transport or urllib_transport
    data = catalogue()
    types = [entry["id"] for entry in data["types"]]
    prompt = labels.TEACHER_PROMPT.read_text(encoding="utf-8")
    if prompt != labels.teacher_prompt():
        raise Stop(f"{labels.TEACHER_PROMPT} does not match the catalogue")
    reviews = load_subset(subset, runs, samples)
    folder = runs / "teacher" / model_key / subset
    folder.mkdir(parents=True, exist_ok=True)
    batches = [reviews.iloc[start:start + BATCH] for start in range(0, len(reviews), BATCH)]
    # Every saved response counts, including tries of a batch the run stopped in.
    spent = sum(usage(path.read_bytes(), model_key)["cost_usd"] for path in folder.glob("batch_*.response.json"))
    for number, part in enumerate(batches, start=1):
        record_path = folder / f"batch_{number:04d}.json"
        if record_path.exists():
            continue
        items = part["item"].tolist()
        body = request_body(MODELS[model_key], prompt, [{"item": i, "text": t} for i, t in zip(
            items, part["text"], strict=True)], types)
        attempts, found = [], None
        for attempt in range(1, TRIES + 1):
            path = folder / f"batch_{number:04d}.try{attempt}.response.json"
            if path.exists():
                raw = path.read_bytes()  # a try made before the run stopped: it counts, and is not sent again
            else:
                if spent >= max_usd:
                    raise Stop(f"--max-usd {max_usd} reached (${spent:.4f} spent); rerun with a higher limit "
                               "to resume")
                raw = send(transport, body, api_key, sleep)
                path.write_bytes(raw)
                spent += usage(raw, model_key)["cost_usd"]
            cost = usage(raw, model_key)
            try:
                found = parse(raw, items, types)
                outcome = "ok"
            except OutputError as error:
                outcome = str(error)
            attempts.append({"try": attempt, "sha256": hashlib.sha256(raw).hexdigest(), **cost, "outcome": outcome})
            if found is not None:
                break
        result = {"items": items, "labels": {str(i): found[i] for i in items} if found else None,
                  "failed": [] if found else items, "attempts": attempts,
                  "cost_usd": sum(a["cost_usd"] for a in attempts)}
        record_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return write_labels(model_key, subset, reviews, folder, labels_dir, types, len(batches))


def write_labels(model_key: str, subset: str, reviews: pd.DataFrame, folder: Path, labels_dir: Path,
                 types: list[str], batches: int) -> dict:
    found: dict[int, list[str]] = {}
    failed: set[int] = set()
    records = [json.loads((folder / f"batch_{n:04d}.json").read_text(encoding="utf-8")) for n in range(1, batches + 1)]
    for record in records:
        failed |= set(record["failed"])
        found.update({int(item): ts for item, ts in (record["labels"] or {}).items()})
    rows = []
    for item, raw_row, set_name in zip(reviews["item"], reviews["raw_row"], reviews["set"], strict=True):
        present = found.get(item, [])
        rows.append({"raw_row": int(raw_row), "set": set_name, **{t: int(t in present) for t in types},
                     "none": int(item not in failed and not present), "failed": int(item in failed)})
    table = pd.DataFrame(rows).sort_values("raw_row")
    name = f"teacher_{model_key}_{subset}"
    labels_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(labels_dir / f"{name}.csv", index=False, lineterminator="\n")
    attempts = [a for record in records for a in record["attempts"]]
    record = {
        "labels": name, "model": MODELS[model_key], "subset": subset,
        "protocol_sha256": sample.trends._sha256(sample.PROTOCOL),
        "teacher_prompt_sha256": sample.trends._sha256(labels.TEACHER_PROMPT),
        "code_commit": sample.pilot.git_commit(), "finished_on": dt.date.today().isoformat(),
        "reviews": len(table), "failed_reviews": len(failed), "batches": batches, "calls": len(attempts),
        "retries": len(attempts) - batches,
        "prompt_tokens": sum(a["prompt_tokens"] for a in attempts),
        "completion_tokens": sum(a["completion_tokens"] for a in attempts),
        "cost_usd": round(sum(a["cost_usd"] for a in attempts), 6),
        "cost_sources": sorted({a["cost_source"] for a in attempts}),
        "responses": {f"batch_{n:04d}": [a["sha256"] for a in r["attempts"]] for n, r in enumerate(records, start=1)},
    }
    (labels_dir / f"{name}_record.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8",
                                                    newline="\n")
    return record


def choose(labels_dir: Path = labels.LABELS, samples: Path = labels.SAMPLES) -> dict:
    """Apply version 4's rule on the development set and record the teacher."""
    out = labels_dir / "teacher_choice.json"
    if out.exists():
        raise Stop(f"{out} exists; the teacher is chosen once")
    data = catalogue()
    urgent = metrics.urgent_departments(data["types"])
    key = pd.read_csv(samples / "key.csv")
    weights = key[key["set"] == "development"].set_index("raw_row")["weight"]
    reference = pd.read_csv(labels_dir / "reference_chatgpt.csv").set_index("raw_row").loc[weights.index]
    result: dict = {"rule": f"sonnet if rec_dev(sonnet) - rec_dev(haiku) >= {MARGIN}, otherwise haiku",
                    "rec_dev": {}, "fa_dev": {}, "cost_usd": {}, "failed_reviews": {}}
    for model_key in MODELS:
        name = f"teacher_{model_key}_development"
        if not (labels_dir / f"{name}.csv").exists():
            raise Stop(f"{name}.csv is missing: both candidates label the development set first")
        pred = pd.read_csv(labels_dir / f"{name}.csv").set_index("raw_row")
        counts = metrics.per_review(reference, pred, urgent)
        result["rec_dev"][model_key] = round(metrics.department_urgent_recall(counts, weights), 6)
        result["fa_dev"][model_key] = round(metrics.false_urgent_per_100(counts, weights), 4)
        record = json.loads((labels_dir / f"{name}_record.json").read_text(encoding="utf-8"))
        result["cost_usd"][model_key] = record["cost_usd"]
        result["failed_reviews"][model_key] = record["failed_reviews"]
    lead = result["rec_dev"]["sonnet"] - result["rec_dev"]["haiku"]
    result["sonnet_lead"] = round(lead, 6)
    result["teacher"] = "sonnet" if lead >= MARGIN else "haiku"
    result["model"] = MODELS[result["teacher"]]
    result["code_commit"] = sample.pilot.git_commit()
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("label", help="label a subset with a candidate or the chosen teacher")
    first.add_argument("--model", choices=[*MODELS, "chosen"], required=True)
    first.add_argument("--subset", choices=sorted(SUBSETS), required=True)
    first.add_argument("--max-usd", type=float, required=True, help="stop before a call once this much is spent")
    steps.add_parser("choose", help="apply the development rule and record the teacher")
    args = parser.parse_args()
    try:
        if args.step == "label":
            record = label(args.model, args.subset, args.max_usd)
            print(f"{record['labels']}: {record['reviews']} reviews, {record['failed_reviews']} failed, "
                  f"{record['calls']} calls ({record['retries']} retries), ${record['cost_usd']:.4f}")
        else:
            result = choose()
            print(f"rec_dev {result['rec_dev']}, fa_dev {result['fa_dev']}: teacher {result['model']}")
    except Stop as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
