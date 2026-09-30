"""Benchmark TypeSafe Jev (System One) against the five-topic complaint gold.

Pre-registered comparison — the rule lives in
docs/experiments/jev_topic_benchmark/DECISION.md and is committed before the
first API call. The question rubrics are distilled from
complaint_topics_guideline.md; the gold key is used by the scorer only.

One request per review, five independent `choice` questions (one per topic,
answer space {1, 0, unsure}), sequential with backoff on 429/529.

The gold labels are read from the handed-in, adjudicated sheet_A.csv itself
(item + five topic columns). key.csv is never opened.

Usage (from the repository root):

    set TYPESAFE_API_KEY=sk-...
    python scripts/benchmark_jev_topics.py ^
        --items "C:\\path\\to\\pilot_sample\\sheet_A.csv" ^
        --limit 100 --seed 0 ^
        --output runs/jev_topic_benchmark

    # offline smoke test: no API call, writes the sample list and the first
    # request payload so you can inspect them before spending anything
    python scripts/benchmark_jev_topics.py --items ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOPICS = ["bathroom", "cleanliness", "air_conditioning", "pests", "responsiveness"]
API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# Rubrics distilled from complaint_topics_guideline.md only.
QUESTION_RUBRICS = {
    "bathroom": {
        "instructions": "Does the guest complain about the bathroom or shower?",
        "criteria": {
            "1": (
                "Complains about the bathroom, shower, bath/tub, toilet, sink or taps, "
                "or how they work or are built: size, layout, privacy, doors and locks, "
                "fittings, leaks, drainage, water pressure, hot water; or a missing bath "
                "or shower."
            ),
            "0": (
                "No such complaint. Dirt, hair, stains or mould in the bathroom is "
                "cleanliness, not bathroom. Towels, bathrobes, bath mats, hair dryer, "
                "toiletries and slippers are not bathroom. Hearing other rooms' toilets "
                "is noise."
            ),
            "unsure": "After two readings the text supports both readings.",
        },
    },
    "cleanliness": {
        "instructions": "Does the guest complain about cleanliness?",
        "criteria": {
            "1": (
                "Dirt, dust, stains, hair, mould or rubbish; a room or area that was not "
                "cleaned or was cleaned badly; cleaners who did not clean."
            ),
            "0": (
                "Praise ('clean', 'spotless'); housekeeping knocking, entering or waking "
                "the guest (staff behaviour); smells alone; worn or shabby furniture with "
                "no dirt described; missing supplies with no cleaning problem; dirt "
                "outside the hotel such as dirty streets."
            ),
            "unsure": "After two readings the text supports both readings.",
        },
    },
    "air_conditioning": {
        "instructions": "Does the guest complain about air conditioning, heating or ventilation?",
        "criteria": {
            "1": (
                "Air conditioning, heating, fans or ventilation; the room's temperature "
                "or its control; stuffy air; noise made by the air conditioner itself."
            ),
            "0": (
                "Shower water temperature (bathroom); the weather outside; the temperature "
                "of a pool or other facilities; a view of air-conditioning units; outside "
                "noise that made the guest close the windows."
            ),
            "unsure": "After two readings the text supports both readings.",
        },
    },
    "pests": {
        "instructions": "Does the guest complain about pests?",
        "criteria": {
            "1": (
                "Insects, bed bugs or their bites, cockroaches, ants, flies, mosquitoes, "
                "spiders, mice or rats, anywhere in the hotel."
            ),
            "0": (
                "'Bite' meaning food; a computer mouse; a 'flea market'; a place named "
                "after an animal; a figure of speech."
            ),
            "unsure": "After two readings the text supports both readings.",
        },
    },
    "responsiveness": {
        "instructions": "Does the guest complain about the hotel's responsiveness?",
        "criteria": {
            "1": (
                "The hotel or its staff did not answer, reply, call back or act on a "
                "request or complaint, or did so very late: unanswered phone calls or "
                "emails, ignored requests, promises to follow up that were not kept, a "
                "reported problem left for days, a change the guest was not told about."
            ),
            "0": (
                "Staff who answered but were rude, unfriendly or unhelpful (attitude); a "
                "language barrier; an answer the guest did not like; a device that is "
                "'unresponsive'; praise; Booking.com, not the hotel, failing."
            ),
            "unsure": "After two readings the text supports both readings.",
        },
    },
}


def build_questions() -> dict:
    return {
        topic: {"type": "choice", **QUESTION_RUBRICS[topic]} for topic in TOPICS
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_items(path: Path) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r]
    header = [c.strip().lower() for c in rows[0]]
    try:
        i_item, i_text = header.index("item"), header.index("text")
    except ValueError as e:
        raise SystemExit(f"{path}: expected 'item' and 'text' columns, got {header}") from e
    return {int(r[i_item]): r[i_text] for r in rows[1:] if len(r) > max(i_item, i_text)}


def load_gold(path: Path) -> dict[int, dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r]
    header = [c.strip().lower() for c in rows[0]]
    if not all(t in header for t in TOPICS):
        raise SystemExit(f"{path}: expected topic columns {TOPICS}, got {header}")
    gold: dict[int, dict[str, str]] = {}
    for r in rows[1:]:
        if not r:
            continue
        item = int(r[header.index("item")])
        gold[item] = {t: str(r[header.index(t)]).strip().lower() for t in TOPICS}
    return gold


def stratified_sample(gold: dict[int, dict[str, str]], limit: int, seed: int) -> list[int]:
    """Gold-positives first per topic (deterministic, seed 0), then fill with the rest."""
    rng = random.Random(seed)
    if limit <= 0 or limit >= len(gold):
        return sorted(gold)
    chosen: set[int] = set()
    for topic in TOPICS:
        positives = [i for i, g in gold.items() if g.get(topic) == "1"]
        rng.shuffle(positives)
        for i in positives:
            if len(chosen) >= limit:
                break
            chosen.add(i)
    rest = sorted(set(gold) - chosen)
    rng.shuffle(rest)
    for i in rest:
        if len(chosen) >= limit:
            break
        chosen.add(i)
    return sorted(chosen)


def call_jev(state: str, api_key: str, timeout: float, max_retries: int = 5) -> tuple[dict, float]:
    """One review -> five answers. Returns (response, elapsed_seconds)."""
    payload = json.dumps({"state": state, "model": MODEL, "questions": build_questions()}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())
            return body, time.perf_counter() - start
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise SystemExit(f"API error {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise SystemExit(f"network error: {e}") from e
    raise SystemExit("unreachable")


def parse_answers(resp: dict) -> dict[str, tuple[str, float, dict]]:
    out: dict[str, tuple[str, float, dict]] = {}
    for topic in TOPICS:
        ans = resp["answers"][topic]
        out[topic] = (str(ans["choice"]).strip().lower(), float(ans["confidence"]), ans)
    return out


def prf(preds: list[str], golds: list[str], positive: str = "1") -> tuple[float, float, float, int]:
    tp = sum(p == positive and g == positive for p, g in zip(preds, golds))
    fp = sum(p == positive and g != positive for p, g in zip(preds, golds))
    fn = sum(p != positive and g == positive for p, g in zip(preds, golds))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp + fp + fn


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def ece(confs: list[float], correct: list[bool], bins: int = 10) -> float:
    if not confs:
        return 0.0
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(c, ok) for c, ok in zip(confs, correct) if lo <= c < hi or (i == bins - 1 and c == hi)]
        if not bucket:
            continue
        acc = sum(ok for _, ok in bucket) / len(bucket)
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        total += len(bucket) / len(confs) * abs(acc - avg_conf)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", required=True, help="CSV with item,text (e.g. the handed-in sheet_A.csv)")
    ap.add_argument("--gold", default=None, help="gold CSV: item + five topic columns; default = the topic columns of --items")
    ap.add_argument("--limit", type=int, default=100, help="stratified sample size (0 = all items)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="runs/jev_topic_benchmark")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--eur-per-1k-input-tokens", type=float, default=None, help="price, for the cost estimate")
    ap.add_argument("--eur-per-1k-output-tokens", type=float, default=None, help="price, for the cost estimate")
    ap.add_argument("--dry-run", action="store_true", help="no API call: write sample + first payload, exit")
    args = ap.parse_args()

    items = load_items(Path(args.items))
    gold = load_gold(Path(args.gold or args.items))
    gold_sha = sha256_file(Path(args.gold or args.items))
    sample = stratified_sample(gold, args.limit, args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample_items.json"
    sample_path.write_text(json.dumps({"seed": args.seed, "limit": args.limit, "items": sample}, indent=2))
    print(f"sample: {len(sample)} items (seed={args.seed}) -> {sample_path}")

    if args.dry_run:
        first = sample[0]
        payload = {"state": items[first], "model": MODEL, "questions": build_questions()}
        (out_dir / "first_request_payload.json").write_text(json.dumps(payload, indent=2)[:200000])
        print(f"dry-run: wrote first_request_payload.json for item {first}; no API call made")
        return

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit("set the TYPESAFE_API_KEY environment variable")

    predictions: list[dict] = []
    raw_path = out_dir / "responses.jsonl"
    with open(raw_path, "w", encoding="utf-8") as raw_f:
        for n, item in enumerate(sample, start=1):
            resp, elapsed = call_jev(items[item], api_key, args.timeout)
            answers = parse_answers(resp)
            raw_f.write(json.dumps({"item": item, "elapsed_s": elapsed, "response": resp}) + "\n")
            row: dict = {"item": item, "elapsed_s": round(elapsed, 4)}
            for topic in TOPICS:
                row[topic], row[f"{topic}_conf"] = answers[topic][0], round(answers[topic][1], 4)
            predictions.append(row)
            usage = resp.get("usage", {})
            print(f"[{n}/{len(sample)}] item {item} in {elapsed:.2f}s tok={usage.get('input_tokens', '?')}+{usage.get('output_tokens', '?')}")

    pred_path = out_dir / "predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    summary: dict = {
        "model": MODEL,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256_file(Path(__file__)),
        "items_csv_sha256": sha256_file(Path(args.items)),
        "gold_sha256": gold_sha,
        "sample_sha256": sha256_file(sample_path),
        "seed": args.seed,
        "n": len(sample),
        "per_topic": {},
        "token_usage": {"input": 0, "output": 0},
        "latency_p50_s": statistics.median(r["elapsed_s"] for r in predictions),
        "latency_p95_s": sorted(r["elapsed_s"] for r in predictions)[int(0.95 * len(predictions)) - 1],
    }

    for topic in TOPICS:
        p = [r[topic] for r in predictions]
        g = [gold[r["item"]][topic] for r in predictions]
        prec, rec, f1, n_eval = prf(p, g)
        unsure_rate = sum(x == "unsure" for x in p) / len(p)
        conf, correct = [], []
        for r in predictions:
            if r[topic] in ("0", "1"):
                conf.append(r[f"{topic}_conf"])
                correct.append(r[topic] == gold[r["item"]][topic])
        summary["per_topic"][topic] = {
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "n_scored": n_eval, "unsure_rate": round(unsure_rate, 4),
            "kappa": round(cohen_kappa(p, g), 4), "ece": round(ece(conf, correct), 4),
        }

    usage_in = usage_out = 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)["response"].get("usage", {})
            usage_in += u.get("input_tokens", 0)
            usage_out += u.get("output_tokens", 0)
    summary["token_usage"] = {"input": usage_in, "output": usage_out}
    if args.eur_per_1k_input_tokens is not None or args.eur_per_1k_output_tokens is not None:
        summary["cost_eur_per_1000_reviews"] = round(
            (usage_in / 1000 * (args.eur_per_1k_input_tokens or 0) + usage_out / 1000 * (args.eur_per_1k_output_tokens or 0)) / len(sample) * 1000,
            4,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["per_topic"], indent=2))
    print(f"wrote {pred_path} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()