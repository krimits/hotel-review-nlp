"""Compare TypeSafe's Jev with the complaint lexicon on the pilot's random texts.

The design and the decision rule are fixed in
docs/experiments/jev_topic_benchmark/DECISION_v2.md, committed before any request is sent.
It replaces the first draft, DECISION.md, under which no request was sent.

- Texts: the pilot's 200 random texts (in_R in key.csv). The other 100 were picked because
  the lexicon matched them, so they would favour its recall; they are left out.
- Gold: the pilot's final labels (labels_final.csv).
- Lexicon: its outputs for the same texts (the lex_* columns of key.csv).
- Jev: one request per text, one `choice` question per topic with the labels 1, 0 and
  unsure. The rules come from the labelling guideline only.

The texts come from a local sheet with `item` and `text` columns, such as the handed-in
sheet_A.csv; no other column is read. Nothing leaves this machine unless
--allow-external-api is given, and everything a run writes goes under runs/, which git
ignores. The API key is read from TYPESAFE_API_KEY and never written anywhere.

    # offline: checks the inputs and writes the first request to runs/; sends nothing
    python scripts/benchmark_jev_topics.py --texts path/to/sheet_A.csv

    # the real run, only once the provider's data terms have been checked
    set TYPESAFE_API_KEY=...
    python scripts/benchmark_jev_topics.py --texts path/to/sheet_A.csv --allow-external-api

A run that stops part-way resumes from its saved answers when the same command is run
again. --score-only scores the saved answers without sending anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "case_study" / "results" / "annotation"
RUNS = ROOT / "runs"
DESIGN = "docs/experiments/jev_topic_benchmark/DECISION_v2.md"

TOPICS = ("bathroom", "cleanliness", "air_conditioning", "pests", "responsiveness")
LABELS = ("1", "0", "unsure")
API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
KEY_ENV = "TYPESAFE_API_KEY"

# The decision rule of DECISION_v2.md.
PRIMARY_TOPIC = "responsiveness"
ALPHA = 0.05
MIN_PRECISION = 0.5

# Reporting, as in the pilot: a proportion on fewer than 10 texts is shown as counts only.
MIN_DENOMINATOR = 10
Z95 = 1.959964
CALIBRATION_BINS = 10

# Retried as the provider's SDK does: timeouts, rate limits and server errors.
RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
MAX_ATTEMPTS = 6

# The questions: the labelling guideline's rules (docs/annotation/complaint_topics_guideline.md).
CONTEXT = ("The text is what a hotel guest wrote when Booking.com asked what they did not like. "
           "It has no punctuation, so sentences run together.")
RULES = ("A complaint is a problem, a lack or a negative judgement; a suggestion or wish that "
         "implies a lack; a problem anywhere in the hotel, not only in the room; or a mild "
         "complaint. Not complaints: praise, a neutral mention, saying nothing was wrong, and a "
         "topic that is only the setting of another complaint. Judge what the guest complains "
         "about: 'kept the windows closed and the air con on because of the street noise' is a "
         "noise complaint, so air conditioning is 0. Each topic is judged on its own.")
UNSURE = ("Only when the text supports both readings, for example 'bathroom disgusting' with no "
          "reason given, or 'the fan' with no hint of which fan. Not for mild complaints or rare "
          "topics: a mild complaint is still 1.")
TOPIC_RULES = {
    "bathroom": (
        "Does the guest complain about the bathroom or shower?",
        "The bathroom, the shower, the bath or tub, the toilet, the sink and taps, or how they "
        "work or are built: size, layout, privacy, doors and locks, fittings, leaks, drainage, "
        "water pressure, hot water; or a missing bath or shower.",
        "No such complaint. These do not count: dirt, hair, stains or mould in the bathroom, "
        "which are cleanliness (mark bathroom 1 only if something else about the bathroom is "
        "also criticised); towels, bathrobes and bath mats, which are linen; a hair dryer, "
        "toiletries and slippers, which are amenities; hearing other rooms' toilets, which is "
        "noise.",
    ),
    "cleanliness": (
        "Does the guest complain about cleanliness?",
        "Dirt, dust, stains, hair, mould or rubbish; a room or area that was not cleaned or was "
        "cleaned badly; cleaners who did not clean.",
        "No such complaint. These do not count: praise ('clean', 'spotless'); housekeeping "
        "knocking, entering or waking the guest, which is about staff; smells alone, which is "
        "another topic; worn or shabby furniture with no dirt described; missing supplies (milk, "
        "toilet paper) with no cleaning problem; dirt outside the hotel, such as dirty streets.",
    ),
    "air_conditioning": (
        "Does the guest complain about air conditioning, heating or ventilation?",
        "Air conditioning, heating, fans or ventilation; the room's temperature or its control; "
        "stuffy air; noise made by the air conditioner itself.",
        "No such complaint. These do not count: the water temperature in the shower, which is "
        "bathroom; the weather outside; the temperature of a pool or other facilities; a view of "
        "air-conditioning units; outside noise that made the guest close the windows.",
    ),
    "pests": (
        "Does the guest complain about pests?",
        "Insects, bed bugs or their bites, cockroaches, ants, flies, mosquitoes, spiders, mice or "
        "rats, anywhere in the hotel.",
        "No such complaint. These do not count: 'bite' meaning food ('a quick bite', "
        "'bite-size'); a computer mouse; a 'flea market'; a place named after an animal; a "
        "figure of speech ('we felt like rats').",
    ),
    "responsiveness": (
        "Does the guest complain about the hotel's responsiveness?",
        "The hotel or its staff did not answer, reply, call back or act on a request or "
        "complaint, or did so very late: unanswered phone calls or emails, ignored requests, "
        "promises to follow up that were not kept, a reported problem left for days, a change "
        "the guest was not told about.",
        "No such complaint. These do not count: staff who did answer but were rude, unfriendly "
        "or unhelpful, which is about attitude; a language barrier; an answer the guest did not "
        "like ('the answer was no'); a device that is 'unresponsive'; praise ('they responded "
        "quickly'); Booking.com, not the hotel, failing to pass something on.",
    ),
}
QUESTIONS = {topic: {"type": "choice", "instructions": f"{ask} {CONTEXT} {RULES}",
                     "criteria": {"1": complaint, "0": other, "unsure": UNSURE}}
             for topic, (ask, complaint, other) in TOPIC_RULES.items()}
QUESTIONS_SHA256 = hashlib.sha256(
    json.dumps({"model": MODEL, "questions": QUESTIONS}, sort_keys=True).encode("utf-8")).hexdigest()


def request_body(text: str) -> dict:
    return {"state": text, "model": MODEL, "questions": QUESTIONS}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: Path, columns: set[str]) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = columns - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def read_texts(path: Path) -> dict[int, str]:
    """Item -> review text from a local sheet; only its item and text columns are read."""
    texts: dict[int, str] = {}
    for row in _rows(path, {"item", "text"}):
        item = int(row["item"])
        if item in texts:
            raise SystemExit(f"{path}: item {item} appears twice")
        texts[item] = row["text"]
    return texts


def read_key(path: Path) -> dict[int, dict]:
    """Item -> whether it is a random text, and whether the lexicon matched each topic."""
    return {int(row["item"]): {"in_R": row["in_R"] == "1", **{t: row[f"lex_{t}"] == "1" for t in TOPICS}}
            for row in _rows(path, {"item", "in_R", *(f"lex_{t}" for t in TOPICS)})}


def read_gold(path: Path) -> dict[int, dict[str, str]]:
    """Item -> final label per topic: '1', '0' or 'unsure'."""
    gold = {}
    for row in _rows(path, {"item", "annotator", *TOPICS}):
        if row["annotator"] != "final" or any(row[t] not in LABELS for t in TOPICS):
            raise SystemExit(f"{path}: item {row['item']} is not a final label of 1, 0 or unsure")
        gold[int(row["item"])] = {t: row[t] for t in TOPICS}
    return gold


def random_items(key: dict[int, dict], gold: dict[int, dict], limit: int | None) -> list[int]:
    """The random texts in item order; --limit keeps the first N, for a smoke run."""
    items = sorted(item for item, row in key.items() if row["in_R"])
    if not set(items) <= set(gold):
        raise SystemExit("the gold labels miss some random texts")
    return items if limit is None else items[:limit]


def output_dir(path: Path) -> Path:
    """The run folder. It holds review texts and raw answers, so it must be inside runs/."""
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    runs = RUNS.resolve()
    if resolved == runs or not resolved.is_relative_to(runs):
        raise SystemExit(f"--output must be a folder inside {runs}, which git ignores")
    return resolved


# Sending ---------------------------------------------------------------------------------------

def post(body: dict, api_key: str, timeout: float) -> tuple[int, dict[str, str], bytes]:
    """One POST to the System One endpoint: (status, headers, body). Network failures raise OSError."""
    request = urllib.request.Request(
        API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()


def retry_delay(headers: dict[str, str], attempt: int) -> float:
    """Seconds to wait: the server's retry-after-ms or retry-after, else 2, 4, 8... up to 60."""
    lowered = {name.lower(): value for name, value in headers.items()}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            return min(max(float(lowered[name]) * scale, 0.0), 300.0)
        except (KeyError, ValueError):
            continue
    return min(2.0 ** attempt, 60.0)


def ask(text: str, api_key: str, timeout: float) -> tuple[dict, float, int]:
    """Send one text: (response, seconds of the answered attempt, attempts). Stops the run on an error it cannot retry."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            status, headers, raw = post(request_body(text), api_key, timeout)
        except (OSError, http.client.HTTPException) as error:
            status, headers, problem = None, {}, f"network error: {error!r}"
        seconds = time.perf_counter() - start
        if status == 200:
            try:
                return json.loads(raw), seconds, attempt
            except json.JSONDecodeError:
                return {"unreadable_body": raw.decode("utf-8", "replace")[:2000]}, seconds, attempt
        if status is not None:
            problem = f"HTTP {status}: {raw.decode('utf-8', 'replace')[:300]}"
        if (status is None or status in RETRY_STATUSES) and attempt < MAX_ATTEMPTS:
            time.sleep(retry_delay(headers, attempt))
            continue
        raise SystemExit(f"stopped: {problem}\nThe answers so far are saved; run the same command "
                         "again to go on from there.")
    raise AssertionError("unreachable")


def read_answers(response: dict) -> dict[str, dict]:
    """Topic -> the chosen label and its probabilities, from an answer in the documented form."""
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("no answers")
    out = {}
    for topic in TOPICS:
        answer = answers.get(topic)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"{topic}: not a choice answer")
        choice, probabilities = answer.get("choice"), answer.get("probabilities")
        if choice not in LABELS or not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
            raise ValueError(f"{topic}: choice {choice!r} or probabilities {probabilities!r} "
                             f"do not use the labels {LABELS}")
        try:
            out[topic] = {"choice": choice, "p_choice": float(probabilities[choice]),
                          "p1": float(probabilities["1"]), "confidence": float(answer["confidence"])}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{topic}: a probability or the confidence is not a number ({error!r})") from None
    return out


def read_log(path: Path) -> tuple[dict[int, dict], list[dict]]:
    """(item -> its first readable record, every record). Records made with other questions are refused."""
    usable: dict[int, dict] = {}
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # a line cut short when a run was killed
                continue
            if record.get("questions_sha256") != QUESTIONS_SHA256:
                raise SystemExit(f"{path} holds answers to other questions; use a new --output")
            records.append(record)
            try:
                read_answers(record["response"])
            except ValueError:
                continue
            usable.setdefault(record["item"], record)
    return usable, records


def check_run_record(output: Path, inputs: dict, create: bool) -> None:
    """The inputs of the saved answers must be the ones given now."""
    path = output / "run.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != inputs:
            raise SystemExit(f"{path}: the texts, labels, key or questions differ from those of the "
                             "saved answers; use a new --output")
    elif create:
        path.write_text(json.dumps(inputs, indent=1) + "\n", encoding="utf-8")
    else:
        raise SystemExit(f"{path} is missing: there are no saved answers to score")


def send_all(items: list[int], texts: dict[int, str], output: Path, api_key: str, timeout: float) -> None:
    log_path = output / "responses.jsonl"
    usable, _ = read_log(log_path)
    todo = [item for item in items if item not in usable]
    print(f"{len(items) - len(todo)} of {len(items)} texts already answered; sending {len(todo)}")
    with open(log_path, "a", encoding="utf-8") as log:
        for count, item in enumerate(todo, 1):
            sent = datetime.now(timezone.utc).isoformat(timespec="seconds")
            response, seconds, attempts = ask(texts[item], api_key, timeout)
            log.write(json.dumps({"item": item, "questions_sha256": QUESTIONS_SHA256, "sent_utc": sent,
                                  "seconds": round(seconds, 4), "attempts": attempts,
                                  "response": response}) + "\n")
            log.flush()
            try:
                read_answers(response)
            except ValueError as error:
                raise SystemExit(f"item {item}: the answer is not in the documented form ({error}). "
                                 f"It is saved in {log_path}; nothing more was sent.") from None
            tokens = (response.get("usage") or {}).get("input_tokens", "?")
            print(f"[{count}/{len(todo)}] item {item}: {seconds:.2f} s, {tokens} input tokens")


# Scoring ---------------------------------------------------------------------------------------

def wilson(successes: int, total: int) -> tuple[float, float]:
    """Wilson score interval, clipped to [0, 1]."""
    p = successes / total
    denominator = 1 + Z95 * Z95 / total
    centre = (p + Z95 * Z95 / (2 * total)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / total + Z95 * Z95 / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def proportion(numerator: int, denominator: int) -> dict:
    result = {"numerator": int(numerator), "denominator": int(denominator)}
    if denominator < MIN_DENOMINATOR:
        return {**result, "status": "insufficient sample"}
    low, high = wilson(numerator, denominator)
    return {**result, "status": "ok", "estimate": round(numerator / denominator, 4),
            "wilson95": [round(low, 4), round(high, 4)]}


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: a binomial test of b out of b + c with p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def expected_calibration_error(pairs: list[tuple[float, bool]]) -> float | None:
    """ECE of the chosen label's probability against being right, over equal-width bins."""
    if not pairs:
        return None
    bins: dict[int, list[tuple[float, bool]]] = {}
    for p, right in pairs:
        bins.setdefault(min(int(p * CALIBRATION_BINS), CALIBRATION_BINS - 1), []).append((p, right))
    return sum(abs(sum(r for _, r in b) / len(b) - sum(p for p, _ in b) / len(b)) * len(b)
               for b in bins.values()) / len(pairs)


def _system(found: list[bool], flagged: list[bool]) -> dict:
    """Counts, precision and recall from the gold-1 texts found and the gold-0 texts flagged."""
    tp, fp = sum(found), sum(flagged)
    fn = len(found) - tp
    return {"true_positives": tp, "false_positives": fp, "missed": fn,
            "precision": proportion(tp, tp + fp), "recall": proportion(tp, tp + fn),
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else None}


def _paired(pairs: list[tuple[bool, bool]]) -> dict:
    """(lexicon, Jev) per text: how often they agree, the discordant counts, and McNemar's p."""
    only_jev = sum(jev and not lex for lex, jev in pairs)
    only_lexicon = sum(lex and not jev for lex, jev in pairs)
    return {"both": sum(lex and jev for lex, jev in pairs), "only_jev": only_jev,
            "only_lexicon": only_lexicon, "neither": sum(not (lex or jev) for lex, jev in pairs),
            "mcnemar_p": round(mcnemar_p(only_jev, only_lexicon), 4)}


def score(items: list[int], key: dict[int, dict], gold: dict[int, dict],
          answers: dict[int, dict]) -> dict:
    """Per topic, the lexicon and Jev against the final labels, on the same answered texts."""
    answered = [item for item in items if item in answers]
    report = {}
    for topic in TOPICS:
        # (gold label, lexicon matched, Jev's label, Jev's probability of that label) per text
        rows = [(gold[i][topic], key[i][topic], answers[i][topic]["choice"], answers[i][topic]["p_choice"])
                for i in answered if gold[i][topic] != "unsure"]
        positives = [(lex, choice) for label, lex, choice, _ in rows if label == "1"]
        negatives = [(lex, choice) for label, lex, choice, _ in rows if label == "0"]
        jev = _system([choice == "1" for _, choice in positives], [choice == "1" for _, choice in negatives])
        unsure_on_1 = sum(choice == "unsure" for _, choice in positives)
        jev["unsure"] = {"on_gold_1": unsure_on_1, "on_gold_0": sum(choice == "unsure" for _, choice in negatives)}
        if positives:
            jev["recall_if_unsure_counted_as_found"] = round((jev["true_positives"] + unsure_on_1) / len(positives), 4)
        report[topic] = {
            "texts": len(rows), "gold_unsure_left_out": len(answered) - len(rows),
            "lexicon": _system([lex for lex, _ in positives], [lex for lex, _ in negatives]),
            "jev": jev,
            "paired_recall": _paired([(lex, choice == "1") for lex, choice in positives]),
            "paired_false_positives": _paired([(lex, choice == "1") for lex, choice in negatives]),
            "jev_calibration_error": expected_calibration_error(
                [(p, choice == label) for label, _, choice, p in rows if choice != "unsure"]),
        }
    return report


def decide(report: dict, reasons: list[str]) -> dict:
    """The rule of DECISION_v2.md, on the primary topic."""
    topic = report[PRIMARY_TOPIC]
    b, c = topic["paired_recall"]["only_jev"], topic["paired_recall"]["only_lexicon"]
    p = mcnemar_p(b, c)
    tp, fp = topic["jev"]["true_positives"], topic["jev"]["false_positives"]
    precision = tp / (tp + fp) if tp + fp else None
    better = b > c and p < ALPHA
    guard = precision is not None and precision >= MIN_PRECISION
    if reasons:
        outcome = "not decided: " + "; ".join(reasons)
    elif better and guard:
        outcome = "Jev finds more responsiveness complaints: confirm on a new sample before any use"
    else:
        outcome = "keep the lexicon"
    return {"primary_topic": PRIMARY_TOPIC, "only_jev": b, "only_lexicon": c, "mcnemar_p": round(p, 4),
            "jev_better": better, "jev_precision": None if precision is None else round(precision, 4),
            "guard_precision_at_least": MIN_PRECISION, "guard_holds": guard,
            "rule_applies": not reasons, "outcome": outcome}


def nearest_rank(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[max(math.ceil(share * len(ordered)) - 1, 0)]


def write_results(items: list[int], n_random: int, key: dict, gold: dict, output: Path,
                  inputs: dict, args: argparse.Namespace) -> dict:
    usable, records = read_log(output / "responses.jsonl")
    used = {item: usable[item] for item in items if item in usable}
    answers = {item: read_answers(record["response"]) for item, record in used.items()}
    reasons = []
    if len(items) < n_random:
        reasons.append(f"smoke run on {len(items)} of {n_random} texts")
    if len(used) < len(items):
        reasons.append(f"only {len(used)} of {len(items)} texts answered")
    if not inputs["texts_are_the_handed_in_sheet_a"]:
        reasons.append("the texts file is not the handed-in sheet A")
    topics = score(items, key, gold, answers)

    fields = ("choice", "p_choice", "p1", "confidence")
    with open(output / "predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", *(f"{t}_{field}" for t in TOPICS for field in fields)])
        for item in sorted(answers):
            writer.writerow([item, *(answers[item][t][field] for t in TOPICS for field in fields)])

    seconds = [record["seconds"] for record in used.values()]
    tokens = {name: sum((record["response"].get("usage") or {}).get(f"{name}_tokens") or 0
                        for record in records) for name in ("input", "output")}
    cost = None
    if args.price_per_million_input_tokens is not None:
        total = (tokens["input"] * args.price_per_million_input_tokens
                 + tokens["output"] * args.price_per_million_output_tokens) / 1e6
        cost = {"currency": args.currency, "total": round(total, 6),
                "per_1000_texts": round(total / len(used) * 1000, 6) if used else None,
                "price_per_million_input_tokens": args.price_per_million_input_tokens,
                "price_per_million_output_tokens": args.price_per_million_output_tokens}
    sent = sorted(record["sent_utc"] for record in records)
    summary = {
        "design": DESIGN,
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": inputs,
        "models_answered": sorted({record["response"].get("model", "?") for record in used.values()}),
        "texts": {"random": n_random, "selected": len(items), "answered": len(used),
                  "item_ids_sha256": hashlib.sha256(",".join(map(str, items)).encode()).hexdigest()},
        "requests": {"sent": len(records), "unreadable_answers": len(records) - len(usable),
                     "retried": sum(record["attempts"] - 1 for record in records),
                     "first_utc": sent[0] if sent else None, "last_utc": sent[-1] if sent else None},
        "decision": decide(topics, reasons),
        "topics": topics,
        "jev_calibration_error_all_topics": expected_calibration_error(
            [(a[t]["p_choice"], a[t]["choice"] == gold[i][t]) for i, a in answers.items() for t in TOPICS
             if a[t]["choice"] != "unsure" and gold[i][t] != "unsure"]),
        "seconds_per_text": {"p50": nearest_rank(seconds, 0.5), "p95": nearest_rank(seconds, 0.95),
                             "measured": "on the machine that ran the script, network included"}
        if seconds else None,
        "tokens": tokens,
        "cost": cost,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def dry_run(items: list[int], texts: dict[int, str], key: dict, gold: dict, output: Path) -> None:
    example = output / "request_example.json"
    example.write_text(json.dumps(request_body(texts[items[0]]), indent=1) + "\n", encoding="utf-8")
    usable, _ = read_log(output / "responses.jsonl")
    print(f"{len(items)} random texts, each with a text. Labelled 1, and found by the lexicon:")
    for topic in TOPICS:
        positives = [i for i in items if gold[i][topic] == "1"]
        print(f"  {topic:17} {len(positives):3}  {sum(key[i][topic] for i in positives):3}")
    print(f"questions sha256 {QUESTIONS_SHA256}")
    print(f"wrote {example}, the request for item {items[0]}")
    print(f"Nothing was sent. With --allow-external-api, {len([i for i in items if i not in usable])} "
          f"texts would go to {API_URL}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=Path, required=True, help="local CSV with item and text columns")
    parser.add_argument("--key", type=Path, default=RESULTS / "key.csv")
    parser.add_argument("--gold", type=Path, default=RESULTS / "labels_final.csv")
    parser.add_argument("--output", type=Path, default=Path("runs/jev_topic_benchmark"), help="a folder inside runs/")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--allow-external-api", action="store_true", help="send the texts; without it nothing is sent")
    mode.add_argument("--score-only", action="store_true", help="score the saved answers; send nothing")
    parser.add_argument("--limit", type=int, default=None, help="only the first N texts: a smoke run, which the rule ignores")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    parser.add_argument("--price-per-million-input-tokens", type=float, default=None)
    parser.add_argument("--price-per-million-output-tokens", type=float, default=0.0)
    parser.add_argument("--currency", default="USD")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    output = output_dir(args.output)
    key, gold = read_key(args.key), read_gold(args.gold)
    n_random = sum(row["in_R"] for row in key.values())
    items = random_items(key, gold, args.limit)
    texts = read_texts(args.texts)
    missing = [item for item in items if not texts.get(item, "").strip()]
    if missing:
        raise SystemExit(f"{args.texts}: no text for items {missing[:10]}")
    handed_in = json.loads((RESULTS / "agreement.json").read_text(encoding="utf-8"))["sheets"]["A"]["sha256"]
    texts_sha256 = sha256_file(args.texts)
    inputs = {"texts_sha256": texts_sha256, "texts_are_the_handed_in_sheet_a": texts_sha256 == handed_in,
              "key_sha256": sha256_file(args.key), "gold_sha256": sha256_file(args.gold),
              "questions_sha256": QUESTIONS_SHA256, "model_requested": MODEL}
    output.mkdir(parents=True, exist_ok=True)

    if not (args.allow_external_api or args.score_only):
        dry_run(items, texts, key, gold, output)
        return
    if args.allow_external_api:
        api_key = os.environ.get(KEY_ENV, "").strip()
        if not api_key:
            raise SystemExit(f"set {KEY_ENV} first; nothing was sent")
        check_run_record(output, inputs, create=True)
        send_all(items, texts, output, api_key, args.timeout)
    else:
        check_run_record(output, inputs, create=False)
    summary = write_results(items, n_random, key, gold, output, inputs, args)
    for topic, result in summary["topics"].items():
        cells = [f"{name} P {result[name]['precision'].get('estimate', '-')} R {result[name]['recall'].get('estimate', '-')}"
                 for name in ("lexicon", "jev")]
        print(f"{topic:17} " + "   ".join(cells))
    print(f"decision: {summary['decision']['outcome']}")
    print(f"wrote {output / 'summary.json'} and {output / 'predictions.csv'}")


if __name__ == "__main__":
    main()
