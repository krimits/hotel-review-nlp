"""Smoke test of the triage chain with the real models, on invented reviews.

What it is: a check that DistilBERT, Jev and Qwen run together, that what each returns fits the pipeline, and a
way to look at what each stage said about a few reviews. What it is not: an evaluation. There are no labels, no
metrics and no thresholds. The fixture's `expected_*` fields are its author's reading, to be looked at, and
nothing may be tuned on them (docs/experiments/triage_smoke/README.md).

The reviews are in docs/experiments/triage_smoke/reviews.json. They are invented, so no guest's text is sent
anywhere. Jev is called only with --allow-external-api, as in the benchmark script, and then the 14 invented
reviews go to the provider. Everything the run writes goes under runs/, which git ignores.

Run (Colab notebook 13 does this):

    python scripts/triage_smoke.py                       # DistilBERT -> Qwen, nothing leaves the machine
    set OPENROUTER_API_KEY=...
    python scripts/triage_smoke.py --allow-external-api  # both arms: without Jev, and with Jev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from reviewnlp.triage import jev_client
from reviewnlp.triage.jev_client import JevClient, JevConfig
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.questions import QUESTIONS_SHA256, QUESTIONS_VERSION
from reviewnlp.triage.qwen_generator import BASE_MODEL, PROMPT_VERSION, QwenActionGenerator
from reviewnlp.triage.schemas import OTHER, TOPICS

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEFAULT_REVIEWS = ROOT / "docs" / "experiments" / "triage_smoke" / "reviews.json"

# The clean-split DistilBERT, as published (docs/experiments/results/distilbert_v2/README.md).
HUB_REPO = "krimits/distilbert-hotel-reviews"
HUB_REVISION = "7306aebcaaebc00d579f5d0a91001ae376f18158"

KINDS = ("positive", "negative", "mixed", "edge")
LABEL_NAMES = frozenset({"negative", "positive"})
SHORT = {"bathroom": "bath", "cleanliness": "clean", "air_conditioning": "ac", "pests": "pests",
         "responsiveness": "resp", OTHER: "other"}
ARMS = ("without-jev", "with-jev")
# The suggestion stage's statuses for a review it was asked about. 'not_triggered' and 'disabled' mean it was not.
ASKED = ("ok", "no_grounded_actions", "error")


# The reviews ------------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reviews(path: Path) -> list[dict]:
    """The invented reviews, checked: every id once, a known kind, only known topics, text a request accepts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    reviews, seen = data["reviews"], set()
    for review in reviews:
        problems = []
        if review["id"] in seen:
            problems.append("id used twice")
        seen.add(review["id"])
        if review["kind"] not in KINDS:
            problems.append(f"kind {review['kind']!r}")
        if review["expected_sentiment"] not in (None, "positive", "negative"):
            problems.append("expected_sentiment")
        if not set(review["expected_topics"]) <= {*TOPICS, OTHER}:
            problems.append("expected_topics")
        if not 3 <= len(review["text"]) <= 8000:  # the limits of TriageRequest
            problems.append("text length")
        if problems:
            raise SystemExit(f"{path}: review {review['id']}: {', '.join(problems)}")
    return reviews


def output_dir(path: Path) -> Path:
    """The run folder. It must be inside runs/, which git ignores."""
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if resolved == RUNS.resolve() or not resolved.is_relative_to(RUNS.resolve()):
        raise SystemExit(f"--output must be a folder inside {RUNS.resolve()}, which git ignores")
    return resolved


# The stages -------------------------------------------------------------------------------------------------------

def build_jev(allow_external_api: bool, route: str, model: str,
              environ: Mapping[str, str] | None = None) -> JevClient:
    """Switched on only by --allow-external-api, and then only with the route's key. Otherwise nothing is sent."""
    environ = os.environ if environ is None else environ
    if not allow_external_api:
        return JevClient(JevConfig(enabled=False, route=route, model=model))
    key_env = jev_client.ROUTES[route]["key_env"]
    key = environ.get(key_env, "").strip()
    if not key:
        raise SystemExit(f"set {key_env} first (the key for --route {route}); nothing was sent")
    return JevClient(JevConfig(enabled=True, route=route, api_key=key, model=model))


def resolve_model_path(model_path: str | None, repo: str, revision: str) -> str:
    """A folder with the DistilBERT, given or downloaded at the pinned revision."""
    if model_path:
        if not Path(model_path).is_dir():
            raise SystemExit(f"--model-path {model_path} is not a folder")
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, revision=revision)


class RecordingGenerator:
    """The suggestion model, with what it wrote or how it failed kept for the report.

    The pipeline keeps neither: in use a review must not travel with a raw generation or an exception, so a failure
    is reported by its kind alone ('generation_failed'). Here the reviews are invented, and a failure that says only
    that would cost a round trip to Colab. The pipeline still sees exactly what it saw before: the result, or the
    exception.
    """

    def __init__(self, inner):
        self.inner, self.model_name, self.calls = inner, inner.model_name, []
        self.worked, self.failure = False, None

    def generate(self, review: str, signals):
        if self.failure:  # it has never worked: trying again would repeat the same failure, slowly, for every review
            self.calls.append({"error": self.failure, "seconds": 0.0})
            raise RuntimeError(f"not tried again after its first failure: {self.failure}")
        start = time.perf_counter()
        try:
            result = self.inner.generate(review, signals)
        except Exception as error:  # noqa: BLE001 - recorded, then raised again for the pipeline to handle as before
            text = f"{type(error).__name__}: {str(error)[:300]}"
            self.calls.append({"error": text, "seconds": round(time.perf_counter() - start, 3)})
            self.failure = None if self.worked else text  # a failure after it has worked belongs to that review alone
            raise
        self.worked = True
        self.calls.append({"raw": result.raw, "hit_token_budget": result.hit_token_budget, "model": result.model,
                           "seconds": round(time.perf_counter() - start, 3)})
        return result

    def drain(self) -> list[dict]:
        calls, self.calls = self.calls, []
        return calls


def build_arms(args: argparse.Namespace) -> dict[str, TriagePipeline]:
    """One pipeline per arm, sharing the sentiment model and the generator. Nothing is loaded until it is used."""
    from reviewnlp.serving.model_wrapper import ModelWrapper

    wrapper = ModelWrapper(model_type="encoder",
                           model_path=resolve_model_path(args.model_path, args.hub_repo, args.revision))
    jev = build_jev(args.allow_external_api, args.route, args.jev_model)
    qwen = None if args.no_qwen else RecordingGenerator(
        QwenActionGenerator(model_id=args.qwen_model, device=args.device))
    wanted = args.arms if args.arms != "auto" else ("both" if jev.enabled else "without-jev")
    if wanted in ("both", "with-jev") and not jev.enabled:
        raise SystemExit("--arms with-jev needs --allow-external-api")
    arms = {}
    if wanted in ("both", "without-jev"):
        arms["without-jev"] = TriagePipeline(wrapper, None, qwen)
    if wanted in ("both", "with-jev"):
        arms["with-jev"] = TriagePipeline(wrapper, jev, qwen)
    return arms


# Running ----------------------------------------------------------------------------------------------------------

def _dump(result) -> dict:
    return {"status": result.status, "sentiment": result.sentiment.model_dump(mode="json"),
            "complaints": result.complaints.model_dump(mode="json"),
            "routing": result.routing.model_dump(mode="json"), "actions": result.actions.model_dump(mode="json"),
            "timings": result.timings.model_dump(mode="json")}


def smoke(arms: Mapping[str, TriagePipeline], reviews: list[dict], progress=None) -> list[dict]:
    """Every review through every arm. A pipeline that raises is recorded by the kind of exception, and goes on."""
    records = []
    for number, review in enumerate(reviews, 1):
        for arm, pipeline in arms.items():
            record = {"id": review["id"], "kind": review["kind"], "arm": arm,
                      "expected_sentiment": review["expected_sentiment"],
                      "expected_topics": review["expected_topics"], "result": None, "crashed": None,
                      "generation": None}
            try:
                record["result"] = _dump(pipeline.run(review["text"]))
            except Exception as error:  # noqa: BLE001 - a smoke test reports whatever happened
                record["crashed"] = f"{type(error).__name__}: {str(error)[:200]}"
            drain = getattr(pipeline.generator, "drain", None)  # what the suggestion model wrote, if it is recorded
            calls = drain() if callable(drain) else []
            record["generation"] = calls[-1] if calls else None
            records.append(record)
            if progress:
                progress(number, len(reviews), record)
    return records


def _found(record: dict) -> list[str]:
    complaints = (record["result"] or {}).get("complaints", {})
    answers = [*complaints.get("topics", []), *([complaints["other_complaint"]] if complaints.get("other_complaint") else [])]
    return [item["topic"] for item in answers if item["answer"] == "yes"]


def _generated(actions: dict) -> bool:
    """Whether the suggestion model wrote something for this review, good or not."""
    return actions["status"] in ("ok", "no_grounded_actions") or actions["error"] in ("invalid_output", "hit_token_budget")


def check(records: list[dict], qwen_enabled: bool = True) -> tuple[list[str], list[str]]:
    """(problems, notes). A problem is the chain not working: a crash, labels the routing cannot read, a stage
    that was switched on and failed, a suggestion model that did not run or was never asked. A note is something to
    look at in what a model said, such as invalid output, an empty list or a cut-off generation."""
    problems, notes = [], []
    labels_seen, failures = set(), {}
    for record in records:
        where = f"{record['id']} ({record['arm']})"
        result = record["result"]
        if result is None:
            problems.append(f"{where}: the pipeline raised {record['crashed']}")
            continue
        sentiment = result["sentiment"]
        distribution = sentiment["probabilities"]
        if distribution is None:
            problems.append(f"{where}: the sentiment model gave no distribution")
        else:
            labels_seen |= set(distribution)
            if abs(sum(distribution.values()) - 1.0) > 1e-3:
                problems.append(f"{where}: the sentiment probabilities add up to {sum(distribution.values()):.4f}")
        complaints = result["complaints"]
        if record["arm"] == "with-jev":
            if complaints["status"] != "ok":
                problems.append(f"{where}: the Jev stage is {complaints['status']}: {complaints['error']}")
            else:
                answered = [item["topic"] for item in complaints["topics"]] + [complaints["other_complaint"]["topic"]]
                if answered != [*TOPICS, OTHER]:
                    problems.append(f"{where}: Jev answered {answered}, not the six questions")
                if complaints["questions_sha256"] != QUESTIONS_SHA256:
                    problems.append(f"{where}: the answers are to other questions than the committed ones")
        elif complaints["status"] != "disabled":
            problems.append(f"{where}: the complaint stage ran in the arm without Jev")
        actions = result["actions"]
        raised = (record.get("generation") or {}).get("error")
        if actions["error"] == "generation_failed" or raised:
            failures.setdefault(raised or "generation_failed", []).append(where)
        elif actions["status"] in ("error", "no_grounded_actions"):
            notes.append(f"{where}: the suggestion stage gave {actions['status']}"
                         + (f" ({actions['error']})" if actions["error"] else f", {actions['dropped']} rejected"))
        elif actions["status"] == "ok" and not actions["actions"]:
            notes.append(f"{where}: the suggestion stage was asked and gave an empty list")
        label = sentiment["label"]
        if record["expected_sentiment"] and label != record["expected_sentiment"]:
            notes.append(f"{where}: sentiment is {label}, the fixture reads {record['expected_sentiment']}")
        if record["arm"] == "with-jev" and complaints["status"] == "ok":
            missing = sorted(set(record["expected_topics"]) - set(_found(record)))
            extra = sorted(set(_found(record)) - set(record["expected_topics"]))
            if missing:
                notes.append(f"{where}: expected {missing}, not found")
            if extra:
                notes.append(f"{where}: found {extra}, not expected")
    asked = sum(1 for r in records if r["result"] and r["result"]["actions"]["status"] in ASKED)
    for error, wheres in failures.items():  # one line for one failure, however many reviews it hit
        problems.append(f"the suggestion model did not run on {len(wheres)} of the {asked} reviews it was asked about "
                        f"(first: {wheres[0]}): {error}")
    if qwen_enabled:  # a chain whose suggestion stage was never asked has not been tested
        for arm in sorted({r["arm"] for r in records if r["result"]}):
            statuses = [r["result"]["actions"]["status"] for r in records if r["arm"] == arm and r["result"]]
            if all(status == "disabled" for status in statuses):
                problems.append(f"{arm}: the suggestion stage is off in every review, though it was not switched off")
            elif not any(status in ASKED for status in statuses):
                problems.append(f"{arm}: the suggestion stage was not asked about any of the {len(statuses)} reviews, "
                                "so nothing about it was tested")
    if labels_seen and not labels_seen <= LABEL_NAMES:
        problems.insert(0, f"the sentiment labels are {sorted(labels_seen)}, not negative and positive: the routing "
                           "would read every review as not negative")
    return problems, notes


def stage_counts(records: list[dict]) -> dict:
    """Per arm: how each stage answered, and for how many reviews the suggestion model wrote text."""
    counts: dict[str, dict] = {}
    for record in records:
        arm = counts.setdefault(record["arm"], {"reviews": 0, "crashed": 0, "complaints": {}, "actions": {},
                                                "suggestion_model_wrote": 0})
        arm["reviews"] += 1
        result = record["result"]
        if result is None:
            arm["crashed"] += 1
            continue
        actions = result["actions"]
        for stage, key in (("complaints", result["complaints"]["status"]),
                           ("actions", actions["status"] + (f":{actions['error']}" if actions["error"] else ""))):
            arm[stage][key] = arm[stage].get(key, 0) + 1
        arm["suggestion_model_wrote"] += _generated(actions)
    return counts


# The report -------------------------------------------------------------------------------------------------------

def _row(record: dict) -> list[str]:
    result = record["result"]
    if result is None:
        return [record["id"], record["kind"], record["arm"], "CRASHED", "", "", "", ""]
    sentiment, complaints = result["sentiment"], result["complaints"]
    found = _found(record)
    jev = "-" if complaints["status"] == "disabled" else (
        f"error {complaints['error']}" if complaints["status"] == "error" else ",".join(SHORT[t] for t in found) or "none")
    expected = ",".join(SHORT[t] for t in record["expected_topics"]) or "none"
    actions = result["actions"]
    qwen = actions["status"] + (f":{actions['error']}" if actions["error"] else "") + (
        f" x{len(actions['actions'])}" if actions["status"] == "ok" else "")
    flags = ",".join(result["routing"]["review_reasons"]) or "-"
    timings = result["timings"]
    ms = "/".join("-" if value is None else f"{value:.0f}" for value in
                  (timings["sentiment_ms"], timings["complaints_ms"], timings["actions_ms"]))
    return [record["id"], record["kind"], record["arm"],
            f"{sentiment['label']} {sentiment['confidence']:.2f}" if sentiment["confidence"] is not None else sentiment["label"],
            f"{jev} (author: {expected})", qwen, flags, ms]


HEADER = ["id", "kind", "arm", "sentiment", "jev yes (author's reading)", "qwen", "needs a look", "ms sent/jev/qwen"]


def format_table(records: list[dict]) -> str:
    rows = [HEADER, *(_row(record) for record in records)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(HEADER))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows]
    return "\n".join([lines[0], "  ".join("-" * width for width in widths), *lines[1:]])


def run_info(args: argparse.Namespace, records: list[dict], reviews_path: Path, started: str, finished: str) -> dict:
    answered = sorted({(r["result"]["complaints"] or {}).get("model") for r in records
                       if r["result"] and r["result"]["complaints"]["status"] == "ok"} - {None})
    versions = {}
    for name in ("torch", "transformers"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            pass
    return {"started_utc": started, "finished_utc": finished, "script_sha256": sha256_file(Path(__file__)),
            "reviews_file": str(Path(reviews_path).relative_to(ROOT)) if Path(reviews_path).is_relative_to(ROOT)
            else str(reviews_path), "reviews_sha256": sha256_file(reviews_path),
            "reviews": len({r["id"] for r in records}), "arms": sorted({r["arm"] for r in records}),
            "sentiment": {"hub_repo": args.hub_repo, "revision": args.revision} if not args.model_path
            else {"model_path": args.model_path},
            "jev": {"route": args.route, "model_requested": args.jev_model, "models_answered": answered,
                    "questions_version": QUESTIONS_VERSION, "questions_sha256": QUESTIONS_SHA256},
            "qwen": None if args.no_qwen else {"model": args.qwen_model, "prompt_version": PROMPT_VERSION},
            "stages": stage_counts(records), "python": platform.python_version(), **versions}


def _fenced(text: str) -> str:
    """The text in a code fence that no backticks inside it can close."""
    fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    return f"{fence}\n{text}\n{fence}"


def _stages_table(info: dict) -> list[str]:
    def cells(counts: dict) -> str:
        return ", ".join(f"{key} {number}" for key, number in sorted(counts.items())) or "-"

    rows = ["| arm | reviews | complaint stage | suggestion stage | suggestion model wrote text for |",
            "|---|---|---|---|---|"]
    for arm, counts in sorted(info["stages"].items()):
        rows.append(f"| {arm} | {counts['reviews']} | {cells(counts['complaints'])} | {cells(counts['actions'])} "
                    f"| {counts['suggestion_model_wrote']} |")
    return rows


def _unusable_generations(records: list[dict], limit: int = 6, width: int = 800) -> list[str]:
    """What the suggestion model wrote where it gave no usable actions: the cases to read."""
    shown = [r for r in records if (r.get("generation") or {}).get("raw") is not None and r["result"]
             and not (r["result"]["actions"]["status"] == "ok" and r["result"]["actions"]["actions"])]
    lines = []
    for record in shown[:limit]:
        raw, actions = record["generation"]["raw"], record["result"]["actions"]
        lines += [f"### {record['id']} ({record['arm']}): {actions['status']}"
                  + (f", {actions['error']}" if actions["error"] else ""), "",
                  _fenced(raw[:width] + (f"\n... ({len(raw) - width} more characters)" if len(raw) > width else "")), ""]
    if len(shown) > limit:
        lines.append(f"{len(shown) - limit} more are in results.jsonl.")
    return lines


def summary_markdown(info: dict, records: list[dict], problems: list[str], notes: list[str]) -> str:
    out = ["# Triage smoke test", "",
           "Invented reviews, no labels and no metrics. The last column of the author's reading is for looking at, "
           "not for scoring.", "",
           f"- Run: {info['started_utc']} to {info['finished_utc']}",
           f"- Arms: {', '.join(info['arms'])}; reviews: {info['reviews']}",
           f"- Jev: {info['jev']['route']}, asked for {info['jev']['model_requested']}, answered by "
           f"{', '.join(info['jev']['models_answered']) or 'nobody'}", "", "```", format_table(records), "```", ""]
    out += ["## Stages", "", *_stages_table(info), "", "## Problems", ""]
    out += ([f"- {p}" for p in problems] or ["None."]) + ["", "## Notes", ""]
    out += ([f"- {n}" for n in notes] or ["None."]) + [""]
    unusable = _unusable_generations(records)
    if unusable:
        out += ["## What the suggestion model wrote where it gave no usable actions", "",
                "The full text of every generation is in results.jsonl.", "", *unusable]
    return "\n".join(out).rstrip("\n") + "\n"


def write_outputs(output: Path, info: dict, records: list[dict], problems: list[str], notes: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (output / "run.json").write_text(json.dumps({**info, "problems": problems, "notes": notes}, indent=1) + "\n",
                                     encoding="utf-8")
    (output / "summary.md").write_text(summary_markdown(info, records, problems, notes), encoding="utf-8")


# The command ------------------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS, help="the invented reviews (default: committed)")
    parser.add_argument("--limit", type=int, help="only the first N reviews")
    parser.add_argument("--hub-repo", default=HUB_REPO)
    parser.add_argument("--revision", default=HUB_REVISION, help="the DistilBERT's revision on the Hub")
    parser.add_argument("--model-path", help="a local folder with the DistilBERT, instead of the Hub")
    parser.add_argument("--allow-external-api", action="store_true",
                        help="call Jev: the invented reviews are sent to the provider")
    parser.add_argument("--route", choices=sorted(jev_client.ROUTES), default="openrouter")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--no-qwen", action="store_true", help="leave the suggestion stage off")
    parser.add_argument("--qwen-model", default=BASE_MODEL)
    parser.add_argument("--device", help="cuda or cpu for Qwen (default: cuda if there is one)")
    parser.add_argument("--arms", choices=("auto", "both", "with-jev", "without-jev"), default="auto")
    parser.add_argument("--output", type=Path, help="a folder inside runs/ (default: runs/triage_smoke_<time>)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, build=build_arms) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    reviews = load_reviews(args.reviews)[: args.limit]
    started = datetime.now(timezone.utc)
    output = output_dir(args.output or Path("runs") / f"triage_smoke_{started.strftime('%Y%m%dT%H%M%SZ')}")
    arms = build(args)

    def progress(number: int, total: int, record: dict) -> None:
        print(f"[{number}/{total}] {record['id']} ({record['arm']}): "
              + ("CRASHED " + record["crashed"] if record["result"] is None
                 else f"{record['result']['timings']['total_ms'] / 1000:.1f} s"), flush=True)

    clock = time.perf_counter()
    records = smoke(arms, reviews, progress)
    finished = datetime.now(timezone.utc)
    problems, notes = check(records, qwen_enabled=not args.no_qwen)
    info = run_info(args, records, args.reviews, started.isoformat(timespec="seconds"),
                    finished.isoformat(timespec="seconds"))
    write_outputs(output, info, records, problems, notes)
    print("\n" + format_table(records))
    for arm, counts in sorted(info["stages"].items()):
        print(f"{arm}: the suggestion model wrote text for {counts['suggestion_model_wrote']} of {counts['reviews']} reviews; "
              f"suggestion stage {counts['actions']}")
    print(f"\n{len(problems)} problem(s), {len(notes)} note(s); {time.perf_counter() - clock:.0f} s; wrote {output}")
    for line in problems:
        print("PROBLEM:", line)
    for line in notes:
        print("note:", line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
