"""Measure the Space's review triage against labels made by people.

    python scripts/eval_space_triage.py user6                     the owner's six reviews (set A)
    python scripts/eval_space_triage.py booking dev|test          Booking guests' liked/disliked split (set B)
    python scripts/eval_space_triage.py topics                    the owner's topic labels (set C)
    python scripts/eval_space_triage.py labels [dev test1 test2]  TripAdvisor labels made by Claude

It runs spaces/hotel-ops-demo/triage.py with its real model, so it needs
transformers, torch, sentencepiece and protobuf, and downloads the model once.
`--triage PATH` runs another version of triage.py instead, e.g. the one that
was deployed before, for a before/after comparison on the same reviews.

Set A (data/eval/space_triage_user6.json): six apartment reviews the project
owner checked finding by finding. Each lists topics the demo must report,
must not report, and the flags a finding must carry.

Set B (data/eval/space_triage_booking.json): English Booking.com reviews from
crawlfeeds/Booking-Hotel-Reviews-Dataset (CC BY-NC 4.0; downloaded, not
stored here). Every guest wrote what they liked and what they disliked in two
separate fields, so the guest labelled the polarity of each sentence. The
reviews are split by property into DEV (tuning) and TEST (run once). Counted
per review, with each field analysed on its own:
  complaints found      reviews with a substantive "disliked" text in which
                        at least one complaint is found in that text
  complaints real       complaints (review x category) that rest on at least
                        one quote from the "disliked" text
  praise found / real   the same with the "liked" text
Guests sometimes put a criticism in "liked", so "real" is a lower bound.

Set C (data/eval/space_triage_booking_labels.json): the owner's topic labels
for 40 TEST reviews, made before seeing the demo's output. Precision and
recall of (topic, sentiment) pairs, and of complaints alone.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
REVIEWS = ROOT / "data" / "raw" / "hotel_reviews.csv"
SHORT = {"positive": "pos", "negative": "neg"}
TRIVIAL = re.compile(
    r"^(?:nothing|none|no|n/?a|na|nil|-+|\.+|no complaints?|nothing (?:at all|really|to (?:complain|report"
    r"|dislike)(?: about)?|special|bad|negative)|all (?:good|great|perfect|fine)|everything (?:was )?(?:perfect"
    r"|great|fine|good|excellent)|no negatives?|not applicable)[\s.!]*$",
    re.I,
)
# A "disliked" text that opens like this says there was nothing to dislike
# ("Nothing - everything was awesome"), so it does not count as a complaint.
NO_COMPLAINT = re.compile(
    r"^\W*(?:nothing|none|no\s+complaints?|all\s+(?:was\s+)?(?:good|great|perfect|fine)"
    r"|everything\s+(?:was\s+)?(?:perfect|great|fine|good|excellent|wonderful|awesome|fantastic|amazing"
    r"|in\s+(?:well\s+)?working\s+order))\b",
    re.I,
)


def load_triage(path: Path | None = None):
    path = path or ROOT / "spaces" / "hotel-ops-demo" / "triage.py"
    spec = importlib.util.spec_from_file_location("space_triage", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(triage, model, texts: list[str]) -> tuple[list[list[dict]], float]:
    started = time.perf_counter()
    found = triage.analyze(texts, model)
    return found, time.perf_counter() - started


def wilson(hits: int, total: int) -> str:
    """Share with a 95% Wilson interval."""
    if not total:
        return "n/a"
    p, z = hits / total, 1.96
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return f"{p:.0%} ({hits}/{total}, 95% CI {centre - half:.0%}-{centre + half:.0%})"


# --- Set A -------------------------------------------------------------------

def evaluate_user6(triage, model) -> None:
    data = json.loads((EVAL / "space_triage_user6.json").read_text(encoding="utf-8"))
    texts = [review["text"] for review in data["reviews"]]
    found, seconds = run(triage, model, texts)
    passed = 0
    for number, (review, mentions) in enumerate(zip(data["reviews"], found, strict=True), start=1):
        grouped: dict[tuple[str, str], dict] = {}
        for m in mentions:
            entry = grouped.setdefault((m["topic"], m["sentiment"]),
                                       {"quotes": [], "flags": set(), "terms": set(), "sure": False})
            entry["quotes"].append(m["quote"])
            entry["flags"] |= set(m["flags"]) - {"check"}
            entry["sure"] |= "check" not in m["flags"]
            entry["terms"].add(m["term"].lower())
        for entry in grouped.values():
            if not entry["sure"]:
                entry["flags"].add("check")
        problems = [f"missing {topic} {sentiment}" for topic, sentiment in review["must"]
                    if (topic, sentiment) not in grouped]
        problems += [f"unwanted {topic} {sentiment}: {grouped[(topic, sentiment)]['quotes']}"
                     for topic, sentiment in review["must_not"] if (topic, sentiment) in grouped]
        for key, flags in review.get("flags", {}).items():
            topic, sentiment = key.split()
            have = grouped.get((topic, sentiment), {"flags": set()})["flags"]
            problems += [f"{topic} {sentiment} lacks flag {flag}" for flag in flags if flag not in have]
        for key, word in review.get("not_from", {}).items():
            if word in grouped.get(tuple(key.split()), {"terms": set()})["terms"]:
                problems.append(f"{key} comes from '{word}'")
        missed_targets = [f"{t} {s}" for t, s in review.get("target", []) if (t, s) not in grouped]
        passed += not problems
        print(f"== review {number}: {'PASS' if not problems else 'FAIL'}")
        for (topic, sentiment), entry in sorted(grouped.items()):
            flags = f" [{', '.join(sorted(entry['flags']))}]" if entry["flags"] else ""
            print(f"   {topic:22} {sentiment:8}{flags} x{len(entry['quotes'])}: {entry['quotes'][0][:90]!r}")
        for problem in problems:
            print(f"   !! {problem}")
        if missed_targets:
            print(f"   target not reached: {', '.join(missed_targets)}")
    print(f"== set A: {passed}/{len(texts)} reviews pass, {seconds / len(texts):.2f}s per review")


# --- Set B -------------------------------------------------------------------

def booking_reviews() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    source = json.loads((EVAL / "space_triage_booking.json").read_text(encoding="utf-8"))["_source"]
    path = hf_hub_download(source["repo"], source["file"], repo_type="dataset", revision=source["revision"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["language"] == "en"].copy()

    def field(value: str) -> str:
        text = str(value).strip()
        return "" if text in ("\\N", "nan") else text

    df["pos"] = df["positive_review_text"].map(field)
    df["neg"] = df["negative_review_text"].map(field)
    df["pos_ok"] = df["pos"].map(lambda t: len(t.split()) >= 3 and not TRIVIAL.match(t))
    df["neg_ok"] = df["neg"].map(lambda t: len(t.split()) >= 3 and not TRIVIAL.match(t))
    return df[df["pos_ok"] | df["neg_ok"]].drop_duplicates("uniq_id")


def booking_split(df: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    """DEV 150 / TEST 300 by property (at most 3 reviews each), and 40 TEST reviews for topic labels."""
    rng = random.Random(seed)
    hotels = sorted(df["url"].unique())
    rng.shuffle(hotels)
    per_hotel = {}
    for url, group in df.groupby("url"):
        ids = sorted(group["uniq_id"])
        random.Random(f"{seed}-{url}").shuffle(ids)
        per_hotel[url] = ids[:3]
    dev: list[str] = []
    test: list[str] = []
    for url in hotels:
        if len(dev) < 150:
            dev += per_hotel[url]
        elif len(test) < 300:
            test += per_hotel[url]
    dev, test = dev[:150], test[:300]
    by_id = df.set_index("uniq_id")
    with_neg = [i for i in test if by_id.loc[i, "neg_ok"]]
    without_neg = [i for i in test if not by_id.loc[i, "neg_ok"]]
    rng.shuffle(with_neg)
    rng.shuffle(without_neg)
    return {"dev": dev, "test": test, "label": sorted(with_neg[:28] + without_neg[:12])}


def booking_sets() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    record = json.loads((EVAL / "space_triage_booking.json").read_text(encoding="utf-8"))
    df = booking_reviews()
    split = booking_split(df, record["seed"])
    for name, ids in split.items():
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if digest != record["sha256"][name]:
            raise SystemExit(f"The {name} split does not match the recorded one; the dataset or the code changed.")
    return df.set_index("uniq_id"), split


def evaluate_booking(triage, model, name: str) -> None:
    df, split = booking_sets()
    rows = df.loc[split[name]]
    texts = list(rows["pos"]) + list(rows["neg"])
    found, seconds = run(triage, model, texts)
    liked, disliked = found[:len(rows)], found[len(rows):]
    c = dict.fromkeys(("comp_found", "comp_gold", "comp_real", "comp_pred", "praise_found", "praise_gold",
                       "praise_real", "praise_pred", "unsure_real", "unsure_pred"), 0)
    by_confidence: dict[str, list[int]] = {}
    for (_, row), pos_mentions, neg_mentions in zip(rows.iterrows(), liked, disliked, strict=True):
        if row["neg_ok"] and not NO_COMPLAINT.match(row["neg"]):
            c["comp_gold"] += 1
            c["comp_found"] += any(m["sentiment"] == "negative" for m in neg_mentions)
        if row["pos_ok"]:
            c["praise_gold"] += 1
            c["praise_found"] += any(m["sentiment"] == "positive" for m in pos_mentions)
        units: dict[tuple[str, str], dict] = {}
        for field, mentions in (("pos", pos_mentions), ("neg", neg_mentions)):
            for m in mentions:
                unit = units.setdefault((m["aspect"], m["sentiment"]), {"fields": set(), "sure": False,
                                                                         "confidence": 0.0})
                unit["fields"].add(field)
                unit["sure"] |= "check" not in m.get("flags", [])
                unit["confidence"] = max(unit["confidence"], m.get("confidence", 1.0))
        for (_, sentiment), unit in units.items():
            side, home = ("comp", "neg") if sentiment == "negative" else ("praise", "pos")
            c[f"{side}_pred"] += 1
            c[f"{side}_real"] += home in unit["fields"]
            if side == "comp" and not unit["sure"]:
                c["unsure_pred"] += 1
                c["unsure_real"] += home in unit["fields"]
            if side == "comp":
                band = next(f"<{edge:.2f}" for edge in (0.6, 0.7, 0.8, 0.9, 1.01) if unit["confidence"] < edge)
                by_confidence.setdefault(band, [0, 0])[0] += home in unit["fields"]
                by_confidence[band][1] += 1
    print(f"== booking {name}: {len(rows)} reviews, {seconds / len(rows):.2f}s per review")
    print(f"  complaints found        {wilson(c['comp_found'], c['comp_gold'])}")
    print(f"  complaints real         {wilson(c['comp_real'], c['comp_pred'])}")
    print(f"  praise found            {wilson(c['praise_found'], c['praise_gold'])}")
    print(f"  praise real             {wilson(c['praise_real'], c['praise_pred'])}")
    if c["unsure_pred"]:
        print(f"  complaints marked 'check' that are real {wilson(c['unsure_real'], c['unsure_pred'])}")
    for band in sorted(by_confidence):
        print(f"  complaints with model confidence {band}: real {wilson(*by_confidence[band])}")


# --- Set C -------------------------------------------------------------------

def evaluate_topics(triage, model) -> None:
    labels = json.loads((EVAL / "space_triage_booking_labels.json").read_text(encoding="utf-8"))["labels"]
    df, _ = booking_sets()
    ids = sorted(labels)
    rows = df.loc[ids]
    texts = [f"{row['pos']}\n{row['neg']}".strip() for _, row in rows.iterrows()]
    found, seconds = run(triage, model, texts)
    c = dict.fromkeys(("hit", "pred", "gold", "comp_hit", "comp_pred", "comp_gold", "cat_hit", "cat_pred",
                       "cat_gold"), 0)
    for uid, mentions in zip(ids, found, strict=True):
        gold = set()
        for topic, sentiment in labels[uid]["topics"]:
            gold |= {(topic, "positive"), (topic, "negative")} if sentiment == "mixed" else {(topic, sentiment)}
        pred = {(m["topic"], m["sentiment"]) for m in mentions}
        c["hit"] += len(pred & gold)
        c["pred"] += len(pred)
        c["gold"] += len(gold)
        comp_pred = {p for p in pred if p[1] == "negative"}
        comp_gold = {g for g in gold if g[1] == "negative"}
        c["comp_hit"] += len(comp_pred & comp_gold)
        c["comp_pred"] += len(comp_pred)
        c["comp_gold"] += len(comp_gold)
        cat = lambda pairs: {(t.split(".")[0], s) for t, s in pairs}  # noqa: E731
        c["cat_hit"] += len(cat(pred) & cat(gold))
        c["cat_pred"] += len(cat(pred))
        c["cat_gold"] += len(cat(gold))
    print(f"== topics: {len(ids)} reviews labelled by the owner, {seconds / len(ids):.2f}s per review")
    print(f"  findings that are right (topic + sentiment)   {wilson(c['hit'], c['pred'])}")
    print(f"  labelled topics found                         {wilson(c['hit'], c['gold'])}")
    print(f"  complaints that are right                     {wilson(c['comp_hit'], c['comp_pred'])}")
    print(f"  labelled complaints found                     {wilson(c['comp_hit'], c['comp_gold'])}")
    print(f"  category level: right {wilson(c['cat_hit'], c['cat_pred'])}, found {wilson(c['cat_hit'], c['cat_gold'])}")


# --- TripAdvisor labels made by Claude (regression) ---------------------------

def load_labels(name: str) -> dict:
    data = json.loads((EVAL / f"space_triage_{name}.json").read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def predictions(mentions: list[dict]) -> dict[str, set]:
    found: dict[str, set] = {}
    for item in mentions:
        found.setdefault(item["aspect"], set()).add(SHORT[item["sentiment"]])
    return found


def score(labels: dict, predicted: dict[str, dict[str, set]]) -> dict[str, int]:
    """Per review and category: a predicted category that is labelled, a labelled one that is found."""
    c = dict.fromkeys(("reviews", "covered", "asp_ok", "asp_pred", "asp_found", "asp_gold",
                       "sent_ok", "comp_found", "comp_gold", "comp_real", "comp_pred",
                       "praise_found", "praise_gold", "praise_real", "praise_pred"), 0)
    for key, label in labels.items():
        gold, optional = label["gold"], set(label["optional"])
        pred = {aspect: s for aspect, s in predicted.get(key, {}).items() if s}
        c["reviews"] += 1
        c["covered"] += bool(pred)
        for aspect, sentiments in pred.items():
            c["asp_pred"] += 1
            c["asp_ok"] += aspect in gold or aspect in optional
            if aspect in optional:
                continue
            if "neg" in sentiments:
                c["comp_pred"] += 1
                c["comp_real"] += gold.get(aspect) in ("neg", "mixed")
            if "pos" in sentiments:
                c["praise_pred"] += 1
                c["praise_real"] += gold.get(aspect) in ("pos", "mixed")
        for aspect, want in gold.items():
            c["asp_gold"] += 1
            sentiments = pred.get(aspect, set())
            if sentiments:
                c["asp_found"] += 1
                c["sent_ok"] += sentiments == ({"pos", "neg"} if want == "mixed" else {want})
            if want in ("neg", "mixed"):
                c["comp_gold"] += 1
                c["comp_found"] += "neg" in sentiments
            if want in ("pos", "mixed"):
                c["praise_gold"] += 1
                c["praise_found"] += "pos" in sentiments
    return c


def evaluate_labels(triage, model, name: str) -> None:
    labels = load_labels(name)
    keys = list(labels)
    reviews = pd.read_csv(REVIEWS)["Review"]
    texts = [" ".join(html.unescape(str(reviews[labels[k]["row"]])).split()) for k in keys]
    found, seconds = run(triage, model, texts)
    c = score(labels, {k: predictions(f) for k, f in zip(keys, found, strict=True)})
    print(f"== {name} (labels by Claude): {len(keys)} reviews, {seconds / len(keys):.2f}s per review")
    for label, hits, total in (("aspect precision", "asp_ok", "asp_pred"), ("aspect recall", "asp_found", "asp_gold"),
                               ("complaints found", "comp_found", "comp_gold"),
                               ("complaints real", "comp_real", "comp_pred"),
                               ("praise found", "praise_found", "praise_gold"),
                               ("praise real", "praise_real", "praise_pred")):
        print(f"  {label:18} {wilson(c[hits], c[total])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["user6", "booking", "topics", "labels"])
    parser.add_argument("sets", nargs="*")
    parser.add_argument("--triage", type=Path, help="another version of spaces/hotel-ops-demo/triage.py")
    args = parser.parse_args()
    triage = load_triage(args.triage)
    model = triage.AbsaModel()
    if args.mode == "user6":
        evaluate_user6(triage, model)
    elif args.mode == "booking":
        for name in args.sets or ["dev"]:
            evaluate_booking(triage, model, name)
    elif args.mode == "topics":
        evaluate_topics(triage, model)
    else:
        for name in args.sets or ["test2"]:
            evaluate_labels(triage, model, name)


if __name__ == "__main__":
    main()
