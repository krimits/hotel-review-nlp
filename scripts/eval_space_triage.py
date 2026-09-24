"""Measure the Space's review triage against hand-labelled hotel reviews.

    python scripts/eval_space_triage.py [dev] [test1] [test2]      (default: test2)

Runs `spaces/hotel-ops-demo/triage.py` with its real model, so it needs
transformers, torch, sentencepiece and protobuf, and downloads the model once.
Labels are in data/eval/space_triage_*.json. Each set was labelled before any
model or rule ran on it; `_role` in each file says how it was used afterwards.
Report the TEST2 numbers: it stayed untouched until the final rule was fixed.

What is counted (per review, summed over the set; an aspect in `optional` is
never required and never an error):
  aspect P / R        predicted aspects that are labelled / labelled aspects found
  sentiment           on labelled aspects found: pos -> {pos}, neg -> {neg},
                      mixed -> {pos, neg}
  complaints R / P    neg|mixed aspects surfaced as a complaint /
                      complaints surfaced that are real
  praise R / P        the same for praise
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "data" / "raw" / "hotel_reviews.csv"
SHORT = {"positive": "pos", "negative": "neg"}


def load_triage():
    path = ROOT / "spaces" / "hotel-ops-demo" / "triage.py"
    spec = importlib.util.spec_from_file_location("space_triage", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_labels(name: str) -> dict:
    data = json.loads((ROOT / "data" / "eval" / f"space_triage_{name}.json").read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def predictions(findings: list[dict]) -> dict[str, set]:
    found: dict[str, set] = {}
    for item in findings:
        found.setdefault(item["aspect"], set()).add(SHORT[item["sentiment"]])
    return found


def score(labels: dict, predicted: dict[str, dict[str, set]]) -> dict[str, int]:
    """Counts behind the metrics in the module docstring."""
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


def report(c: dict[str, int]) -> str:
    def pct(a: str, b: str) -> str:
        return f"{c[a] / c[b]:.0%} ({c[a]}/{c[b]})" if c[b] else "n/a"
    return "\n".join([
        f"  reviews with a finding  {pct('covered', 'reviews')}",
        f"  aspect precision        {pct('asp_ok', 'asp_pred')}",
        f"  aspect recall           {pct('asp_found', 'asp_gold')}",
        f"  sentiment on found      {pct('sent_ok', 'asp_found')}",
        f"  complaints found        {pct('comp_found', 'comp_gold')}",
        f"  complaints that are real {pct('comp_real', 'comp_pred')}",
        f"  praise found            {pct('praise_found', 'praise_gold')}",
        f"  praise that is real     {pct('praise_real', 'praise_pred')}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="*", default=["test2"], choices=["dev", "test1", "test2"])
    args = parser.parse_args()
    triage = load_triage()
    model = triage.AbsaModel()
    reviews = pd.read_csv(REVIEWS)["Review"]
    for name in args.sets:
        labels = load_labels(name)
        keys = list(labels)
        texts = [" ".join(html.unescape(str(reviews[labels[k]["row"]])).split()) for k in keys]
        started = time.perf_counter()
        findings = triage.analyze(texts, model)
        seconds = time.perf_counter() - started
        counts = score(labels, {k: predictions(f) for k, f in zip(keys, findings, strict=True)})
        print(f"== {name}: {len(keys)} reviews, {seconds / len(keys):.2f}s per review ==")
        print(report(counts))


if __name__ == "__main__":
    main()
