"""Planning input for the triage-diagnosis protocol: the whole-review population and the stratifier.

Prints counts only, never a review. Reads the raw Booking CSV (checked by SHA-256), the committed keys and
labels of the pilot and the Jev confirmation, and rebuilds the v2 random split to count the texts the
sentiment model was trained or tuned on.

    .venv/Scripts/python -I docs/experiments/triage_diagnosis/planning/population.py .

The pilot labels are used here for planning only: to size the sample and to see how a candidate stratifier
divides urgent complaints. Nothing is tuned on them, and their 300 texts are excluded from every pool.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
from reviewnlp.data.integrity import canonical_text  # noqa: E402
from reviewnlp.data.preprocess import (  # noqa: E402
    clean_text,
    extract_labeled_reviews,
    split_and_cap,
)

RAW = ROOT / "data" / "raw" / "booking_reviews_515k.csv"
RAW_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"
SEEN_KEYS = ("docs/case_study/results/annotation/key.csv",
             "docs/experiments/jev_topic_benchmark/confirmation/key.csv")
# The stratifier: a deliberately broad word list for the four urgent groups. It only decides how often a
# review is sampled; it never decides whether a review is scored. Copied verbatim into PROTOCOL.md.
TRIGGER = re.compile(
    r"\b(dirt\w*|filth\w*|stain\w*|mou?ld\w*|hairs?|smel\w*|stink\w*|odou?r\w*|cockroach\w*|roach\w*|bugs?|bed ?bugs?|"
    r"insects?|ants|mice|mouse|rats?|fleas?|mosquito\w*|lock\w*|safe|stolen|theft|thief|unsafe|danger\w*|injur\w*|"
    r"fire|smoke|air ?con\w*|a/?c|heat\w*|cold|hot|temperature|freez\w*|leak\w*|flood\w*|toilet\w*|blocked|drain\w*)\b",
    re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    assert sha256(RAW) == RAW_SHA256, "not the Kaggle file"
    trends = load_module(ROOT / "scripts" / "complaint_trends.py", "complaint_trends")
    raw = pd.read_csv(RAW)
    frame, info = trends.load_reviews(RAW)
    frame["positive"] = raw.loc[frame["raw_row"], "Positive_Review"].fillna("").astype(str).str.strip().to_numpy()
    complaint = frame["has_complaint_text"]
    praise = (frame["positive"] != "") & (frame["positive"] != "No Positive")
    trigger = complaint & frame["negative"].map(lambda text: bool(TRIGGER.search(text)))
    print("reviews:", info)
    print("strata: H (complaint field with a trigger word)", int(trigger.sum()),
          "| N (complaint field without one)", int((complaint & ~trigger).sum()),
          "| B (no complaint field)", int((~complaint).sum()),
          "| complaint field and praise field both written", int((complaint & praise).sum()))

    seen = set()
    for path in SEEN_KEYS:
        seen.update(int(row["raw_row"]) for row in csv.DictReader(open(ROOT / path, encoding="utf-8")))
    print("texts the owner has labelled:", len(seen))

    labelled = extract_labeled_reviews(raw, max_chars=1200, min_chars=15)
    train, dev, test = split_and_cap(labelled, seed=42, train_frac=0.8, dev_frac=0.1, train_cap=100000,
                                     test_cap=10000, policy="random")
    print("v2 random split rebuilt:", len(train), len(dev), len(test), "(manifest: 117880 14734 13263)")
    used = set(train["text"].map(canonical_text)) | set(dev["text"].map(canonical_text))

    def used_by_sentiment(text: str) -> bool:
        return canonical_text(clean_text(text, 1200)) in used

    either = frame["negative"].map(used_by_sentiment) | frame["positive"].map(used_by_sentiment)
    print("reviews with a field in the sentiment model's train or dev:", int(either.sum()))

    annotation = ROOT / "docs" / "case_study" / "results" / "annotation"
    labels = {r["item"]: r for r in csv.DictReader(open(annotation / "labels_final.csv", encoding="utf-8"))}
    key = {r["item"]: r for r in csv.DictReader(open(annotation / "key.csv", encoding="utf-8"))}
    negative_of = dict(zip(frame["raw_row"], frame["negative"], strict=True))
    table = {(hit, urgent): 0 for hit in (True, False) for urgent in (True, False)}
    for item, row in key.items():
        if row["in_R"] != "1":
            continue
        hit = bool(TRIGGER.search(negative_of[int(row["raw_row"])]))
        urgent = any(labels[item][topic] == "1" for topic in ("cleanliness", "pests", "air_conditioning"))
        table[(hit, urgent)] += 1
    print("pilot's 200 random texts, (trigger word, cleanliness/pests/AC labelled 1):", table)


if __name__ == "__main__":
    main()
