"""ABSA runner: A/B (base | adapter) on legacy or verified clean test splits.

Fixes vs v8: adapter optional (--variant), strict gold mapping (hard fail on
unknown labels), metrics condition on the denominator, per-review aspect
frequencies (one mention per review), row_id + gold saved per record,
code_sha256 + model/adapter/dataset versions stamped into the summary.

Usage (from the repository root):

    python scripts/run_absa.py \
        --variant adapter \
        --test-parquet data/processed/test.parquet \
        --output-dir runs/absa_adapter

    python scripts/run_absa.py --variant base  --test-parquet data/processed/test.parquet \
        --output-dir runs/absa_base

The frozen test parquet is the archived legacy file (hash-checked below); it
is git-ignored, so point --test-parquet at your local copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = "krimits/hotel-review-nlp-frozen-splits"
ADAPTER_REPO = "krimits/hotel-review-nlp-qwen-qlora-adapter"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_PARQUET_SHA256 = "8b37675028c8081951ee0ebd1181aaf484d63058122b7088288266fe992a7be1"

REPO_ROOT = Path(__file__).resolve().parents[1]
ABSA_CODE_DIR = REPO_ROOT / "src" / "reviewnlp" / "absa"

sys.path.insert(0, str(REPO_ROOT / "src"))


def code_sha256() -> str:
    """Hash the ABSA code as it lives in the repository — this is the stamp
    that goes into the summary, making code <-> results matchable."""
    digest = hashlib.sha256()
    sources = list(ABSA_CODE_DIR.rglob("*.py"))
    runner_path = REPO_ROOT / "scripts" / "run_absa.py"
    if runner_path.is_file():
        sources.append(runner_path)
    for path in sorted(sources):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def map_gold(value) -> str:
    """Strict label mapping — hard fail on anything unexpected (review #5b)."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "1.0", "positive"):
            return "positive"
        if text in ("0", "0.0", "negative"):
            return "negative"
    else:
        number = float(value)  # raises on None/strings — never silently coerced
        if number == 1.0:
            return "positive"
        if number == 0.0:
            return "negative"
    raise ValueError(f"unexpected gold label: {value!r}")


def parse_summary_fields(records: list[dict], parse_fields: tuple, n: int) -> dict:
    boolean_fields = {"json_valid", "salvaged", "empty_valid", "generation_hit_token_budget"}
    fields = {
        field: int(sum(bool(r.get(field)) if field in boolean_fields else int(r.get(field, 0))
                       for r in records))
        for field in parse_fields
    }
    fields["json_valid_rate"] = round(fields["json_valid"] / n, 4) if n else None
    fields["salvaged_rate"] = round(fields["salvaged"] / n, 4) if n else None
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="ABSA base/adapter runner")
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    parser.add_argument("--test-parquet", required=True, help="Path to frozen test.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-per-class", type=int, default=25)
    parser.add_argument("--data-manifest", type=Path,
                        help="manifest from make data; required for new clean splits")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    from reviewnlp.absa.extract import overall_from_aspects
    from reviewnlp.absa.pipeline import extract_aspects_batch

    # --- data: frozen test, row_ids preserved, strict gold mapping ---
    test_path = Path(args.test_parquet)
    assert test_path.is_file(), f"missing {test_path}"
    test_sha = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if args.data_manifest:
        from reviewnlp.data.preprocess import load_processed
        from reviewnlp.utils.experiments import (
            assert_clean_splits,
            fingerprint_splits,
            frame_fingerprint,
        )

        manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema_version") not in (2, 3):
            raise ValueError("a clean, versioned data manifest is required")
        if manifest["schema_version"] == 3 and manifest.get("source_type") != "uploaded_processed_parquets_without_raw_csv":
            raise ValueError("unknown uploaded data provenance")
        directory = test_path.parent
        assert_clean_splits(load_processed(str(directory)))
        if fingerprint_splits(directory) != manifest["splits"]:
            raise ValueError("ABSA data splits differ from the clean manifest")
        expected_test = manifest["splits"]["test"]
        actual_test = frame_fingerprint(pd.read_parquet(test_path))
        if actual_test != expected_test:
            raise ValueError("ABSA test split differs from the clean data manifest")
        dataset_source = "local clean splits"
    else:
        if test_sha != TEST_PARQUET_SHA256:
            raise ValueError("legacy frozen test parquet hash mismatch; pass --data-manifest for new splits")
        dataset_source = REPO
    frame = pd.read_parquet(test_path).reset_index().rename(columns={"index": "row_id"})
    frame["gold"] = frame["label"].map(map_gold)
    sampled = pd.concat([
        frame[frame["gold"] == "positive"].iloc[: args.sample_per_class],
        frame[frame["gold"] == "negative"].iloc[: args.sample_per_class],
    ]).reset_index(drop=True)
    assert len(sampled) == 2 * args.sample_per_class
    print("sample:", sampled["gold"].value_counts().to_dict())

    # --- adapter (only when variant=adapter) ---
    adapter_local = None
    if args.variant == "adapter":
        adapter_dir = snapshot_download(
            ADAPTER_REPO, allow_patterns=["adapter/*"], token=None
        )
        adapter_local = str(Path(adapter_dir) / "adapter")
        assert (Path(adapter_local) / "adapter_config.json").is_file()

    # --- run ---
    records = extract_aspects_batch(
        sampled["text"].tolist(), adapter_dir=adapter_local, batch_size=8
    )
    for record, (_, row) in zip(records, sampled.iterrows(), strict=True):
        record["row_id"] = int(row["row_id"])
        record["gold"] = row["gold"]

    # --- metrics (denominator-checked; per-review aspect frequencies) ---
    gold = sampled["gold"].tolist()
    votes = [overall_from_aspects(r["aspects"]) for r in records]
    conditional = {}
    for cls in ("positive", "negative"):
        total = sum(1 for g in gold if g == cls)
        ok = sum(1 for v, g in zip(votes, gold, strict=True) if g == cls and v == cls)
        conditional[cls] = {
            "ok": ok,
            "total": total,
            "rate": round(ok / total, 4) if total else None,  # denominator, not numerator
        }
    aspect_reviews: dict[str, int] = {}
    for record in records:
        for aspect in {item["aspect"] for item in record["aspects"]}:  # one mention per review
            aspect_reviews[aspect] = aspect_reviews.get(aspect, 0) + 1

    parse_fields = (
        "json_valid", "salvaged", "entries_total", "entries_dropped",
        "empty_valid", "quote_absent", "quote_not_in_review", "quote_truncated",
        "generation_hit_token_budget",
    )
    n = len(records)
    agree = sum(1 for v, g in zip(votes, gold, strict=True) if v == g)
    summary = {
        "schema_version": 10,
        "code_sha256": code_sha256(),
        "model": BASE_MODEL,
        "variant": args.variant,
        "adapter": ADAPTER_REPO if args.variant == "adapter" else None,
        "prompt": "apply_chat_template(system+user, add_generation_prompt=True)",
        "max_new_tokens": 320,
        "dataset": {
            "repo": dataset_source,
            "test_parquet_sha256": test_sha,
            "data_manifest": str(args.data_manifest) if args.data_manifest else None,
            "selection": f"first {args.sample_per_class} per class in frozen order",
            "gold_split": {
                "positive": gold.count("positive"),
                "negative": gold.count("negative"),
            },
        },
        "n_reviews": n,
        "parse": parse_summary_fields(records, parse_fields, n),
        "overall": {
            "headline_agreement": round(agree / n, 4),
            "abstentions": int(sum(1 for v in votes if v is None)),
            "conditional": conditional,
        },
        "aspect_frequency_reviews": dict(
            sorted(aspect_reviews.items(), key=lambda kv: -kv[1])
        ),
    }
    print(json.dumps(summary, indent=2))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("DONE:", out / "summary.json")


if __name__ == "__main__":
    main()
