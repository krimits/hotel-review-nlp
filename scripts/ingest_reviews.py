"""Idempotent hotel review CSV import through the authenticated ABSA batch API."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def load_reviews(csv_path: Path, hotel_id: str) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"review_id", "text"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires review_id,text (optional: hotel_id,source,language,review_date)")
        seen = set()
        rows = []
        for number, row in enumerate(reader, start=2):
            if row.get("hotel_id") and row["hotel_id"] != hotel_id:
                raise ValueError(f"line {number}: CSV belongs to another hotel")
            review_id, source = (row.get("review_id") or "").strip(), (row.get("source") or "csv").strip()
            text = (row.get("text") or "").strip()
            if not review_id or not source or not 3 <= len(text) <= 8000:
                raise ValueError(f"line {number}: review_id/source or text length is invalid")
            if (source, review_id) in seen:
                raise ValueError(f"line {number}: repeated source/review_id in CSV")
            seen.add((source, review_id))
            language = (row.get("language") or "en").strip()
            if language != "en":
                raise ValueError(f"line {number}: ABSA supports only English reviews (language=en)")
            rows.append({
                "hotel_id": hotel_id, "review_id": review_id, "source": source,
                "language": language, "text": text,
                "review_date": (row.get("review_date") or "").strip() or None,
            })
    if not rows:
        raise ValueError("CSV is empty")
    return rows


def import_reviews(base_url: str, hotel_id: str, key: str, rows: list[dict],
                   batch_size: int, timeout: int = 300) -> int:
    url = urlparse(base_url)
    if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1")):
        raise ValueError("HTTPS is required outside localhost")
    if not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be between 1 and 256")
    stored = 0
    for index in range(0, len(rows), batch_size):
        payload = json.dumps({"hotel_id": hotel_id, "reviews": rows[index:index + batch_size]}).encode("utf-8")
        request = Request(
            base_url.rstrip("/") + "/absa/batch", data=payload, method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": key},
        )
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if len(result["results"]) != len(rows[index:index + batch_size]):
            raise RuntimeError("batch response length mismatch; stop and investigate before continuing")
        stored += sum(bool(item["stored"]) for item in result["results"])
        print(f"Imported {index + len(result['results'])}/{len(rows)} reviews")
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--hotel-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("runs/ingestion/summary.json"))
    args = parser.parse_args()
    key = os.getenv("REVIEWNLP_API_KEY")
    if not key:
        parser.error("set REVIEWNLP_API_KEY with the raw key for this hotel")
    rows = load_reviews(args.csv, args.hotel_id)
    start = time.perf_counter()
    saved = import_reviews(args.base_url, args.hotel_id, key, rows, args.batch_size)
    if saved != len(rows):
        raise RuntimeError("server did not persist all reviews; check REVIEWNLP_DB_PATH")
    summary = {"hotel_id": args.hotel_id, "n_reviews": saved,
               "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
               "duration_seconds": round(time.perf_counter() - start, 2)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Complete:", args.output)


if __name__ == "__main__":
    main()
