# Hotel Review Operations: local pilot

This guide runs the actual review-to-recommendation flow for one or more
hotels. It requires two checkpoints for the complete English workflow:
DistilBERT for binary sentiment (`/predict`) and the base Qwen instruct model
for eight-aspect extraction (`/absa`). The latter downloads on first use.
The sentiment classifier and the ABSA model do different jobs; do not treat a
binary-label score as a measure of aspect accuracy.

## 1. Install and configure

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[llm,serving]'
hf download krimits/distilbert-hotel-reviews --local-dir models/distilbert
python scripts/create_api_key.py --hotel-id hotel-athens-01
```

The key generator displays a raw key **once** for the hotel administrator and
a JSON object containing only its SHA256 digest for the server. Store the raw
key in a secret manager, not in source control. For an initial pilot, set the
four variables below using the generated JSON. Keep the raw key outside the
server configuration.

```bash
export MODEL_TYPE=encoder
export MODEL_PATH=models/distilbert
export REVIEWNLP_DB_PATH=data/processed/hotel-operations.sqlite3
export REVIEWNLP_API_KEYS_JSON='{"PASTE_GENERATED_DIGEST_HERE":["hotel-athens-01"]}'
uvicorn reviewnlp.serving.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/dashboard`; enter the hotel's ID and the raw key.
The key stays in this browser tab's memory. A reverse proxy **must provide
HTTPS** for access beyond localhost. Run with `REVIEWNLP_ENV=production` to
require API keys even if someone accidentally disables the database setting.
If `REVIEWNLP_DB_PATH` is set but no key mapping exists, API requests fail
closed. A second hotel needs its own independent key and hotel ID.

On Windows PowerShell, replace `source` with `.venv\Scripts\Activate.ps1`
and `export` with `$env:NAME="value"`; do not paste secret values into tickets
or screenshots. Mount model weights and the database volume when using Docker.
The plain `MODEL_TYPE=stub` mode exercises API contracts without a model and
is **never** an inference result for a guest.

## 2. A review becomes a recommendation

Submit an English review using the dashboard, or POST:

```bash
curl -X POST http://127.0.0.1:8000/absa \
  -H 'Content-Type: application/json' -H 'X-API-Key: YOUR_RAW_KEY' \
  -d '{"hotel_id":"hotel-athens-01","review_id":"booking-123","source":"booking","language":"en","text":"The staff were kind but the room was dirty."}'
```

The response says `stored: true` if the write succeeded. Repeating the same
`hotel_id`, `source` and `review_id` replaces the prior extraction atomically:
that review contributes at most once per aspect. Without a `review_id`, the
server creates one and returns it. The database stores a hash of the review,
its classification metadata and short aspect quotes, **not the raw review**.
It records rejected outputs but excludes them from aspect counts. Repeated
or contradicted sentiments for one aspect are resolved to a neutral stored
entry. Missing or invented quote spans are dropped before storage.

Get the real complaint list:

```bash
curl -H 'X-API-Key: YOUR_RAW_KEY' \
  'http://127.0.0.1:8000/hotels/hotel-athens-01/recommendations?days=30'
```

Only aspects with negative mentions appear in the action list. `trend` is the
change in negative *rate* from the preceding equal-length window; it is null
if no comparable prior examples exist. The priority score's fixed weights are
a heuristic, not a measured estimate of revenue. Check low-volume topics
against individual reviews before acting. The dashboard never fabricates
examples when the database is empty.

To remove an incorrectly imported review and its quotes:

```bash
curl -X DELETE -H 'X-API-Key: YOUR_RAW_KEY' \
  'http://127.0.0.1:8000/hotels/hotel-athens-01/reviews/booking-123?source=booking'
```

The source, external ID and hotel ID jointly identify a review. An API key
for another hotel cannot read, insert or delete it. Protect and back up the
SQLite database; removal from live rows does not remove older backups.
Quotes can still contain personal data: obtain the right to process reviews,
restrict access, define a retention period and follow your deletion policy.

To ingest more than one review, prepare a UTF-8 CSV with `review_id,text`
and optional `source,language,review_date,hotel_id`. The importer requires
stable external IDs; rerunning a partially completed import safely replaces
earlier rows. Set the hotel's raw key on the **client** and run:

```bash
export REVIEWNLP_API_KEY='YOUR_RAW_KEY'
python scripts/ingest_reviews.py --hotel-id hotel-athens-01 \
  --csv hotel_reviews_to_import.csv --batch-size 8
```

The receipt includes the CSV SHA256 and row count, not raw guest comments.
The importer refuses a row assigned to another hotel or cleartext HTTP
outside localhost. A failed batch might have written earlier rows; rerun it
with the same IDs. Manage permission to import reviews from the source site
before using third-party data.

## 3. Evaluate what the product claims

**English classification.** Obtain `data/raw/booking_reviews_515k.csv` as
documented in `data/raw/README.md`. `make data` now creates clean processed
splits and `data_manifest.json` with a raw-file hash and split fingerprints.
Rerun `make baselines`, `make bilstm`, `make distilbert`, and `make qlora` on
these **same new splits**; only then run `make benchmark`. The benchmark refuses
old cached logits, misordered labels and mismatched fingerprints. The historical
0.9634 macro-F1 is not a measured result on the new splits. Training the full
set needs significant compute; none of those expensive jobs runs in CI.

The repository also now contains three **older, uploaded** processed parquets.
Their test parquet matches the archived frozen ABSA test hash exactly. In an
independent audit of the three uploads, their normalized text has **zero
cross-split overlaps**, but 652 duplicate rows within train, 16 within dev,
8 within test, and 19 train text groups with conflicting labels. These source
files have **no raw CSV provenance or clean manifest**. Do not use them as-is
for a new benchmark. To run a verifiable *derived* experiment without the raw
CSV, preserve the originals and generate a separate output:

```bash
python scripts/clean_uploaded_splits.py --source data/processed --output data/processed_clean
python -m reviewnlp.baselines.classical --config configs/baselines_clean_uploaded.yaml
python -m reviewnlp.evaluation.benchmark --config configs/baselines_clean_uploaded.yaml
```

The derived manifest records hashes of the **uploaded processed files**, the
rows rejected, ordered clean split identities and the lack of raw CSV lineage.
This does not prove that the source files were sampled or labeled correctly.
The new test set differs from the archived frozen test because its internal
duplicates are removed; train every compared model on the new clean train/dev
and keep all historical results labeled as legacy.

**ABSA.** Use a locally available frozen test parquet to make an annotation
template. Two reviewers should annotate independently, settle disagreements,
and mark every row `status: "annotated"`. Each aspect needs one taxonomy name,
polarity and a literal quote from its review; an empty annotated array means
that no aspect was present. Do not turn the model's own outputs into gold.

```bash
python scripts/prepare_absa_annotations.py --test-parquet data/processed/test.parquet \
  --output runs/absa_gold_template.jsonl
python scripts/run_absa.py --variant base --test-parquet data/processed/test.parquet \
  --data-manifest data/processed/data_manifest.json --output-dir runs/absa_base
python scripts/evaluate_absa.py --gold runs/absa_gold_reviewed.jsonl \
  --predictions runs/absa_base/records.jsonl --output runs/absa_base/aspect_evaluation.json
```

The scorer rejects unfinished annotations, duplicate IDs and text mismatches;
it reports aspect F1 and sentiment accuracy on correctly identified aspects.
The uploaded legacy test parquet has a pinned hash in `scripts/run_absa.py`;
derived clean splits must pass their ordered fingerprint check using
`--data-manifest`. An unlabeled 50-review template was verified against the
uploaded frozen test, but humans must supply its aspect annotations.
There are **no publishable aspect
accuracy numbers** until this reviewed annotation work is complete.

**Greek hotels.** `configs/greek_bert.yaml` trains on Greek tweets, not hotel
reviews. Set `GREEK_MODEL_PATH` to a saved checkpoint for `/greek/predict`,
or run `notebooks/07_greek_sentiment_colab.ipynb`. After independent labeling
of a Greek hotel `text,label` CSV, use:

```bash
python scripts/evaluate_greek.py --config configs/greek_bert.yaml \
  --model-dir runs/greek_bert --hotel-test-csv greek_hotel_test.csv
```

Do not train or tune on the hotel test CSV. The Greek API currently reports its
training domain and marks hotel-domain validation as pending; update that
claim only after independently reviewed results are published. The ABSA taxonomy has not been validated on Greek
hotel text; the dashboard deliberately asks for English reviews.

**CPU and INT8.** Run the corrected `scripts/quantize_distilbert.py` on the
full held-out test split and keep its JSON artifact. It measures agreement and
accuracy on that split, serialized state-dict sizes and latency on a fixed
batch. Load-test the *actual* deployed checkpoint separately with
`scripts/load_test.py`. No INT8 or ABSA production-latency figure is claimed
for this new code until these runs are saved and independently checked.

## Pilot boundary

This implementation gives a concrete local or single-worker pilot. A public,
multi-tenant hosted service additionally needs managed database migrations,
key rotation, TLS and access logging at the edge, scheduled backups and
retention, real-model load tests, labeled in-domain evaluation, and operational
monitoring. Shipping the API and dashboard does not substitute for that
evidence.
