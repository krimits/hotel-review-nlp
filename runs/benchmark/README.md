# Audited classical baseline on clean derived uploaded splits

Run on 2026-09-23. These figures use the three processed parquet files already
uploaded to this repository, after removing within-split duplicate texts and
text groups with contradictory labels. The **raw Booking CSV was unavailable**;
the manifest identifies the uploaded files by SHA256 and explicitly records
that raw-file provenance has not been verified.

| Split | Rows | Positive | Negative |
| --- | ---: | ---: | ---: |
| Train | 117,890 | 91,794 | 26,096 |
| Dev | 14,830 | 11,557 | 3,273 |
| Test | 13,270 | 9,993 | 3,277 |

The best classical pipeline was selected on **dev**: word TF-IDF +
Multinomial Naive Bayes (dev macro-F1 **0.9319**). On the clean test it reached
**0.9347 macro-F1** and **0.9510 accuracy**. There were **2,992 true negatives,
285 false positives, 365 false negatives and 9,628 true positives**. These are
real measured outcomes for this new derived test set; they are not comparable
to the legacy 0.9634 macro-F1, which used different data. No BiLSTM,
DistilBERT or Qwen checkpoint was available for a valid comparison.

The recorded p50/p95 milliseconds per text divide a preloaded pipeline's
**batch-of-64** time by 64 over three repeats. They are neither single-review
request latency nor a serving/load-test result. The trained joblib pipeline is
not bundled; reproduce it and the figures with:

```bash
python scripts/clean_uploaded_splits.py --source data/processed --output data/processed_clean
python -m reviewnlp.baselines.classical --config configs/baselines_clean_uploaded.yaml
python -m reviewnlp.evaluation.benchmark --config configs/baselines_clean_uploaded.yaml
```

See `results.json` for the ordered split hash, SHA256s of the uploaded source
parquets, all class metrics, skipped model families and the latency protocol.
The [four candidate metrics](../../docs/experiments/results/classical_clean_uploaded_metrics.json)
retain the dev-based selection and the corrected word/character feature views.
