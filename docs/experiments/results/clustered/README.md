# Cluster-robust intervals on the unseen-hotels split

Every macro-F1 interval published for the three v2 splits resamples **reviews**. The hotel
view asks a different question — what happens on hotels the model has never seen — and
drawing a different sample of hotels is not the same as drawing different reviews from the
same ones. Reviews of one hotel share its staff, its building and its season.

This records what the row-level interval costs.

## Result (13,289 test reviews, 147 hotels, median 90 reviews per hotel)

| Model | Macro-F1 | 95% CI, resampling reviews | 95% CI, resampling hotels | Width |
| :--- | ---: | :--- | :--- | ---: |
| word TF-IDF + Naive Bayes (selected on dev) | 0.9347 | 0.9296–0.9394 | 0.9287–0.9405 | ×1.20 |
| BiLSTM, seed 42, saved logits | 0.9514 | 0.9472–0.9557 | 0.9452–0.9566 | ×1.34 |

- **Dependence inside a hotel costs 20–34% more interval width.** The row-level interval is
  the optimistic one, as expected when rows are clustered.
- **No conclusion changes.** The BiLSTM's clustered lower bound (0.9452) is still above
  Naive Bayes's clustered upper bound (0.9405), so the ordering the decision note relies on
  survives the correction.
- **The point estimates are unchanged** (0.9347 and 0.9514 against the committed 0.9348 and
  0.9514 — the Naive Bayes difference is the fourth decimal of a re-fit on this machine).

## Method

Percentile bootstrap of macro-F1: 1,000 resamples, seed 42, whole hotels drawn with
replacement. The same seed and the same rows are used for the row-level interval, so the
two differ only in what is resampled. `reviewnlp.evaluation.significance.cluster_bootstrap_ci`
does the resampling; `scripts/clustered_hotel_intervals.py` runs it here.

The script refuses to run unless the rebuilt split matches the committed manifest on
`split`, `split_policy`, `splits` (per-split fingerprints), `raw_csv_sha256`, `hotels` and
`review_dates`. It also refuses if the saved BiLSTM labels do not line up element-wise with
the rebuilt test rows. An interval over other rows is not an interval over these rows.

## Limits

- One split, one seed, one bootstrap size. Three seeds would give a spread of point
  estimates, which is a different quantity.
- 147 clusters is a small number to bootstrap over; the interval is a description of this
  sample of hotels, not a guarantee for hotels not in the data.
- The DistilBERT checkpoint was trained on the random and out-of-time splits only, so there
  is no transformer entry here. Its hotel-split gap remains an open item.
- The Naive Bayes model is re-trained by the script. The BiLSTM row reads the committed
  logits.

## Reproduce

```bash
python scripts/fetch_booking_515k.py
python -m reviewnlp.data.preprocess --config configs/baselines_hotel.yaml
python scripts/clustered_hotel_intervals.py   # writes this directory's hotel.json
```

Artifact: [`hotel.json`](hotel.json) — rows, clusters, seed, resample count, both intervals,
the per-model width ratio and the split fingerprints it was computed on.
