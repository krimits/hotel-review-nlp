# Pilot of the complaint lexicon: labels and agreement

**What this pilot evaluates.** The lexicon that names five complaint topics in the
[complaint-trend analysis](../../complaint_trends.md): bathroom, cleanliness, air conditioning,
pests and responsiveness. It does **not** evaluate DistilBERT or any other sentiment model.

**Where it stands.** This is step 2 of the [protocol](../../../annotation/pilot_protocol.md):
the agreement between the two annotators. It was committed before any discussion. Next:
1. The annotators discuss their disagreements on the 80 double-labelled texts and record the agreed
   labels.
2. The scorer's `metrics` step computes the lexicon's precision and recall.

## Provenance

- **Drawn** on 28 September 2026 by
  [notebook 09](../../../../notebooks/09_annotation_sample_colab.ipynb), from code commit
  [`06bf23de83f1c882980ba0dc5112605c27e424fd`](https://github.com/krimits/hotel-review-nlp/commit/06bf23de83f1c882980ba0dc5112605c27e424fd).
  - The draw's manifest is [`sample_manifest.json`](sample_manifest.json), byte for byte.
  - It records the SHA-256 of the protocol, `9c84f3c9…`, which is that of the locked protocol at
    that commit.
  - It records the SHA-256 of the raw Booking CSV, `a4810c27…`, which is the value
    `fetch_booking_515k.py` checks.
- **Handed in** on 30 September 2026 as `pilot_sample.zip` (SHA-256 `cb7de81e…`). The zip also held
  the key, the manifest and the guideline, all unchanged since the draw. The guideline is byte for
  byte the repository's.
- **The scored sheets:**
  - sheet A, 300 texts: SHA-256 `6bc124aa951631ccc2d12ecf382dcd516b1aca9f1cc8fb5672fb972a601dd1be`;
  - sheet B, 80 texts: SHA-256 `461653221c303e1dae153743dc0a34516bfeb1dd7e84c1bc16fe137f4d233a39`.
  - Both passed the scorer's check: every topic is 1, 0 or `unsure`, `done` is `yes`, and no item is
    missing, repeated or unknown.
- **A first sheet B was set aside before scoring.**
  - The zip's sheet B (SHA-256 `2a237745…`) was byte for byte sheet A's rows for the same 80 texts,
    including four free-text notes.
  - It was a wrong export, not a second annotation.
  - The second annotator's own sheet was then handed in separately, and it is the one scored above.
- **What is committed.** The sheets hold review text, so they stay out of the repository. Only ids and
  labels are committed here.

## Agreement on the 80 double-labelled texts

Cohen's kappa per topic, computed from the raw sheets. A text counts for a topic when both annotators
gave it 1 or 0.

| Topic | Counted | Left out (`unsure`) | Positives A / B | Both 1 | A 1, B 0 | A 0, B 1 | Both 0 | Raw agreement | Kappa (95% percentile bootstrap) | Resamples dropped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Bathroom | 79 | 1 | 11 / 11 | 11 | 0 | 0 | 68 | 1.000 | 1.000 (1.000–1.000) | 0 |
| Cleanliness | 80 | 0 | 12 / 13 | 12 | 0 | 1 | 67 | 0.988 | 0.953 (0.836–1.000) | 0 |
| Air conditioning | 80 | 0 | 10 / 10 | 10 | 0 | 0 | 70 | 1.000 | 1.000 (1.000–1.000) | 0 |
| Pests | 80 | 0 | 8 / 8 | 8 | 0 | 0 | 72 | 1.000 | 1.000 (1.000–1.000) | 0 |
| Responsiveness | 79 | 1 | 15 / 13 | 12 | 3 | 1 | 63 | 0.949 | 0.827 (0.632–0.962) | 0 |

- **The bootstrap uses** 2,000 resamples of the counted texts, with seed 0, as the protocol fixed.
- **Perfect agreement in this sample, not certain agreement.**
  - For bathroom, air conditioning and pests, every counted text agrees. So every resample agrees too,
    and the percentile interval collapses to 1.000–1.000.
  - That interval does not express the uncertainty of 80 texts with 8 to 11 positives. A larger sample
    could well contain disagreements.
  - The interval method was fixed in the protocol and is reported as it is.
- **To discuss: seven label cells.**
  - Cleanliness: 1.
  - Responsiveness: 4.
  - Two texts where one annotator wrote `unsure`: one for bathroom, one for responsiveness.
- **Judgement.** One of the two annotators is the project owner. The guideline limits judgement but does
  not remove it (protocol, "What the pilot cannot show").

## Files

- [`agreement.json`](agreement.json): the scorer's output, with the SHA-256 of both sheets and the
  kappa report per topic.
- [`labels_raw.csv`](labels_raw.csv): both annotators' raw labels, with the item, the annotator and the
  five topics. No text and no notes.
- [`key.csv`](key.csv): the draw's key without text. For each item: the raw row, the review id, the
  period, whether it came from the random draw, the topic it was drawn for, the lexicon's output, and
  whether it was double-labelled.
- [`sample_manifest.json`](sample_manifest.json): the draw's manifest.

[`tests/test_pilot_results.py`](../../../../tests/test_pilot_results.py) checks these files in CI:
- no review text is committed;
- the labels cover exactly the drawn items;
- kappa is reproduced from `labels_raw.csv` and `key.csv`;
- the protocol at the draw commit has the manifest's SHA-256;
- only the protocol's list of changes has grown since the draw.
