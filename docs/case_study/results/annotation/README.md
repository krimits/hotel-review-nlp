# Pilot of the complaint lexicon: labels, agreement, precision and recall

**What this pilot evaluates.** The lexicon that names five complaint topics in the
[complaint-trend analysis](../../complaint_trends.md): bathroom, cleanliness, air conditioning,
pests and responsiveness. It does **not** evaluate DistilBERT or any other sentiment model.

**Where it stands.** Every step of the [protocol](../../../annotation/pilot_protocol.md) is done:
1. Two annotators labelled the texts independently.
2. Their agreement was committed before any discussion (commit
   [`1d2c1aa`](https://github.com/krimits/hotel-review-nlp/commit/1d2c1aa)).
3. They discussed their disagreements.
4. The final labels are the agreed ones for the 80 double-labelled texts, and annotator A's for the
   rest.
5. The scorer computed precision and recall from the final labels.

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
  - sheet B, 80 texts: SHA-256 `461653221c303e1dae153743dc0a34516bfeb1dd7e84c1bc16fe137f4d233a39`;
  - the agreed labels for the 80 texts: SHA-256
    `53c77ce18f2643ac7616943035a61b24ff9f62c7f055cd71a5a07768ad27e0a0`.
  - All three passed the scorer's check: every topic is 1, 0 or `unsure`, `done` is `yes`, and no item
    is missing, repeated or unknown.
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

## The agreed labels

- **Seven cells were disagreements:**
  - cleanliness: 1;
  - responsiveness: 4;
  - two texts where one annotator wrote `unsure`, one for bathroom and one for responsiveness.
- **All seven were settled on annotator A's label.** Annotator A is the project owner.
  - Each decision cites a rule of the [guideline](../../../annotation/complaint_topics_guideline.md).
    Four of the seven concern responsiveness: whether a request was ignored or answered very late,
    or only answered slowly or unwelcomely.
  - The notes that give the reasons quote the reviews, so they are not committed.
  - That every decision went one way is reported here, because it may reflect the owner's weight in
    the discussion as well as the guideline.
- **No other cell changed.** Every cell where the annotators already agreed kept their label.

## Precision

**The question.** Of the texts in which the lexicon names a topic, the share that are really about
it. Each estimate has a Wilson 95% interval.

| Topic | Base, Feb–Jul 2016 | Recent, Feb–Jul 2017 | Change, recent − base (95% Newcombe) |
|---|---|---|---|
| Bathroom | 12/16 = 0.750 (0.505–0.898) | 12/16 = 0.750 (0.505–0.898) | +0.000 (−0.286 to +0.286) |
| Cleanliness | 8/17 = 0.471 (0.262–0.690) | 14/26 = 0.538 (0.355–0.712) | +0.068 (−0.219 to +0.340) |
| Air conditioning | 11/12 = 0.917 (0.646–0.985); 1 `unsure`, 0.846–0.923 if it were 0 or 1 | 18/20 = 0.900 (0.699–0.972) | −0.017 (−0.229 to +0.263) |
| Pests | 7/7, insufficient sample | 6/7, insufficient sample | not reported |
| Responsiveness | 5/15 = 0.333 (0.152–0.583) | 11/19 = 0.579 (0.363–0.769) | +0.246 (−0.085 to +0.508) |

## Recall

**The question.** Of the random texts that are really about a topic, the share the lexicon names.

| Topic | Base | Recent | Both periods pooled |
|---|---|---|---|
| Bathroom | 6/7, insufficient sample | 6/6, insufficient sample | 12/13 = 0.923 (0.667–0.986) |
| Cleanliness | 1/2, insufficient sample | 8/9, insufficient sample; 1 `unsure` | 9/11 = 0.818 (0.523–0.949) |
| Air conditioning | 5/5, insufficient sample | 11/13 = 0.846 (0.578–0.957) | 16/18 = 0.889 (0.672–0.969) |
| Pests | 1/1, insufficient sample | 1/1, insufficient sample | 2/2, insufficient sample |
| Responsiveness | 0/8, insufficient sample | 2/8, insufficient sample | 2/16 = 0.125 (0.035–0.360) |

**Reporting rules.** These are the protocol's thresholds:
- An estimate needs a denominator of at least 10.
- A pooled recall is reported when a period falls short.
- A change is reported only when both periods are.

## What the pilot shows, and what it does not

- **The lexicon is reliable for air conditioning, fairly so for bathroom, and weak for cleanliness and
  responsiveness.**
  - Precision is about 0.9 for air conditioning and 0.75 for bathroom in both periods.
  - It is about half for cleanliness, and a third to a half for responsiveness.
- **It misses most responsiveness complaints.** In the random texts it named 2 of the 16 that people
  judged to be about responsiveness. For bathroom, air conditioning and cleanliness it named 0.82 to
  0.92 of them, pooled over the periods.
- **No change in precision between the periods can be told apart from zero.**
  - The protocol expected intervals of about ±0.2 for precision in each period. The intervals for the
    change are wider, about ±0.25 to ±0.3, so only large changes could have shown.
  - The intervals also contain the falls in precision that would erase the rises in the
    [trend analysis](../../complaint_trends.md#reading-the-quotes): 12% to 17% of the base precision
    for bathroom, air conditioning, cleanliness and responsiveness. So the pilot neither confirms
    those rises nor explains them away.
- **Pests have too few texts** for any estimate.
- **A change in recall between the periods is not measured.** Recall is reported pooled, or for
  one period only.
- **What the protocol rules out.**
  - It combines nothing into a corrected trend.
  - The estimates are per period, over the texts that passed the filter, at each period's own hotel
    mix.

## Files

- [`agreement.json`](agreement.json): the scorer's agreement output, with the SHA-256 of both sheets
  and the kappa report per topic.
- [`labels_raw.csv`](labels_raw.csv): both annotators' raw labels, with the item, the annotator and the
  five topics.
- [`labels_final.csv`](labels_final.csv): the final labels, in the same columns.
- [`metrics.json`](metrics.json): precision and recall per topic and period, with counts, intervals,
  `unsure` bounds and the SHA-256 of the agreed labels.
- [`key.csv`](key.csv): the draw's key without text. For each item: the raw row, the review id, the
  period, whether it came from the random draw, the topic it was drawn for, the lexicon's output, and
  whether it was double-labelled.
- [`sample_manifest.json`](sample_manifest.json): the draw's manifest.

None of these files holds review text or notes.

[`tests/test_pilot_results.py`](../../../../tests/test_pilot_results.py) checks these files in CI:
- no review text is committed;
- the labels cover exactly the drawn items;
- kappa is reproduced from the raw labels;
- the final labels are A's outside the double-labelled texts, and the agreed label wherever the two
  annotators already agreed;
- precision and recall are reproduced from the final labels;
- the protocol at the draw commit has the manifest's SHA-256, and only its list of changes has grown
  since.
