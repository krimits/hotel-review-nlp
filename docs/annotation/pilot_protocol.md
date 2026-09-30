# Pilot evaluation of the complaint lexicon: protocol

**Status: locked.**
- **When.** The commit that added
  [`scripts/annotation_sample.py`](../../scripts/annotation_sample.py) and
  [`scripts/score_annotations.py`](../../scripts/score_annotations.py) locked
  this protocol, before any sheet was drawn.
- **Record.** The sample's manifest records this file's SHA-256 and the code
  commit.
- **Changes.** Any later change is listed at the end, with the reason.

## Question

For the five topics that the second complaint-trend run flagged as rising:
- **Precision.** How often is the lexicon right?
- **Recall.** How much does it miss?
- **Agreement.** Do two annotators agree (Cohen's kappa)?

Precision and recall are measured in each period.

This is a **pilot**: 300 texts, 80 of them labelled twice. Many estimates will
be too imprecise to report. The thresholds below, fixed before labelling, say
when a number is reported.

## Population

- **Unit.** The negative field of one review, called a text below.
- **`P_p`.** The texts that pass `has_complaint`
  ([`scripts/complaint_trends.py`](../../scripts/complaint_trends.py)).
  - They come from reviews in the 863 hotels with at least 30 reviews in each
    period.
  - They were written in period p: February–July 2016 (base) or February–July
    2017 (recent).
  - These are the same hotels and months as the second run.
- **`A_{t,p}`.** The texts of `P_p` in which `complaint_topics` names topic t.
- **Topics.**

  | Sheet column | Lexicon topic |
  |---|---|
  | `bathroom` | `room.bathroom` |
  | `cleanliness` | `cleanliness.general` |
  | `air_conditioning` | `room.climate` |
  | `pests` | `cleanliness.pests` |
  | `responsiveness` | `staff.response` |

- **The topics are defined by the guideline.**
  - The topics are defined by the [guideline](complaint_topics_guideline.md),
    not by the lexicon's word lists.
  - Where they differ, the lexicon is scored as wrong. For example, the lexicon
    counts a hair dryer as bathroom and the guideline does not.
- **Outside the scope:**
  - complaints written in the positive field;
  - texts that `has_complaint` excluded. The manifest counts those that are not
    Booking's placeholder "No Negative" or empty. They are a blind spot of the
    recall estimate.

## Design, in each period

- **R.** A simple random sample of 100 texts from `P_p`.
- **M_t.** For each topic, a simple random sample from `A_{t,p}` minus R.
  - Sizes:

    | Topic | Size |
    |---|---:|
    | cleanliness | 14 |
    | responsiveness | 14 |
    | bathroom | 8 |
    | air conditioning | 8 |
    | pests | 6 |

  - Each topic is drawn independently.
  - A text drawn for two topics is labelled once and belongs to both strata.
  - If `A_{t,p}` minus R holds fewer texts than the size, all of them are taken,
    and the manifest records the shortfall.
- **B.** A simple random sample of 20 texts from R and 20 from the union of the
  M strata. These 80 texts are labelled by a second annotator as well.
- **Total.** 2 × (100 + 50) = 300 draws. Duplicates across M strata would give
  fewer unique texts, and the manifest records how many.
- **Seed and order.** The seed is 0, recorded in the manifest. Items are
  numbered in a random order.

## Sheets and labels

**Two sheets.**
- `sheet_A.csv` holds every text, and `sheet_B.csv` holds the 80.
- Columns: `item, text, bathroom, cleanliness, air_conditioning, pests,
  responsiveness, done, note`.

**What goes in each cell.**
- **Each topic** takes exactly one of:
  - `1`: the guest complains about this topic;
  - `0`: the guest does not;
  - `unsure`: the text supports both readings (see the guideline).
- **`done`** is `yes` once the text is finished.
- **`note`** is optional.

**Blind labelling.**
- The sheets carry no period, hotel, date, stratum or lexicon output.
- The key (`key.csv`, no text) is kept apart. The annotators do not open it
  before both sheets are submitted.

**Valid submission.** The scorer rejects the whole submission, and lists the
offending items, unless all of these hold:
- every item of the sheet appears exactly once, and there are no unknown items;
- every topic cell is `1`, `0` or `unsure` (case and surrounding spaces are
  ignored);
- `done` is `yes`;
- sheet B's items are all in sheet A.

A blank cell is never read as 0.

## Order

1. **Label.** The two annotators label independently and do not discuss the texts.
2. **Agreement.** Both raw sheets are submitted.
   - The scorer's `agreement` step computes kappa from them and records their
     SHA-256.
   - Kappa, the raw labels and the hashes are committed before any discussion.
3. **Discussion.** The annotators then discuss the disagreements on the 80 texts.
   They record the agreed labels in `adjudicated.csv`, with the same columns;
   `unsure` is allowed.
4. **Final labels.** The adjudicated labels for the 80 texts, and annotator A's
   labels for the rest.
5. **Metrics.** The scorer's `metrics` step computes precision and recall from
   the final labels.

## Estimators

**Rules for all estimators.**
- **Final labels.** Every estimator uses the final labels.
- **`unsure`.** A text labelled `unsure` for topic t is left out of every
  estimate for t. How many there are is reported.

**Precision.**
- Formula: `precision_{t,p}` = texts in `S_{t,p}` labelled 1 for t / texts in
  `S_{t,p}` labelled 1 or 0 for t.
- `S_{t,p}` is the set `(R_p ∩ A_{t,p}) ∪ M_{t,p}`.
- **Why this is a simple random sample of `A_{t,p}`.**
  - `R_p ∩ A_{t,p}` is a random subset of `A_{t,p}`.
  - `M_{t,p}` is a random sample of the rest of `A_{t,p}`.
- **What does not count.** A text drawn for another topic's M stratum is not
  used for t, even if the lexicon matched t in it.

**Recall.**
- Formula: `recall_{t,p}` = texts in `R_p` labelled 1 for t and in `A_{t,p}` /
  texts in `R_p` labelled 1 for t.
- **Scope.** It refers to `P_p`, the texts that passed the filter.
- **Mix.** It is not reweighted to the base period's hotel mix.

**Intervals.**
- Each proportion gets a Wilson score 95% interval, reported with its counts.

**Change between the periods.**
- Recent minus base, with Newcombe's hybrid score interval (method 10).
- Only when both periods are reported.

**Pooled recall.**
- The sum of both numerators over the sum of both denominators.
- It is used only as the thresholds say.

**Bounds for `unsure`.**
- Each estimate is recomputed twice: with every `unsure` for t set to 1, and
  with every one set to 0.
- Both are reported next to it.

## Agreement

**Kappa.**
- Cohen's kappa per topic, on the 80 double-labelled texts, from the raw sheets.
- **Which texts count.** Only texts where both annotators gave 1 or 0 for the
  topic.
- **What is reported with it:**
  - the number of texts left out because of `unsure`;
  - the number of texts counted;
  - each annotator's positives;
  - the 2 × 2 table;
  - the raw agreement.

**Outcomes.** They are fixed before labelling:
- **`undefined`** in any of these cases:
  - no text counts;
  - either annotator has no positive or no negative among the counted texts;
  - expected agreement is 1.
- **`too few positives`** when either annotator has fewer than 5 positives.
  Only the counts are reported, not kappa.
- **Otherwise**, kappa with a percentile bootstrap 95% interval:
  - 2,000 resamples of the counted texts, with a fixed seed;
  - resamples where kappa is undefined are dropped, and their number is
    reported.

## Reporting thresholds

- **Precision** is reported if its denominator is at least 10. Otherwise:
  "insufficient sample".
- **Recall** is reported for each period whose denominator is at least 10.
  - If a period is below 10, the pooled recall is reported as well, if the
    pooled denominator is at least 10.
  - Otherwise: "insufficient sample".
- **A change between the periods** is reported only when both periods are
  reported.
- **Kappa** follows the outcomes above.

## What the pilot cannot show

- **No corrected trend.**
  - The second run's rates are within hotels, at the base period's hotel mix.
  - The estimates here are per period, over `P_p`.
  - Combining them needs a common population and weights, and is not attempted
    here.
- **Recall for rare topics.**
  - Expected positives in R per period, from the lexicon's rates:

    | Topic | Expected positives |
    |---|---:|
    | Bathroom | about 13 |
    | Cleanliness | 7 |
    | Air conditioning | 6 |
    | Responsiveness | 1–2 |
    | Pests | 0–1 |

  - So recall will likely be an "insufficient sample" for responsiveness and
    pests.
  - Per-period recall may also be insufficient for cleanliness and air
    conditioning.
- **Small changes.** Precision intervals will be about ±20 points, so only large
  changes between the periods can show.
- **Judgement.** There are two annotators, and one of them is the project owner.
  The guideline limits judgement but does not remove it.

## Files

- **Drawn by**
  [`notebooks/09_annotation_sample_colab.ipynb`](../../notebooks/09_annotation_sample_colab.ipynb),
  which runs `scripts/annotation_sample.py --expect-hotels 863`.
- **Not committed** (`runs/annotation/`, these hold review text):
  - `sheet_A.csv`, `sheet_B.csv`;
  - `key.csv`, `manifest.json`.
- **Committed after labelling** (no text, no notes):
  - `docs/case_study/results/annotation/key.csv`;
  - the raw labels of both annotators;
  - `agreement.json`, with the SHA-256 of the raw sheets;
  - the final labels;
  - `metrics.json`.

## Changes after locking

- **30 September 2026: the draw's manifest is committed as well.**
  - The draw's `manifest.json` holds counts, hashes and commits, and no text.
  - It is committed as `docs/case_study/results/annotation/sample_manifest.json`, so that the sample's
    provenance stays with its results.
  - This adds a record. It changes no part of the design, the labels or the estimators.
