# Confirmation of Jev's responsiveness result: protocol

**Status: locked.**
- **When.** The commit that added
  [`scripts/confirmation_sample.py`](../../scripts/confirmation_sample.py) locked
  this protocol, before any sheet was drawn.
- **Record.** The draw's manifest records this file's SHA-256 and the code
  commit.
- **Changes.** Any later change is listed at the end, with the reason.

## Question

In the [first comparison](../experiments/jev_topic_benchmark/results/README.md),
on the pilot's 200 random texts, Jev found 13 of the 16 responsiveness
complaints and the lexicon found 2. Ten of the 23 texts Jev flagged were not
complaints. Under [its rule](../experiments/jev_topic_benchmark/DECISION_v2.md),
that result had to be confirmed on new texts before any use. This is the
confirmation.

On 400 new random texts:
- **Recall.** Does Jev find more responsiveness complaints than the lexicon?
- **Guard.** Are at least half of the texts Jev flags complaints?

Only responsiveness is confirmed. The other four topics are not labelled here,
so nothing is said about them.

## Population

- **Unit.** The negative field of one review, called a text below.
- **`P_p`.** The same texts as in the [pilot](pilot_protocol.md):
  - they pass `has_complaint`
    ([`scripts/complaint_trends.py`](../../scripts/complaint_trends.py));
  - they come from the 863 hotels with at least 30 reviews in each period;
  - they were written in period p: February–July 2016 (base) or February–July
    2017 (recent).
- **Left out.** The pilot's 300 texts. They are a test set that has been seen.
  Their raw rows are in the pilot's committed
  [`key.csv`](../case_study/results/annotation/key.csv).
- **The topic is defined by the [guideline](complaint_topics_guideline.md),**
  not by the lexicon's word lists. Where they differ, the lexicon is scored as
  wrong.
- **The lexicon** is the one at the draw's code commit. When this protocol was
  locked, `spaces/hotel-ops-demo/triage.py` and `scripts/complaint_trends.py`
  had not changed since the pilot's draw (commit `06bf23d`).
- **Outside the scope:**
  - complaints written in the positive field;
  - texts that `has_complaint` excluded. The manifest counts them.

## Design

- **Sample.** In each period, a simple random sample of 200 texts from `P_p`
  minus the pilot's texts. That is 400 texts.
- **Seed and order.** The seed is 1 (the pilot's was 0), recorded in the
  manifest. Items are numbered in a random order.
- **Nothing else.** No stratum drawn by the lexicon, and no second annotator.
- **Drawn once.** The sample is not redrawn because of what it contains.
- **Why 400.**
  - The pilot found 16 complaints in 200 texts: a rate of 8%, with a 95%
    interval of 5% to 13%.
  - So 400 texts hold about 32 complaints, and between 20 and 50 are plausible.
  - At 32, a recall of 0.8 has a 95% interval of about 0.65 to 0.91.

## Sheet and labels

- **One sheet,** `sheet.csv`, with the columns `item, text, responsiveness,
  done, note`.
- **What goes in each cell.**
  - `responsiveness` takes exactly one of `1`, `0` or `unsure`, as the
    guideline's responsiveness section says.
  - `done` is `yes` once the text is finished.
  - `note` is optional.
- **Who labels.** The project owner, alone.
- **Blind labelling.**
  - The sheet carries no period, hotel, date or system output.
  - `key.csv` holds the lexicon's verdict on each text. It is not opened before
    the sheet is submitted.
  - No request with any of these texts has been sent to any service before the
    labels are submitted. Jev has not seen them.
- **Valid submission.** The check rejects the whole sheet, and lists the
  offending items, unless all of these hold:
  - every item appears exactly once, and there are no unknown items;
  - every `responsiveness` cell is `1`, `0` or `unsure` (case and surrounding
    spaces are ignored);
  - `done` is `yes`;
  - every text is the one drawn. Texts are compared by a hash that ignores white
    space.

  A blank cell is never read as 0.

## Order

1. **Draw.** The notebook draws the sample.
2. **Label.** The owner labels `sheet.csv` without opening `key.csv`.
3. **Submit and record.**
   - The `finish` step checks the sheet and records its SHA-256 and the count of
     each label.
   - The labels, the key and the manifest, which hold no text, are committed
     **before** Jev is run.
4. **Run Jev,** once, as below.
5. **Score and report.**

## The Jev run

- **Script.** [`scripts/benchmark_jev_topics.py`](../../scripts/benchmark_jev_topics.py)
  with `--stage confirmation`. The run records the script's SHA-256.
- **Texts.** All 400, taken from the returned sheet, in item order. A run is
  complete only when all 400 are answered.
- **Questions.** The same five as in the first comparison, with the SHA-256
  `96462e1e7c03e823d860742124eea948173418bb98336dc0f3a063633d619754`. The answers
  for the other four topics are not scored.
- **Model.** `jev-latest`. The version that answers is recorded. The pilot's was
  `typesafe/jev-1.13-20260917`. If another version answers, this confirms that
  version, and the report says so.
- **Route.** TypeSafe's own API or OpenRouter, as the owner chooses, and
  recorded. The owner checks each provider's data terms before any text is sent.
- **One run.** The questions and the script do not change between the submitted
  sheet and the end of the run. A run that stops resumes with the same command.

## The rule

**Labels and lexicon.**
- The labels are the final labels, which are the project owner's.
- The lexicon finds a complaint when `lex_responsiveness` in the key is 1.
- Jev finds a complaint when it answers `1`. An answer of `0` or `unsure`, or no
  answer, counts as not found, as in the first comparison.
- A text labelled `unsure` is left out. How many there are is reported.

**Recall.**
- Among the texts labelled 1, let b be the texts only Jev finds and c the texts
  only the lexicon finds.
- Jev finds more if b > c and the exact two-sided McNemar p-value is below 0.05.
- With c = 0, that takes b of at least 6.

**Guard.** Among the texts Jev flags and that are labelled 1 or 0, the share
labelled 1 is at least 0.5. This is judged on the estimate.

**Outcomes.**

| Recall | Guard | Outcome |
|---|---|---|
| Jev finds more | holds | **Confirmed.** Jev finds more responsiveness complaints on new texts too, and at least half of its flags are right. |
| Jev finds more | fails | **Not confirmed as a whole.** Jev finds more, but fewer than half of its flags are right. |
| Jev does not find more | either | **Not confirmed.** The pilot's result did not repeat, and the lexicon stays. |

**Complete runs only.** The rule applies only if all 400 texts are answered,
`--limit` was not used, and the SHA-256 of the texts file is that of the
recorded sheet. Otherwise the summary says "not decided" and gives the reason.

**The new texts alone decide.** The pilot's 200 and the new 400 are not pooled
in the rule. Pooled counts are reported as a description.

## Reported, not deciding

- Recall and precision of both systems, with Wilson 95% intervals and counts.
- The texts labelled 0 that only one system flags.
- Counts per period for each system. A change in precision between the periods
  would look like a change in complaints, so it matters for trend work.
- How often Jev answers `unsure`.
- Jev's probability for `1` on its right and wrong flags. This is a
  description, and no threshold is chosen from it.
- Time per text, tokens and OpenRouter's cost.
- The model version.

## What the confirmation can and cannot show

- **Both parts of the rule can fail by chance.** A
  [test](../../tests/test_confirmation_sample.py) recomputes these figures.
  - **Recall.**
    - The plausible range is 20 to 50 complaints, and the lexicon finds about an
      eighth of them.
    - If Jev's true recall is 0.65 or more, the recall part passes in at least
      98 runs in 100.
    - At 0.5 it passes in about 85 runs in 100, and at 0.4 in about 60.
    - These figures take the lexicon and Jev as finding texts independently,
      which is the harder case.
  - **Guard.**
    - With about 46 flags, which is the pilot's rate doubled, a true precision
      of 0.57 as in the pilot would give an estimate below 0.5 in about 13 runs
      in 100.
    - A true precision of 0.50 would give such an estimate in about 44 runs in
      100.
    - So a failed guard with an estimate near 0.5 would not show that the true
      precision is lower.
- **0.5 is the bar of this confirmation.** It does not say that a hotel owner
  would accept that many false alarms. That is decided separately, before any
  use.
- **One annotator.** The only evidence on how reliable the labels are is the
  pilot's. Its kappa for responsiveness was 0.83, with a 95% bootstrap interval of
  0.63 to 0.96, on the 79 texts that both annotators labelled 1 or 0.
- **Not a trend.** This measures one topic in two periods taken together. It
  corrects no trend.
- **Not a licence.** A pass does not put Jev in the demo. That needs a separate
  decision on:
  - the data: guest reviews would leave the demo for two companies;
  - the cost, and the dependence on a vendor and on a model version;
  - the false-alarm rate a hotel owner would accept.
- **A failed confirmation is a result.** It is reported the same way.

## Files

- **Drawn by**
  [`notebooks/12_confirmation_sample_colab.ipynb`](../../notebooks/12_confirmation_sample_colab.ipynb),
  which runs `scripts/confirmation_sample.py draw --expect-hotels 863`.
- **Not committed** (`runs/confirmation/`, these hold review text or the
  lexicon's verdicts):
  - `sheet.csv`;
  - `key.csv`, `manifest.json`.
- **Committed after labelling** (no text, no notes), in
  `docs/experiments/jev_topic_benchmark/confirmation/`:
  - `labels.csv`, `key.csv`, `sample_manifest.json`;
  - `sheet_record.json`, with the SHA-256 of the returned sheet.
- **Committed after the run,** in
  `docs/experiments/jev_topic_benchmark/confirmation/results/`:
  - `summary.json`, `predictions.csv`, `responses.jsonl`, and a README.

## Changes after locking

None yet.
