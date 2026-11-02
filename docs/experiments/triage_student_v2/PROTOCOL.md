# Triage student, second attempt: protocol — version 1, DRAFT

**Status: draft, 2 November 2026, awaiting the owner's approval.** It is locked only by the commit that adds its
lock test. Nothing in it has been drawn, labelled, trained or scored.

**Why a second attempt.** The [triage diagnosis protocol](../triage_diagnosis/PROTOCOL.md) closed on 2 November
2026 with **Not met** ([decision](../decision_triage_diagnosis.md)): the student caught 0.709 of the urgent
problems ChatGPT labelled, while its teacher caught 0.967 on the same reviews. The student never emitted the two
rare urgent types `security_fault` and `safety_hazard` at its one urgent threshold. The owner then revised the
catalogue to [`catalogue_v2`](../triage_diagnosis/catalogue_v2.yaml) and chose, on 2 November 2026:

- a new small model, with **three variants fixed here and one chosen on the development set**;
- a teacher sample of **30,000** reviews;
- **no** new hand-labelled check;
- the **same decision rule**: 0.90, a lower bound of 0.80, at most 5 false urgent flags per 100.

## 1. Question

A small model, trained only on a teacher model's labels, diagnoses each problem a guest reports and routes it to
its department. **On new whole reviews from the public Booking.com dataset, does it catch at least 90% of the
urgent problems that a reference LLM (ChatGPT, OpenAI) labels with `catalogue_v2`?**

Catching a problem means both: the model flags it as urgent, and sends it to the department that owns it.

**Scope of any claim:** agreement with one OpenAI model's labels, made through ChatGPT. Not agreement with the
owner, a hotel's staff or any person.

## 2. Data

### Source, unit, strata, exclusions, pools

As in the first protocol: `data/raw/booking_reviews_515k.csv` (238,154,765 bytes, SHA-256
`a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9`); the whole review as the unit; strata H, N
and B by the trigger words of
[`../triage_diagnosis/planning/population.py`](../triage_diagnosis/planning/population.py); the owner's 700 earlier
texts and duplicate whole-review texts left out; pools by text hash, 0–79 teacher, 80–89 development, 90–99 test.

### Samples

- **Development, 600 reviews.** The first protocol's 300 development reviews (`../triage_diagnosis/samples/key.csv`)
  and **300 new** from the development pool outside them: H 90, N 180, B 30, **seed 20261103**. The 300 old ones
  were used there to set thresholds and choose a teacher; here they are labelled again with `catalogue_v2`, and
  no model of this protocol has seen them. Each review carries `N_s / n_s` within the pool.
- **Test, 500 new reviews** from the test pool, none of the first protocol's 500 test or 100 check reviews:
  H 150, N 300, B 50, **seed 20261102**. Each carries `N_s / n_s`, with `N_s` counted after those 600 are removed.
- **Teacher sample, 30,000** from the teacher pool: H 12,000, N 15,000, B 3,000, **seed 20261104**. It is split
  by a fixed hash of the text into **27,000 for training** and **3,000 held out** (`int(sha256(text)[:8], 16) % 10
  == 0`, about a tenth). The held-out part is never trained on; variant C sets its thresholds there.
- **Not used:** the first protocol's 100 check reviews. They shaped `catalogue_v2`.

Every draw is stratified as before (per stratum, sorted by `raw_row`, `DataFrame.sample(n=n_s,
random_state=seed)`), first redraws the first protocol's samples and stops unless they match its committed keys,
and is recorded in a key and a manifest without text.

### Labels

**Catalogue:** [`catalogue_v2.yaml`](../triage_diagnosis/catalogue_v2.yaml), 21 types, SHA-256
`5c592af14016cba5f97a0ee021af0a54e2196d1f50ad1f01b938be3a384f966d`, approved by the owner on 2 November 2026. Its
urgent types are `cleanliness`, `pests`, `security_fault`, `safety_fault`, `safety_incident`, `hvac_failure` and
`plumbing_major`.

**Reference labels, `reference_chatgpt_v2` — the gold for the decision.**

- ChatGPT, run by the owner **after this protocol is locked**, with the model the app shows as "GPT 6.1 Terra" (the
  first protocol's), recorded as shown and the same for every batch.
- The prompt is `reference_prompt_v2.md`, generated from `catalogue_v2` by the same code as the first protocol's,
  committed and hashed before any batch.
- **22 batches of 50**: 12 for the 600 development reviews, 10 for the 500 test reviews, in item order. Only blank
  batches are uploaded.
- A batch is rerun if ChatGPT shows code or keyword matching that assigns labels, or returns a file the checks
  refuse; the owner reports reruns. The files are checked as in the first protocol, recorded by `raw_row` with no
  text, and never edited.

**Teacher labels.** `anthropic/claude-haiku-5.5-20261007` through OpenRouter, the first protocol's teacher, with
no new choice of model. The prompt is `teacher_prompt_v2.md`, generated from `catalogue_v2` by the same code. It
uses structured output, 25 reviews per call, at most two retries, every response hashed. The teacher labels
the 30,000, the 600 development and the 500 test reviews, each once.

**All labels are committed before any student is scored on either set.**

## 3. The student: three variants, one chosen on development

All three start from `distilbert/distilbert-base-uncased` at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`
and train on the 27,000 training reviews with the first protocol's version 6 settings. Those are a sigmoid per
type, 256 tokens, 4 epochs, batch 32, lr 2e-5, 6% warmup then linear decay, weight decay 0.01, fp16, seed 42, and
the last epoch kept. They differ only as follows.

| Variant | Loss | Thresholds |
| :--- | :--- | :--- |
| **A** | binary cross-entropy, no class weights (the first protocol's) | one per priority group, by the first protocol's §4 rule, on development |
| **B** | binary cross-entropy with `pos_weight` per type = min(negatives / positives in the training labels, 20) | as A |
| **C** | B's model | routine and structural as A. **Each urgent type has its own threshold:** the grid value that maximises F1 against the teacher on the 3,000 held-out reviews; then, if development `fa` exceeds 5, every urgent threshold rises by 0.05 together until it does not, or reaches 0.95 |

The grid is 0.05, 0.10, …, 0.95; ties go to the smallest value. A type is emitted when the sigmoid of its logit is
at least its threshold. One Colab run trains A and B, and writes their logits for the development, test and
held-out reviews; C uses B's logits.

**The choice, on the 600 development reviews against `reference_chatgpt_v2`:** among the variants whose
development `fa` is at most 5, the one with the highest development `rec`. On a tie, the simpler wins, in the
order A, B, C. If none qualifies, A is chosen. The thresholds of all three variants, their development
figures and the choice are committed before the test set is scored. **Only the chosen variant is scored on the
test set, once.**

## 4. One primary comparison and the decision rule

As in the first protocol, now against `reference_chatgpt_v2` with `catalogue_v2`'s urgent types:

- **Unit and catch:** a (review, reference urgent type) pair is caught if the student emits an urgent type of
  the same department.
- **The three numbers:**
  - `rec` is the IPW ratio by stratum on the 500 test reviews;
  - `lo` is its 2.5th percentile over 2,000 bootstrap resamples, seed 0, within strata;
  - `fa` is false urgent flags per 100, IPW.
- **Completeness:** all 500 reviews count; a review the student fails on catches nothing.

The rule applies only to a complete run, with labels and thresholds committed before scoring; otherwise "not
decided".

| Condition | Outcome |
| :--- | :--- |
| `rec >= 0.90 and lo >= 0.80 and fa <= 5` | **Met.** |
| `rec >= 0.90 and lo >= 0.80 and fa > 5` | **Recall met, over budget.** |
| `rec >= 0.90 and lo < 0.80` | **Point estimate met, not established.** |
| `rec < 0.90` | **Not met.** |

### Reported, not deciding

- All three variants' development `rec` and `fa`, and their thresholds.
- The chosen variant on the test set:
  - per-type recall and precision (counts only below 10 support);
  - strict type recall;
  - department accuracy on emitted types;
  - per-stratum and unweighted recall.
- **Baselines on the same 500 test reviews:**
  - the teacher;
  - `complaint-lexicon-v1` through `catalogue_v2`'s lexicon mapping.
- **The teacher against the student** on the 3,000 held-out reviews, per type: how much the student loses in
  distillation, apart from any LLM disagreement.
- **Not produced:** clause evidence and CPU latency, unless the Colab run keeps the chosen model's weights. The
  notebook keeps them in the owner's Google Drive if the owner chooses; otherwise they are listed as not produced.

## 5. Limits

- **The gold is an LLM's labels.** The reference is one ChatGPT session, which cannot be reproduced.
- **No human check.** The owner's agreement with the reference under `catalogue_v2` is not measured in this
  protocol. Under `catalogue_v1`, on 100 reviews, the owner and the LLMs agreed on about two thirds of the
  urgent problems. `catalogue_v2` was written to close part of that gap, which is not shown here.
- **Development is partly reused.** 300 of the 600 development reviews served the first protocol. The test set is
  new.
- **A choice among three variants on development** flatters their development figures; the test set, scored once
  for the chosen variant only, does not share that bias.
- **The teacher is fixed** as `claude-haiku-5.5` and its labels limit the student. The held-out comparison shows
  how much.
- **No production accuracy claim:** public Booking.com reviews, 2015–2017, European hotels, English, after checkout.
- `selection_frozen` and `promoted` stay false; no Space change follows.

## 6. Order of work and human gates

| Step | What | Human gate |
| :--- | :--- | :--- |
| 1 | Lock with its lock test; generate and commit `reference_prompt_v2.md` and `teacher_prompt_v2.md` | the owner approves this draft |
| 2 | Draw the 300 new development, 500 test and 30,000 teacher reviews; write the 22 blank ChatGPT batches | — |
| 3 | The owner runs the 22 batches in ChatGPT; `reference_chatgpt_v2` recorded | ChatGPT session, about 2 hours |
| 4 | The owner runs the teacher on the 30,000, the 600 and the 500 | OpenRouter key with a spending limit, about $2.10, about 2.5 hours unattended |
| 5 | One Colab run trains A and B; logits committed | Colab T4, about 20–30 minutes |
| 6 | Thresholds of A, B and C on development; the choice; committed | — |
| 7 | The chosen variant scored once on the test set; the rule applied; decision note | — |

## 7. Fields

| Field | Value | Set by |
| :--- | :--- | :--- |
| Catalogue | `catalogue_v2`, SHA-256 `5c592af1…` | owner, 2 November 2026 |
| Rule | 0.90, `L` 0.80, `F` 5 | owner (first protocol), kept on 2 November 2026 |
| Samples | development 300 old + 300 new (seed 20261103); test 500 new (seed 20261102); teacher 30,000 (seed 20261104), 27,000 / 3,000 by text hash | builder |
| Reference | ChatGPT, "GPT 6.1 Terra" as shown, `reference_prompt_v2.md`, 22 batches of 50 | owner's choice |
| Teacher | `anthropic/claude-haiku-5.5-20261007`, `teacher_prompt_v2.md`, 25 per call | owner's choice |
| Student | three variants A, B, C as in §3; one chosen on development | owner's choice of the approach; variants by the builder |
