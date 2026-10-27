# Triage diagnosis and routing: protocol — version 2, DRAFT

**Status: draft of version 2, 27 October 2026, awaiting the owner's approval.** When approved, this text
replaces `PROTOCOL.md` (version 1) in the commit that updates its lock test. Version 1 stays readable in that
file's history and is summarised in [§10](#10-version-history).

**Why a version 2.** The owner returned the version 1 labelling sheet on 27 October 2026, filled entirely by
ChatGPT (OpenAI). Version 1 takes the owner's own blind labels as gold, so the sheet could not be recorded as
`owner_initial` and was withdrawn. The owner then chose, the same day:

- to keep the ChatGPT sheet as the **reference labels**;
- a **second rater from a third provider**, Google Gemini, reported only;
- a **human check on 100 new reviews**, labelled by the owner by hand, reported only.

The question therefore changes from agreement with the owner's judgement to agreement with an LLM's labels.
Everything else that version 1 fixed stays: the data, the samples, the catalogue, the teacher, the student, the
metric, the decision rule and its numbers. Nothing has been trained or scored.

## 1. Question

A small model, trained only on a teacher model's labels, diagnoses each problem a guest reports and routes it
to its department. **On new whole reviews from the public Booking.com dataset, does it catch at least 90% of the
urgent problems that a reference LLM (ChatGPT, OpenAI) labels?**

Catching a problem means two things, both required:

- the model flags the problem as urgent;
- it sends the problem to the department that owns it.

**Scope of any claim:** agreement with one OpenAI model's labels, made through ChatGPT. Not agreement with the
owner, a hotel's staff or any person.

## 2. Data

### Source, unit, strata, exclusions, pools

Unchanged from version 1:

- **Source:** `data/raw/booking_reviews_515k.csv`, 238,154,765 bytes, SHA-256
  `a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9`; 515,173 reviews after exact duplicates
  are dropped; `raw_row` is the identity.
- **Unit:** the whole review, `Disliked: …`, a blank line, `Liked: …`, placeholders and empty fields left out,
  whitespace collapsed.
- **Strata:** H 63,325, N 285,436, B 166,412, by the trigger words of version 1, read from
  [`planning/population.py`](planning/population.py). They decide only how often a review is sampled.
- **Exclusions before pooling:** the owner's 700 earlier labelled texts, and duplicate whole-review texts (the
  earliest `raw_row` kept).
- **Pools** by `int(sha256(canonical text)[:8], 16) % 100`: 0–79 teacher, 80–89 development, 90–99 test.
- **Student base:** `distilbert/distilbert-base-uncased` at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.

### Samples

The two samples of version 1, drawn on 21 October 2026 and recorded in [`samples/`](samples/), are kept:

- **Development:** 300 reviews, H 90, N 180, B 30, seed 20261021.
- **Test:** 500 reviews, H 150, N 300, B 50, seed 20261022.

Each review carries the weight `N_s / n_s` of its stratum within its pool.

**New: the check sample.** 100 reviews from the test pool, none of them among the 500 test reviews: H 30, N 60,
B 10, drawn the same way with **seed 20261024**. Each carries `N_s / n_s`, with `N_s` counted in the test pool
after the 500 test reviews are removed. It is used only for the owner's check (§4, "Reported").

### Labels

Each review gets the set of problem types present, from [`catalogue_v1.yaml`](catalogue_v1.yaml) (20 types,
SHA-256 `2575996133afaa6e0178060df3bd5288d01ce9b7d842a6b3bf67d362d29445e9`). The empty set is allowed.

**Reference labels, `reference_chatgpt` — the gold for the decision.**

- The version 1 sheet of the 800 development and test reviews, returned on 27 October 2026, SHA-256
  `6a9fa2bb32cf44cd91d5e182fbc8287cc7603c0278f51e53a2649568543a0c6a`.
- Filled entirely by ChatGPT, according to the owner, with the model the owner names **"Terra MAX"** in ChatGPT.
  Its exact OpenAI identifier and version were not recorded.
- The owner's prompt, given with the sheet, verbatim:

  > Θέλω να είσαι ο αξιολογητής των κριτικών που έκαναν οι πελάτες. Συμπλήρωσε το αρχείο σύμφωνα με τις
  > παρακάτω οδηγίες.
  > Σε κάθε γραμμή:
  > - βάζεις 1 στη στήλη κάθε τύπου προβλήματος που αναφέρει η κριτική·
  > - βάζεις 1 στο none αν δεν υπάρχει κανένα πρόβλημα·
  > - βάζεις 1 στο done όταν τελειώσεις τη γραμμή.

- **The catalogue's definitions, exclusions and labelling rules were not given.** The model saw only the column
  names (`cleanliness`, `room_and_facilities`, …) and inferred what each type covers.
- It passed every check of `scripts/triage_diagnosis_labels.py finish` (all items, unchanged texts, every row
  done, a type or `none` on each row and never both).
- Recorded by `raw_row`, with no text, and never edited.

**Second rater, `second_rater` — reported only.**

- `google/gemini-4-argon-20261013` through OpenRouter, with structured output. Its prompt holds the catalogue's
  definitions and labelling rules and is committed before it runs.
- It labels the 800 reviews and the 100 check reviews, independently, once.
- Its provider and model family differ from both the teacher's (Anthropic) and the reference's (OpenAI).

**Owner's check, `owner_check` — reported only.**

- The owner labels the 100 check reviews **by hand, with no AI tool**, on a sheet with no model output, stratum
  or weight.
- It is committed before the owner sees any model output on these reviews and before any agreement figure.

**All labels and their SHA-256 are committed before the student is scored on either set.**

## 3. One primary comparison

### The metric: department-urgent recall on the test set, against `reference_chatgpt`

- **The unit** is a (review, reference urgent type) pair.
- **It is caught** if, for that review, the student emits at least one **urgent** type whose department is the
  reference type's department.
- **The estimate** is the inverse-probability-weighted ratio, by stratum, of caught pairs to all reference urgent
  pairs:
  - `rec` is the point estimate;
  - `lo` is the 2.5th percentile of 2,000 bootstrap resamples, seed 0, resampling reviews within each stratum.

All 500 reviews count. A review the student fails on, or abstains on, catches nothing. No review is filtered
out. The unweighted sample recall is reported, never used for the decision.

### The guard: false urgent flags per 100 reviews, IPW-estimated on the test set

`fa` counts the urgent types the student emits that are not reference urgent types of the same department. The
budget is **`F = 5` false urgent flags per 100 reviews**, set by the owner on 20 October 2026.

## 4. Decision rule

The rule applies only to a complete run: all 500 test reviews scored with no abstention dropped, labels
committed before scoring, thresholds committed before the test set is scored. Otherwise the outcome is "not
decided", with the reason.

The lower bound `lo` must reach **`L = 0.80`**, set by the owner on 20 October 2026.

| Condition | Outcome |
| :--- | :--- |
| `rec >= 0.90 and lo >= 0.80 and fa <= 5` | **Met.** The student catches at least 90% of the urgent problems the reference labels, within the false-alarm budget. |
| `rec >= 0.90 and lo >= 0.80 and fa > 5` | **Recall met, over budget.** |
| `rec >= 0.90 and lo < 0.80` | **Point estimate met, not established.** |
| `rec < 0.90` | **Not met.** |

**Feasibility.** Version 1 planned on about 102 urgent pairs in the test sample. The reference labels hold
**106** (development 67); the builder saw these counts when the sheet was first recorded on 27 October 2026, and
no model, threshold or rule was set from them. The version 1 feasibility table therefore still applies: with a
true recall of 0.90, `lo` lies near 0.83.

### Thresholds are chosen on the development set only

Against `reference_chatgpt`, as version 1 fixed them: three thresholds, one per priority group, committed in
`thresholds_locked.json` before the test set is scored. Urgent group: the smallest grid value (0.05, 0.10, …,
0.95) whose development `fa` is at most 5, else the largest. Routine and structural groups: the grid value that
maximises development micro-F1 within the group.

### Reported, not deciding

- **Per type:** recall and precision with IPW estimates and support; counts only below 10. The reference has no
  `pests` label in the test sample, so its test recall cannot be reported.
- **Strict type recall**, **department accuracy** on detected problems, the **unweighted sample recall**, and
  **recall per stratum**.
- **Baselines on the same 500 reviews, paired:** `complaint-lexicon-v1` through the catalogue's lexicon mapping,
  and the teacher LLM itself. The teacher against the reference shows how far two LLM families agree, which
  bounds what a student trained on the teacher can reach against the reference.
- **Agreement:** the second rater against the reference on the 800, per type, as Cohen's kappa with prevalence.
- **The owner's check on the 100 check reviews:**
  - the student's department-urgent recall and `fa` against `owner_check`, IPW-estimated, with the support stated
    (about 20 urgent pairs are expected, so the interval is wide);
  - the second rater against `owner_check`, per type, as Cohen's kappa with prevalence;
  - the reference cannot be compared on these reviews: ChatGPT did not label them, and the session that made the
    reference cannot be repeated.
- **Evidence:** the quoted clause is the clause with the highest logit for the type; it is a substring of the
  review. Whether it supports the problem is judged by the owner on 100 caught problems drawn at random; this is
  not blind to the model and is labelled as such.
- **CPU latency** per review, with and without the clause pass, on one thread.

## 5. Limits

- **The gold is an LLM's labels, not a person's.** Every recall and precision is agreement with one OpenAI model
  through ChatGPT. Where the reference is wrong, the student is scored as wrong for being right, and the reverse.
- **The reference cannot be reproduced.** The prompt is recorded, but "Terra MAX" names a ChatGPT model choice,
  not a pinned version, and a ChatGPT session is not deterministic. Only the returned sheet and its SHA-256 fix
  the labels.
- **The reference did not use the catalogue's definitions.** It labelled from the type names alone, while the
  teacher and the second rater are given the definitions, exclusions and labelling rules. Where a name and its
  definition differ (for example `room_and_facilities`, which excludes anything broken or dirty, or
  `noise_internal` against `noise_external`), the reference may follow the name, and the student is scored
  against that reading.
- **Two LLM families meet in the score.** The student learns from Claude (Anthropic) and is scored against
  ChatGPT (OpenAI). Part of any miss may be a disagreement between the two families rather than a student error;
  the teacher's own score against the reference shows how large that part is.
- **The human anchor is small.** The owner's check covers 100 reviews, with about 20 urgent pairs; it shows the
  direction of any gap between the reference and a person, not its size with any precision. It is one person.
- **No production accuracy claim.** Public Booking.com reviews, 2015–2017, European hotels, English, written after
  checkout.
- **The teacher's labels limit what the student can learn**, and that ceiling is not measured apart from the
  student's own errors.
- **The measures are only checked to match the diagnosed type**; whether a catalogue measure is the right action
  rests on the owner's catalogue review.
- **No Space change follows automatically.** `selection_frozen` and `promoted` stay false.
- **Wording.** The README and any decision note say "agreement with ChatGPT reference labels", never accuracy,
  correctness or human validation.

## 6. Order of work and human gates

| Step | What | Human gate |
| :--- | :--- | :--- |
| 1 | Lock this version with its updated lock test | the owner approves this draft |
| 2 | Record `reference_chatgpt` from the returned sheet; draw the check sample and write its blind sheet | — |
| 3 | The owner labels the 100 check reviews by hand; `owner_check` committed | labelling, no AI tool |
| 4 | Second rater on the 800 and the 100; labels committed | API key; the owner checks Google's data terms through OpenRouter |
| 5 | Teacher labels the teacher pool; student trained in Colab; thresholds on development, committed | API key, Anthropic's data terms, Colab |
| 6 | One scoring run on the test set; rule applied; decision note | — |

## 7. Files

- **At lock:** this protocol, its lock test, `catalogue_v1`, the planning scripts.
- **After the draws, without text:** the sample keys and manifests, including the check sample's; the blank-sheet
  hashes.
- **After labelling, by `raw_row`, without text:** `reference_chatgpt`, `owner_check`, `second_rater`, with a
  record of each sheet's SHA-256; the second rater's prompt and raw responses' hashes.
- **After the run:** the student's test and check logits, `thresholds_locked.json`, the scored summary, and a test
  that recomputes the decision from them.

Review texts stay under `runs/`.

## 8. Departures from the design

| Design | Here | Why |
| :--- | :--- | :--- |
| Gold labels from the owner, with an LLM second rater and the owner's re-check | Gold from ChatGPT; a Gemini second rater; the owner labels 100 other reviews by hand | The owner's choice on 27 October 2026, after the version 1 sheet was filled by ChatGPT |
| Samples from "the clean splits' test partition" | Whole reviews from the raw CSV, in hash pools | The v2 splits contain no mixed reviews (unchanged from version 1). |
| Reviews with complaint text oversampled | Three strata; the IPW estimate decides | Unchanged from version 1. |
| Thresholds "per problem type" | One threshold per priority group | Unchanged from version 1. |
| Plumbing and noise each one type | Split into major/minor and internal/external | Unchanged from version 1. |

## 9. Fields

| Field | Value | Set by |
| :--- | :--- | :--- |
| False-alarm budget `F` | 5 per 100 reviews | owner, 20 October 2026 |
| Lower bound `L` | 0.80 | owner, 20 October 2026 |
| `catalogue_v1` SHA-256 | `2575996133afaa6e0178060df3bd5288d01ce9b7d842a6b3bf67d362d29445e9` | owner, 20 October 2026 |
| Reference labels | ChatGPT (OpenAI), model "Terra MAX" as named by the owner, the prompt in §2, no catalogue definitions; sheet SHA-256 `6a9fa2bb32cf44cd91d5e182fbc8287cc7603c0278f51e53a2649568543a0c6a` | owner, 27 October 2026 |
| Teacher | `anthropic/claude-sonnet-5.5-20260928` | builder, version 1 |
| Second rater | `google/gemini-4-argon-20261013` | owner's choice of a third provider, 27 October 2026; model by the builder |
| Student base | `distilbert/distilbert-base-uncased` @ `12040accade4e8a0f71eabdb258fecc2e7e948be` | builder, version 1 |
| Sample seeds | development 20261021, test 20261022, check 20261024 | builder |

## 10. Version history

| Version | Commit | Change |
| :--- | :--- | :--- |
| 1 | `6a6dc63` | Locked 20 October 2026. Gold from the owner's own blind labels, `openai/gpt-6.1-sol-20260929` as second rater. Under it: the samples were drawn (`a61b51f`, `02092f4`); the returned sheet was recorded as `owner_initial` (`3a5d7f6`) and withdrawn the same day (`8d0c8bf`) when the owner reported it had been filled by ChatGPT. The builder saw the sheet's label counts (urgent pairs: development 67, test 106). **No model was trained, no threshold set and nothing scored under version 1.** |
| 2 | this commit | The owner's choices of 27 October 2026: the ChatGPT sheet as reference labels, so the question becomes agreement with an LLM; the second rater moved to a third provider (Google), as the reference is OpenAI's; a check sample of 100 test-pool reviews labelled by the owner by hand, reported only. The owner-only steps of version 1 (adjudication, the 80-review relabel) are dropped. The data, samples, catalogue, teacher, student, metric, rule and its numbers are unchanged. |
