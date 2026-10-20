# Triage diagnosis and routing: protocol — version 1, locked

**Status: locked on 20 October 2026,** by the commit that adds its lock test,
[`tests/test_triage_diagnosis_protocol.py`](../../../tests/test_triage_diagnosis_protocol.py). Nothing had been
drawn, labelled, trained or scored at lock. Any later change needs a new version, not an edit.

It turns the design in
[`2026-10-16-triage-diagnosis-routing-design.md`](../../superpowers/specs/2026-10-16-triage-diagnosis-routing-design.md)
into a protocol. It also takes in the review of 19 October:

- the decision metric is the weighted recall;
- the plumbing and noise types are split in two;
- the evidence comes from clauses;
- results are reported against both sets of labels;
- thresholds are set per group, not per type;
- texts the owner has already labelled are left out.

Departures from the design are listed in [§8](#8-departures-from-the-design).

## 1. Question

A small model, trained only on a teacher model's labels, diagnoses each problem a guest reports and routes it
to its department. **On new whole reviews from the public Booking.com dataset, does it catch at least 90% of
urgent problems?**

Catching a problem means two things, both required:

- the model flags the problem as urgent;
- it sends the problem to the department that owns it.

## 2. Data

### Source

`data/raw/booking_reviews_515k.csv`:

- 238,154,765 bytes, SHA-256 `a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9`, the Kaggle file;
- 515,173 reviews after `scripts/complaint_trends.py` drops exact duplicate rows;
- the review's 0-based `raw_row` is its identity.

### The unit: a whole review

The model reads, and the annotator labels, the same string:

- `Disliked: <negative field>`;
- a blank line;
- `Liked: <positive field>`.

A field that is Booking's placeholder (`No Negative`, `No Positive`) or empty is omitted, with its prefix.
The text is whitespace-collapsed and nothing else. The v2 sentiment splits cannot serve here. They keep only
reviews with a single written field, so no mixed review is in them
(`src/reviewnlp/data/preprocess.py`).

### Strata

The strata are planning inputs, counted by
[`planning/population.py`](planning/population.py):

| Stratum | Definition | Reviews |
| :--- | :--- | ---: |
| H | complaint field (`has_complaint`) with a trigger word | 63,325 |
| N | complaint field without a trigger word | 285,436 |
| B | no complaint field | 166,412 |

Of the 348,761 reviews with a complaint field, 313,118 also have a written praise field.

**Trigger words.** They decide **only how often a review is sampled**, never whether it is scored. Matching is
case-insensitive, with word boundaries, on the negative field:

```
dirt* filth* stain* mold*/mould* hair(s) smel* stink* odor*/odour* cockroach* roach* bug(s) bed bug(s)/bedbug(s)
insect(s) ants mice mouse rat(s) flea(s) mosquito* lock* safe stolen theft thief unsafe danger* injur* fire smoke
aircon*/air con* a/c ac heat* cold hot temperature freez* leak* flood* toilet* blocked drain*
```

The regular expression is in `planning/population.py`, and the lock test checks the two agree.

### Exclusions, applied before any pool is formed

- **The 700 texts the owner has already labelled:** the pilot's 300 and the confirmation's 400, by `raw_row`
  in their committed `key.csv` files.
- **Reviews whose whole-review text duplicates another review's.** The earliest `raw_row` is kept.

### Pools

The pools come from a fixed hash, not a seed:

- `int(sha256(canonical whole-review text)[:8], 16) % 100` gives a number from 0 to 99;
- **0–79** is the teacher pool, the only source of training labels;
- **80–89** is the development pool;
- **90–99** is the test pool.

No text is in two pools.

**The student starts from `distilbert/distilbert-base-uncased` at revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`**, not from the
project's sentiment checkpoint. That checkpoint was trained on 165,511 reviews' fields drawn from every part
of the dataset. Starting from it would put training text into the development and test pools.

### Samples

Each sample is a stratified random draw without replacement, recorded in a manifest. Within each stratum the
pool is sorted by `raw_row` and drawn with `DataFrame.sample(n=n_s, random_state=seed)`, with **seed 20261021
for development and 20261022 for test**:

- **Development:** 300 reviews from the development pool, split H 90, N 180, B 30.
- **Test:** 500 reviews from the test pool, split H 150, N 300, B 50.

Each review carries the weight `N_s / n_s` of its stratum, with `N_s` counted within its pool. Neither sample is
redrawn for what it contains.

### Labels

Each review gets the set of problem types present, from the catalogue's 15–20 types. The empty set is allowed.
Each type maps to one department and one priority level, fixed by the catalogue.

- **The catalogue** is [`catalogue_v1.yaml`](catalogue_v1.yaml), 20 types, SHA-256
  `2575996133afaa6e0178060df3bd5288d01ce9b7d842a6b3bf67d362d29445e9`. The owner approved it as is on
  20 October 2026.
- **Plumbing** is split in two:
  - `plumbing_major`: no hot water, an active leak, or a toilet that does not work. Urgent; maintenance.
  - `plumbing_minor`: slow drain or low pressure. Routine; maintenance.
- **Noise** is split in two:
  - `noise_internal`: building, corridor or equipment noise. Routine; reception.
  - `noise_external`: street or location noise. Structural; management.

**The two LLMs**, both called through OpenRouter at their dated identifiers, with structured output:

- **Teacher:** `anthropic/claude-sonnet-5.5-20260928`. It labels the teacher pool and is a baseline.
- **Second rater:** `openai/gpt-6.1-sol-20260929`, a different provider and model family.

Labelling and checking go in this order:

1. **The owner labels** all 800 reviews. The sheets carry no model output, no stratum and no weight. The result
   is `owner_initial`.
2. **A second-rater LLM** labels the same 800, independently. It must come from a different provider and model
   family than the teacher. Agreement with `owner_initial` is recorded per type, before any change.
3. **The owner looks again at every disagreement.** The sheet shows the review and the disputed types, but not
   which side proposed which. The result is `owner_adjudicated`, written to a new column. `owner_initial` is
   never edited.
4. **The owner relabels 80 reviews (30 development, 50 test) blind, at least 7 days later,** before seeing any
   agreement figure. That measures self-consistency.
5. **All labels and their SHA-256 are committed** before the student is scored on either set.

## 3. One primary comparison

### The metric: department-urgent recall on the test set, against `owner_adjudicated`

- **The unit** is a (review, gold urgent type) pair.
- **It is caught** if, for that review, the student emits at least one **urgent** type whose department is the
  gold type's department.
- **The estimate** is the inverse-probability-weighted ratio, by stratum, of caught pairs to all gold urgent
  pairs:
  - `rec` is the point estimate;
  - `lo` is the 2.5th percentile of 2,000 bootstrap resamples, seed 0, resampling reviews within each stratum.

All 500 reviews count:

- **A review the student fails on, or abstains on, catches nothing.**
- **No review is filtered out by any lexicon or confidence.**
- **The unweighted sample recall is reported, never used for the decision.** In
  [`planning/power.py`](planning/power.py) a 50/35/15 stratified sample gives an unweighted 0.939 when the true
  recall is 0.895. That happens because complaints without a trigger word are harder and are under-sampled.

### The guard: false urgent flags per 100 reviews, IPW-estimated on the test set

`fa` counts the urgent types the student emits that are not gold urgent types of the same department. The
budget is **`F = 5` false urgent flags per 100 reviews**, set by the owner on 20 October 2026: for 80–100 reviews a day, about 4–5 needless urgent checks a day.

## 4. Decision rule

The rule applies only to a complete run:

- all 500 test reviews scored, with no abstention dropped;
- labels committed before scoring;
- thresholds committed before the test set is scored.

Otherwise the outcome is "not decided", with the reason.

The lower bound `lo` must reach **`L = 0.80`**, set by the owner on 20 October 2026, knowing the feasibility table below.

| Condition | Outcome |
| :--- | :--- |
| `rec >= 0.90 and lo >= 0.80 and fa <= 5` | **Met.** The student catches at least 90% of urgent problems, within the false-alarm budget. |
| `rec >= 0.90 and lo >= 0.80 and fa > 5` | **Recall met, over budget.** It catches enough but raises more false urgent flags than the budget. |
| `rec >= 0.90 and lo < 0.80` | **Point estimate met, not established.** The interval reaches below the required bound. |
| `rec < 0.90` | **Not met.** |

**Feasibility, against which `L` was set.** The table comes from [`planning/power.py`](planning/power.py),
covering cleanliness, pests and AC only:

| Allocation (H/N/B) | True recall | 95% spread of `rec` | Urgent pairs in the sample |
| :--- | :--- | :--- | :--- |
| 150/300/50 | 0.90 everywhere | 0.828–0.957 | ~102 |
| 150/300/50 | 0.95 with a trigger word, 0.75 without | 0.808–0.964 | ~102 |
| 72/249/179 (simple random) | 0.90 everywhere | 0.814–0.971 | ~55 |

- Even when the true recall is exactly 0.90, `lo` lies near 0.83.
- An `L` above about 0.80 therefore makes "Met" unlikely for a model whose true recall is 0.90.
- A bound of 0.85 would need several times this test set.

### Thresholds are chosen on the development set only

They are committed in `thresholds_locked.json` before the test set is scored. There are **three thresholds,
one per priority group**, not one per type. The rare types would have only a handful of development positives
each.

- **Urgent group:** the smallest value on the grid 0.05, 0.10, …, 0.95 whose development `fa` is at most 5.
  If none qualifies, the largest value.
- **Routine and structural groups:** the grid value that maximises development micro-F1 within the group.

### Reported, not deciding

- **The same metrics against `owner_initial`.** If the outcome differs from the one against
  `owner_adjudicated`, both are reported and the difference is stated.
- **Recall and precision per type**, with IPW estimates and support. Counts only below 10.
- **Strict type recall:** the gold type itself is emitted.
- **Department accuracy** on the problems the student detected.
- **The unweighted sample recall.**
- **Recall per stratum.** Stratum N is where a lexicon fails and a reader of the whole text should not.
- **Baselines on the same 500 reviews, paired:**
  - `complaint-lexicon-v1`, through a lexicon-topic → type mapping fixed at lock;
  - the teacher LLM itself.
- **Evidence.** The quoted clause is the clause with the highest logit for the type, after the existing clause
  splitter in `spaces/hotel-ops-demo/triage.py`.
  - By construction it is a substring of the review.
  - Whether it supports the problem is judged by the owner on 100 caught problems drawn at random. This is
    not blind to the model and is labelled as such.
- **Agreement:**
  - owner vs second rater, per type, before adjudication, as Cohen's kappa with prevalence;
  - owner self-consistency on the 80 relabelled reviews.
- **CPU latency** per review, with and without the clause pass, on one thread.

## 5. Limits

- **No production accuracy claim.** The reviews are public Booking.com reviews from 2015–2017, of European hotels, in English, written after
  checkout. They are not one hotel's current reviews.
- **One annotator.** The second-rater LLM and the self-relabelling are partial substitutes for a second person,
  not equivalents. An LLM second rater can pull the adjudicated labels toward what LLMs find, so the
  `owner_initial` results are reported beside them.
- **The teacher's labels limit what the student can learn.** A ceiling set by the teacher is not measured
  separately from the student's own errors.
- **The urgent rate behind the sample sizes comes from three pilot topics.** Safety, heating and
  `plumbing_major` were not labelled there, so the real support may be somewhat larger.
- **The pilot labels were used to plan, not to tune.** They shaped the trigger words' allocation, and no model,
  threshold or catalogue entry was fitted on them. Their texts are excluded.
- **The measures are only checked to match the diagnosed type.** Whether a catalogue measure is the right action
  rests on the owner's one-time catalogue review, not on this test.
- **No Space change follows automatically.** Replacing the generative suggestions in the Space depends on the
  outcome and is a separate, gated step.
- `selection_frozen` and `promoted` stay false.

## 6. Order of work and human gates

| Step | What | Human gate |
| :--- | :--- | :--- |
| 0 | `catalogue_v1`: types, departments, priority, measure templates | the owner reviews it |
| 1 | Fill the fields in §9; lock this protocol with its lock test | done 20 October 2026: the owner set `F = 5` and `L = 0.80` and approved the catalogue |
| 2 | Draw the development and test samples; produce the blind sheets | — |
| 3 | The owner labels 800 reviews (about two weeks) | labelling |
| 4 | Second rater on the 800; disagreements re-checked; 80 relabelled; labels committed | API key, data terms |
| 5 | Teacher labels the teacher pool; student trained in Colab; thresholds on development, committed | API key, Colab |
| 6 | One scoring run on the test set; rule applied; decision note | — |

## 7. Files

**Committed at lock:**

- this protocol;
- its lock test;
- `catalogue_v1`;
- the planning scripts.

**Committed after the draw, without review text:**

- the sample manifests (`raw_row`, stratum, weight, pool, text SHA-256);
- the blank-sheet hashes.

**Committed after labelling:**

- the labels `owner_initial`, `second_rater`, `owner_adjudicated` and `owner_relabel`, by `raw_row`.

**Committed after the run:**

- the student's test logits;
- `thresholds_locked.json`;
- the scored summary;
- a test that recomputes the decision from them.

Review texts stay under `runs/`.

## 8. Departures from the design

| Design | Here | Why |
| :--- | :--- | :--- |
| Samples from "the clean splits' test partition" | Whole reviews from the raw CSV, in hash pools | The v2 splits contain no mixed reviews. |
| Reviews with complaint text oversampled | Three strata with a broad trigger word list; the IPW estimate decides | The unweighted estimate would favour complaints written with the obvious words. |
| Thresholds "per problem type" | One threshold per priority group | Too few development positives per rare type. |
| Point estimate ≥ 0.90; a lower bound "to be decided" | Point estimate ≥ 0.90 and `lo >= 0.80`, with the feasibility table above | The owner chose the bound knowing what 500 reviews can show. |
| Plumbing and noise each one type | Split into major/minor and internal/external | The urgent and structural levels need the split. |

## 9. Fields filled at lock

| Field | Value | Set by |
| :--- | :--- | :--- |
| False-alarm budget `F` | 5 per 100 reviews | owner, 20 October 2026 |
| Lower bound `L` | 0.80 | owner, 20 October 2026 |
| `catalogue_v1` SHA-256 | `2575996133afaa6e0178060df3bd5288d01ce9b7d842a6b3bf67d362d29445e9` | approved by the owner, 20 October 2026 |
| Teacher | `anthropic/claude-sonnet-5.5-20260928` | builder |
| Second rater | `openai/gpt-6.1-sol-20260929` | builder; the owner checks the data terms before step 4 |
| Student base | `distilbert/distilbert-base-uncased` @ `12040accade4e8a0f71eabdb258fecc2e7e948be` | builder |
| Sample seeds | development 20261021, test 20261022 | builder |
