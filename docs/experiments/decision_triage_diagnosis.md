# Decision: does the triage student catch 90% of the urgent problems a reference LLM labels?

**2 November 2026. The rule of the [triage diagnosis protocol, version 6](triage_diagnosis/PROTOCOL.md), applied
as written to the one scoring run made after its labels and thresholds were committed. Scope: development
evidence on a public 2015–2017 dataset, against an LLM's labels; no selection is frozen and no model is
promoted.**

## Question

A small model, trained only on a teacher model's labels, diagnoses each problem a guest reports and routes it to
its department. On new whole reviews from the public Booking.com dataset, does it catch at least 90% of the
urgent problems that a reference LLM (ChatGPT, OpenAI) labels?

## Primary comparison

Department-urgent recall on the 500 test reviews against `reference_chatgpt`: a (review, reference urgent type)
pair is caught when the student emits an urgent type of the same department; `rec` is the inverse-probability
weighted ratio by stratum, `lo` the 2.5th percentile of 2,000 bootstrap resamples (seed 0, within strata). Guard:
`fa`, false urgent flags per 100 reviews, IPW. Rule, in order: `rec >= 0.90 and lo >= 0.80 and fa <= 5` Met;
`rec >= 0.90 and lo >= 0.80 and fa > 5` Recall met, over budget; `rec >= 0.90 and lo < 0.80` Point estimate met,
not established; `rec < 0.90` Not met.

## Result

| Test, 500 reviews, against `reference_chatgpt` | Value |
| :--- | ---: |
| `rec`, department-urgent recall (IPW) | **0.7087** |
| `lo`, bootstrap 2.5th percentile | 0.6185 |
| `fa`, false urgent flags per 100 (IPW) | 3.62 |
| Urgent pairs caught (unweighted) | 76 of 104 (0.7308) |

- **The run is complete:** all 500 test reviews have logits; nothing was abstained or dropped.
- **The order held:** the labels and the thresholds (`817698a`) were committed before the one scoring run
  (`3f7a09b`).
- **Thresholds, chosen on development by §4:** urgent 0.30 (development `fa` 4.85, the smallest grid value within
  5), routine 0.35 and structural 0.40 (best development micro-F1 within each group).

Source: `triage_diagnosis/student/score.json` (SHA-256 `9827f687…`), from `logits_test.csv` (`6293ec64…`) and
`thresholds_locked.json` (`e1578e32…`); the student's run is `student/run.json` (`81989d7a…`).

## Decision

**Not met.** `rec` = 0.7087 is below 0.90, so the fourth branch fires: `rec < 0.90`.

## Reported, not deciding

**Where the student falls short.** The teacher, `claude-haiku-5.5`, on the same 500 reviews against the same
reference, catches 0.9669 (`fa` 3.81). The two LLM families agree well; the gap of about 26 points is lost in
distilling the teacher into the student, not in a disagreement between ChatGPT and Claude.

| Urgent type, test | Support | Student emitted | Caught by type | Recall (IPW) |
| :--- | ---: | ---: | ---: | ---: |
| `cleanliness` | 37 | 44 | 32 | 0.845 |
| `hvac_failure` | 38 | 59 | 36 | 0.947 |
| `plumbing_major` | 13 | 7 | 6 | 0.507 |
| `safety_hazard` | 9 | 0 | 0 | counts only |
| `security_fault` | 7 | 0 | 0 | counts only |
| `pests` | 0 | 0 | 0 | — |

- **The student never emits `security_fault` or `safety_hazard` at the locked urgent threshold**, although the
  teacher gave them 293 and 251 training examples. One threshold for the whole urgent group, as the protocol
  fixed it, suits the frequent types and silences these two.
- **The ceiling under the locked rule.** 16 of the 104 urgent pairs cannot be reached at
  the locked urgent threshold of 0.30: 7 are `security_fault` and 9 `safety_hazard`, two
  types the student never emits at that threshold. The most this rule could have scored is
  therefore 88 of 104 pairs = 0.846, below the 0.90 bar. The shortfall has two parts, and
  they are different claims: distillation took the score from 0.846 to 0.7087, and the
  single-threshold choice fixed 0.846 as the ceiling, leaving 0.90 out of reach whatever
  the student did inside the rule. Recovering it needs per-type thresholds, class weights
  or more teacher data (a new protocol and new test reviews), not a re-read of this run.
- **By stratum:** H (trigger word) 68 of 90 pairs (0.756); N (no trigger word) 8 of 14 (0.571); B none.
- **Strict type recall** 0.692; **department accuracy on emitted types** 0.824.
- **Baselines on the same 500 reviews:** the teacher 0.9669 (100 of 104); `complaint-lexicon-v1` 0.0083 (1 of
  104): its negation guard blocks almost every whole review.
- **Second rater** (Gemini Flash) against the reference on the 800: Cohen's kappa 0.74–0.95 on the urgent types,
  median 0.86 over all types.
- **The owner's check, 100 hand-labelled test-pool reviews:** the student's recall against the owner is 0.469
  (13 of 23 pairs unweighted, IPW 0.469), `fa` 3.07. Gemini against the owner: kappa 0.63–0.65 on
  `cleanliness`, `hvac_failure` and `plumbing_major`, 0.0 on `security_fault` and `safety_hazard`. These rest on
  23 urgent pairs and one person.
- **The teacher choice** on development: `claude-haiku-5.5` 0.926, `claude-sonnet-5.5` 0.956; Sonnet's lead of
  0.030 was below the 0.05 the rule required. Teacher cost in all: about $0.94.
- **Not produced:** the clause evidence and the CPU latency. Both need the student's weights, which notebook 29
  did not keep.

## Scope and limits

- **The gold is an LLM's labels.** Every figure is agreement with ChatGPT (OpenAI, model shown as "GPT 6.1
  Terra"), run once in a chat session that cannot be reproduced; not agreement with the owner, a hotel's staff or
  any person.
- **No production accuracy claim.** Public Booking.com reviews from 2015–2017, European hotels, English, written
  after checkout.
- **The human anchor is small:** 100 reviews, 23 urgent pairs, one person.
- **The second rater is Gemini Flash**, run in a web app; it is reported only.
- **What the result does not say.** It does not say that a small student cannot reach 90%. It says this student,
  trained once with the version 6 settings and one urgent threshold, does not. The settings that shaped it, such
  as no class weights and one threshold for the group, were fixed before any result.
- **The test set is now used.** Any new attempt, such as per-type thresholds, class weights or more teacher
  data, needs a new protocol and new test reviews. These 500 cannot judge a model tuned after seeing this
  result.
- `selection_frozen` and `promoted` stay false. No Space change follows.

## Artifacts

| What | Where | SHA-256 |
| :--- | :--- | :--- |
| Protocol, version 6 | `docs/experiments/triage_diagnosis/PROTOCOL.md` | `18d2555c…` |
| Score | `triage_diagnosis/student/score.json` (`3f7a09b`) | `9827f687…` |
| Thresholds | `triage_diagnosis/student/thresholds_locked.json` (`817698a`) | `e1578e32…` |
| Student logits and run | `triage_diagnosis/student/` from notebook 29 at `dba52d0` (`7e8c382`) | `6293ec64…` (test logits), `81989d7a…` (run) |
| Reference labels | `triage_diagnosis/labels/reference_chatgpt.csv` | `3f2538f2…` |
| Teacher labels | `triage_diagnosis/labels/teacher_haiku_teacher.csv` | `41c1f005…` |
| Owner's check | `triage_diagnosis/labels/owner_check.csv` | `2be23b58…` |
| Run log | `docs/RUNS.md`, entries of 19 October to 2 November 2026 | |
