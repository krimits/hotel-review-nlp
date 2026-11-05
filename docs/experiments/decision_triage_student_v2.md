# Decision: does the second triage student catch 90% of the urgent problems a reference LLM labels?

**5 November 2026. The rule of the [triage student v2 protocol, version 1](triage_student_v2/PROTOCOL.md), applied
as written to the one scoring run made after its labels, thresholds and choice of variant were committed. Scope:
development evidence on a public 2015–2017 dataset, against an LLM's labels under `catalogue_v2`; no selection is
frozen and no model is promoted.**

## Question

A small model, trained only on a teacher model's labels, diagnoses each problem a guest reports and routes it to
its department. On new whole reviews from the public Booking.com dataset, does it catch at least 90% of the
urgent problems that a reference LLM (ChatGPT, OpenAI) labels with `catalogue_v2`?

## Primary comparison

Department-urgent recall of the chosen variant on the 500 test reviews against `reference_chatgpt_v2`: a
(review, reference urgent type) pair is caught when the student emits an urgent type of the same department;
`rec` is the inverse-probability weighted ratio by stratum, `lo` the 2.5th percentile of 2,000 bootstrap
resamples (seed 0, within strata). Guard: `fa`, false urgent flags per 100 reviews, IPW. Rule, in order:
`rec >= 0.90 and lo >= 0.80 and fa <= 5` Met; `rec >= 0.90 and lo >= 0.80 and fa > 5` Recall met, over budget;
`rec >= 0.90 and lo < 0.80` Point estimate met, not established; `rec < 0.90` Not met.

The variant was chosen on the 600 development reviews by the protocol's rule: among the variants with development
`fa` at most 5, the highest development `rec`; ties to the simpler (A, B, C); none within the budget, A.

## Result

| Variant on development, 600 reviews, 144 urgent pairs | Thresholds | `rec` (IPW) | `fa` |
| :--- | :--- | ---: | ---: |
| A: BCE, no class weights (the first protocol's recipe) | urgent 0.20, routine 0.45, structural 0.20 | 0.847 | 4.99 |
| **B: BCE with `pos_weight` = min(neg / pos, 20)** | urgent 0.80, routine 0.90, structural 0.45 | **0.875** | 4.18 |
| C: B's model, one threshold per urgent type from the held-out F1 against the teacher | urgent 0.60–0.90 by type; B's others | 0.836 | 4.57 |

All three were within the budget; **B was chosen**. C needed no rise: its development `fa` was already 4.57.

| Test, 500 reviews, against `reference_chatgpt_v2`, variant B | Value |
| :--- | ---: |
| `rec`, department-urgent recall (IPW) | **0.8423** |
| `lo`, bootstrap 2.5th percentile | 0.7517 |
| `fa`, false urgent flags per 100 (IPW) | 4.39 |
| Urgent pairs caught (unweighted) | 104 of 118 (0.8814) |

- **The run is complete:** all 500 test reviews have logits; nothing was abstained or dropped.
- **The order held:** every label (`78c3ba0`, `fa0ed37`), the student bundle (`255b6fa`) and the thresholds with
  the choice (`655ab18`) were committed before the one scoring run (`0341194`). The thresholds read no test file.

Source: `triage_student_v2/student/score.json` (SHA-256 `caf551c4…`), from `logits_B_test.csv` (`72e0ad3d…`) and
`thresholds_locked.json` (`a03f45b3…`); the training run is `student/run.json` (`92310e7c…`), notebook 30 at
`fa0ed37`.

## Decision

**Not met.** `rec` = 0.8423 is below 0.90, so the fourth branch fires: `rec < 0.90`.

## Reported, not deciding

**Where the student falls short.** The teacher, `claude-haiku-5.5`, on the same 500 reviews against the same
reference, catches 0.9491 (114 of 118, `fa` 2.85). On the 2,948 held-out teacher-sample reviews, the student
catches 0.845 of the teacher's own urgent pairs (IPW; 866 of 966 unweighted). So most of the gap to 0.90 is lost
in distilling the teacher into the student, as in the first attempt.

| Urgent type, test | Support | Student emitted | Caught by type | Recall (IPW) |
| :--- | ---: | ---: | ---: | ---: |
| `cleanliness` | 47 | 54 | 44 | 0.912 |
| `hvac_failure` | 42 | 44 | 37 | 0.849 |
| `plumbing_major` | 10 | 17 | 8 | 0.764 |
| `safety_fault` | 10 | 10 | 4 | 0.472 |
| `security_fault` | 8 | 18 | 7 | counts only |
| `safety_incident` | 1 | 0 | 0 | counts only |
| `pests` | 0 | 1 | 0 | — |

- **The rare urgent types are now emitted.** The first student never emitted `security_fault` or `safety_hazard`.
  With class weights and 27,052 training reviews (the teacher gave `security_fault` 613 examples,
  `safety_fault` 587, `safety_incident` 158), this one catches 7 of 8 `security_fault` pairs and 4 of 10
  `safety_fault`. The safety types remain the weakest: against the teacher on the held-out reviews,
  `safety_incident` 4 of 13 and `safety_fault` 34 of 60.
- **By stratum:** H (trigger word) 85 of 91 pairs (0.934); N (no trigger word) 19 of 27 (0.704); B none. The N
  stratum carries a larger weight per review, so its misses pull the IPW `rec` below the unweighted 0.881.
- **Strict type recall** 0.816; **department accuracy on emitted types** 0.822.
- **Baselines on the same 500 reviews:** the teacher 0.9491; `complaint-lexicon-v1` 0.0264 (4 of 118).
- **Development figures of all three variants** are in the first table; they come from the set the choice was made
  on, so they flatter all three. None reached 0.90 on development either.
- **The first attempt is not a baseline for this one.** It scored 0.7087 under `catalogue_v1`, on other test
  reviews, against other reference labels. The two numbers answer different questions and are not compared as
  a gain.
- **Weights.** Both models are in the owner's Google Drive (`run.json`), so the clause evidence and the CPU
  latency can be measured later; this protocol's scoring did not measure them.

## Scope and limits

- **The gold is an LLM's labels.** Every figure is agreement with ChatGPT (OpenAI, model shown as "GPT 6.1
  Terra"), run once in a chat session that cannot be reproduced, with `catalogue_v2`; not agreement with the
  owner, a hotel's staff or any person.
- **No human check.** The owner's agreement with the reference under `catalogue_v2` was not measured. Under
  `catalogue_v1`, on 100 reviews, the owner and the LLMs agreed on about two thirds of the urgent problems;
  whether `catalogue_v2` closed part of that gap is not shown.
- **No production accuracy claim.** Public Booking.com reviews from 2015–2017, European hotels, English, written
  after checkout.
- **Development is partly reused**: 300 of its 600 reviews served the first protocol. The test set was new.
- **What the result does not say.** It does not say that a small student cannot reach 90%. It says that this
  DistilBERT, with the version 6 settings and the best of three variants fixed in advance, does not, while its
  teacher does reach that point estimate on the same reviews. The teacher's figure is reported, not judged by the
  rule: no lower bound was computed for it.
- **The test set is now used.** Any new attempt needs a new protocol and new test reviews.
- `selection_frozen` and `promoted` stay false. No Space change follows.

## Artifacts

| What | Where | SHA-256 |
| :--- | :--- | :--- |
| Protocol, version 1 | `docs/experiments/triage_student_v2/PROTOCOL.md` | `fd72c554…` |
| Score | `triage_student_v2/student/score.json` (`0341194`) | `caf551c4…` |
| Thresholds and choice | `triage_student_v2/student/thresholds_locked.json` (`655ab18`) | `a03f45b3…` |
| Student logits and run | `triage_student_v2/student/` from notebook 30 at `fa0ed37` (`255b6fa`) | `72e0ad3d…` (B test logits), `92310e7c…` (run) |
| Reference labels | `triage_student_v2/labels/reference_chatgpt_v2.csv` | `60342b53…` |
| Teacher labels | `triage_student_v2/labels/teacher_haiku_teacher.csv` | `71ffe775…` |
| Catalogue | `triage_diagnosis/catalogue_v2.yaml` | `5c592af1…` |
| Run log | `docs/RUNS.md`, entries of 2 to 5 November 2026 | |
