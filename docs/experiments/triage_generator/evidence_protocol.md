# Evidence-first development comparison (notebook 17)

Status: implementation and offline wiring checked; real-weight GPU execution and human ratings pending.
No selected production generator, independent final evaluation or Space deployment is recorded here.

## Paired scope

Compare **F (corrected actions-v4-staged)** and **G (actions-v5-evidence)** on the same 24 authored
development reviews, cached DistilBERT/Jev predictions and Qwen2.5-1.5B weights. Reuse the archived
`20261003T145901Z_a29e8d68` upstream, not an API request. Reference bytes, dataset, code, model revisions
and prompt fingerprints are verified and recorded. Historical model outputs and human sheets stay unchanged.

Both workflows have a maximum of two calls of 400 new tokens each, with `do_sample=False`. One independent
warmup per candidate is excluded from case latency. Actual calls, failures and latency can differ. Record
hardware, library versions and precision policy; GPU billing is unknown. Reused historical Jev usage is not
charged again by the comparison. The final reserved set is not opened by notebook 17.

G jointly changes the prompt, issue schema, evidence consistency policy and category-to-department mapping.
This is a workflow comparison, not an isolated causal test of one prompt sentence or evidence field.

## Evidence policy

Each item has a problem, exact excerpt, category, status and three separate nullable exact quotes:
`reported`, `hypothetical`, `resolved`. `REAL_PENDING` describes an actual guest report without a stated
successful fix; it does not verify today's condition. A response or workaround is not proof of resolution.

The parser requires one whole JSON object (optionally one complete JSON Markdown fence), unique keys,
closed fields and verbatim quote presence. Missing or conflicting evidence downgrades a claimed exclusion
to `UNCERTAIN`, retaining `model_status` and review reasons. Uncertain items remain visible and do not receive
corrective measures. A Jev complaint signal without a supported reported/uncertain issue requests human review.

G also preserves the literal excerpt through the final action parser; the existing API and earlier
variants keep their default normalized quote policy. This policy difference is part of the G package.

Presence checks cannot determine whether a copied quote supports the claim. The model can supply an
irrelevant quote or misread a failed repair as successful. Human assessment of exclusions is essential.
There is no constrained decoding, lexical override of model semantics or external automated judge.

Departments come from `CATEGORY_DEPARTMENTS`, with constructor overrides validated against the category
and department lists. This prevents stage two from reassigning a department; it does not guarantee correct
categorization. `access` and `other` require department confirmation. Stage two addresses supplied pending
issue IDs only. Each accepted measure adds a current-condition check. Pending items without a measure are
flagged, including items beyond the two-measure cap.

## Human review

Keep the downloaded source ZIP untouched. Save completed sheets as separate files. Never edit review text,
issue assessments, actions, blind IDs or machine execution fields. No historical labels are transferred to
new outputs. Inspect the whole review and every excluded issue, not just proposals.

`human_review.csv` keeps the existing four dimensions: usefulness, grounded evidence, correct department,
and no invented facts. Use the existing `0`/`1`/`na` protocol; incomplete or partial judgments must be resolved
by a person before a frozen selection.

`coverage_review.csv` uses the same shuffled blind IDs. Give nonnegative integer counts (zero is explicit):

| Field | Human definition |
|---|---|
| `actual_problem_count` | Distinct actual guest-reported problems without a stated successful fix. Same count for this review in F and G. |
| `missed_problem_count` | Actual problems not correctly represented as reported pending or uncertain issues. |
| `false_exclusion_count` | Subset of missed problems wrongly excluded as resolved, hypothetical or praise. |
| `unaddressed_problem_count` | Actual problems lacking an appropriate grounded operational measure. Uncertain items can be recognized yet unaddressed. |
| `invented_problem_count` | Distinct invented problems in the issue assessments or proposals; do not count the same invention twice. |
| `unnecessary_action_count` | Accepted proposals for praise, imaginary problems or already successfully resolved issues. |
| `useful_action_count` | Accepted proposals correct in all four quality dimensions, concrete and actionable by the hotel. |
| `wrong_department_count` | Accepted proposals assigned to an incorrect department; adjudicate ambiguous cases before counting. |

Record classification details and ambiguous cases in `notes`. A correctly quoted placeholder, guest-directed
instruction or unsupported cause is not a useful measure. Code/schema errors remain machine findings and
are not removed from denominators. Counts of accepted outputs alone do not establish improvement.

Score completed coverage sheets without modifying the source run:

```bash
python scripts/score_triage_coverage.py \
  --run runs/generator_evidence_dev_RUN \
  --ratings completed_coverage_review.csv \
  --output runs/coverage_review_scored.json
```

The scorer verifies source hashes, fixed columns, complete unique IDs, logical count bounds and matching
paired human ground truth. It returns totals and human issue/action coverage. Zero-denominator fractions
are unknown (`null`). It does not freeze a selection or judge content automatically.

## Decision and next stages

Review coverage, wrong exclusions, invented issues, all four action dimensions, workload, failures and
cost together. A larger output count or structurally valid JSON is insufficient. Do not promote G based on
this implementation or a parser replay. Complete and reconcile both human sheets before choosing a version.
This small, author-known synthetic set is development evidence, not production accuracy.

After explicit selection, freeze code, model revisions, prompts, mapping and decoding settings, then run
the existing reserved evaluation once. A separate real whole/mixed-review pilot with independent human
annotations is still needed before a reliability claim for hotel operations.

## Space preparation

`spaces/hotel-triage-demo/` is a separate experimental UI for this chain; the existing ABSA and sentiment
Spaces are not replaced. All reviews reach issue extraction there, including confident positives. It
shows excluded and uncertain issues, evidence, proposals, department, checks, stage failure, per-request
time and provider-reported cost. Keys are server-side, Jev is opt-in and no review database is used.

`scripts/prepare_triage_space.py` builds a small source-only package with an explicit allowlist and SHA-256
manifest. It performs no Hub write or model download. Live rebuild, real-weight inference and pilot checks
are pending; a stubbed Gradio build cannot establish those results.
