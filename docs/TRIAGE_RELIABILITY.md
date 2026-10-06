# From experimental triage to a validated, human-assisted hotel workflow

The current diagnostic update and development capture are recorded at
[e1e37e5](experiments/triage_space_publication/e1e37e5/README.md). The
[earlier release](experiments/triage_space_publication/ea8c7f8/README.md) passed its publication run,
but a later independent API request had a visible GPU worker failure.
Source-span extraction removes the quote-copying failure; it does not prove coverage, sound exclusions,
correct departments or useful measures. No candidate has been selected or promoted.

## Current engineering changes

The diagnostic update recognizes the exact error titles emitted by the pinned `spaces 0.51.3` SDK:
quota exceeded, pending-credit allocation, illegal duration, queue timeout, client scheduling and worker
failure. Unknown exceptions remain unknown; searching a review or arbitrary exception message for the
word “quota” would be an unreliable diagnosis. Logs retain only a closed error code, without review text,
raw model output, provider response or token. The UI explains the next action without treating a missing
GPU result as an absence of problems. There is no automatic retry or paid fallback.

The publication check preserves earlier successes and the failing live case in `published.json`, with
`runtime_verified: false`. A successful upload alone never changes that flag. Hardware and model revision
pins remain recorded; shortening the 60-second allocation without testing long inputs is not a remedy.
These changes were published through [notebook 20](../notebooks/20_triage_reliability_colab.ipynb).
The owner-run receipt, all 21 deployed files and loaded snapshot are verified in the current record.
The fresh capture completed 24 requests, including four issue-stage failures and eleven UNCERTAIN
assessments without measures. A measure for a successfully replaced bulb is also a review priority.
The original capture sheets remain blank. Separately submitted
[completed v3 sheets and adjudication](experiments/triage_space_publication/e1e37e5/human_review_v3/README.md)
now pass the scorer with no remaining disagreements: seventeen actual problems, thirteen without an
appropriate measure, and four useful measures among five emitted. These are adjudicated synthetic
development judgments; reviewer independence is unverified and the reserved set remains closed.

## Fresh development review of the deployed workflow

The notebook first builds and tests the exact package with real pinned weights, updates the existing
Space and checks the loaded snapshot. Only after these succeed does it capture all 24 explicitly
**authored development cases** through the live API. This is a development review, not the independent
real-review pilot or the reserved evaluation.

`scripts/capture_triage_review.py` defaults to Jev off. HF authentication is taken only from `HF_TOKEN`;
no key appears in an argument, log or result. `--jev` explicitly opts in to provider requests using the
Space's own server-side key; report usage cost and resolved model, and do not infer currency or GPU cost.
An enabled checkbox is not proof that a provider call succeeded. Avoid interpreting agreement with Qwen
as a Jev quality score; the whole/mixed-review questions need their own human ground truth.

The capture binds source manifest, model/runtime information, fixed outputs, case identities, capture
script hash and request condition. It verifies the snapshot before, during and after the run. Client jobs
have a 180-second deadline. A GPU/provider/transport failure stops further requests, is recorded, and
leaves all remaining planned cases explicitly `not_executed`, not empty successful outputs. Interrupted
captures cannot be scored. Repeated captures use new folders; existing evidence is never overwritten.

The bundle contains `coverage_A.csv` and `coverage_B.csv`, identical blank sheets for two reviewers.
Keep the source bundle intact and save completed copies separately. Do not transfer the historical F/G
ratings or alter reviews, issue assessments, accepted actions, blind IDs or execution findings.

Each reviewer independently fills these nonnegative integer counts (zero must be explicit):

| Column | Meaning |
| --- | --- |
| `actual_problem_count` | Actual guest-reported problems without a stated successful fix |
| `missed_problem_count` | Actual problems not represented correctly as pending or uncertain |
| `false_exclusion_count` | Missed problems wrongly excluded as resolved, hypothetical or praise |
| `unaddressed_problem_count` | Actual problems lacking an appropriate grounded hotel action |
| `invented_problem_count` | Invented problems in assessments or actions; count an invention once |
| `unnecessary_action_count` | Actions for praise, hypothetical or already resolved issues |
| `useful_action_count` | Actions correct in all four dimensions and useful to hotel staff |
| `wrong_department_count` | Accepted actions assigned to the wrong department |

Also fill `useful`, `grounded`, `department_correct`, `no_invented_facts` with `1` or `0`. With no action,
`useful=1` only for a justified abstention; the other three fields may be `na`. A failed workflow cannot
be rated fully useful. `grounded` evaluates whether evidence supports the specific claim, beyond a
substring match. Record unsupported causes, guest-directed instructions, placeholders, missed multiple
issues and questionable exclusions in `notes`. A fully quoted invented claim is still incorrect.

```bash
python scripts/score_triage_review.py --run runs/triage_review_RUN \
  --ratings-a completed_A.csv --ratings-b completed_B.csv \
  --output runs/triage_review_scored.json
```

The scorer verifies all capture-file hashes, completeness, immutable outputs, logical count bounds and
agreement on every count and quality dimension. It reports both reviewers and every disagreement;
there is no averaged-away adjudication. Agreeing sheets do not prove reviewer independence, and no
selection or promotion is performed. Resolve disagreements through human discussion and new completed
copies, retaining the originals. Machine failures remain in the development evidence.

## Next corrections and release gates

Use the scored findings to choose the next change: extraction misses, false exclusions, unsupported
claims, department categorization or weak measures. Rerun fresh development outputs after each meaningful
prompt/model/policy change and do not reuse ratings of different outputs. Current F/G summaries and the
seven functional regressions cannot supply the missing source-span quality assessment.

Before changing the experimental label:

1. Verify the updated live package and the Jev off/on behavior from the owner's session; test long inputs,
   repeated requests, first request after sleep, quota messages and mobile layout. Resolve unexplained
   runtime failures. ZeroGPU remains quota-based; no availability guarantee follows from a smoke test.
2. Complete and reconcile the new development review, correct blocking failures, and explicitly decide
   the supported use case and acceptance thresholds. These thresholds must precede final evaluation;
   they cannot be chosen to fit observed final scores.
3. Select and freeze a workflow with code/model revisions, prompts, taxonomy, decoding and runtime policy.
   Run its reserved evaluation once. The old F/G selection code does not automatically select source-span
   G; it needs a reviewed, version-specific evaluation path before the reserved set is opened.
4. Complete a separate usability pilot (15–20 cases) and an independent real whole/mixed-review pilot.
   Establish human issue ground truth before exposing model proposals, record reviewer agreement, and
   evaluate missed issues, false exclusions, invented facts, departments and usefulness with denominators
   and uncertainty. Authored development cases cannot replace this evidence.
5. Document the measured scope, remaining failures, escalation to a person and operational monitoring.
   Only then consider a validated human-assisted release for that scope. Automated work orders and a
   review database are outside the present system.

The change in label is a result of these checks, not a code switch. Deployment and developer tests are
necessary evidence; they do not establish hotel-use reliability on their own.
