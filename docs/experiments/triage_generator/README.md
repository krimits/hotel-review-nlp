# Generator development after the real triage smoke run

The [recorded smoke run](../triage_smoke/20261003T071410Z_18020ced/README.md)
completed, but accepted zero actions. This experiment tests the generator
without changing production prompts, routing, sentiment model, or
complaint questions. There is no training and no automatic deployment.
The parser now rejects duplicate JSON keys instead of silently choosing the last value;
see the [recorded v3 diagnosis](prompt_v3_review.md).

## Controlled variants

| Candidate | Base instruct model, no adapter | Prompt | Comparison |
|---|---|---|---|
| A | Qwen2.5-0.5B | current actions-v1 | baseline |
| B | Qwen2.5-0.5B | actions-v2, shorter rules and two examples | prompt package vs A |
| C | Qwen2.5-1.5B | same actions-v2 | model size vs B |
| D | Qwen2.5-0.5B | actions-v2, sentiment metadata omitted | sentiment metadata vs B |
| E | Qwen2.5-1.5B | actions-v3, explicit issue eligibility and manager actions | system instruction package vs C |
| F | Qwen2.5-1.5B | actions-v4-staged, issue extraction then measures linked to issue IDs | workflow package vs C |

B changes the instruction package and adds examples together; it does not
isolate the contribution of few-shot examples. C is a capacity experiment in
the same family, not a promise of improvement. D tests a hypothesis suggested
by copied metadata in the smoke output.

The default development command and notebook 14 still run A–D. E is a follow-up
to C: it keeps C's two examples, user message, metadata and model size. Only
the system instruction changes. It distinguishes reported pending issues
from resolved, hypothetical and positive mentions, asks for evidence that
supports the specific problem, and directs operational measures to the hotel
manager. It returns the existing `actions` schema without a reasoning block.
These instructions are a hypothesis, not verified semantic enforcement.

Single-pass variants A–E use greedy decoding, a 400 new-token budget, and the
action parser and department names. The new prompt asks for at most two concise
actions to fit that budget. Budget exhaustion remains a failed completion,
even if a fragment passes the parser. Raw output and accepted actions are
stored separately. A still uses the production prompt exactly as written.

The experiment also records `full_json_valid`, meaning the entire output is
one JSON object without repeated keys or non-JSON constants. This is separate
from `parser_json_valid`, which may accept recovered fragments. Neither is a
measure of usefulness, evidence entailment, or strict schema conformance.
The case/whitespace-normalized action quote check stays unchanged. Duplicate keys in a
decoded object now make an output invalid, including when the second value is empty.

## New cases and the reserved set

[dev.json](dev.json) contains 24 newly authored synthetic texts.
[holdout.json](holdout.json) contains another 24, reserved for after selection.
Both include negative, mixed, positive, resolved, conditional, request, and
short cases. These scenario names describe the author's intention; they are
**not ground truth and are never passed to a model**. The two demonstration
examples in the new prompt are separate from both sets and the smoke fixture.
Tests check IDs and normalized-text overlap.

The old fourteen smoke texts and the old Space dev/test sets are excluded.
Do not read reserved outputs to revise the same configuration and then report
them as an untouched test. If final results lead to changes, create another
reserved set. These few synthetic texts do not establish performance on real
hotel reviews; an independently annotated real-review pilot comes afterwards.

## Run and record

Use [notebook 14](../../../notebooks/14_triage_generator_comparison_colab.ipynb)
on Colab GPU, with the OPENROUTER_API_KEY Secret enabled, or:

~~~bash
python scripts/compare_triage_generators.py dev \
  --allow-external-api --output runs/generator_dev_NEW_ID
~~~

Only these invented development texts go to Jev. DistilBERT is pinned to its
published commit. Jev uses the existing jev-latest request alias and questions;
the responding model is recorded and must stay the same within the run.
It may differ from the historical smoke model. DistilBERT, Jev and routing
run once per text and are frozen in upstream.json, reused across candidates.
Qwen model commits are resolved once and pinned for tokenizer and weights.

All cases are queried for **generator diagnosis**, including positive controls
that production routing would skip. production_routed preserves that decision
for subgroup analysis; this forced querying does not alter application routing.
Never present counts over all cases as the production pipeline's performance.

Files include results.jsonl (raw output, accepted entries, failure and case
latency), run.json (revisions, prompt and dataset hashes, hardware, warmups,
completion counts and cost availability), upstream.json, summary.md, and
the human-review sheet/key. Warmup/loading is timed separately for each
candidate. Case times are observations on this GPU/session, not a Space CPU
benchmark. Failed/partial runs stay available and cannot be used for selection.

The transport records each Jev attempt, including retries and failures.
Only numeric provider-reported usage token counts and usage.cost are kept.
An omitted cost is null, never zero; a partial reported sum is not a total
bill. Currency is not inferred, and local GPU billing is unavailable.
If cost is absent, obtain the provider's usage/billing record before making
a cost-benefit claim. Qwen comparison itself makes no extra Jev calls.

## Follow-up: actions-v2 versus actions-v3 with the same upstream and weights

Use [notebook 15](../../../notebooks/15_triage_prompt_v3_comparison_colab.ipynb)
on Colab GPU and upload the previous generator-development ZIP
(`20261003T084949Z_51c7a13e`). It checks the known run/upstream hashes, then
compares C and E on the same 24 development cases with the same Qwen 1.5B
commit. It makes **no new Jev requests**, does not load DistilBERT, and needs
no OpenRouter Secret. It produces 48 outputs and a fresh human-review sheet.

For another verified completed development run, extract its `run.json` and
`upstream.json`, then use:

~~~bash
python scripts/compare_triage_generators.py dev \
  --candidates C E --reference-run /path/to/previous_dev \
  --output runs/generator_prompt_v3_dev_NEW_ID
~~~

Reference reuse checks the dataset, upstream configuration, responding Jev
model, cache fingerprint, completed-run status, decoding settings and baseline
prompt. Model commits come from that run rather than today's mutable Hub main.
The reused upstream bytes and original run metadata are preserved in the new
output. Current API attempts are zero; historical provider cost is labeled
separately and local GPU billing is still unknown. Without `--reference-run`,
fresh upstream calls still require `--allow-external-api`.

Review C and E blindly using the same rubric below. Count false problems from
praise/conditional/resolved passages, useful measures for pending problems,
and missed genuine issues. Do not use more accepted actions as a quality score.
The reserved set remains unopened until a human choice and freeze.

## Human usefulness, separately from parser acceptance

Give the reviewer human_review.csv, not annotation_key.json. Rows are
shuffled and model identities are replaced with opaque output IDs. A single
reviewer is an initial screen, not inter-annotator agreement. For each row,
read the review and accepted actions and fill:

| Field | Values | Question |
|---|---|---|
| useful | 0 / 1 | Does it propose a concrete, feasible step for the main unresolved issue, or appropriately abstain when none exists? |
| grounded | 0 / 1 / na | Do all proposed problems and excerpts actually concern a problem reported here? |
| department_correct | 0 / 1 / na | Are the assigned departments appropriate for all retained actions? |
| no_invented_facts | 0 / 1 / na | Does it avoid invented dates, room numbers, diagnoses, promises or completed work? |

Use na for the last three fields when there are no accepted actions.
Do not edit review, accepted_actions or execution_issue: they are machine columns.
execution_issue records a model/runtime, token-budget or output-structure failure;
an unhelpful measure is graded in the human columns, not as a runtime exception.
`partial` is not a final rating under this binary rubric. Re-examine whether all
retained actions satisfy the field's criterion; a mixed acceptable/unacceptable
set does not satisfy an all-actions criterion. Resolved issues do not qualify
as pending problems, even if a preventive measure might be sensible elsewhere.
An empty answer to a concrete unresolved complaint gets useful=0; a sound
abstention on praise can get useful=1. An execution failure or exhausted
budget gets useful=0 regardless of any retained fragment. Record why in
notes. Reading raw output can help diagnose a failure, but it is not what the
hotel would receive. Neither valid JSON nor an exact quote establishes useful
advice. An exact quote can still quote praise.

After all rows are reviewed, compare usefulness, failures, unsupported content
and latency. Human selection is explicit; the script never declares a winner.
If no candidate provides useful measures, revise on development cases or try
another model before proceeding.

### Audit an externally edited rating sheet

~~~bash
python scripts/audit_generator_review.py --run /path/to/intact_extracted_dev \
  --ratings /path/to/completed_human_review.csv --output runs/rating_audit_NEW_ID
~~~

The intact archive is matched by its manifest, exact blind IDs, original review
text and accepted actions. Unambiguous yes/no/n/a tokens become 1/0/na in a
separate output. The execution_issue column comes from the original export;
overwrites are recorded. Partial, missing and invalid judgments are listed
for human adjudication; they are never silently rounded to a success or failure.
With unresolved judgments the command returns 2 and writes adjudication.csv,
not a freeze-ready CSV. It does not select a model, open the reserved set,
regrade semantics, or replay a historical sheet through today's parser.

## Follow-up: issue extraction and measures as separate generations

Use [notebook 16](../../../notebooks/16_triage_staged_comparison_colab.ipynb) on
Colab GPU. Upload the **notebook 15 ZIP** (`20261003T121429Z_876f0781`), which
has been verified and [archived](runs/20261003T121429Z_876f0781/manifest.json).
It compares C and F on the same 24 development reviews and frozen upstream
predictions, with the same Qwen 1.5B revision and no new Jev calls or API key.

F first extracts short issue records with a verbatim excerpt, department and
one of REAL_PENDING, REAL_RESOLVED, HYPOTHETICAL or POSITIVE_COMMENT. Only
structurally valid, literally quoted REAL_PENDING records reach the second
generation. The second output has issue_id, measure and to_confirm only;
code copies problem, excerpt and department from the referenced issue. It
cannot silently introduce another issue or change its evidence or department.
The extraction can still misclassify praise, miss an issue or choose the wrong
department, and a measure can still be inappropriate. Human review is essential.

Each stage requires one JSON object with unique keys and its own exact schema.
One complete JSON/unlabelled Markdown fence is also accepted as an envelope;
prose outside it, incomplete fences and multiple objects remain failures. The
original model text remains in stages[].raw. Literal whole-JSON diagnostics
describe that text, including rejected workflows.
Invalid output and either exhausted token budget remain explicit workflow
failures, not valid abstentions. Actual model text, stage timings, extracted
issues and failure stage are saved in results.jsonl. F's top-level raw is
assembled application output; stages[].raw contains the actual generations.
When there are no pending issues, the measure call is skipped and this is visible.

This changes the workflow, examples and schema together. It is **not a pure
prompt ablation or an equal-budget comparison**: C permits one 400-token call,
F up to two 400-token calls. The manifest records that budget and case times
include both calls. Weights are shared; model load and warmup are excluded.
Both candidates use the updated duplicate-key guard; fill the new shuffled
48-row sheet rather than transferring historical ratings. The new shuffle
seed differs from C/E, but one reviewer's repeated development ratings remain
an initial screen, not independent agreement or held-out performance.

The notebook checks that the second stage was actually exercised at least once,
then downloads diagnostic artifacts even on failure. It performs no training,
freeze, reserved evaluation or Space deployment. The
[3 October staged run and offline parser replay](staged_run_review.md) found
five JSON-wrapper rejections and serious first-stage classification errors.
The original GPU run produced zero accepted F actions; replaying the exact
saved text with wrapper support recovers five structurally valid measures.
Human quality remains unvalidated, and F is not selected for the demo.

## Freeze, then one reserved evaluation

Keep the completed development folder and ratings. Select a candidate:

~~~bash
python scripts/compare_triage_generators.py freeze \
  --dev-run runs/generator_dev_NEW_ID \
  --ratings runs/generator_dev_NEW_ID/human_review.csv \
  --candidate B --selection runs/generator_selection.json
python scripts/compare_triage_generators.py holdout \
  --allow-external-api --selection runs/generator_selection.json \
  --output runs/generator_holdout_NEW_ID
~~~

B above is an example, not a recommended winner. The freeze step requires a
complete development run, complete human ratings, unchanged machine artifacts,
and at least one retained measure rated useful, grounded, appropriate and
free of invented facts for the selected candidate. It
records the choice, human judgments' hash, prompt hash and pinned model commit.
This is a minimal selection guard, not a quality threshold.

The holdout command requires that record and uses exactly its configuration.
It also requires the actual responding Jev version recorded during development;
if the request alias has changed models, it stops before final Qwen generation.
A marker in runs/ blocks a repeated final evaluation in that checkout, even
after failure. This is a workflow guard, not a security boundary: preserve the
record across Colab sessions and do not reset it to tune on the reserved set.
Human review is also needed for the final output.

Only after that review should the chosen configuration be proposed for the
triage application. Then run the independently annotated real-review pilot,
compare sentiment-only and Jev routing at fixed thresholds, and evaluate
missed complaints, false alerts and useful measures against human labels.
The existing complaint-lexicon pilot has a different target and does not by
itself validate generated measures.
