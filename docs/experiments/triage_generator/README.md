# Generator development after the real triage smoke run

The [recorded smoke run](../triage_smoke/20261003T071410Z_18020ced/README.md)
completed, but accepted zero actions. This experiment tests the generator
without changing production prompts, parser, routing, sentiment model, or
complaint questions. There is no training and no automatic deployment.

## Controlled variants

| Candidate | Base instruct model, no adapter | Prompt | Comparison |
|---|---|---|---|
| A | Qwen2.5-0.5B | current actions-v1 | baseline |
| B | Qwen2.5-0.5B | actions-v2, shorter rules and two examples | prompt package vs A |
| C | Qwen2.5-1.5B | same actions-v2 | model size vs B |
| D | Qwen2.5-0.5B | actions-v2, sentiment metadata omitted | sentiment metadata vs B |

B changes the instruction package and adds examples together; it does not
isolate the contribution of few-shot examples. C is a capacity experiment in
the same family, not a promise of improvement. D tests a hypothesis suggested
by copied metadata in the smoke output.

All variants use greedy decoding, a 400 new-token budget, and the existing
parser and department names. The new prompt asks for at most two concise
actions to fit that budget. Budget exhaustion remains a failed completion,
even if a fragment passes the parser. Raw output and accepted actions are
stored separately. A still uses the production prompt exactly as written.

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
