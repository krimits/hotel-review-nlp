# Triage smoke test

## Recorded execution

The [3 October 2026 real run](20261003T071410Z_18020ced/README.md) completed both
arms on a Colab T4: Qwen wrote text for 6 cases without Jev and 9 with Jev.
**Neither arm produced an accepted action.** This is execution evidence, not
an evaluation. The follow-up is a
[controlled generator comparison](../triage_generator/README.md) on new texts.

A check that the triage chain runs with the real models, and a way to look at what each stage says about 14
invented reviews: 4 positive, 4 negative, 5 mixed (praise and a complaint together) and one very short one.

**It is not an evaluation.** There are no labels, no metrics and no thresholds. The `expected_*` fields in
[`reviews.json`](reviews.json) are the author's own reading, so that the output can be looked at quickly. Nothing
is scored on them, and no threshold or prompt may be tuned on them. Fourteen reviews could not support a
conclusion about quality anyway.

The reviews are invented, so no guest's text is sent anywhere. With `--allow-external-api`, these 14 invented
reviews go to the Jev provider.

## What it does

[`scripts/triage_smoke.py`](../../../scripts/triage_smoke.py) runs every review through two arms that share the
sentiment model and the generator:

- **without Jev:** DistilBERT, then Qwen if the sentiment is negative or doubtful. Nothing leaves the machine.
- **with Jev:** DistilBERT, Jev, then Qwen if anything was flagged. Only with `--allow-external-api` and the
  route's key.

The sentiment model is the clean-split DistilBERT as published
([`krimits/distilbert-hotel-reviews`](https://huggingface.co/krimits/distilbert-hotel-reviews)), at the revision
recorded in the
[publication record](../results/distilbert_v2/README.md), unless `--model-path` names a folder. Qwen is the base
`Qwen/Qwen2.5-0.5B-Instruct`.

It writes `results.jsonl` (one stage-by-stage result per review and arm, with what the suggestion model wrote or
how it failed), `run.json` (what ran: script and fixture hashes, the model revisions, the versions of the libraries,
the Jev model that answered, and per arm how many reviews each stage answered how and for how many the suggestion
model wrote text) and `summary.md` (a table, the stage counts, the problems and the notes, and the model's own words
where it gave no usable actions) to a folder inside `runs/`, which git ignores.

The pipeline keeps neither a raw generation nor an exception, because in use a review must not travel with them. The
script wraps the suggestion model only to keep both for this report, so that a failure says what went wrong and not
only `generation_failed`.

## Problems and notes

A **problem** means the chain is not working, and the exit code is 1:

- a stage raised an error;
- the sentiment labels are not `negative` and `positive`. The routing would then read every review as not
  negative, without any error;
- the sentiment probabilities do not add up to 1, or are missing;
- Jev was switched on and failed (for example `http_402` when the credit has run out), did not answer the six
  questions, or answered other questions than the committed ones;
- **the suggestion model did not run:** it raised an error, with the message in the report (a weights file that
  would not load, a GPU that ran out of memory), or it was never asked about any review in an arm, so nothing about
  it was tested. A run where Qwen never ran is not a pass.

A **note** is something to look at in what a model said. The suggestion model ran, and it gave invalid output; or it
ran out of tokens (an error in the result, flagged, even if what it wrote was valid JSON); or its actions were not
in the review; or it gave an empty list. Or the sentiment disagrees with the author's reading, or Jev found a
topic the author did not expect, or missed one the author did.

## What to look at

- **Do mixed reviews reach the suggestion stage under Jev, when the sentiment alone would miss them?** That is the
  reason for the Jev arm. Compare the two arms on `mix-1` to `mix-5`.
- **Is a positive review with nothing wrong left alone?** `pos-1` to `pos-4`, including `pos-3` ("No problems at
  all"), where saying that nothing was wrong is not a complaint.
- **Does Qwen quote the review, and is the measure sensible?** The code checks that the quote is in the review. It
  cannot check that the measure is any good.
- **Time per stage.** The first review that reaches Qwen includes loading its weights. On a CPU the generation is
  slow.
- **What `needs a look` says,** and whether the reasons make sense for the review.

## What it does not show

It does not show how well any stage does. Nothing here is a measured rate. A review the models get right says
nothing about the next one, and 14 invented reviews are easier than guests' real ones. The evaluation that would
show it is described in [TRIAGE.md](../../TRIAGE.md#what-has-not-been-done).

## Running it

Colab notebook [`13_triage_smoke_colab.ipynb`](../../../notebooks/13_triage_smoke_colab.ipynb), with a GPU if
there is one. The Jev key is read from Colab's Secrets (a secret named `OPENROUTER_API_KEY`) and is never typed
into a cell. Or, from a checkout with the package installed:

```
python scripts/triage_smoke.py                       # without Jev
python scripts/triage_smoke.py --allow-external-api  # both arms (OPENROUTER_API_KEY set)
```
