# Jev extension: dev-only comparison with the recorded Microsoft result

**Declared 12 October 2026, after Microsoft dev was inspected and before
any Jev primary-topic inference.** This is an additional development arm,
not a prospectively unseen confirmation of a four-model winner. The original
protocol, sample, human gold and three-arm capture remain unchanged.

Use `typesafe/jev-1.13-20260917`, the dated canonical slug found in
OpenRouter's public Decisions catalog on 12 October. Never use `jev-latest`,
silently accept a model change, or replace failure with a different model.
OpenRouter's documented `/api/alpha/decisions` supports Jev and a `choice`
question. Use exactly the original `primary_topic` rubric, all five criteria
and `state.review`; no hotel metadata, labels, split, notes or sentiment.
Five separate yes/no scores normalized together would be a different task.

## Fixed comparison

- Exactly the same locked 100 dev reviews and human labels as notebook 23;
  no new labels, review resampling or test predictions.
- Exactly one Jev request per review, serial batch 1, no retries or paid
  warm-up. At most 100 requests. The owner explicitly enables the API, sets
  an observed USD budget and uses a newly capped OpenRouter key as a Secret.
- Reuse the full recorded Microsoft build `microsoft-decision-1-20261009`
  from `results/comparison_dev_20261012`. Verify the capture hashes, freeze,
  task/question, labels, all paired review fingerprints and recomputed
  Microsoft scores before any new provider call. No Microsoft calls occur.
- Raw macro-F1 over five classes is the primary development comparison;
  also accuracy, per-class F1/support, ECE (10 fixed equal-width bins), Brier,
  log loss, p95 and provider-reported USD per 1,000 attempted requests.
- Fit one Jev temperature with the original dev-only bounded log-temperature
  optimizer. Preserve the recorded Microsoft temperature. These dev-fitted
  ECE/referral figures are in-sample diagnostics, not independent calibration.
- Max class probability >=0.7 remains an illustrative preset. Report coverage,
  referrals and errors among retained. No retained cases means undefined
  selective error, not zero. All failed/stopped cases go to human review.
- For complete captures, record paired accuracy disagreements and a paired
  review bootstrap of the macro-F1 difference, Jev minus Microsoft:
  2,000 ordinary resamples, seed 20261012, percentile 95% interval,
  absent-class F1=0. Sparse classes make this a dev diagnostic; it does not
  establish deployment reliability. Do not automatically choose a winner.

## Failure, timing, cost and scope

Reject missing/malformed distributions, a choice inconsistent with the
distribution, a changed returned model ID or invalid usage/cost. Retain every
planned row. A partial Jev capture has null quality scores, no comparison and
no freeze. After a failure/unknown cost, or observed spend reaching the cap,
stop subsequent calls; a single request can overshoot. Never print raw
provider errors, exception text, review text or secrets.

Microsoft's API timings were collected earlier; Jev is a new session. Their
p95 values describe these observed paths, not a simultaneously controlled
speed experiment. Record this explicitly. Use reported `usage.cost` in USD;
unknown local compute and failed/unknown API costs remain unknown, not zero.
USD/1,000 is sample scaling, not a tariff or spending forecast.

Notebook 24 is CPU-only and does not import/download Qwen or require a GPU.
It does not resolve the previous Qwen T4 BF16 policy issue. Its freeze is a
Jev dev-extension record, not authorization to run a final four-arm test.
No deployment, review database, automatic hotel actions or HF writes.

The two NYC hotels (2008--2009), sparse class support, unverified reviewer
independence and one primary topic remain material limitations. This is not
negative-complaint detection, quote extraction or hotel measure generation.
If Jev looks better, freeze the eventual final configuration and evaluate
the untouched test once before considering any model promotion.

References verified 12 October 2026:

- [Official OpenRouter Decisions request/response schema](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
- [Public Decisions catalog, including the canonical Jev build](https://openrouter.ai/api/v1/models?output_modalities=decisions)
