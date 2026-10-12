# Primary-topic comparison

**Human labels validated; first logistic dev run complete.** All 500 final
labels passed the identity/completeness checks on 12 October 2026. The
[first supervised baseline](results/logistic_dev_20261012/README.md) predicts
`other` on every dev review: accuracy 64%, macro-F1 0.1561. Reviewer
independence remains unverified. Qwen/Microsoft dev inference and the final
test are pending. This does not select a model or change the Space.

[Protocol](PROTOCOL.md) is fixed before any labels or inference. Five classes
describe one main topic regardless of sentiment. A sample of 500 whole
reviews is split 300 train / 100 dev / 100 test. This is a pilot on the two
NYC hotels in the already committed corpus, not 500 test reviews or a hotel
operations reliability score.

## Start without an API key

Run [notebook 22](../../../notebooks/22_decision_topics_sample_colab.ipynb)
on Colab CPU. It exports `decision_topics_sample.zip` with blank A (500), B
(100 independent overlap), final (500) CSVs and hashes. Preserve the original
ZIP. Complete copies of the sheets following the protocol's table, then
adjudicate disagreements in a completed final sheet. Do not edit texts, IDs
or hashes, and do not look at model predictions first.

The same preparation can be run locally, without model downloads:

```bash
python scripts/benchmark_decision_topics.py sample --output runs/decision_topics/sample
python scripts/benchmark_decision_topics.py labels \
  --sample runs/decision_topics/sample \
  --a completed_A.csv --b completed_B.csv --final completed_final.csv \
  --output runs/decision_topics/labels.json --independent-reviewers
```

The last flag is an attestation of independent review, not a validator's
finding. Omit it if the reviews were not independent. All three sheets must
be complete; model-invented labels are not accepted as a ground-truth method.

## Dev first, then a frozen test

Use an isolated environment. For Qwen use the project's tested Transformers
4.56.2, huggingface-hub 0.36.2, and the pinned model revision. A supervised
CPU-only run is available first:

```bash
python scripts/benchmark_decision_topics.py run \
  --sample runs/decision_topics/sample --labels runs/decision_topics/labels.json \
  --partition dev --arms logistic --output runs/decision_topics/lr_dev
```

The full comparison includes local Qwen and **one paid Microsoft request
per dev review**. Only enable it after choosing a capped OpenRouter key,
checking the data terms, and deciding the account budget. Put the key in
the environment or Colab Secrets; never in a notebook cell or a command:

```bash
python scripts/benchmark_decision_topics.py run \
  --sample runs/decision_topics/sample --labels runs/decision_topics/labels.json \
  --partition dev --device cuda --arms logistic qwen decision \
  --allow-external-api --max-cost-usd 0.10 --output runs/decision_topics/dev
```

`0.10` is an example software observed-spend limit, **not an approved charge
or an account-level cap**. One request can overshoot; missing cost stops the
next call. A partial arm has no quality scores and no complete freeze.

When all three dev arms complete, the script writes `freeze.json`. For one
test run, use identical source/environment/device, inputs and arm order:

```bash
python scripts/benchmark_decision_topics.py run \
  --sample runs/decision_topics/sample --labels runs/decision_topics/labels.json \
  --partition test --device cuda --arms logistic qwen decision \
  --freeze runs/decision_topics/dev/freeze.json \
  --allow-external-api --max-cost-usd 0.10 --output runs/decision_topics/test
```

Outputs are `predictions.jsonl`, `run.json`, `human_queue.csv`, and a dev
freeze only when complete. The queue uses calibrated max class probability
<0.7; every failed or unexecuted pair is also referred. `run.json` contains
raw/calibrated macro-F1, ECE with bin counts, coverage/selective risk,
request p95 and provider USD/1,000 attempts, with unknown local compute
cost explicitly null. Test scores need class support and the 100-review
denominator when reported. There is no automatic model selection or deploy.

Local/offline tests exercise real scikit-learn fits, frozen dev calibration,
API request/response contracts, failure accounting, and a tiny random Qwen2
through real Transformers APIs. They establish wiring, **not** the quality
of the published Qwen or Microsoft model. The 500-review labels, real Qwen
weights and real Microsoft responses were pending at initial preparation;
the validated submission and actual logistic dev run are now recorded above.

The [preparation record](preparation_record.json) documents a sequential local
execution of notebook 22's code cells with a real fresh clone, isolated
package installation, runtime preflight, 500-review sample and verified ZIP
hashes. Only paths and the Colab download callback were substituted. A local
Jupyter kernel could not start because of socket restrictions; execution in
Colab and visual inspection of the exported notebook remain pending. This
is a preparation check, not a model-quality result.
