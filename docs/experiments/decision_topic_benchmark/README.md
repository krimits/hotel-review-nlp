# Primary-topic comparison

**Three-arm development comparison completed and independently checked.**
On the same 100 dev reviews, macro-F1 is logistic 0.1561, Qwen 0.2299 and
Microsoft Decision-1 0.6524. See the
[actual Colab result](results/comparison_dev_20261012/README.md) for raw and
dev-fitted calibration, retained errors, class support, timing/cost and the
Qwen T4 precision caveat. All 500 human labels remain locked; reviewer
independence is unverified. The final test, model promotion and Space update
remain pending. Resolve precision policy and freeze any changed configuration
on dev before the one final test.

An additional Jev development arm is prepared under the
[Jev extension protocol](JEV_EXTENSION.md): dated `typesafe/jev-1.13-20260917`,
the same five-way question and 100 dev reviews, with the archived Microsoft
capture reused as the reference. No new Jev provider result exists yet.
The extension is CPU/API-only and does not resolve the Qwen precision caveat.
Run [notebook 24](../../../notebooks/24_jev_primary_topics_dev_comparison_colab.ipynb)
in a fresh Colab CPU session with the same `decision_topics_inputs_20261012.zip`.
Default checks only. Set `ALLOW_JEV=True`, choose a positive `MAX_COST_USD`
and enable the `OPENROUTER_API_KEY` Colab Secret to make the 100 new requests.
Return `jev_primary_topics_dev_comparison.zip`, including failed-run diagnostics.
No new annotation sheet, Microsoft run or Qwen/GPU work is needed for this extension.

For the next dev comparison, run [notebook 23](../../../notebooks/23_decision_topics_dev_comparison_colab.ipynb)
on a fresh Colab GPU and upload the validated `decision_topics_inputs_20261012.zip`
returned with the annotation check. Default is logistic + Qwen with no provider
key. Add Microsoft only by explicitly setting `ALLOW_MICROSOFT=True`, choosing
the USD observed-spend budget and enabling the Colab `OPENROUTER_API_KEY` Secret.
The notebook never runs the test partition and retains failed-run diagnostics.
The returned capture records real GPU/Qwen and paid provider execution;
it is archived and reviewed above, without starting a further provider run.

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
the validated submission, initial logistic run and returned three-arm dev
capture are now recorded above.

The [preparation record](preparation_record.json) documents a sequential local
execution of notebook 22's code cells with a real fresh clone, isolated
package installation, runtime preflight, 500-review sample and verified ZIP
hashes. Only paths and the Colab download callback were substituted. A local
Jupyter kernel could not start because of socket restrictions; execution in
Colab and visual inspection of the exported notebook remain pending. This
is a preparation check, not a model-quality result.
