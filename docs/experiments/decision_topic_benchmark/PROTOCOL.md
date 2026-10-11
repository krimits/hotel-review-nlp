# Primary-topic decision model experiment — version 1

**Fixed before human labels or inference, 11 October 2026. Status: preparation;
no measured model quality and no Space change.**

Question: on the same held-out whole reviews, does a decision model improve
primary-topic classification, calibration, selective coverage or request cost
relative to a small supervised baseline and our existing general Qwen model?
This is a separate experiment from sentiment, complaint detection, evidence
extraction and suggested hotel actions. Success here does not resolve the
triage chain's semantic failures or establish production reliability.

## Population and split

Use the already tracked `data/raw/hotel_reviews.csv`: 658 unlabeled whole
reviews from Hudson Hotel and Westin New York at Times Square, 2008–2009.
Input is title, two newlines, body. Strip field-edge whitespace only. Remove
empty bodies and duplicate case-folded, whitespace-collapsed title/body texts
before sampling. With seed 20261011, draw 500 reviews without replacement;
assign the first 300 to train, next 100 to dev and last 100 to test, before
any topic label or prediction is read. Do not filter by sentiment or lexicon.

**500 is the total labeling budget, not 500 held-out test cases.** The paired
comparison uses the same 100 test reviews for every arm. Sample manifest
records raw-source, protocol and artifact hashes, split/hotel counts and seed.
Hotels occur in multiple splits. This estimates performance on similar
reviews from these two hotels; it does not estimate unseen-hotel, current
traffic, multilingual or operational triage performance. The original 24
authored triage cases and the triage reserved set are not accessed.

If a class is missing from train, stop. Do not redraw or balance after seeing
test labels. Design a new, separately documented sample if necessary. Small
or absent test-class support must accompany macro-F1 and ECE, which can be
unstable on 100 reviews; do not attach production claims to this pilot.

## Exactly one main topic, regardless of sentiment

| CSV label | Greek label | Inclusion |
| --- | --- | --- |
| `cleanliness` | καθαριότητα | Dirt, hygiene, cleaning quality, stains, pests |
| `service` | εξυπηρέτηση | Staff attitude/helpfulness, requests, communication, service delays |
| `location` | τοποθεσία | Position, surroundings, transport, nearby places |
| `value` | τιμή / σχέση ποιότητας–τιμής | Price, charges, affordability, value; room size alone is not price |
| `other` | άλλο | A main subject outside these four, including room size/equipment, maintenance, noise, food, amenities, or no specific topic |

The label describes the **main evaluation** in the entire review, positive or
negative. Choose the topic the author explicitly emphasizes as the overall
reason for their evaluation. Otherwise choose the topic evaluated in the most
distinct statements. For a tie, use the first tied topic evaluated in the
title, otherwise the first in the body. Incidental mentions and settings do
not count. A multi-topic review still receives one primary label and
`multiple_topics=yes`; `other` is not a substitute for annotation uncertainty.
Discuss uncertain judgments during adjudication before finalizing labels.

Examples (invented, not included in the 500):

- “Spotless room, but reception ignored three requests. That ruined the stay.”
  → `service`, `multiple_topics=yes`.
- “Ideal location. The room was clean too.” → `location`, `multiple_topics=yes`.
- “The air conditioner leaked every night.” → `other`, `multiple_topics=no`.
- “Overpriced for what we received.” → `value`, `multiple_topics=no`.

## Human ground truth

Reviewer A labels all 500. Reviewer B independently labels a fixed random
overlap of 100, without A's labels or model predictions. Sheets show no split,
hotel metadata or model outputs. Preserve the blank originals and submit
completed copies under different names. Both fill:

1. `label`: exactly one of the five CSV strings above.
2. `multiple_topics`: `yes` or `no`.
3. `done`: `yes` only after reviewing the whole text.
4. `note`: optional explanation for discussion.

Do not change `review_id`, `review_sha256` or `review_text`. The final sheet
contains all 500 labels: retain uncontested judgments and adjudicate the
overlap disagreements using the text and this rule. Never inspect model
predictions before human final labels are locked. Human judgment is the gold;
another model's labels are not a substitute. Kappa is measured on the overlap
**before adjudication**. A structural validator checks exact identity,
completeness, allowed values and duplicates. It does not prove independence
or correct semantic judgment. `--independent-reviewers` records the owner's
attestation; without it independence remains `unverified`. The other 400 have
single-reviewer labels, a material limitation.

## Fixed arms and scoring

- **Supervised baseline:** word TF-IDF, unigram/bigram, max 20,000 features,
  `min_df=1`, logistic regression `C=1`, max 2,000 iterations, fixed seed.
  Fit vectorizer and classifier on train only; no test vocabulary or labels
  enter the fit. No hyperparameter search in this first experiment.
- **General LLM:** `Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Prompt it with the same topic
  rubric and fixed A–E mapping. One forward pass scores the five next-token
  letters, normalized over those five. These are **conditional option-token
  probabilities**, not confidence written by the model or a free-form
  generation benchmark. Record total full-vocabulary mass on these letters.
  Fixed order may introduce position bias. CPU float32; CUDA bf16 if natively
  supported, otherwise fp16. No truncation; reject context overflow.
- **Decision model:** `microsoft/microsoft-decision-1-20261009`, canonical
  dated build verified in OpenRouter's public catalog on 11 October 2026.
  `POST /api/alpha/decisions`; one `choice` question with the same rubric and
  five criteria. Require the exact returned build and all five finite,
  nonnegative probabilities summing to one. A mismatched choice/distribution,
  missing probabilities or revision change is an error, never `other`.

Clef-omni is a possible separately declared later arm, not an automatic
replacement if Microsoft fails. Both vendor announcements are evidence of
availability, not evidence of quality on hotel reviews. Raw API probabilities
must be evaluated here even when the vendor calls them calibrated.

Run dev first. Fit one scalar temperature per arm on dev by minimizing log
loss, with log-temperature bounded to [-3, 3]. Freeze task/source hashes,
model IDs/revisions, LR settings, temperatures, label/sample hashes and device.
All chosen arms need complete dev captures before a freeze is written.
Report calibrated dev scores as in-sample diagnostics, not held-out quality.
Test requires this verified freeze and a new output directory. Run test once;
do not tune the rubric, classes, threshold, model or calibration on it.
Device must match the freeze; actual versions, precision, GPU, interpreter and
model setup/warm-up timings are recorded. A source/configuration change means
a new experiment and new dev freeze, not relabeling the old test as untouched.

Report raw **and** dev-calibrated test macro-F1 (all five classes, zero for
undefined class F1), per-class F1/support, accuracy, top-label ECE (10 fixed
equal-width bins with bin counts), multiclass Brier and log loss. Missing gold
classes are explicit. ECE is an estimate sensitive to sample size and bins;
lower ECE alone does not imply a more useful classifier.

For every arm, define confidence as **maximum class probability**, not the
provider's separate `confidence` field. Use calibrated max probability for
the CSV human queue once a dev temperature exists. The requested **0.7** is a
predeclared illustration, not a demonstrated safe business threshold. Report
retained count, coverage, human-referral count and selective error rate. The
error rate is null if no predictions are retained. Errors and non-executed
pairs are also routed to human review.

## Latency, cost and failures

Batch 1, serial calls with fixed seeded arm-order interleaving. Time includes
text processing/tokenization, inference and response validation. API latency
also includes network and provider queue time. Local load and synthetic
warm-up are recorded separately and excluded from steady-state latency.
Report p95 over attempted requests, including failed attempts, and their
denominator. GPU and API request timings measure different deployment paths;
do not present their difference as intrinsic architecture speed.

Keep every planned review/arm record. No automatic retries. Failures and
budget-stopped pairs stay visible. With any missing/failed prediction, that
arm's quality scores are null rather than biased successful-subset estimates;
availability and attempt latency/cost can still be reported. No automatic
winner or promotion. For cost, use reported `usage.cost`, which the new
OpenRouter Decisions contract defines in USD. Report USD per 1,000 attempted
requests only if every attempt has known cost. Missing costs and local
CPU/GPU billing are unknown, never zero; observed partial spend is a lower
bound. Training cost is separate and not priced here.

External inference needs `--allow-external-api`, server-side
`OPENROUTER_API_KEY`, and `--max-cost-usd`. The owner checks provider data
terms and configures a capped API key before submitting reviews. Send only
review text, not hotel metadata, labels, split, notes or secrets. This run's
software limit stops before the next call after observed spend reaches the
cap or cost is unknown; **one request can overshoot**, so it is not an account
spending guarantee. No provider call is made by sample/labels preparation.
No redirect carries a key; provider error bodies/exception messages are not
logged. Captures contain IDs, probabilities, timings and usage metadata.
Sample and rating sheets stay under ignored `runs/`; no HF writes or review
database is part of the experiment.

## Primary references checked on 11 October 2026

- [Microsoft announcement](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)
- [OpenRouter Decisions OpenAPI](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
- [OpenRouter's request/response reference](https://github.com/OpenRouterTeam/skills/blob/33597a44e38b0cdc38ea1256ef8bcf50c2704e3d/skills/openrouter-decisions/references/decisions-api.md)
- [OpenRouter public Decisions catalog](https://openrouter.ai/api/v1/models?output_modalities=decisions)
- [Cloudflare Clef-omni announcement](https://blog.cloudflare.com/clef-faster-cheaper-multimodal/)

The Microsoft Foundry API page linked from its announcement was inaccessible
from this session. The adapter follows the independently accessible official
OpenRouter OpenAPI, not a guessed Foundry schema or the old Jev endpoint.
