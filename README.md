# 🏨 Hotel Review Sentiment API

[![CI](https://github.com/krimits/hotel-review-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/krimits/hotel-review-nlp/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Reviewing this for a role? [Start here](docs/START_HERE.md)** — a ten-minute path through the evidence, including what is still open.

> Hotel review classification and an emerging hotel-operations product, with an API, aspect extraction and hotel-scoped recommendations. On test sets that share no review text with training, fine-tuned DistilBERT reaches **0.9642 macro-F1** on a random split and **0.9491** on reviews written after the training period. That is about 3 points above TF-IDF + Naive Bayes chosen on dev, a gain that meets a rule fixed before training ([decision note](docs/experiments/decision_distilbert_vs_nb.md)). Earlier figures, from a split whose test set shares 170 texts with training, are kept as [history](#historical-results-legacy-split-text-overlap). What this project demonstrates, with links to the evidence: [evidence map](docs/EVIDENCE_MAP.md).

🚀 **[hotel-review-demo](https://huggingface.co/spaces/krimits/hotel-review-demo)** · 🇬🇷 **[greek-review-sentiment-demo](https://huggingface.co/spaces/krimits/greek-review-sentiment-demo)** · 🛠️ **[hotel-triage-demo](https://huggingface.co/spaces/krimits/hotel-triage-demo)** (experimental) · 📦 **[Model on HF Hub](https://huggingface.co/krimits/distilbert-hotel-reviews)**

---

## At a glance

| Area | What is here | Evidence |
| :--- | :--- | :--- |
| **Data hygiene** | The raw file is refused unless its size and SHA-256 match; duplicates are removed before splitting; three splits (random, out-of-time, unseen hotels) whose manifests record zero text overlap | [`data/raw/README.md`](data/raw/README.md) · [`v2_splits/`](docs/experiments/results/v2_splits/) |
| **Evaluation fixed in advance** | The comparisons and the rule for a meaningful gain were committed before training; paired bootstrap intervals (2,000 resamples) and exact McNemar tests; CI recomputes the comparison from saved logits on every push | [plan](docs/experiments/phase2_analysis_plan.md) · [`distilbert_v2/`](docs/experiments/results/distilbert_v2/) |
| **Model internals** | LoRA written from the paper's equations in PyTorch and checked numerically against Hugging Face `peft` | [`lora.py`](src/reviewnlp/lora/lora.py) · [`test_lora.py`](tests/test_lora.py) |
| **Serving** | FastAPI with pluggable backends (`stub`, `classical`, `encoder`, `qwen_qlora`); a Docker image with weights mounted at runtime; an archived Locust load test; an optional SQLite hotel dashboard for a local single-worker pilot | [product setup](docs/PRODUCT_SETUP.md) · [serving benchmark](#serving-benchmark) |
| **Analytics** | SQL analysis of complaint topics rising within the same hotels, with a hotel-level bootstrap; the topic lexicon measured by two annotators (kappa 0.83–1.00), which leaves the rises unconfirmed | [case study](docs/case_study/complaint_trends.md) · [annotation](docs/case_study/results/annotation/README.md) |

---

## The problem

Hotel chains receive **thousands of guest reviews every day** across Booking.com, TripAdvisor, Google, and internal channels. A single negative review left unaddressed can cost bookings; a missed pattern across hundreds of reviews can hide a systemic issue (broken AC, rude staff, misleading photos). Manual triage doesn't scale, keyword rules miss sarcasm, mixed sentiment and context, and traditional dashboards are backward-looking.

## The solution

A **binary sentiment classifier** trained on about **118,000 real Booking.com reviews**, behind an API:

- **Accuracy.** Fine-tuned DistilBERT reaches **0.9642 macro-F1** on a test set that shares no review text with training, and 0.9491 on reviews written after the training period. A TF-IDF + Naive Bayes baseline, 3 points lower, answers in about 1 ms on a CPU.
- **Service.** A **FastAPI service**: `/predict` for scoring and `/absa` for per-aspect extraction, each with a batch path, plus health checks. The archived load test measured `/predict` p50 **360 ms under load** on its Windows CPU.
- **Deployment.** A Docker image and an optional [hotel operations dashboard](docs/PRODUCT_SETUP.md).
- **CPU cost.** Dynamic INT8 makes it 3.0 times faster on one CPU thread (17 ms per review) with macro-F1 unchanged within 0.001, measured on the full test set ([Quantization](#quantization)).
- **Experimental, not selected or promoted:** a triage endpoint and a topic-model comparison (below).

For teams, this is the difference between reading reviews and *acting* on them.

<details>
<summary><b>Experimental work: triage endpoint, topic-model comparison</b></summary>

- **`POST /triage`, unvalidated.** It chains sentiment, local lexical complaint-topic hints (default primary), optional Jev enhancement and suggested actions (Qwen, off by default in the API): [what it does, what it sends out and what has not been measured](docs/TRIAGE.md). Unknown topics require human review; the [whole-review pilot](docs/experiments/triage_lexicon_pilot/PROTOCOL.md) comes before new paid comparisons.
- **[Primary-topic decision-model comparison](docs/experiments/decision_topic_benchmark/README.md):** 500 validated final annotations, split 300/100/100.
  - The [actual three-arm dev run](docs/experiments/decision_topic_benchmark/results/comparison_dev_20261012/README.md) completed 100 reviews per arm: macro-F1 logistic 0.1561, Qwen 0.2299, Microsoft Decision-1 0.6524. These are development results with limited rare-class support, in-sample calibration and a documented Qwen T4 precision caveat.
  - The [Jev extension](docs/experiments/decision_topic_benchmark/results/jev_dev_20261012/README.md) completed the same 100 dev cases: macro-F1 0.6783, with a paired Jev-minus-Microsoft interval crossing zero. Its selective risk is recorded alongside coverage and in-sample calibration caveats.
  - The paired held-out test is pending; no model was selected or promoted.

</details>

---

## Key results

**Clean test set.** The uploaded processed parquets were audited and cleaned:
duplicates and texts with contradictory labels removed, and no review text
shared between train, dev and test. On the resulting **13,270-review** test set,
a word TF-IDF + Naive Bayes model selected on dev reached **0.9347 macro-F1** and
**95.10% accuracy**. Its fingerprints, rejected duplicates, per-class metrics and
explicit missing-model list are in [the benchmark](runs/benchmark/README.md), the
[four candidate metrics](docs/experiments/results/classical_clean_uploaded_metrics.json) and
[`results.json`](runs/benchmark/results.json). Only the classical baseline was run on
this cleaned split; DistilBERT was re-run on the rebuilt splits below.

**Rebuilt from the raw file.** The Booking CSV is not in the repository, but a
byte-identical copy of the Kaggle file is on the Hugging Face Hub. `make data`
downloads it at a pinned revision, refuses it unless the size and SHA-256 match,
and builds three splits of the same reviews: random, **out-of-time** (train on
older reviews, test on newer) and **unseen hotels**. See
[`data/raw/README.md`](data/raw/README.md) and
[`notebooks/08_phase1_data_and_trends_colab.ipynb`](notebooks/08_phase1_data_and_trends_colab.ipynb).
The same pipeline, chosen on dev, gives these results on each split:

| Test set | What it asks | Test reviews | Macro-F1 (95% bootstrap CI) | Accuracy |
| :--- | :--- | ---: | ---: | ---: |
| Random | reviews like the training ones | 13,263 | 0.9351 (0.9303–0.9400) | 95.14% |
| **Out-of-time** | reviews written after every training review (26 May–3 Aug 2017) | 13,921 | **0.9184** (0.9132–0.9233) | 93.32% |
| Unseen hotels | 147 hotels with no review in training | 13,289 | 0.9348 (0.9298–0.9396) | 95.12% |

The selected pipeline is word TF-IDF + Naive Bayes on all three splits.
- **Unseen hotels.** The observed performance is similar to the random split in
  this comparison (0.9348 against 0.9351). This is not a test of equivalence.
- **Later reviews.** Macro-F1 is 1.7 points lower, and the two intervals do not
  overlap. The test sets hold different reviews, so this is not a paired
  comparison.
- **Recall falls for both classes** (negative 0.913 → 0.900, positive
  0.964 → 0.946). So the drop is not only a change in the share of negative
  reviews. The later reviews differ from the training ones in ways that cost
  recall in both classes: wording, topics or season. This split cannot say
  which.
- **What it means for deployment.** A deployed model needs monitoring on recent
  reviews and periodic retraining.

The results, manifests and test predictions are in
[`docs/experiments/results/split_views.json`](docs/experiments/results/split_views.json) and
[`v2_splits/`](docs/experiments/results/v2_splits/).

**DistilBERT on the same test sets.**
- **What was trained.** DistilBERT was fine-tuned on the random and out-of-time splits, in
  full and with the scratch LoRA, with the settings of the legacy run.
- **Fixed in advance.** The comparisons and the rule for a meaningful gain were committed
  before training ([plan](docs/experiments/phase2_analysis_plan.md)).

| Model | Random: macro-F1 (95% CI) | Out-of-time: macro-F1 (95% CI) | Errors, out-of-time |
| :--- | ---: | ---: | ---: |
| TF-IDF + Naive Bayes, chosen on dev | 0.9351 (0.9303–0.9398) | 0.9184 (0.9133–0.9230) | 930 |
| **DistilBERT, full fine-tune** | **0.9642** (0.9605–0.9677) | **0.9491** (0.9450–0.9531) | **578** |
| DistilBERT, scratch LoRA | 0.9576 (0.9537–0.9613) | 0.9424 (0.9380–0.9465) | 657 |

- **The primary comparison meets the rule.**
  - On the out-of-time test, DistilBERT beats Naive Bayes by 3.1 points, with a paired 95%
    interval of 2.6–3.5 points and 38% fewer errors.
  - The rule asks for an interval that excludes zero and a gain of at least one point.
- **Not more robust over time.** All three models lose 1.5–1.7 points on later reviews.
  DistilBERT keeps its lead.
- **Negative reviews.** DistilBERT misses 233 of the 3,921 negative reviews instead of 393,
  and flags 345 positive reviews as negative instead of 537.
- **LoRA.**
  - It is 0.7 points below the full fine-tune on both tests: clear but small.
  - **The gap holds across three training seeds** on the out-of-time test (0.68, 0.67 and
    0.45 points; every paired interval excludes zero), so it is not an accident of seed 42
    ([decision note](docs/experiments/decision_distilbert_seeds.md)).
  - It trains in a third less time with 39% less GPU memory.
- **Cost**, timed with the models of the out-of-time split.
  - On one CPU thread, one review at a time, from text to label: DistilBERT takes 57 ms per
    review (median; 124 ms at the 95th percentile) against 1.2 ms.
  - It weighs 255 MiB against 3.4 MiB.
  - On a T4 GPU, in batches of 64, it labels 254 reviews per second. That is a throughput, not
    the time to answer one review.
- **Decision.**
  - For scoring reviews in batches: DistilBERT.
  - For one review at a time on a CPU: it depends on the time allowed
    ([decision note](docs/experiments/decision_distilbert_vs_nb.md)).

- **The intervals** in this table use 2,000 resamples, as the plan fixed. The table of the
  three splits used 1,000, hence the small differences for Naive Bayes.
- **Evidence.** The runs, logits and checks are in
  [`results/distilbert_v2/`](docs/experiments/results/distilbert_v2/). The CI recomputes the
  comparison from the logits on every push.
- **Published.** The random-split full fine-tune is now the Hub model
  [`krimits/distilbert-hotel-reviews`](https://huggingface.co/krimits/distilbert-hotel-reviews/tree/7306aebcaaebc00d579f5d0a91001ae376f18158)
  and the demo's DistilBERT, both at commit `7306aeb`.
  - Before any write, its weights reproduced the saved test logits on 1,024 reviews.
  - The previous model is kept under the tag `legacy-split-v1`.
  - The steps, checks and records are in the
    [publication record](docs/experiments/results/distilbert_v2/README.md#published).

**The BiLSTM on the same test sets.** Re-run on all three clean splits under a
[protocol](docs/experiments/bilstm_clean_splits/PROTOCOL.md) locked before training, with the
committed settings and no tuning.

| BiLSTM, seed 42 | Random | Out-of-time | Unseen hotels |
| :--- | ---: | ---: | ---: |
| Macro-F1 (95% CI) | 0.9486 (0.9445–0.9528) | 0.9357 (0.9310–0.9400) | 0.9514 (0.9471–0.9556) |
| Against Naive Bayes (paired) | +1.35 points | +1.73 points | +1.66 points |

- **The protocol's question: DistilBERT is meaningfully better on clean out-of-time data.**
  - The BiLSTM is 1.35 points below the full fine-tune, with a paired 95% interval of
    0.99–1.69 points; it makes 735 errors against 578.
  - Seeds 43 and 44 land in the same branch of the rule, so the conclusion holds across three
    training seeds of the BiLSTM (macro-F1 0.9357, 0.9363, 0.9372).
- **It beats Naive Bayes on every split**, each gain meaningful by the plan's rule.
- **Speed.** On one CPU thread, one review at a time, the BiLSTM takes 1.7 ms per review (median)
  and weighs 8.4 MB, against 1.2 ms for Naive Bayes and 56 ms for DistilBERT.
  - By the protocol's rule, no model dominates another on that CPU, so all four are on the
    measured frontier.
  - The scratch LoRA stays on it only because it timed 4% faster than the full fine-tune, which
    the protocol itself calls a tie; merged, it has the same architecture and size
    ([decision note](docs/experiments/decision_bilstm_clean_splits.md)).
- **Evidence.** [`results/bilstm_v2/`](docs/experiments/results/bilstm_v2/); CI recomputes both
  rules from the logits on every push.

**Which complaints are rising.** A SQL analysis of the same reviews asks which
complaint topics became more frequent within the same hotels.
- **How.** It compares February–July 2016 with February–July 2017.
  - Both periods come after a step in the data in February 2016, possibly a
    change in how reviews were collected or recorded.
  - A hotel-level bootstrap and a stricter interval for the 29 topics tested
    decide what counts as a rise.
- **Worth a look, not confirmed:**
  - bathroom & shower (+14% within the same hotels);
  - air conditioning (+17%);
  - pests (rare, +40%).
- **Not yet findings: cleanliness and responsiveness.** The lexicon that names
  the topics is right in only about half of their sampled quotes.
- **Not rising faster than other topics: bed.** Most of its rise in the first run
  came from the February 2016 step.
- **The lexicon, measured by two people.** A pilot tested the lexicon, not a
  sentiment model, under a protocol locked before the sample was drawn:
  300 texts, 80 labelled twice, kappa 0.83–1.00.
  - Precision is about 0.9 for air conditioning, 0.75 for bathroom, about half
    for cleanliness, and a third to a half for responsiveness.
  - The lexicon finds most bathroom, air conditioning and cleanliness complaints,
    but only 2 of 16 responsiveness ones.
  - No change in precision between the periods can be told apart from zero, nor
    from the fall that would erase each rise. So the rises above stay unconfirmed.
  - [Results and caveats](docs/case_study/results/annotation/README.md).

The queries, checks, charts and caveats are in the
[case study](docs/case_study/complaint_trends.md).

### Historical results (legacy split, text overlap)

These were measured on the first frozen test set of 13,278 reviews (10,000
positive, 3,278 negative), which shares **170 normalized review texts with its
training set** (audit finding F02). They are kept as recorded and are not
comparable with the clean result above. Every row is the number recorded in the
preserved run artifacts under [`docs/experiments/results/`](docs/experiments/results/);
the DistilBERT pair is also recomputed from saved logits by
[`scripts/verify_distilbert_handoff.py`](scripts/verify_distilbert_handoff.py) on every CI run.

| Model | Macro-F1 | Accuracy | Trainable params | Notes |
| :--- | ---: | ---: | ---: | :--- |
| TF-IDF + Naive Bayes | 0.9345 | 95.09% | — | Classical baseline, 0.04 ms/text |
| BiLSTM (pure PyTorch) | 0.9503 | 96.29% | All | Custom training loop, seed 42 |
| **DistilBERT (full FT)** | **0.9634** | **97.3%** | 67.0M (100%) | 🏆 Best accuracy |
| DistilBERT + scratch LoRA | 0.9573 | 96.8% | 0.74M (**1.1%**) | 35% faster training, 38% less GPU memory |
| Qwen2.5-0.5B + QLoRA | 0.9571 | 96.7% | 8.8M (adapter) | 4-bit, 20k training rows; first-step label scores |

**How the Qwen row was scored.** Its [archived metrics](docs/experiments/results/qwen_legacy_v1/README.md)
hold two scores for the same 13,278 test reviews.
- **From its generated answers, with a strict parser: macro-F1 0.0.** The parser accepts only a
  single label word. Qwen kept writing after the label (for example `positive\npositive`), so every
  answer was rejected.
- **The 0.9571 above** reads the prediction at the first generated position: the higher-scoring of
  the two label tokens.
- **Not yet scored:** the lenient parser that the demo uses, which takes the first label word.

<details>
<summary><b>Pairwise McNemar tests (exact, on classification errors)</b></summary>

| Pair | Discordant | p-value | Significant at 0.05 |
| :--- | ---: | ---: | :---: |
| Classical vs BiLSTM | 542 | 9.4 × 10⁻⁹ | ✅ |
| Classical vs DistilBERT full | 506 | 2.3 × 10⁻³⁸ | ✅ |
| Classical vs DistilBERT LoRA | 510 | 2.5 × 10⁻²³ | ✅ |
| Classical vs Qwen QLoRA | 544 | 1.1 × 10⁻¹⁹ | ✅ |
| BiLSTM vs DistilBERT full | 366 | 2.8 × 10⁻¹⁵ | ✅ |
| BiLSTM vs DistilBERT LoRA | 374 | 6.2 × 10⁻⁶ | ✅ |
| BiLSTM vs Qwen QLoRA | 380 | 1.1 × 10⁻⁴ | ✅ |
| DistilBERT full vs LoRA | 182 | 5.0 × 10⁻⁶ | ✅ |
| DistilBERT full vs Qwen | 320 | 4.2 × 10⁻⁵ | ✅ |
| **DistilBERT LoRA vs Qwen QLoRA** | **300** | **0.53** | ❌ |

**On this legacy test set, 9 of 10 pairwise differences are statistically significant.** LoRA and QLoRA are indistinguishable there: 739K adapter parameters match a 0.5B LLM fine-tuned with 4-bit quantization.

</details>

On this legacy test set, **full fine-tuning beats scratch LoRA by 0.61 pp macro-F1** (McNemar p = 5.0 × 10⁻⁶), and **Qwen QLoRA is statistically indistinguishable from scratch LoRA** (p = 0.53; see how its row
was scored, above).

**Since then:**
- **DistilBERT and the scratch LoRA were re-run on the clean splits** (above).
  - They score about the same there (0.9642 and 0.9576).
  - Full fine-tuning again beats LoRA, by 0.66–0.68 points.
- **The BiLSTM was re-run on the clean splits** (above).
  - DistilBERT's full fine-tune beats it meaningfully on later reviews, as on the legacy split,
    and across three BiLSTM seeds.
  - It beats Naive Bayes on all three splits.
- **Qwen QLoRA is not carried forward for binary scoring**
  ([decision, 14 October 2026](docs/experiments/decision_qwen_binary_scope.md)). Its legacy
  comparisons stay unconfirmed on clean splits.

### Archived API load test (CPU, FastAPI)

| Metric | Value |
| :--- | ---: |
| Mixed traffic: `/predict`, `/predict/batch`, `/health` (20 users, 60 s) | **20.5 req/s total** |
| `/predict` p50 / p95 / p99 | 360 ms / 1.2 s / 2.2 s |
| Failures | **0 / 873** |
| Dynamic INT8, one CPU thread | **3.0×** faster per review, macro-F1 within 0.001 ([full test](#quantization)) |

---

## Try it

### Option 1 — Interactive demo

👉 Try the deployed models interactively:

- **[hotel-review-demo](https://huggingface.co/spaces/krimits/hotel-review-demo)** —
  DistilBERT vs Qwen2.5-0.5B QLoRA vs GreekBERT, side by side on one review text.
- **[greek-review-sentiment-demo](https://huggingface.co/spaces/krimits/greek-review-sentiment-demo)** —
  dedicated Greek-language demo (GreekBERT, single tab).
  
Paste any hotel review and get a live prediction. No setup required.
- **DistilBERT** is the clean-split model: 0.9642 macro-F1 on a test set that shares no review
  text with training. The demo loads it at Hub commit
  [`7306aeb`](https://huggingface.co/krimits/distilbert-hotel-reviews/tree/7306aebcaaebc00d579f5d0a91001ae376f18158)
  ([publication record](docs/experiments/results/distilbert_v2/README.md#published)).
- **Qwen QLoRA** was trained on the legacy split (see
  [Historical results](#historical-results-legacy-split-text-overlap)), and **GreekBERT** on Greek tweets.
  Their numbers come from other test sets and are not comparable with DistilBERT's.
- **Speed.** Each answer shows its server processing time on the Space's CPU: tokenization,
  inference and output processing, without queueing, network or rendering in the browser. No
  benchmark of the demo has been run.
  - The project's benchmark ([`latency.json`](docs/experiments/results/distilbert_v2/latency.json))
    timed the out-of-time DistilBERT checkpoint, which has the same architecture and size as the
    demo's. It is not a measurement of the published random-split checkpoint in the Space.
  - On one CPU thread, one review at a time, tokenization included: 57.2 ms median, 124.4 ms at the
    95th percentile.
  - On a T4, in batches of 64: 254.5 reviews per second. That is a throughput, not the time to
    answer one request.
  - Qwen generates the label token by token. Its speed relative to the other models in the Space
    has not been measured.

![The hotel-review-demo Space comparing DistilBERT and Qwen QLoRA on one review](docs/images/space_demo.png)

#### Experimental hotel actions demo

**[hotel-triage-demo](https://huggingface.co/spaces/krimits/hotel-triage-demo)** takes one English
review and shows its sentiment, the issues it reports with quoted evidence, and suggested measures.
- **How.** DistilBERT gives the sentiment and a small local lexicon gives topic hints. Qwen2.5-1.5B,
  on ZeroGPU, extracts the issues and proposes measures; its quotes are cut from the review itself.
  Jev, an optional external check, is disabled on the Space, and the app stores no reviews.
- **Status: experimental, not selected or promoted.** An adjudicated review of the 6 October version
  covered 24 authored development reviews; the reviewers' independence is unverified.
  - It gave an appropriate measure for 4 of 17 actual problems, and 4 of its 5 measures were useful.
  - Issues it marks uncertain get no measure by design
    ([review](docs/experiments/triage_space_publication/e1e37e5/human_review_v3/README.md)).
- **In the demo.** The two examples are stored after their first run and replay without GPU. A new
  review needs ZeroGPU quota, which anonymous visitors exhaust quickly (`gpu_quota_exceeded`).
- **More:** [how it works](docs/TRIAGE.md) ·
  [publications and their checks](docs/TRIAGE_SPACE_DEPLOYMENT.md), latest at Space commit `3f2aff8` ·
  [reliability plan](docs/TRIAGE_RELIABILITY.md) ·
  [whole-review pilot](docs/experiments/triage_lexicon_pilot/PROTOCOL.md).

### Option 2 — Run locally

```bash
git clone https://github.com/krimits/hotel-review-nlp.git
cd hotel-review-nlp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serving]"

# Download the fine-tuned encoder from HF Hub, at the published commit
# (the previous, legacy-split model: --revision legacy-split-v1)
hf download krimits/distilbert-hotel-reviews --revision 7306aebcaaebc00d579f5d0a91001ae376f18158 \
  --local-dir models/distilbert

# Serve it
MODEL_TYPE=encoder MODEL_PATH=models/distilbert make serve
```

Then hit the API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Perfect stay, spotless room, incredibly friendly staff."}'
```

```json
{ "label": "positive", "confidence": 0.9985, "latency_ms": 24.9 }
```

This response is an illustrative example, not a recorded request. `latency_ms` is the server's
processing time for that one request, and it depends on the hardware.

`bash scripts/demo_api.sh` exercises `/health`, both `/predict` paths and `/predict/batch` against a
real encoder checkpoint:

![Terminal output of scripts/demo_api.sh against the DistilBERT checkpoint](docs/images/api_demo.png)

### Option 3 — Docker

```bash
docker build -f docker/Dockerfile -t hotel-review-nlp .

# Smoke-test the API contract with no weights at all
docker run -p 8000:8000 -e MODEL_TYPE=stub hotel-review-nlp
```

Weights are **never baked into the image** — mount them and point `MODEL_PATH` at the
mount, or the container exits at startup:

```bash
hf download krimits/distilbert-hotel-reviews --revision 7306aebcaaebc00d579f5d0a91001ae376f18158 \
  --local-dir models/distilbert

docker run -p 8000:8000 \
  -v "$(pwd)/models:/models:ro" \
  -e MODEL_TYPE=encoder -e MODEL_PATH=/models/distilbert \
  hotel-review-nlp
```

`MODEL_TYPE` accepts `stub`, `classical`, `encoder` or `qwen_qlora`; everything except
`stub` requires `MODEL_PATH`.

---

## API endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness plus the loaded model's info |
| `POST` | `/predict` | One review → label, confidence, latency |
| `POST` | `/predict/batch` | Up to 256 reviews |
| `POST` | `/absa` | One review → aspects, sentiments, supporting quotes |
| `POST` | `/absa/batch` | Up to 256 reviews, one padded forward pass per batch |
| `GET` | `/hotels/{hotel_id}/recommendations` | Stored aspects for a hotel, worst first |
| `GET` | `/hotels/{hotel_id}/aspects/{aspect}/evidence` | Recent negative quote spans and review source IDs, hotel scoped |
| `DELETE` | `/hotels/{hotel_id}/reviews/{review_id}` | Delete a stored review and its extracted quotes |
| `POST` | `/greek/predict`, `/greek/predict/batch` | Greek sentiment (requires `GREEK_MODEL_PATH`; trained on tweets) |
| `GET` | `/dashboard` | Hotel operations view over stored review aspects |

Interactive docs are at `/docs` once the service is running.

To enable hotel-scoped storage and API-key authorization, set `REVIEWNLP_DB_PATH`
and `REVIEWNLP_API_KEYS_JSON` before starting the server. The dashboard does
not keep your key in browser storage. See [the product setup guide](docs/PRODUCT_SETUP.md)
for a complete local walkthrough. The API without a configured store is a
demo: it will not claim to have calculated real hotel recommendations.

The `/absa` paths load `Qwen2.5-0.5B-Instruct` on first use and keep it for the
process, so the first request pays for the load and the rest do not. Configure
with `ABSA_ADAPTER_DIR` (omit for the base model, which is what the prototype
uses), `ABSA_DEVICE` and `ABSA_BATCH_SIZE`.

<details>
<summary><b>POST /absa</b> — aspects from one review</summary>

```bash
curl -X POST http://localhost:8000/absa \
  -H "Content-Type: application/json" \
  -d '{"hotel_id": "acme-athens", "review_id": "r-1042",
       "text": "The room was fine but it was very loud all night."}'
```

```json
{
  "hotel_id": "acme-athens",
  "review_id": "r-1042",
  "stored": false,
  "aspects": [
    {"aspect": "noise", "sentiment": "negative", "quote": "very loud", "confidence": null}
  ],
  "overall_sentiment": "negative",
  "json_valid": true,
  "salvaged": false,
  "entries_dropped": 0,
  "quote_absent": 0,
  "quote_not_in_review": 0,
  "generation_hit_token_budget": false,
  "processing_time_ms": 39.82
}
```

`overall_sentiment` is `null` for a genuinely mixed review rather than a coin
flip — see the abstention note in the ABSA section. `json_valid`, `salvaged`
and `entries_dropped` describe how well the model followed the JSON
instruction, so a caller can tell a clean extraction from a salvaged one.

</details>

<details>
<summary><b>POST /absa/batch</b> — many reviews in one forward pass</summary>

```json
{
  "hotel_id": "acme-athens",
  "results": [
    {
      "hotel_id": "acme-athens",
      "review_id": "r-1",
      "stored": false,
      "aspects": [
        {"aspect": "staff", "sentiment": "positive", "quote": "kind staff", "confidence": null}
      ],
      "overall_sentiment": "positive",
      "json_valid": true,
      "salvaged": false,
      "entries_dropped": 0,
      "quote_absent": 0,
      "quote_not_in_review": 0,
      "generation_hit_token_budget": false,
      "processing_time_ms": null
    }
  ],
  "total_processing_time_ms": 0.69
}
```

Per-review `processing_time_ms` is **`null` in a batch**, on purpose. The
reviews are generated together in one padded pass, so no per-review share of
that time was measured; `total_processing_time_ms` is the figure that was.
A single `/absa` call is generated on its own, so there it is a real number.

</details>

<details>
<summary><b>GET /hotels/{hotel_id}/recommendations</b> — aspects worth acting on</summary>

Every figure is computed from stored aspect rows for that hotel and window.
With `REVIEWNLP_DB_PATH` configured, it returns actual complaints sorted by
priority and isolates hotels by API key. Without a configured store it
answers **501** rather than serving examples:

```json
{
  "detail": "No aspect store is configured. Set REVIEWNLP_DB_PATH to enable hotel-scoped persistence and recommendations; no example figures are returned."
}
```

The priority score is a heuristic, not an estimate of business impact.
Positive-only topics are excluded from the complaint list. An empty store
returns `200` with an empty list.

</details>

> The millisecond figures above come from the offline test harness, which drives
> the endpoints with a fake model. They show the response shape, not latency.
> `/predict` has benchmarked numbers in [Serving benchmark](#serving-benchmark);
> ABSA does not.

---

## How it works

```mermaid
flowchart LR
    A[Booking.com CSV<br/>515,738 rows] --> V[Pinned download<br/>size + SHA-256 check]
    V --> B[Labelling<br/>duplicates removed before splitting]
    B --> C[Three v2 splits with manifests<br/>random · out-of-time · unseen hotels]
    C --> D[Classical baseline<br/>chosen on dev, bootstrap CI]
    C --> F[DistilBERT full FT · scratch LoRA<br/>random + out-of-time, paired bootstrap]
    C --> G[BiLSTM, three splits<br/>three seeds on out-of-time]
    V --> S[SQLite + SQL<br/>complaint trends]
    L[Legacy frozen split<br/>test shares texts with train] -.-> E[BiLSTM · DistilBERT · LoRA · Qwen<br/>historical results]
    D & F & G & E --> H[Saved artifacts]
    H --> I[FastAPI serving<br/>stub / classical / encoder / Qwen]
```

**Pipeline:**

1. **Raw data.** `make data` downloads the Booking CSV from a pinned revision.
   It refuses the file unless its size and SHA-256 match the Kaggle download
   ([`data/raw/README.md`](data/raw/README.md)).
2. **Labelling.** Each raw review has separate `Positive_Review` /
   `Negative_Review` fields. The pipeline labels positive-only and negative-only
   reviews, excludes mixed reviews, and does not impose a `Reviewer_Score`
   cutoff.
3. **Splitting.**
   - **Duplicates first.** Duplicate texts are removed before splitting, so one
     text cannot land in two splits.
   - **Three splits.** Built from the same deduplicated reviews, they answer
     different questions: random (117,880 / 14,734 / 13,263 train / dev / test),
     out-of-time and unseen hotels.
   - **Manifests.** Every split directory has a manifest with the raw file's
     hash, the split fingerprints, date ranges, hotels and a zero-overlap check.
4. **Training and selection.**
   - **Classical baseline.** Chosen on dev, on all three splits.
   - **DistilBERT.** Fine-tuned in full and with the scratch LoRA on the random
     and out-of-time splits, with two extra training seeds on out-of-time. It
     keeps the epoch with the best dev macro-F1.
   - **BiLSTM.** Trained on all three splits, with two extra seeds on
     out-of-time; early stopping on dev macro-F1.
   - **Qwen2.5-0.5B QLoRA.** Trained only on the legacy frozen split (118,990 /
     14,872 / 13,278), whose test set shares 170 texts with training, so its
     results are kept as history.
5. **Evaluation.**
   - Macro-F1 with bootstrap intervals, per-class metrics and confusion
     matrices.
   - For pairs of models: exact paired McNemar tests, and a paired bootstrap
     interval of the difference in macro-F1.
6. **Analysis.** A SQLite database and SQL queries ask which complaint topics
   are rising within the same hotels
   ([case study](docs/case_study/complaint_trends.md)).
7. **Serving.** FastAPI backend with pluggable model types (`stub`, `classical`,
   `encoder`, `qwen`) for CI-safe testing and flexible deployment.

---

## Serving benchmark

Measured with **Locust** on a Windows CPU (no GPU), 20 concurrent users, 60-second run:

| Endpoint | Requests | p50 | p95 | RPS |
| :--- | ---: | ---: | ---: | ---: |
| `POST /predict` | 638 | 360 ms | 1.2 s | 15.0 |
| `POST /predict/batch` | 152 | 500 ms | 1.3 s | 3.6 |
| `GET /health` | 83 | 2 ms | 5 ms | 2.0 |
| **Aggregated** | **873** | **320 ms** | **1.2 s** | **20.5** |

Zero failures across the run.

---

## Quantization

The published DistilBERT (`7306aeb`), quantized with dynamic INT8 on its linear layers, measured
on all 13,263 random-split test reviews under a
[protocol locked before the run](docs/experiments/int8_and_batching/PROTOCOL.md):

| One CPU thread | FP32 | INT8 |
| :--- | ---: | ---: |
| Macro-F1 | 0.9642 | 0.9637 |
| Median time per review, one at a time | 51.5 ms | **17.2 ms** (3.0× faster) |
| Serialized size | 255.5 MB | 132.3 MB |

- **The protocol's rule: INT8 keeps macro-F1 within one point and is meaningfully faster.**
  The paired difference is −0.0005 (95% interval −0.0024 to 0.0013), and the two models agree
  on 99.4% of the reviews.
- **Batching on a CPU.** Batches of 32 in test order are slower per review than single reviews
  (93 ms against 51.5 for FP32), because every review is padded to the longest in its batch.
  Sorting the reviews by length first halves FP32's time per review (25 ms). For INT8 it saves
  nothing (21 ms against 17).
- **Limits.** One Colab CPU and one thread; only the linear layers are quantized, so the
  embeddings keep the size from quartering; one checkpoint. The Space and the API still serve
  FP32 ([decision note](docs/experiments/decision_int8_and_batching.md)).

The 32-review experiment this replaces (1.38× faster, with a size estimator that counted
parameters) is superseded.

---

## Why the scratch LoRA matters

Most portfolios use `peft` and call it a day. This one **implements LoRA from the paper's equations** and verifies it numerically against PEFT.

[`src/reviewnlp/lora/lora.py`](src/reviewnlp/lora/lora.py) follows [Hu et al. (2021)](https://arxiv.org/abs/2106.09685):

```
h = W₀x + (α/r) · B(A(x))
```

It initializes `A` with Kaiming uniform and `B` with zeros, applies scaling and dropout on the adapter path, and supports merge/unmerge. [`tests/test_lora.py`](tests/test_lora.py) checks initial behavior, gradient flow, frozen base weights, merge/unmerge, and numerical agreement with PEFT on a locally constructed tiny BERT after copying weights.

**Result on the legacy split**: LoRA trains **1.10% of parameters** with **35% less time** and **38.3% less peak GPU memory** for **0.61 pp lower macro-F1** than full fine-tuning — a trade-off worth documenting rather than hiding.

**On the clean splits the trade-off holds.**
- **Savings.** LoRA trains 1.10% of the parameters, in a third less time, with 39% less
  peak GPU memory.
- **Price.** Its macro-F1 is 0.66 points lower on the random test and 0.68 on the
  out-of-time test.
- **Verdict.** The paired intervals exclude zero. By the plan's rule, the full fine-tune is
  better by a clear but small margin
  ([decision note](docs/experiments/decision_distilbert_vs_nb.md)).
- **Across seeds.** On the out-of-time test the lead holds in each of three training seeds,
  between 0.45 and 0.68 points ([decision note](docs/experiments/decision_distilbert_seeds.md)).

---

## Aspect-based sentiment (prototype)

A binary label answers "was this review good?". A hotel manager needs "good at
what?" — *"the sheets were dirty, but the staff were wonderful and the view was
stunning"* is one `positive` that hides the only line worth acting on.

[`src/reviewnlp/absa/`](src/reviewnlp/absa/) extracts one record per aspect
instead:

```json
[
  {"aspect": "cleanliness", "sentiment": "negative", "quote": "sheets were dirty"},
  {"aspect": "staff",       "sentiment": "positive", "quote": "staff were wonderful"},
  {"aspect": "location",    "sentiment": "positive", "quote": "the view was stunning"}
]
```

The taxonomy is **fixed at eight aspects** (cleanliness, staff, location, room,
food, noise, value, facilities) rather than free-form, so that counts are
comparable from one week to the next. Unrecognized aspect strings are dropped
and counted, never coerced onto the nearest name — a coerced aspect invents
evidence for something the review never said.

**Two models, two roles.** Extraction runs on the **base**
`Qwen2.5-0.5B-Instruct`, zero-shot; the QLoRA adapter stays on binary scoring
where it was trained. The adapter was fine-tuned completion-only on
single-word targets, and asked for JSON it emits that trained single-word loop
instead — the instruction-following the task needs is what the fine-tune traded
away. `--variant base|adapter` keeps the comparison runnable rather than
asserted, and the failure mode itself is pinned in
[`tests/test_absa.py`](tests/test_absa.py).

```bash
# Per-aspect records + summary.json for a sample of the v2 test split built by `make data`
python scripts/run_absa.py --variant base \
    --test-parquet data/processed_v2/test.parquet \
    --data-manifest data/processed_v2/data_manifest.json \
    --output-dir runs/absa_base
```

The runner checks the split against its manifest. Without `--data-manifest` it
accepts only the archived legacy test parquet, checked by its SHA-256.

Every generation is parsed with quality flags: JSON recovered from surrounding
prose is flagged as `salvaged`, and aspects with missing or invented supporting
quotes are dropped. `scripts/prepare_absa_annotations.py` prepares a frozen
sample for independent human labeling; `scripts/evaluate_absa.py` scores aspect
detection and sentiment against completed annotations. No labeled evaluation
set or resulting performance figures have been published yet.

**Prototype status — what this section does not claim:**

- **No accuracy numbers are published here.** Runs land in git-ignored `runs/`,
  and no ABSA artifact is committed under `docs/experiments/`. As with the
  five-family benchmark above, the README publishes figures only once the
  artifact backing them is in the repository.
- **The overall vote abstains on genuinely mixed reviews.** One aspect positive
  and one negative returns no label rather than a coin flip. That is a declared
  behaviour, but it means any headline agreement score is bounded by how many
  reviews are mixed — a real open problem in ABSA, not a bug to tune away.
- **Zero-shot, with no supervised ABSA baseline** to compare against, and no
  gold aspect annotations — only the binary gold label the frozen split
  carries, which can check the aggregate vote but not the aspects themselves.
- **The generation loop is not covered offline.** `absa/pipeline.py` needs a
  model download, so CI exercises the taxonomy, parser, vote and runner helpers
  around it, not the batched `generate` call.

---

## Hotel-operations demo (Greek)

[`spaces/hotel-ops-demo/`](spaces/hotel-ops-demo/) is a demo for the owner of a
hotel or an apartment. Paste English reviews and it answers in Greek: complaints
and praise by topic, with quotes and a suggested action.

**How it differs from the Qwen prototype above.**
- Rules split each review into clauses.
- A lexicon names the topic, out of 30 topics in 8 categories.
- An aspect-sentiment model
  ([`yangheng/deberta-v3-base-absa-v1.1`](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1))
  reads each clause once per topic.
- The complaint-trend analysis uses the same lexicon to name its topics.
- The Space itself is private.

**How well it works.** It is measured on three label sets
([details](spaces/hotel-ops-demo/README.md#how-well-it-works)):
- **Set A.** Six apartment reviews that the project owner checked finding by
  finding. Each error became an acceptance test.
- **Set B.** 300 Booking.com TEST reviews in which the guests themselves wrote
  what they liked and disliked. A complaint is found in 77% of the reviews that
  have one (95% CI 70–83%).
- **Set C.** 40 reviews that the project owner labelled by topic, without seeing
  the output.
  - **8 categories.** 63% of the complaints the demo reports are labelled, and
    83% of the labelled ones are found.
  - **30 topics.** 53% and 46%.
  - **One annotator.** There is no agreement measure yet.

---

## Technical details

<details>
<summary><b>Training setup</b></summary>

- **Hardware**: Google Colab Tesla T4 (all runs)
- **Seed**: 42, two epochs, batch size 32, max length 256, FP16; lr 2e-5 (full fine-tune) and 1e-4 (scratch LoRA)
- **Clean splits** ([runs](docs/experiments/results/distilbert_v2/)):
  - full fine-tune: 13.6 / 13.3 min (random / out-of-time), peak 2,543 MiB;
  - scratch LoRA: 9.0 / 8.9 min, peak 1,558 MiB.
- **Legacy split**:
  - full fine-tune: 872.6 s, peak 2,534 MiB, best dev F1 0.9602;
  - scratch LoRA: 567.2 s, peak 1,563 MiB, best dev F1 0.9522.
- **LoRA parameters**: 147,456 adapter + 592,130 task head = 739,586 total

</details>

<details>
<summary><b>Evaluation methodology</b></summary>

- **Primary metric**: macro-F1 (balanced across positive/negative classes)
- **Significance**: exact paired McNemar test on classification errors
- **Comparing two models**: a paired bootstrap interval of the macro-F1 difference on the same test reviews.
  - A gain counts as meaningful when that interval excludes zero and the gain is at least one point.
  - The rule was fixed before training ([plan](docs/experiments/phase2_analysis_plan.md)).
- **Verification**, without a GPU:
  - [`scripts/verify_distilbert_handoff.py`](scripts/verify_distilbert_handoff.py) reproduces the legacy metrics, saved arrays, hashes and the McNemar calculation;
  - [`tests/test_phase2_results.py`](tests/test_phase2_results.py) recomputes the clean-split comparison from the committed logits on every CI run.

</details>

<details>
<summary><b>Known limitations</b></summary>

- The **historical training run** recorded normalized-text overlap (180 train/dev, 170 train/test, 24 dev/test). The newly uploaded three parquets were separately audited: they have **zero cross-split text overlap**, but 676 within-split duplicate rows and 19 train text groups with conflicting labels. The new clean baseline uses derived splits; the legacy five-model results do not become leakage-free by uploading later files. The v2 splits are rebuilt from the hash-checked raw file with duplicates removed before splitting, and their manifests record zero overlap between splits.
- The classical baseline rows are **historical runs** that predate the fix moving model selection to dev. The old `*_char` rows used the wrong analyzer and are excluded from the summary table.
- The Qwen QLoRA run was **trained on 20,000** of the 118,990 legacy training rows and tested on all 13,278 legacy test reviews. Its 0.9571 comes from first-step label scores; its generated answers failed a strict parser ([record](docs/experiments/results/qwen_legacy_v1/README.md)).
- The INT8 and batching timings come from **one Colab CPU and one thread**, for one checkpoint on the random split; other processors, thread counts or quantization backends can change them ([decision note](docs/experiments/decision_int8_and_batching.md)).
- The BiLSTM row is the **seed-42 run that has a saved metrics artifact**. Seed 100 reached a higher 0.9521 / 96.42% and a weighted-loss variant reached 0.9492 / 96.14% (see [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)), but neither has a preserved artifact bundle, so the table reports the reproducible one.
- On clean splits, **four of the five legacy models are measured**; Qwen is the fifth.
  - The historical McNemar table above is transcribed from the archived benchmark rather than regenerated from a committed artifact.
  - On the v2 random and out-of-time splits, the classical baseline and DistilBERT, fine-tuned in full and with the scratch LoRA, are measured and verified from committed logits ([decision note](docs/experiments/decision_distilbert_vs_nb.md)).
  - The BiLSTM is measured on all three clean splits ([decision note](docs/experiments/decision_bilstm_clean_splits.md)). Qwen QLoRA is not carried forward for binary scoring ([decision](docs/experiments/decision_qwen_binary_scope.md)).
  - Training seeds: the full fine-tune against LoRA and the BiLSTM against DistilBERT were each checked across three seeds, on the out-of-time split only ([DistilBERT](docs/experiments/decision_distilbert_seeds.md), [BiLSTM](docs/experiments/decision_bilstm_clean_splits.md)). Three seeds give a rough spread, not an interval; the random split's DistilBERT results come from one seed.
- The complaint-trend analysis names topics with a lexicon. A [pilot with two annotators](docs/case_study/results/annotation/README.md) measured it (300 texts, 80 labelled twice, kappa 0.83–1.00). Precision is about half or lower for cleanliness and responsiveness, and the lexicon finds only 2 of 16 responsiveness complaints. A change in its precision between the two periods cannot be told apart from zero, and a change in recall is not measured, so the rising topics stay unconfirmed.
- The dashboard and SQLite store support a local single-worker pilot. ABSA aspect accuracy, Greek hotel-domain accuracy, access operations, and a real load test still need evidence before a public hosted product.

</details>

<details>
<summary><b>Repository layout</b></summary>

```text
src/reviewnlp/
  data/            schema labels, preprocessing, split policies, manifests
  baselines/       TF-IDF + NB / LR-SGD, pure-PyTorch BiLSTM
  lora/            scratch LoRA layers, merge/unmerge
  llm/             DistilBERT full FT / scratch LoRA, Qwen QLoRA
  evaluation/      metrics, bootstrap intervals, benchmark, McNemar, plots
  serving/         FastAPI; stub, classical, encoder, Qwen backends
  analytics/       SQLite aspect storage, complaint ranking and recommendations
  greek/           Greek fine-tune: config, splits, baseline, evaluation
  absa/            aspect taxonomy, strict JSON parser, per-aspect vote
analysis/complaint_trends/sql/   SQL for the complaint-trend analysis
spaces/hotel-ops-demo/           the Greek operations demo (Gradio Space)
configs/           YAML configs for CLI experiments
data/eval/         label sets for the demo (ids and labels only)
notebooks/         Colab templates (no saved outputs — see notebooks/README.md)
scripts/           data fetch, split views, model comparison and latency, complaint trends, pilot sampling and scoring, evaluation, verification, API utilities
tests/             data, splits, LoRA/PEFT, metrics, SQL, API, demo, Greek, ABSA checks
docs/experiments/  phase 2 plan and decision note, preserved runs, manifests, verified handoffs
docs/case_study/   the complaint-trend report and its results
docs/annotation/   pilot protocol and labelling guideline (locked)
docs/EVIDENCE_MAP.md  what the project shows, with links to the evidence
```

</details>

---

## What I learned

- **A clean test set has to be built, not assumed.**
  - The first split shared 170 texts between train and test.
  - What makes overlap checkable: rebuilding the splits from a hash-checked raw
    file, removing duplicates before splitting, and recording the overlap in a
    manifest.
- **What the test set holds decides what the number means.** The same pipeline
  scores 1.7 macro-F1 points lower on reviews written after the training period.
  Unseen hotels show similar observed performance.
- **A step in the data can look like a trend.**
  - One jump in February 2016 lifted every complaint topic at once.
  - Comparing only the months after it changed which topics looked like rises.
- **Say what was measured, and by whom.**
  - The complaint lexicon's precision was first checked on a small sample, by an
    AI reader, and its recall not at all. The report said both.
  - A pilot with two people then measured precision and recall. Precision was about
    0.9 for air conditioning, but about half or lower for cleanliness and
    responsiveness.
- **Decide what counts as a win before training.**
  - The comparison and the rule were committed before the first DistilBERT run: a
    paired interval above zero and a gain of at least one point.
  - DistilBERT then beat Naive Bayes by 3.1 points on later reviews.
  - CI checks that the rule has not changed since.
- **Full fine-tuning beat LoRA, on the legacy split and again on the clean
  splits** (by 0.6–0.7 points). With 118k training rows, LoRA's parameter savings
  did not translate into accuracy.
- **Decoder models aren't automatically better.**
  - On the legacy split, Qwen2.5-0.5B + QLoRA matched DistilBERT + LoRA only when its label was
    read from the first-step scores of the two label tokens.
  - Its generated answers failed a strict parser, because it kept writing after the label. How the
    label is read out of a generative model is part of the result.
  - Its speed relative to DistilBERT has not been measured.
  - For classification here, the encoder is the pragmatic choice: it has 67M parameters against 0.5B,
    and it returns class probabilities rather than a generated word.
- **The serving layer matters as much as the model.** A 96% accurate model behind a slow API is worse than a 95% model that answers quickly.
- **Quantization deserves a full-test check.**
  - A 32-example observation suggested INT8 was 1.38 times faster.
  - Measured on the whole test set, with one review at a time, it is 3.0 times faster, with macro-F1 unchanged within 0.001.
  - Batching was the surprise: unsorted batches were slower than single reviews, and sorting by length helped FP32 but not INT8.

---

## Roadmap

- [x] Audit the uploaded parquets and re-run the classical baseline, including character n-grams, on independently identified clean splits
- [x] Rebuild the splits from the hash-checked raw file: random, **out-of-time** and **unseen hotels**, with bootstrap intervals
- [x] Ask which complaint topics are rising within the same hotels, in SQL ([case study](docs/case_study/complaint_trends.md))
- [x] Re-run DistilBERT (full fine-tune and scratch LoRA) on the v2 random and out-of-time splits, under a plan fixed before training ([decision note](docs/experiments/decision_distilbert_vs_nb.md))
- [x] Re-run the BiLSTM on the **v2 splits**, with three seeds and CPU latency and size ([decision note](docs/experiments/decision_bilstm_clean_splits.md))
- [x] Check DistilBERT's full fine-tune against LoRA across three training seeds ([decision note](docs/experiments/decision_distilbert_seeds.md))
- [x] Measure dynamic INT8 and length-sorted CPU batching on the full test set ([decision note](docs/experiments/decision_int8_and_batching.md))
- [ ] Serve the INT8 model from the API and the Space, with a check that it reproduces its recorded test labels
- [x] Pilot evaluation of the complaint lexicon by two annotators: precision, recall, Cohen's kappa ([protocol](docs/annotation/pilot_protocol.md), [results](docs/case_study/results/annotation/README.md))
- [ ] Break the rising complaint topics down by hotel and city
- [ ] Add a **streaming inference** endpoint for high-throughput ingestion
- [ ] Experiment with **ONNX Runtime** for further CPU speedups
- [ ] Add **calibration** (temperature scaling) so confidence scores are usable downstream
- [x] Add a local hotel-scoped operations dashboard over stored aspect records
- [ ] Validate and deploy the dashboard with real hotel data and monitoring
- [ ] Publish independently annotated ABSA and Greek hotel evaluation sets and their measured results
- [ ] Replace the small-deployment SQLite store with an audited multi-worker storage and access-management setup before offering a public hosted service

---

## Links

- 📓 **Notebooks**: [executed runs with outputs](docs/experiments/notebooks/) · [Colab templates](notebooks/) ([what each needs](notebooks/README.md)) — the templates are committed without outputs and are meant to be run, not read as results
- 📊 **Results**: [DistilBERT or Naive Bayes? decision note](docs/experiments/decision_distilbert_vs_nb.md) · [preserved run artifacts](docs/experiments/results/) · [verification report](docs/experiments/results/distilbert_legacy_full_v1/verification.json)
- 🏗️ **Design**: [DESIGN.md](DESIGN.md) — evaluation methodology, what I'd do with more compute
- 🧪 **Experiment log**: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)
- 🗺️ **Evidence map**: [docs/EVIDENCE_MAP.md](docs/EVIDENCE_MAP.md) — what the project shows for a junior data scientist role, and what is not done yet
- 📈 **Case study**: [Which complaints are rising?](docs/case_study/complaint_trends.md)
- 🏷️ **Pilot evaluation**: [protocol and labelling guideline](docs/annotation/)
- 🏨 **Hotel-operations demo**: [spaces/hotel-ops-demo/](spaces/hotel-ops-demo/)

## License

MIT — see [LICENSE](LICENSE).
