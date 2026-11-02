# 🏨 Hotel Review NLP

[![CI](https://github.com/krimits/hotel-review-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/krimits/hotel-review-nlp/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Reviewing this for a role? [Start here](docs/START_HERE.md)** — a ten-minute path through the evidence, including what is still open.

Hotel review classification and an emerging hotel-operations product: a binary sentiment
API, aspect extraction, SQL complaint-trend analysis and hotel-scoped recommendations,
trained on about **118,000 real Booking.com reviews**.

🚀 **[hotel-review-demo](https://huggingface.co/spaces/krimits/hotel-review-demo)** · 🇬🇷 **[greek-review-sentiment-demo](https://huggingface.co/spaces/krimits/greek-review-sentiment-demo)** · 🛠️ **[hotel-triage-demo](https://huggingface.co/spaces/krimits/hotel-triage-demo)** (experimental) · 📦 **[Model on HF Hub](https://huggingface.co/krimits/distilbert-hotel-reviews)**

## What this is

- **A classifier with evidence.** Fine-tuned DistilBERT: **0.9642 macro-F1** on a random
  split and **0.9491** on reviews written after the training period, in test sets that share
  no review text with training. Paired bootstrap intervals and exact McNemar tests; the
  rule for a meaningful gain was committed before training.
- **A service.** FastAPI with pluggable backends (`stub`, `classical`, `encoder`,
  `qwen_qlora`), a Docker image with weights mounted at runtime, a hotel dashboard over
  SQLite, and aspect extraction that returns quoted evidence.
- **An analytics study.** SQL over all 515K reviews asking which complaint topics rise
  *within the same hotels*, with a hotel-level bootstrap and a lexicon measured by two
  annotators.

## Results

Macro-F1 on three clean test sets. Naive Bayes was selected on dev; DistilBERT was
fine-tuned after the analysis plan was locked.

| Test set | What it asks | Reviews | TF-IDF + NB | BiLSTM | DistilBERT full FT |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Random | reviews like the training ones | 13,263 | 0.9351 | 0.9486 | **0.9642** |
| **Out-of-time** | reviews written after every training review | 13,921 | 0.9184 | 0.9357 | **0.9491** |
| Unseen hotels | 147 hotels with no review in training | 13,289 | 0.9348 | 0.9514 | not run |

Every value, its 95% interval, its seeds and the artifact that holds it:
**[docs/NUMBERS.md](docs/NUMBERS.md)**. LoRA, the legacy split, quantisation and the
full comparison tables are in **[docs/RESULTS.md](docs/RESULTS.md)**.

Earlier figures from a split whose test set shares **170** review texts with training are
kept, labelled *not comparable*, in
[the historical section](docs/RESULTS.md#historical-results-legacy-split-text-overlap).

## Quickstart (60 seconds)

```bash
git clone https://github.com/krimits/hotel-review-nlp.git && cd hotel-review-nlp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serving]"
hf download krimits/distilbert-hotel-reviews --revision 7306aebcaaebc00d579f5d0a91001ae376f18158 --local-dir models/distilbert
MODEL_TYPE=encoder MODEL_PATH=models/distilbert make serve
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "Perfect stay, spotless room, incredibly friendly staff."}'
```

Interactive API docs are at `/docs` once the service runs. The Docker alternative is in
[RESULTS](docs/RESULTS.md#option-3--docker); the full local product walkthrough is in
[docs/PRODUCT_SETUP.md](docs/PRODUCT_SETUP.md).

## Repository layout

```text
src/reviewnlp/                   data pipeline, baselines, scratch LoRA, DistilBERT, BiLSTM,
                                 evaluation, serving, analytics, ABSA, Greek models
analysis/complaint_trends/sql/   the seven SQL queries behind the case study
spaces/                          the demo apps (sentiment, Greek operations, triage)
configs/                         one YAML per experiment
docs/                            results, numbers, evidence map, case study, protocols
tests/                           offline checks: data, splits, LoRA/PEFT, metrics, SQL, API, demo
notebooks/                       Colab templates (committed without outputs)
```

## Documentation

| Read this | For |
| :--- | :--- |
| [docs/START_HERE.md](docs/START_HERE.md) | the ten-minute reviewer path |
| [docs/RESULTS.md](docs/RESULTS.md) | every result, with tables, limits and the roadmap |
| [docs/NUMBERS.md](docs/NUMBERS.md) | each headline number and the artifact that holds it |
| [docs/EVIDENCE_MAP.md](docs/EVIDENCE_MAP.md) | claim → evidence → what is not done yet |
| [docs/case_study/complaint_trends.md](docs/case_study/complaint_trends.md) | the complaint-trend study |
| [docs/PRODUCT_SETUP.md](docs/PRODUCT_SETUP.md) · [docs/TRIAGE.md](docs/TRIAGE.md) | the local product pilot and the experimental triage endpoint |
| [docs/RUNS.md](docs/RUNS.md) · [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | run log and experiment history |
| [docs/history/](docs/history/) | the superseded 2026-09 action plan and completion audit |

## Known limitations

Prototypes stay labelled: the triage endpoint, ABSA and the Greek demo publish no
accuracy against human labels, the complaint lexicon is reliable on only part of its five
topics, and the legacy split is not comparable with the clean one. The full list, with
scope and what would change each verdict, is in
[docs/RESULTS.md#known-limitations](docs/RESULTS.md#known-limitations).

## License

MIT — see [LICENSE](LICENSE). The Booking.com data is downloaded from a pinned public
copy; each evaluation set is described, with its source and its labelling, in
[docs/RESULTS.md](docs/RESULTS.md).
