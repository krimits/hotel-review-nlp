# Evidence map

What this project shows for a junior data scientist role, and where to check it.
Each row links to code, data or results that a reviewer can open or rerun. The
last column says what is not done yet.

| Requirement | Evidence | Not done yet |
|---|---|---|
| **Python for data work** | The `reviewnlp` package: [data pipeline](../src/reviewnlp/data/preprocess.py), [models](../src/reviewnlp/baselines/), [evaluation](../src/reviewnlp/evaluation/) and a [FastAPI service](../src/reviewnlp/serving/). The [tests](../tests/) run on every push ([CI](../.github/workflows/ci.yml)). | |
| **SQL** | [Seven queries](../analysis/complaint_trends/sql/) on hotels, reviews and complaints: CTEs, joins, window functions (rolling means, shares within partitions), a same-month year-over-year comparison and a within-hotel comparison with a standardised rate. They are [tested](../tests/test_complaint_trends.py) on a hand-built database with known answers. | The results on all 515K reviews, and the [report](case_study/complaint_trends.md) written from them. |
| **Data management and reproducibility** | [`fetch_booking_515k.py`](../scripts/fetch_booking_515k.py) downloads the raw file from a pinned source and refuses it unless its size and SHA-256 match the Kaggle download. Deduplication runs before splitting, and every split directory has a manifest with hashes and overlap checks. The [clean benchmark](../runs/benchmark/results.json) records fingerprints and zero overlap. | The manifests of the three rebuilt splits. |
| **Model development and validation** | Models are chosen on dev only ([classical baselines](../src/reviewnlp/baselines/classical.py)). Results carry a [bootstrap interval](../src/reviewnlp/evaluation/metrics.py), and pairs of models are compared with an [exact McNemar test](../src/reviewnlp/evaluation/significance.py). Three test sets ask different questions: random, [out-of-time](../configs/baselines_time.yaml) and [unseen hotels](../configs/baselines_hotel.yaml). The aspect demo is evaluated on three label sets, with Wilson intervals ([`eval_space_triage.py`](../scripts/eval_space_triage.py)). | DistilBERT on the clean and out-of-time splits, with saved predictions, cost and a short decision note. |
| **Statistics** | Wilson and bootstrap intervals. A hotel-level bootstrap, since reviews of one hotel are not independent. A stricter interval when 30 topics are tested at once. Standardisation to separate a change in the hotel mix from a real change. | |
| **Honest evaluation** | Legacy results are kept, but [labelled](../README.md#historical-results-legacy-split-text-overlap) as measured on a test set that shares texts with training. Labels written by the AI assistant are [declared as such](../spaces/hotel-ops-demo/README.md). A person labelled 40 reviews without seeing the output, and every disagreement was analysed. | A second annotator and an agreement measure (Cohen's kappa). |
| **Working with a user** | A hotel owner reviewed 42 findings of the demo. Each error became an [acceptance test](../data/eval/space_triage_user6.json) and was fixed ([tests](../tests/test_space_demo.py)). | A timed pilot with and without the tool. |
| **Communication** | An English [README](../README.md), a Greek-language demo for hotel owners ([`spaces/hotel-ops-demo`](../spaces/hotel-ops-demo/)), and quotes shown next to every finding. | A five-minute case study in English. |

**Not covered by this project:** financial mathematics, credit-risk modelling
(PD models, calibration, stability), R and SAS.
