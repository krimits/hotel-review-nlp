# First supervised baseline: development only

**Executed 12 October 2026**, with all 500 final human annotations structurally
validated before model predictions. [Annotation record](../../annotation_record.json)
locks source-file hashes and records 88/100 initial A–B agreements, kappa
0.74555; reviewer independence is unverified. The 499 completely empty formatted
Excel rows were omitted from CSV conversion; all 500 populated rows and their
literal fields were preserved. No human judgments were filled by a model.

The unchanged protocol trains word TF-IDF + logistic regression on 300 reviews
and evaluates on the fixed 100 dev reviews. Source `d2a9348`; CPU, Python 3.12.14,
NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0. These actual local versions are
in [run.json](run.json); the later Colab comparison uses its own pinned
environment and reruns the same baseline for a common-runtime comparison.

| Dev measurement | Raw | Dev-fitted temperature |
| --- | ---: | ---: |
| Accuracy | 0.6400 | 0.6400 |
| Macro-F1, all five classes | 0.1561 | 0.1561 |
| Top-label ECE, 10 bins | 0.1528 | 0.1432 |
| Retained at max probability ≥0.7 | 42/100 | 47/100 |
| Human referrals below 0.7 | 58/100 | 53/100 |
| Errors among retained | 6/42 | 8/47 |

**All 100 predictions are `other`.** The 64% accuracy equals always predicting
the majority class; the macro-F1 exposes the failure to distinguish topics.
`other` F1 is 0.7805; every other class has zero F1. Dev gold support is
cleanliness 1, service 18, location 14, value 3, other 64. No hyperparameter,
class-weight or ontology change was made after observing this result.

Temperature 0.96662 was fitted on this same dev set. Its ECE/log-loss and
referral figures are **in-sample development diagnostics**, not an independent
validation of calibration or the 0.7 policy. Calibration does not change the
argmax; it retains five more cases here, with two additional errors.

Request p95 was 1.223 ms on this particular local CPU path, batch 1; no model
load, training or warm-up is included. Local CPU billing is unknown, not zero.
No Qwen or Microsoft inference, reserved triage evaluation, primary-topic test
prediction, model selection or Space change occurred.

- [predictions.jsonl](predictions.jsonl): IDs and complete probabilities; no review text or gold labels.
- [freeze.json](freeze.json): **logistic-only dev freeze**, not the three-model freeze.
- [human_queue.csv](human_queue.csv): model referrals, not a human rating sheet.

Next: run Qwen and the dated Microsoft build on the same dev inputs with
notebook 23. Its software API budget is opt-in; save the capture including
errors. Freeze the chosen full configuration before the single final test.
