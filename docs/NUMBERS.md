# Numbers

One row per headline number, with the committed artifact that holds it and the exact
JSON key inside that artifact. `tests/test_numbers.py` parses this table and fails when a
value does not match its artifact to the precision printed here, or when a path or key
does not resolve.

Two rules for reading the table:

- **Scope decides meaning.** A row marked *legacy; not comparable* was measured on the
  first frozen test set, which shares **170** normalized review texts with its training
  set (audit finding F02). It is kept as recorded and cannot be compared with a clean row.
- **A model score is not a product claim.** The triage rows are agreement with an LLM's
  labels (`reference_chatgpt`), not with a hotel's staff or any person.

## Macro-F1 on the clean splits

Clean here means: rebuilt from the hash-checked raw file, duplicates removed before
splitting, zero review text shared between train, dev and test.

| Value | Metric | Test set | Scope | Artifact | Key |
| --- | --- | --- | --- | --- | --- |
| 0.9642 | DistilBERT, full fine-tune, macro-F1 | random, 13,263 reviews | clean; seed 42 | [metrics.json](experiments/results/distilbert_v2/random/distilbert/metrics.json) | `test.macro_f1` |
| 0.9491 | DistilBERT, full fine-tune, macro-F1 | out-of-time, 13,921 reviews | clean; seed 42 | [metrics.json](experiments/results/distilbert_v2/time/distilbert/metrics.json) | `test.macro_f1` |
| 0.9496 | DistilBERT, full fine-tune, macro-F1 | out-of-time | clean; seed 43 | [metrics.json](experiments/results/distilbert_v2/time/distilbert_seed43/metrics.json) | `test.macro_f1` |
| 0.9480 | DistilBERT, full fine-tune, macro-F1 | out-of-time | clean; seed 44 | [metrics.json](experiments/results/distilbert_v2/time/distilbert_seed44/metrics.json) | `test.macro_f1` |
| 0.9576 | DistilBERT + scratch LoRA, macro-F1 | random | clean; seed 42 | [metrics.json](experiments/results/distilbert_v2/random/distilbert_lora_scratch/metrics.json) | `test.macro_f1` |
| 0.9424 | DistilBERT + scratch LoRA, macro-F1 | out-of-time | clean; seed 42 | [metrics.json](experiments/results/distilbert_v2/time/distilbert_lora_scratch/metrics.json) | `test.macro_f1` |
| 0.9428 | DistilBERT + scratch LoRA, macro-F1 | out-of-time | clean; seed 43 | [metrics.json](experiments/results/distilbert_v2/time/distilbert_lora_scratch_seed43/metrics.json) | `test.macro_f1` |
| 0.9436 | DistilBERT + scratch LoRA, macro-F1 | out-of-time | clean; seed 44 | [metrics.json](experiments/results/distilbert_v2/time/distilbert_lora_scratch_seed44/metrics.json) | `test.macro_f1` |
| 0.9351 | TF-IDF + Naive Bayes, macro-F1 | random | clean; pipeline selected on dev | [classical_metrics.json](experiments/results/v2_splits/random/classical_metrics.json) | `naive_bayes_word.macro_f1` |
| 0.9184 | TF-IDF + Naive Bayes, macro-F1 | out-of-time | clean; pipeline selected on dev | [classical_metrics.json](experiments/results/v2_splits/time/classical_metrics.json) | `naive_bayes_word.macro_f1` |
| 0.9348 | TF-IDF + Naive Bayes, macro-F1 | unseen hotels, 13,289 reviews | clean; pipeline selected on dev | [classical_metrics.json](experiments/results/v2_splits/hotel/classical_metrics.json) | `naive_bayes_word.macro_f1` |
| 0.9486 | BiLSTM, macro-F1 | random | clean; seed 42 | [metrics.json](experiments/results/bilstm_v2/random/bilstm/metrics.json) | `test.macro_f1` |
| 0.9357 | BiLSTM, macro-F1 | out-of-time | clean; seed 42 | [metrics.json](experiments/results/bilstm_v2/time/bilstm/metrics.json) | `test.macro_f1` |
| 0.9514 | BiLSTM, macro-F1 | unseen hotels | clean; seed 42 | [metrics.json](experiments/results/bilstm_v2/hotel/bilstm/metrics.json) | `test.macro_f1` |
| 0.9363 | BiLSTM, macro-F1 | out-of-time | clean; seed 43 | [metrics.json](experiments/results/bilstm_v2/time/bilstm_seed43/metrics.json) | `test.macro_f1` |
| 0.9372 | BiLSTM, macro-F1 | out-of-time | clean; seed 44 | [metrics.json](experiments/results/bilstm_v2/time/bilstm_seed44/metrics.json) | `test.macro_f1` |
| 0.9347 | TF-IDF + Naive Bayes, macro-F1 | derived clean set, 13,270 reviews | older uploaded parquets, deduplicated; no raw-CSV lineage | [classical_clean_uploaded_metrics.json](experiments/results/classical_clean_uploaded_metrics.json) | `naive_bayes_word.macro_f1` |
| 0.9637 | DistilBERT, dynamic INT8, macro-F1 | random, full test set | clean; quantised from the published checkpoint | [int8_benchmark.json](experiments/results/int8_v1/int8_benchmark.json) | `models.int8.test.macro_f1` |

## Macro-F1 on the legacy split

The legacy split is the first frozen test set (13,278 reviews). Its test set shares
**170** normalized review texts with training, so every row below is **legacy; not
comparable** with the clean rows above. Kept because the runs are what they are.

| Value | Metric | Test set | Scope | Artifact | Key |
| --- | --- | --- | --- | --- | --- |
| 0.9345 | TF-IDF + Naive Bayes, macro-F1 | legacy, 13,278 reviews | legacy; not comparable | [classical_legacy_metrics.json](experiments/results/classical_legacy_metrics.json) | `naive_bayes_word.macro_f1` |
| 0.9503 | BiLSTM, macro-F1 | legacy | legacy; not comparable | [bilstm_legacy_metrics.json](experiments/results/bilstm_legacy_metrics.json) | `test.macro_f1` |
| 0.9634 | DistilBERT, full fine-tune, macro-F1 | legacy | legacy; not comparable | [metrics.json](experiments/results/distilbert_legacy_full_v1/distilbert/metrics.json) | `test.macro_f1` |
| 0.9573 | DistilBERT + scratch LoRA, macro-F1 | legacy | legacy; not comparable | [metrics.json](experiments/results/distilbert_legacy_full_v1/distilbert_lora_scratch/metrics.json) | `test.macro_f1` |
| 0.9571 | Qwen2.5-0.5B + QLoRA, macro-F1 | legacy | legacy; not comparable; read from first-step label scores, not generated answers | [metrics.json](experiments/results/qwen_legacy_v1/metrics.json) | `evaluation.first_step_score_metrics.macro_f1` |

## Triage diagnosis (agreement with an LLM's labels, not accuracy)

The 500 test reviews and 104 urgent (review, type) pairs the reference labels carry. The
rule was locked before the run; the verdict is **Not met** (see
[the decision note](experiments/decision_triage_diagnosis.md)).

| Value | Metric | Test set | Scope | Artifact | Key |
| --- | --- | --- | --- | --- | --- |
| 0.7087 | Urgent recall, `rec` (IPW) | 500 test reviews | agreement with `reference_chatgpt` | [score.json](experiments/triage_diagnosis/student/score.json) | `decision.rec` |
| 0.6185 | Bootstrap 2.5th percentile, `lo` | 500 test reviews | 2,000 resamples, seed 0 | [score.json](experiments/triage_diagnosis/student/score.json) | `decision.lo` |
| 3.62 | False urgent flags per 100, `fa` | 500 test reviews | IPW | [score.json](experiments/triage_diagnosis/student/score.json) | `decision.fa` |
| 0.9669 | Teacher (`claude-haiku-5.5`) urgent recall | 500 test reviews | agreement with `reference_chatgpt`; baseline, not the product | [score.json](experiments/triage_diagnosis/student/score.json) | `reported.baselines.teacher_claude_haiku_5_5.rec` |

## Triage student v2 (agreement with an LLM's labels, not accuracy)

The 500 new test reviews and 118 urgent (review, type) pairs the reference labels carry under
`catalogue_v2`. The rule and the choice of variant B were locked before the run; the verdict is
**Not met** (see [the decision note](experiments/decision_triage_student_v2.md)).

| Value | Metric | Test set | Scope | Artifact | Key |
| --- | --- | --- | --- | --- | --- |
| 0.8423 | Urgent recall, `rec` (IPW), variant B | 500 test reviews | agreement with `reference_chatgpt_v2` | [score.json](experiments/triage_student_v2/student/score.json) | `decision.rec` |
| 0.7517 | Bootstrap 2.5th percentile, `lo` | 500 test reviews | 2,000 resamples, seed 0 | [score.json](experiments/triage_student_v2/student/score.json) | `decision.lo` |
| 4.39 | False urgent flags per 100, `fa` | 500 test reviews | IPW | [score.json](experiments/triage_student_v2/student/score.json) | `decision.fa` |
| 0.9491 | Teacher (`claude-haiku-5.5`) urgent recall | 500 test reviews | agreement with `reference_chatgpt_v2`; baseline, not the product | [score.json](experiments/triage_student_v2/student/score.json) | `reported.baselines.teacher_claude_haiku_5_5.rec` |

## What is not in this table

- Confidence intervals are in the artifacts and in
  [RESULTS.md](RESULTS.md#key-results); they are not repeated here, because the 1,000- and
  2,000-resample versions differ slightly and two copies would invite a wrong comparison.
- Demo and Space numbers (77% complaint coverage on 300 reviews, kappa 0.83-1.00 for the
  lexicon) live in the demo README and the case study, with their own scopes.
- Latency, size and throughput numbers are in
  [RESULTS.md](RESULTS.md#serving-benchmark) and
  [RESULTS.md](RESULTS.md#quantization).
