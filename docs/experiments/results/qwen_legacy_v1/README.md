# Qwen2.5-0.5B QLoRA on the legacy split: archived metrics

This is the source of the README's Qwen row (macro-F1 0.9571). It records **two** scores for the
same test reviews, from two prediction rules. The README number is the second one.

## Provenance

- [`metrics.json`](metrics.json): the `metrics.json` of the Hub repo
  [`krimits/hotel-review-nlp-qwen-qlora-adapter`](https://huggingface.co/krimits/hotel-review-nlp-qwen-qlora-adapter/tree/1458bdb9058fbb629e72ce83d9b0f44a09271941)
  at commit `1458bdb9058fbb629e72ce83d9b0f44a09271941`. This is the commit the sentiment Space pins.
  - SHA-256 of the archived file: `f48fc006014fa693912809a5d1533946565e069ea488007cf3971c8559c870af`, 4,561 bytes.
  - It was archived on 14 October 2026 from the file's text as read through the Hub API.
  - Its size equals the Hub's 4,561 bytes. The Hub's own hash of the file could not be read from the
    archiving environment.
- **Producer:** [notebook 06](../../../../notebooks/06_qwen_qlora_colab.ipynb), run on 11 September 2026
  from code commit `e7a3c71f8cbd652a878ed5f19c2b7eae766f1615`, on a Tesla T4.
- **The adapter repo also holds** the run's `raw_generations.jsonl`, `test_logits.npy` (the first-step
  scores of the two label tokens), `test_labels.npy`, `test_row_ids.npy` and `trainer_state.json`.
  They are not copied here.

## What was trained and tested

| | |
| :--- | :--- |
| Split | The legacy frozen split. Its test set shares 170 texts with training (audit finding F02). |
| Training rows | 20,000 of the 118,990 legacy training rows; dev capped at 2,000 |
| Test rows | All 13,278 legacy test reviews |
| Training | 1 epoch, seed 42, 4-bit base, 8,798,208 trainable adapter parameters |
| Time on a T4 | 9,856.9 s training; 916.7 s to generate for the test set (14.5 reviews/s) |

## The two scores

| Prediction rule | Macro-F1 | Accuracy |
| :--- | ---: | ---: |
| Greedy generation, strict parser | 0.0 | 0.0 |
| First-step label scores | **0.9571** | **96.71%** |

- **Greedy generation, strict parser.** The parser (`strict_generated_label` in the notebook) accepts
  an answer only when the whole generated text is exactly one label word.
  - The model wrote the label and then a second label line, for example `positive\npositive\n`.
    `max_new_tokens` was 4.
  - So the parser rejected all 13,278 answers (`invalid_output_rate` 1.0).
- **First-step label scores.** At the first generated position, the notebook takes the logits of the two
  label tokens (`generated.scores[0][:, ordered_label_token_ids]`) and predicts the higher one. That
  label is scored against the gold labels.

## What this does not show

- **No generation-based score on this test set.** The lenient parser (`parse_label` in
  [`src/reviewnlp/llm/prompt_format.py`](../../../../src/reviewnlp/llm/prompt_format.py)) takes the first
  label word in the generated text. It is used by `src/reviewnlp/llm/predict.py` and by the sentiment
  Space. It has not been scored here.
  - It could be, without a GPU, from the archived `raw_generations.jsonl`.
  - The 25 rejected answers recorded in `metrics.json` begin with their gold label. That is consistent
    with, but not a measurement of, close agreement between the two rules.
- **The McNemar rows for Qwen are not regenerated.** The README's table is transcribed from the
  archived benchmark, which does not record which Qwen predictions it compared.
- **Not comparable with the clean splits.** The test set shares texts with training, and Qwen saw a
  sixth of the training rows that the other models saw.
- **Speed.** The 14.5 reviews/s above is the evaluation rate in batches on a T4 GPU. Qwen's speed
  relative to DistilBERT on comparable terms has not been measured.
