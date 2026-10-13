# BiLSTM on the clean splits — version 1

**Fixed before any training run, 13 October 2026. Status: locked; no run has been
executed under it. Scope: development evidence; no selection is frozen and no model is
promoted.**

The legacy five-model table compares the BiLSTM on a test set that shares 170 review
texts with training (audit finding F02). DistilBERT and the scratch LoRA were re-run on
the clean splits; the BiLSTM was not. Every BiLSTM comparison in the README therefore
still rests on the contaminated split. This protocol fixes the terms under which that
gap is closed, before any number exists.

## 1. Question

Does the BiLSTM baseline's standing against the full DistilBERT fine-tune survive on a
split that shares no review text with training — specifically, on reviews written after
the training period?

One sentence, one split, one pair. Everything else in this document is reporting.

## 2. Data

Three clean split views, produced by `scripts/split_views.py` with seed 42 from the
Booking.com 515k CSV. The source file is **not** committed (CLAUDE.md forbids raw data in
the repository); it is pinned by hash instead:

| Item | Value |
| --- | --- |
| raw CSV | `data/raw/booking_reviews_515k.csv` |
| raw CSV sha256 | `a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9` |
| raw CSV bytes | 238,154,765 |
| split seed | 42 |

Per-split fingerprints, as recorded in each view's `data_manifest.json` under
`docs/experiments/results/v2_splits/`:

| View | Split | Rows | Negative | Positive | sha256 (first 16) |
| --- | --- | ---: | ---: | ---: | --- |
| random | train | 117880 | 26105 | 91775 | `37930e6da0c6487a` |
| random | dev | 14734 | 3263 | 11471 | `ba1d62768436766d` |
| random | test | 13263 | 3263 | 10000 | `8cd21cdac358f10b` |
| **time** | **train** | **116547** | **24692** | **91855** | `74f447be7e776135` |
| **time** | **dev** | **14612** | **4018** | **10594** | `5ddec23ec9bb0bbf` |
| **time** | **test** | **13921** | **3921** | **10000** | `07429dd59e7cea22` |
| hotel | train | 116202 | 26162 | 90040 | `e7e942a83bf77b69` |
| hotel | dev | 15086 | 3180 | 11906 | `8df9b1055804d1e3` |
| hotel | test | 13289 | 3289 | 10000 | `877487d90eb77d27` |

`cross_split_overlap` is 0 for `train_dev`, `train_test` and `dev_test` in all three
views. The `time` view's date ranges are train 2015-08-04 → 2017-03-09, dev 2017-03-10 →
2017-05-25, test 2017-05-26 → 2017-08-03.

**The data may not change after this commit.** A run whose `metrics["data"]["test"]`
differs from `docs/experiments/results/split_views.json` is rejected by
`scripts/compare_split_models.py::check_same_test`, which also re-scores the recorded
Naive Bayes predictions on the supplied labels and refuses a mismatch above 1e-4.

## 3. One primary comparison

**BiLSTM versus DistilBERT (full fine-tune), macro-F1, on the `time` view's 13,921 test
reviews.** Paired bootstrap over the same test rows, 2000 resamples, seed 0 — the
procedure `scripts/compare_split_models.py` already applies to the other pairs.

Values already recorded for that split, against which the BiLSTM will be placed:

| Model | macro-F1 | 95% CI |
| --- | ---: | --- |
| DistilBERT, full fine-tune | 0.9491 | [0.9450, 0.9531] |
| DistilBERT, scratch LoRA | 0.9424 | [0.9380, 0.9465] |
| TF-IDF + Naive Bayes | 0.9184 | [0.9133, 0.9230] |

Reported but **unable to overturn the rule**: the BiLSTM on the `random` and `hotel`
views; BiLSTM versus TF-IDF + Naive Bayes on all three views; per-class precision and
recall; the random-to-time drop; training wall time.

## 4. Decision rule

Let `d` be the macro-F1 difference (BiLSTM − DistilBERT) on the `time` test set and
`[lo, hi]` its 95% paired bootstrap interval.

| Condition | Conclusion |
| --- | --- |
| `hi < 0` | DistilBERT is better on clean out-of-time data. The legacy ordering is confirmed and the README keeps its transformer claim, now on uncontaminated data. |
| `lo > 0` | The BiLSTM is better. The legacy ordering is reversed; the README's transformer claim must be restated. |
| `lo <= 0 <= hi` | The two are indistinguishable on this split. The legacy claim that the transformer beats the BiLSTM is **not confirmed on clean data**, and the README must say exactly that. |

This is applied exactly as written once results exist, with no discretion. The other two
views are reported alongside and cannot change the conclusion. If the rule turns out to
be the wrong question for this comparison, the decision note says so in those words —
it is never reinterpreted silently.

## 5. Limits

- Binary positive/negative sentiment over Booking.com review text. **No production
  accuracy claim**: these are development splits of a public dataset.
- **Qwen2.5-0.5B QLoRA is out of scope.** It needs a GPU and Hugging Face Hub access.
  Its legacy claim — statistically indistinguishable from the scratch LoRA, McNemar
  p = 0.53 — stays unconfirmed on clean splits and keeps its legacy caveat in the README
  until a separate protocol covers it.
- **One seed.** The BiLSTM is trained once per view with seed 42. There is no
  seed-variation interval, so a difference smaller than seed noise cannot be separated
  by this run. The existing DistilBERT artifacts carry a second seed on the `time`
  split; the BiLSTM will not.
- **No tuning.** Hyperparameters are the committed `configs/bilstm.yaml`: hidden 128,
  1 layer, bidirectional, dropout 0.3, `last` pooling, embedding 128, batch 128, lr
  1e-3, weight decay 1e-4, grad clip 5.0, 6 epochs, early stopping patience 2 on dev
  macro-F1, `max_tokens` 220. This is a baseline reproduction, not a BiLSTM
  optimisation. Threshold tuning (the unmerged `experiment/bilstm-threshold-tuning`
  branch) is explicitly not part of it.
- **`min_freq` is pinned at its default, not at the configured value.**
  `src/reviewnlp/baselines/bilstm.py` reads `min_freq` from the `train:` section while
  `configs/bilstm.yaml` defines it under `data:`, so the configured key is ignored and
  the vocabulary is built with the code default of 2. Both happen to be 2 today, so the
  run is unaffected, but the pinned value here is the default. Changing the configured
  key would not change the run until that read is corrected.
- No selection is frozen and no candidate is promoted by this run.
  `selection_frozen` and `promoted` stay false.

## Execution

The split views cannot be rebuilt in an agent container: the 238 MB source CSV is not
committed and `kaggle.com` is refused by the session's network policy. Training therefore
runs at a **human gate**, on a machine that holds the source CSV, or in Colab. The
artifacts each run must produce, matching the shape
`docs/experiments/results/distilbert_v2/<view>/<model>/` already uses, are:
`run_config.yaml`, `metrics.json` (including the `data` fingerprints this protocol
pins), `dev_logits.npy`, `dev_labels.npy`, `test_logits.npy`, `test_labels.npy` and
`train_log.txt`.
