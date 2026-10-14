# BiLSTM on the clean splits — version 3

**Fixed before any training run: versions 1 and 2 on 13 October 2026, version 3 on
14 October 2026. Status: locked; no run has been executed under any version (see the
version history at the end). Scope: development evidence; no selection is frozen and no
model is promoted.**

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
recall; the random-to-time drop; training wall time; the two secondary analyses below.

### Secondary: three training seeds

The BiLSTM is trained with seed 42 on all three views, and also with seeds 43 and 44 on
the `time` view. These seeds are fixed here, before any run. **Seed 42 stays the primary
result**: the decision rule in section 4 is applied to it alone.

For seeds 43 and 44, the primary comparison is repeated against the same seed-42
DistilBERT full fine-tune, and each seed's branch of the rule is reported. Then:

| Seed check | What the README says |
| --- | --- |
| all three seeds land in the same branch | the conclusion holds across three training seeds of the BiLSTM |
| any seed lands in another branch | the conclusion depends on the BiLSTM's seed: the seed-42 conclusion is stated with that caveat and all three branches |

The mean and standard deviation of the BiLSTM's macro-F1 over the three seeds are also
reported. Neither can change the primary conclusion.

### Secondary: speed, size and the measured frontier

Four models are timed in one session, on one CPU, with the method of
[`latency.json`](../results/distilbert_v2/latency.json):
- the same 500 `time` test reviews (sample seed 0);
- one PyTorch thread, one review at a time, text to label with tokenization included;
- 3 warm-up calls, then the median and 95th percentile per review.

The four are TF-IDF + Naive Bayes, the BiLSTM (seed 42), the DistilBERT full fine-tune
and the scratch LoRA merged into it. A DistilBERT timing may use any checkpoint of that
architecture trained in the session: the architecture, not the weight values, sets the
time. The BiLSTM's size is its saved weights plus its vocabulary file; the others' size
is as `latency.json` defines it.

Each model is a point: its macro-F1 on the `time` test (seed 42 for the neural models)
and its median CPU latency. Let `[lo, hi]` be the paired 95% interval of macro-F1(A) −
macro-F1(B). Model A dominates model B when:

| Dominance condition |
| --- |
| `lat_a <= lat_b and hi >= 0 and (lat_a < 0.9 * lat_b or lo > 0)` |

That is, A is no slower, not significantly less accurate, and either at least 10% faster
or significantly more accurate. A timing gap under 10% counts as a tie, because timings on
a shared cloud CPU vary. The measured frontier is the set of models that no other model
dominates. The README may call a model "on the measured frontier" only by this rule. The
frontier describes this CPU and this method; it does not change the primary conclusion.

## 4. Decision rule

Let `d` be the macro-F1 difference (BiLSTM − DistilBERT) on the `time` test set and
`[lo, hi]` its 95% paired bootstrap interval.

The rule is the repository's own, from
[`docs/experiments/phase2_analysis_plan.md`](../phase2_analysis_plan.md): a gain is
**meaningful** when the paired interval excludes zero *and* the difference is at least
1 macro-F1 point (0.01); a gain that meets only the first condition is **clear but
small**. Applied to this pair:

| Condition | Conclusion |
| --- | --- |
| `hi < 0 and d <= -0.01` | DistilBERT is meaningfully better on clean out-of-time data. The legacy ordering is confirmed and the README keeps its transformer claim, now on uncontaminated data. |
| `hi < 0 and d > -0.01` | DistilBERT is better, but the gain is clear but small. The ordering is confirmed; the README must state the gap as under one point. |
| `lo <= 0 <= hi` | The two are indistinguishable on this split. The legacy claim that the transformer beats the BiLSTM is **not confirmed on clean data**, and the README must say exactly that. |
| `lo > 0 and d < 0.01` | The BiLSTM is better, but the gain is clear but small. The legacy ordering is reversed by under one point; the README's transformer claim must be restated. |
| `lo > 0 and d >= 0.01` | The BiLSTM is meaningfully better. The legacy ordering is reversed; the README's transformer claim must be restated. |

This is applied exactly as written once results exist, with no discretion. The other two
views are reported alongside and cannot change the conclusion. If the rule turns out to
be the wrong question for this comparison, the decision note says so in those words —
it is never reinterpreted silently.

## 5. Limits

- Binary positive/negative sentiment over Booking.com review text. **No production
  accuracy claim**: these are development splits of a public dataset.
- **Qwen2.5-0.5B QLoRA is out of scope.** It is not carried forward for binary scoring
  ([decision, 14 October 2026](../decision_qwen_binary_scope.md)). Its legacy claim —
  statistically indistinguishable from the scratch LoRA, McNemar p = 0.53 — stays
  unconfirmed on clean splits and keeps its legacy caveat in the README.
- **Three seeds on the `time` view, one elsewhere.** Seed 42 decides; seeds 43 and 44 show
  whether its branch depends on the seed. Three seeds give a rough spread, not an
  interval. The `random` and `hotel` views are trained once.
- **The DistilBERT reference is one seed.** All four DistilBERT runs on the clean splits
  used seed 42. Whether DistilBERT's own results move with the seed is the subject of a
  [separate protocol](../distilbert_seed_robustness/PROTOCOL.md).
- **The frontier is one CPU's.** Timings depend on the processor and its load. The
  frontier holds for this session's CPU and this method only.
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
`train_log.txt`. Seeds 43 and 44 add the same files under `time/bilstm_seed43/` and
`time/bilstm_seed44/`. The session adds a `latency.json` with the four models' CPU
timings and sizes, the CPU model, the thread count and the package versions.

## Version history

| Version | Commit | Change |
| --- | --- | --- |
| 1 | `3ddb358` | Locked with a three-branch rule on the interval alone. |
| 2 | `334eb82` | The rule omitted the analysis plan's 1-point magnitude criterion, so it could not tell a meaningful gain from a clear but small one. Amended to the plan's two-condition rule. **No run had been executed under version 1**, so no result could have influenced the change. |
| 3 | this commit | Added, before any run, seeds 43 and 44 on the `time` view (seed 42 stays primary) and CPU latency and size for a measured frontier. Both are reported beside the primary rule and cannot change it. Corrected version 2's limits, which said the DistilBERT artifacts had another seed on the `time` split: all four DistilBERT runs used seed 42. **No run had been executed under version 2.** |
