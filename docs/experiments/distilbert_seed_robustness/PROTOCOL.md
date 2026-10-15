# DistilBERT across training seeds — version 1

**Fixed before any training run, 14 October 2026. Status: locked; no run has been executed
under it. Scope: development evidence; no selection is frozen and no model is promoted.**

All four DistilBERT runs on the clean splits used training seed 42. The paired bootstrap
intervals in the [decision note](../decision_distilbert_vs_nb.md) cover the sampling of test
reviews, not the randomness of training. The full fine-tune leads the scratch LoRA by under
one macro-F1 point, which is small enough for training randomness to matter. This protocol
fixes, before any run, how that is checked.

## 1. Question

Does the full fine-tune's lead over the scratch LoRA on the out-of-time test hold across
three training seeds?

## 2. Data

The `time` view of the clean splits, as recorded in
`docs/experiments/results/v2_splits/time/data_manifest.json` and pinned in the
[BiLSTM protocol](../bilstm_clean_splits/PROTOCOL.md):

| Item | Value |
| --- | --- |
| raw CSV sha256 | `a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9` |
| raw CSV bytes | 238,154,765 |
| split seed | 42 |

| View | Split | Rows | Negative | Positive | sha256 (first 16) |
| --- | --- | ---: | ---: | ---: | --- |
| time | train | 116547 | 24692 | 91855 | `74f447be7e776135` |
| time | dev | 14612 | 4018 | 10594 | `5ddec23ec9bb0bbf` |
| time | test | 13921 | 3921 | 10000 | `07429dd59e7cea22` |

**The data may not change after this commit.** `scripts/compare_split_models.py::check_same_test`
rejects a run whose test fingerprint differs from `docs/experiments/results/split_views.json`.

## 3. One primary comparison

**Full fine-tune versus scratch LoRA, macro-F1, on the `time` view's 13,921 test reviews,
within each of three training seeds: 42, 43 and 44.** For each seed `s`:
- `d_s` = macro-F1(full fine-tune, seed `s`) − macro-F1(scratch LoRA, seed `s`);
- `[lo_s, hi_s]` is its paired bootstrap interval over the same test rows, 2000 resamples,
  seed 0.

The seed-42 runs are the committed phase 2 artifacts in
`docs/experiments/results/distilbert_v2/time/`. Their recorded values:

| Seed | d (full − LoRA) | 95% paired interval |
| --- | ---: | --- |
| 42 | 0.0068 | [0.0041, 0.0094] |

Seeds 43 and 44 are new runs. They keep every setting of the seed-42 runs except `seed`:
`configs/distilbert_v2_time.yaml` and, for the LoRA, the recorded `lora_scratch_config.json`.
The seeds are fixed here, before any run.

Reported but **unable to overturn the rule**:
- each run's macro-F1, and the mean and standard deviation per method over the three seeds;
- each full fine-tune seed against TF-IDF + Naive Bayes. This is the extra seed that the
  [phase 2 plan](../phase2_analysis_plan.md) already provides for. It is reported beside the
  seed-42 result, which stays the primary result of that plan;
- each LoRA seed against Naive Bayes, and training time and GPU memory.

## 4. Decision rule

Let `d` and `lo` be the lists of the three seeds' differences and interval lower bounds.

| Condition | Conclusion |
| --- | --- |
| `all(x > 0 for x in lo)` | The lead holds across seeds: in every seed the full fine-tune is better and its interval excludes zero. The README says so. The size is "meaningful" if every `d_s` is at least 0.01, otherwise "clear but small". |
| `all(x > 0 for x in d) and any(x <= 0 for x in lo)` | The full fine-tune is ahead in every seed, but in at least one seed the difference cannot be told apart from zero. The README says the lead is about the size of the variation between seeds. |
| `any(x <= 0 for x in d)` | No consistent lead: in at least one seed the LoRA scores as high or higher. The README withdraws "clear but small" for the clean splits and says the two are not separated across seeds. |

This is applied exactly as written once results exist, with no discretion. The
comparisons with Naive Bayes are reported beside it and cannot change it. If the rule
turns out to be the wrong question, the decision note says so in those words.

## 5. Limits

- Binary positive/negative sentiment over Booking.com review text. **No production
  accuracy claim**: these are development splits of a public dataset.
- **Three seeds.** They give a rough spread, not an interval over seeds. A fourth seed
  could land elsewhere.
- **One split.** Only the `time` view, on which the phase 2 decision was made. The random
  split is not re-run.
- **GPU training is not bit-reproducible.** With fp16 on a T4, a seed fixes the
  initialisation and the data order, not every floating-point sum. Seed 42 is not re-run.
- **No tuning.** The settings are the phase 2 settings.
- **The phase 2 decision stands on seed 42.** This protocol cannot change the
  DistilBERT-versus-Naive-Bayes conclusion; the other seeds are reported beside it.
- No selection is frozen and no candidate is promoted. `selection_frozen` and `promoted`
  stay false.

## Execution

Training runs at a **human gate**, in Colab on a T4: the agent container cannot rebuild the
split views. Each new run writes the files the seed-42 runs hold (`run_config.yaml`,
`metrics.json` with the `data` fingerprints and `settings.seed`, `dev_logits.npy`,
`dev_labels.npy`, `test_logits.npy`, `test_labels.npy`, `train_log.txt`). They go under
`docs/experiments/results/distilbert_v2/time/`, in `distilbert_seed43/`,
`distilbert_lora_scratch_seed43/`, `distilbert_seed44/` and `distilbert_lora_scratch_seed44/`.
`scripts/compare_split_models.py` recomputes every comparison from the committed logits.

## Version history

| Version | Commit | Change |
| --- | --- | --- |
| 1 | this commit | Locked before any run. |
