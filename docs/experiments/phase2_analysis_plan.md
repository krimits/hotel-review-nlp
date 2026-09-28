# Phase 2: DistilBERT on the clean splits, analysis plan

**Status: fixed before training, then executed.**
- **When.** This plan was committed before any model of this phase was trained.
- **Executed.** The runs used commit `994a103`, which holds this plan. Their results are in
  the [decision note](decision_distilbert_vs_nb.md).
- **Record.** The notebook records the commit it ran from.
- **Changes.** Any later change is listed at the end, with the reason.

## Question

- Does DistilBERT beat the dev-selected TF-IDF + Naive Bayes on test sets that
  share no text with training? It is fine-tuned in full, or with the
  from-scratch LoRA.
- Does it hold up better on reviews written after the training period?
- Is the gain worth the cost?

The earlier DistilBERT results (0.9634 and 0.9573) were measured on a legacy
test set that shares 170 texts with training. They are not compared here.

## Data

The v2 splits built by `make data` from the hash-checked raw file:

| Split | Train | Dev | Test | Test reviews written |
|---|---:|---:|---:|---|
| Random | 117,880 | 14,734 | 13,263 | August 2015 – August 2017 |
| Out-of-time | 116,547 | 14,612 | 13,921 | 26 May – 3 August 2017 |

The notebook refuses to train unless each split's fingerprints equal those in
[`split_views.json`](results/split_views.json), which holds the Naive Bayes
predictions. So every model is scored on the same test reviews, in the same
order.

## Models and settings

The settings are those of the legacy run, so only the data changes. No model is
tuned.

| | TF-IDF + Naive Bayes | DistilBERT, full fine-tune | DistilBERT, scratch LoRA |
|---|---|---|---|
| Source | `split_views.json`: chosen on dev among the classical candidates | [`configs/distilbert_v2.yaml`](../../configs/distilbert_v2.yaml), [`…_time.yaml`](../../configs/distilbert_v2_time.yaml) | the same configs, with `--r 8 --alpha 16 --lr 1e-4` |
| Settings | as recorded | `distilbert-base-uncased`, max length 256, batch 32, 2 epochs, lr 2e-5, 6% warm-up, weight decay 0.01, fp16, seed 42 | the same, except: rank-8 adapters on `q_lin` and `v_lin`, α = 16, lr 1e-4; the pre-classifier and classifier layers are trained too |
| Checkpoint | none | the epoch with the best dev macro-F1 | the same |

## Comparisons

**Primary comparison.** There is one, and it is fixed in advance.
- **Models.** DistilBERT (full fine-tune) against TF-IDF + Naive Bayes, on the
  out-of-time test.
- **Measure.** The difference in macro-F1 (DistilBERT minus Naive Bayes), with a
  paired bootstrap 95% interval:
  - 2,000 resamples of the test reviews;
  - both models scored on the same resampled reviews;
  - seed 0.

**Secondary comparisons:**
- **The same comparison** on the random test.
- **Scratch LoRA** against Naive Bayes, and against the full fine-tune, on both
  tests.
- **McNemar.** An exact McNemar test on the errors of each pair.
- **Drop over time.** Each model's drop from the random to the out-of-time test.
  - The two test sets hold different reviews.
  - So the drop is reported with each model's own bootstrap interval, and is not
    tested.

## What counts as a meaningful gain

A gain over another model is called meaningful when both of these hold:
- the paired 95% interval of the macro-F1 difference excludes zero;
- the difference is at least 1 macro-F1 point (0.01).

A gain that meets only the first condition is reported as clear but small.

## Cost

Cost is measured and reported, but it is not a criterion on its own.

- **Training:**
  - minutes on the Colab GPU;
  - peak GPU memory;
  - trainable parameters.
- **Size:** the model on disk.
- **Inference**, from text to label, on the out-of-time test
  ([`scripts/benchmark_latency.py`](../../scripts/benchmark_latency.py)):
  - **CPU latency** per review at batch 1 and at batch 32, on the same 500 test
    reviews (seed 0), for DistilBERT and for TF-IDF + Naive Bayes;
  - **the whole test set:** DistilBERT on the GPU, Naive Bayes on the CPU;
  - **hardware:** the script records it.
- **The timed models are the scored ones.** Before its times count, each model
  must give the predictions that were scored, on at least 99% of the reviews.
- **LoRA at inference.** The scratch LoRA is merged into the model before it is
  saved, so at inference it costs the same as the full fine-tune.

## Decision note

It is written after the results. It applies the rule above to each use and
weighs the costs. Two uses:
- scoring reviews in batches, such as a nightly job;
- answering one review at a time on a CPU.

## Limits

- **One seed (42).**
  - The bootstrap covers the sampling of test reviews, not the randomness of
    training.
  - The notebook can add a second seed for the primary comparison. That run is
    compared with Naive Bayes in the same way and reported beside the seed-42
    result, which stays the primary result.
- **No tuning** for any model.
- **Colab hardware.** Absolute times depend on it, so the ratios between models
  matter more than the times themselves.
- **A task near its ceiling.** Naive Bayes already reaches 0.935 on the random
  test and 0.918 on the out-of-time test.

## Outputs

- **`results/distilbert_v2/`**, for each run:
  - `metrics.json`;
  - dev and test logits and labels;
  - the run config and the training log (`train_log.txt`).

  Also `environment.json`, `latency.json` and `artifact_manifest.json`.
- **`results/model_comparison_v2.json`**, written by
  [`scripts/compare_split_models.py`](../../scripts/compare_split_models.py). The
  CI recomputes it from the committed logits.
- **Model weights** are not committed.

## Changes after fixing

None. The runs followed the plan as written.
