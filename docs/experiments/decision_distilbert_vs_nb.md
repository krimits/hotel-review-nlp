# DistilBERT or Naive Bayes for hotel reviews? Decision note

**Status: written after the results.**
- **Rules.** It applies the rules of the [analysis plan](phase2_analysis_plan.md), which
  was committed before any model was trained.
- **Runs.** The runs, their logs and their logits are in
  [`results/distilbert_v2/`](results/distilbert_v2/).
- **Numbers.** Every number below is in
  [`model_comparison_v2.json`](results/model_comparison_v2.json) or
  [`latency.json`](results/distilbert_v2/latency.json). The CI recomputes the comparison from
  the logits.

## The answer

- **DistilBERT, fine-tuned in full, meets the rule fixed before training.**
  - On reviews written after the training period, it beats TF-IDF + Naive Bayes by
    **3.1 macro-F1 points**: 0.9491 against 0.9184, with a paired 95% interval of 2.6 to 3.5
    points.
  - It makes 578 errors on these 13,921 reviews, against 930: 38% fewer.
- **On the random test the gain is about the same:** 2.9 points, with an interval of 2.5 to
  3.4.
- **It does not hold up better over time.**
  - All three models lose about the same from the random to the out-of-time test: 1.5 to 1.7
    points.
  - DistilBERT keeps its lead, but the lead does not grow.
- **The scratch LoRA is 0.7 points below the full fine-tune** on both tests. The difference is
  clear but small. The LoRA still beats Naive Bayes by more than 2 points.
- **The cost is speed and size on a CPU.**
  - One review takes 57 ms instead of 1.2 ms (median, one CPU thread).
  - The model is 255 MiB instead of 3.4 MiB.
- **What to use:**
  - **scoring reviews in batches:** DistilBERT, fine-tuned in full;
  - **answering one review at a time on a CPU:** DistilBERT if about 0.1 s per review is
    acceptable; Naive Bayes if the answer must take a few milliseconds, or one CPU must serve
    many requests.

## Results

Macro-F1 with a 95% bootstrap interval (2,000 resamples of the test reviews, seed 0).

| Model | Random test: macro-F1 | Accuracy | Errors | Out-of-time test: macro-F1 | Accuracy | Errors |
|---|---|---:|---:|---|---:|---:|
| TF-IDF + Naive Bayes, chosen on dev | 0.9351 (0.9303–0.9398) | 95.14% | 644 | 0.9184 (0.9133–0.9230) | 93.32% | 930 |
| DistilBERT, full fine-tune | **0.9642** (0.9605–0.9677) | 97.33% | 354 | **0.9491** (0.9450–0.9531) | 95.85% | 578 |
| DistilBERT, scratch LoRA | 0.9576 (0.9537–0.9613) | 96.83% | 421 | 0.9424 (0.9380–0.9465) | 95.28% | 657 |

- **The two test sets.**
  - The random test holds 13,263 reviews (3,263 negative) like the training ones.
  - The out-of-time test holds 13,921 reviews (3,921 negative), written from 26 May to
    3 August 2017, after every training review.
- **Checkpoint and scoring.**
  - Each DistilBERT run kept the epoch with the best dev macro-F1.
  - Every model was scored once on each test, on the same reviews in the same order.

## Paired comparisons

The models are compared on the same reviews.
- **Difference.** The macro-F1 difference comes with a paired bootstrap interval.
- **McNemar.** It counts the reviews that only one of the two models gets right, and tests
  them with an exact McNemar test.
- **Verdict.** A gain is **meaningful** when its interval excludes zero and it is at least one
  point (0.01). If only the first holds, it is **clear but small**.

| Test | Comparison | Difference (95% interval) | Only the first right | Only the second right | McNemar p | Verdict |
|---|---|---|---:|---:|---:|---|
| **Out-of-time** | **DistilBERT full vs Naive Bayes** (primary) | **+0.0308** (0.0264 to 0.0350) | 500 | 148 | 1.5 × 10⁻⁴⁵ | **meaningful** |
| Out-of-time | Scratch LoRA vs Naive Bayes | +0.0240 (0.0198 to 0.0283) | 459 | 186 | 1.5 × 10⁻²⁷ | meaningful |
| Out-of-time | Scratch LoRA vs DistilBERT full | −0.0068 (−0.0094 to −0.0041) | 80 | 159 | 3.6 × 10⁻⁷ | full better, clear but small |
| Random | DistilBERT full vs Naive Bayes | +0.0291 (0.0246 to 0.0335) | 387 | 97 | 4.9 × 10⁻⁴² | meaningful |
| Random | Scratch LoRA vs Naive Bayes | +0.0225 (0.0183 to 0.0269) | 348 | 125 | 2.2 × 10⁻²⁵ | meaningful |
| Random | Scratch LoRA vs DistilBERT full | −0.0066 (−0.0091 to −0.0039) | 58 | 125 | 8.1 × 10⁻⁷ | full better, clear but small |

**The drop over time** is shown below.
- **Not a paired comparison.** The two test sets hold different reviews, so the drop is not
  tested (as the plan fixed).
- **About the same for all three models:**

| Model | Random | Out-of-time | Drop |
|---|---:|---:|---:|
| TF-IDF + Naive Bayes | 0.9351 | 0.9184 | 0.0167 |
| DistilBERT, full fine-tune | 0.9642 | 0.9491 | 0.0151 |
| DistilBERT, scratch LoRA | 0.9576 | 0.9424 | 0.0152 |

## Negative reviews

For a hotel, the negative reviews are the ones to act on. On the out-of-time test:

| Model | Negative recall | Negative precision | Negative reviews missed | Positive reviews flagged as negative |
|---|---:|---:|---:|---:|
| TF-IDF + Naive Bayes | 0.900 | 0.868 | 393 | 537 |
| DistilBERT, full fine-tune | **0.941** | **0.914** | **233** | **345** |
| DistilBERT, scratch LoRA | 0.936 | 0.900 | 250 | 407 |

- **Fewer misses and fewer false alarms.** DistilBERT misses 160 fewer of the 3,921 negative
  reviews, and raises 192 fewer false alarms.
- **The random test agrees:**
  - negative recall 0.913 for Naive Bayes and 0.953 for DistilBERT;
  - precision 0.892 and 0.939.

## Cost

**Hardware:**
- **Training:** a Colab Tesla T4.
- **CPU times:** the same Colab machine, an Intel Xeon at 2.00 GHz with 2 vCPUs; PyTorch used
  one thread.
- **What the timings mean.** They are single measurements on this hardware. The ratios between
  models matter more than the times themselves.

| | TF-IDF + Naive Bayes | DistilBERT, full fine-tune | DistilBERT, scratch LoRA |
|---|---:|---:|---:|
| Training time | the fit takes under 1 s (0.61 s); the TF-IDF features were not timed | 13.6 min (random), 13.3 min (out-of-time) on a T4 | 9.0 min, 8.9 min: a third less |
| Peak GPU memory in training | — | 2,543 MiB | 1,558 MiB: 39% less |
| Trainable parameters | — | 66,955,010 (100%) | 739,586 (1.10%) |
| Size on disk | 3.4 MiB | 255.4 MiB | 255.4 MiB merged; adapters and head alone 2.8 MiB |
| CPU, one review at a time: median (95th percentile) | 1.2 ms (1.6 ms) | 57 ms (124 ms) | 57 ms (115 ms) |
| CPU, batches of 32: time per review | 0.08 ms | 135 ms | 134 ms |
| Whole out-of-time test (13,921 reviews) | 0.5 s on the CPU | 55 s on the T4 (254 reviews/s) | 61 s on the T4 (228 reviews/s) |

**How to read three of these numbers:**
- **LoRA costs the same as the full fine-tune at inference.** It is merged into the model
  before saving, so the two have the same architecture and size.
  - Their GPU times still differ by 11%.
  - That gap is the variation of a single measurement, not a property of LoRA.
- **Batches of 32 were slower per review than single reviews on one CPU thread.**
  - The time was 135 ms per review in a batch, against 65 ms on average one at a time.
  - Each batch is padded to its longest review.
  - Sorting reviews by length before batching was not tried.
- **The timed Naive Bayes was trained again, only to be timed.**
  - It gives the prediction that was scored on 13,906 of the 13,921 reviews (99.9%), above
    the plan's 99% floor.
  - The comparisons use the predictions scored in phase 1.
  - The cause of the 15 differences was not traced.

## What to use when

**Scoring reviews in batches, such as a nightly job: DistilBERT, fine-tuned in full.**
- **The gain.** It passes the rule fixed before training, and it cuts errors by 38% on later
  reviews.
- **The cost is small at this volume.** A hotel group receiving thousands of reviews a day
  would need:
  - about 40 s for 10,000 reviews on a T4;
  - about 11 minutes on one CPU thread (one review at a time: 15 reviews/s).

**Answering one review at a time on a CPU: it depends on the time allowed and the load.**
- **DistilBERT:**
  - 57 ms per review (median), 124 ms at the 95th percentile;
  - about 15 reviews per second per CPU thread.
  - If that fits, it is the better model.
- **Naive Bayes:**
  - 1.2 ms per review.
  - It is the choice when the answer must be near-instant, or one small CPU must serve many
    requests.
  - The price is 3.1 points of macro-F1 on later reviews: 930 errors instead of 578 in the
    out-of-time test.

**Full fine-tune or LoRA: the full fine-tune, for this task.**
- **Why the full fine-tune.** It is 0.7 points better, and costs the same to run.
- **When LoRA pays off:**
  - **Training memory is tight.** LoRA trains in a third less time with 39% less memory.
  - **Many task variants must be stored.** They can be kept as 2.8 MiB adapters instead of
    255 MiB models; running unmerged adapters was not measured.

## Limits

- **One training seed (42).**
  - The intervals cover the sampling of test reviews, not the randomness of training.
  - The legacy run is consistent with these results, but it is not a replication:
    - it used another split;
    - it scored 0.9634 for the full fine-tune and 0.9573 for the LoRA;
    - it found a gap of 0.61 points between them.
- **No tuning for any model.**
  - Naive Bayes was chosen on dev among four classical candidates.
  - DistilBERT used the settings of the legacy run.
- **The task excludes mixed reviews.**
  - The labels come from reviews with only positive or only negative text; mixed reviews
    were left out.
  - Each test also has at most 10,000 positive reviews.
  - How the gain carries to mixed reviews is not known.
- **Timings.**
  - They are single measurements on Colab hardware.
  - Batching on a CPU was not tuned: one thread, and no sorting by length.
- **Legacy numbers.** The earlier results were 0.9634 for the full fine-tune and 0.9573 for
  the LoRA.
  - They are close to the clean ones (0.9642 and 0.9576).
  - This suggests the 170 texts shared between train and test in the legacy split did not
    inflate them much.
  - The test sets differ, so this is not a formal comparison.

## Reproduce

From the repository root, without a GPU:

```bash
python scripts/compare_split_models.py      # recomputes results/model_comparison_v2.json from the logits
pytest tests/test_phase2_results.py         # the checks that run in CI
```

To train again, run
[`notebooks/10_distilbert_v2_colab.ipynb`](../../notebooks/10_distilbert_v2_colab.ipynb) on a
Colab GPU, in about an hour.
