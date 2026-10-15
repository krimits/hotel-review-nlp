# Decision: the BiLSTM on the clean splits

**15 October 2026. The rule of the [BiLSTM protocol, version 3](bilstm_clean_splits/PROTOCOL.md),
applied as written to runs made after it was locked. Scope: development evidence on a public
dataset; no selection is frozen and no model is promoted.**

## Question

Does the BiLSTM baseline's standing against the full DistilBERT fine-tune survive on a split
that shares no review text with training, specifically on reviews written after the training
period?

## Answer

**Yes. DistilBERT is meaningfully better on clean out-of-time data.** The legacy ordering is
confirmed, now on a test set with no text overlap.

| Out-of-time test, 13,921 reviews | Value |
| :--- | :--- |
| BiLSTM, seed 42 | macro-F1 0.9357 (95% CI 0.9310–0.9400) |
| DistilBERT, full fine-tune, seed 42 (phase 2 run) | macro-F1 0.9491 (0.9450–0.9531) |
| d = BiLSTM − DistilBERT | −0.0135, paired 95% interval [−0.0169, −0.0099] |
| Branch of the rule | `hi < 0 and d <= -0.01`: DistilBERT meaningfully better |
| McNemar (exact) | 274 reviews only DistilBERT gets right, 117 only the BiLSTM; p = 1.3 × 10⁻¹⁵ |

The rule's one-point condition is on d (−0.0135), not on the interval. The interval's end
nearest zero, −0.0099, is just under one point, so the gap could be slightly smaller than a point.

## Secondary results (cannot change the answer)

**Three training seeds of the BiLSTM.** All three land in the same branch, so: *the conclusion
holds across three training seeds of the BiLSTM.*

| Seed | Macro-F1 | d against DistilBERT | Paired 95% interval | Branch |
| ---: | ---: | ---: | :--- | :--- |
| 42 | 0.9357 | −0.0135 | [−0.0169, −0.0099] | DistilBERT meaningfully better |
| 43 | 0.9363 | −0.0128 | [−0.0163, −0.0093] | DistilBERT meaningfully better |
| 44 | 0.9372 | −0.0119 | [−0.0153, −0.0085] | DistilBERT meaningfully better |

Mean 0.9364, standard deviation 0.0008. Three seeds give a rough spread, not an interval.

**The three views, seed 42.** The BiLSTM is meaningfully better than TF-IDF + Naive Bayes on
each of them.

| View | BiLSTM macro-F1 (95% CI) | Accuracy | BiLSTM − Naive Bayes (paired 95% interval) |
| :--- | :--- | ---: | :--- |
| Random | 0.9486 (0.9445–0.9528) | 96.13% | +0.0135 [0.0094, 0.0178] |
| Out-of-time | 0.9357 (0.9310–0.9400) | 94.72% | +0.0173 [0.0131, 0.0216] |
| Unseen hotels | 0.9514 (0.9471–0.9556) | 96.37% | +0.0166 [0.0122, 0.0211] |

- **Later reviews.** The BiLSTM loses 1.29 points from random to out-of-time, against 1.67 for
  Naive Bayes and 1.51 for DistilBERT. The test sets hold different reviews, so these drops are
  not paired comparisons.
- **Negative reviews, out-of-time.** Recall 0.930, precision 0.888.
- **Legacy figure.** The legacy BiLSTM row, 0.9503, was measured on a test set that shares 170
  texts with training. It is not comparable with these.
- **Training.** About 2.1–2.4 minutes per run on a T4, early stopping on dev macro-F1.

**Speed, size and the measured frontier.** Four models, timed in the same Colab session on one
CPU (Intel Xeon @ 2.00 GHz), one PyTorch thread, one review at a time, text to label, on the same
500 out-of-time test reviews.

| Model | Macro-F1, out-of-time | Median ms per review | Size |
| :--- | ---: | ---: | ---: |
| TF-IDF + Naive Bayes | 0.9184 | 1.24 | 3.4 MB |
| BiLSTM | 0.9357 | 1.67 | 8.4 MB |
| DistilBERT, scratch LoRA (merged) | 0.9424 | 54.1 | 255.4 MB |
| DistilBERT, full fine-tune | 0.9491 | 56.4 | 255.4 MB |

By the protocol's dominance rule, no model dominates another, so **all four are on the measured
frontier.** For the BiLSTM: about 1.35 times Naive Bayes' time for a meaningful 1.7-point gain,
and about one thirtieth of DistilBERT's time for 1.35 points less.

**Where the rule gives the wrong answer.** The scratch LoRA is on the frontier only because it
timed 4% faster than the full fine-tune (54.1 against 56.4 ms). The protocol itself calls a gap
under 10% a tie. But the rule demands `lat_a <= lat_b`, so a model that is significantly more
accurate and 4% slower cannot dominate. Merged, the LoRA has the full fine-tune's architecture
and size, and it is significantly less accurate (−0.0068, [−0.0094, −0.0041]). On this evidence
it offers no speed advantage at inference, and its place on the frontier is an artifact of
timing noise and of how the rule was written. The rule is not changed here; this note says so
in those words, as the protocol requires.

## What this decision does not claim

- **Production accuracy.** These are development splits of a public Booking.com dataset.
- **That the BiLSTM was tuned.** These are the committed `configs/bilstm.yaml` settings; it is a
  baseline reproduction, not an optimisation.
- **Anything about Qwen.** It is out of scope ([decision](decision_qwen_binary_scope.md)).
- **Speed on other hardware.** The frontier holds for this session's CPU and this method.
- **That DistilBERT's own seed does not matter.** The reference is the seed-42 run; its seed
  variation is a [separate check](decision_distilbert_seeds.md).

## How it was run

- Notebook 27, one Colab T4 session, pinned to commit `ed4070c`. It rebuilt the three views and
  checked their fingerprints against `split_views.json` before training.
- The bundle came back as `seeds_and_bilstm_results.zip`. Every one of its 78 files matched its
  SHA-256 manifest. Both comparisons were recomputed locally from the logits and matched the
  bundle's JSON field by field. CI recomputes them on every push
  ([`test_seeds_and_bilstm_results.py`](../../tests/test_seeds_and_bilstm_results.py)).
- **Departures from the protocol's wording, none affecting the rule.**
  - The timings are in `latency_frontier.json`, not `latency.json` as the protocol's Execution
    section names it, so as not to overwrite the phase 2 timings.
  - DistilBERT was timed with the seed-43 checkpoints of the session, as the protocol allows; its
    accuracy in the frontier is seed 42's.
  - Naive Bayes was retrained in the session only to be timed. It agreed with the scored
    predictions on 498 of the 500 timed reviews; the script's floor is 99%.

## Artifacts

- [`results/bilstm_v2/`](results/bilstm_v2/): the five BiLSTM runs (`metrics.json`, dev and test
  logits and labels, `run_config.yaml`, `train_log.txt`), `bilstm_comparison_v2.json`,
  `latency_frontier.json`, `environment.json` and `artifact_manifest.json`.
- [`docs/RUNS.md`](../RUNS.md), entry `20261015-seeds-and-bilstm-results`.
