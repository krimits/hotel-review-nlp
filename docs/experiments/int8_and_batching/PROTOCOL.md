# Dynamic INT8 and CPU batching for the published DistilBERT — version 1

**Fixed before any run, 15 October 2026. Status: locked; no run has been executed under it.
Scope: development evidence on one CPU; no selection is frozen and no model is promoted.**

The README carries two caveats that this protocol replaces with a measurement:
- dynamic INT8 was tried on 32 reviews only, with a size estimator that counted parameters
  rather than the serialized weights;
- on a CPU, batches of 32 were slower per review than one review at a time, which padding
  to the longest review in a batch would explain.

## 1. Question

On one CPU thread, does dynamic INT8 quantization keep the published DistilBERT's macro-F1
within one point, and is it meaningfully faster?

Reported beside it, unable to change the answer: whether batches sorted by length beat one
review at a time.

## 2. Data

The `random` view of the clean splits, as recorded in
`docs/experiments/results/v2_splits/random/data_manifest.json`:

| Item | Value |
| --- | --- |
| raw CSV sha256 | `a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9` |
| raw CSV bytes | 238,154,765 |
| split seed | 42 |

| View | Split | Rows | Negative | Positive | sha256 (first 16) |
| --- | --- | ---: | ---: | ---: | --- |
| random | test | 13263 | 3263 | 10000 | `8cd21cdac358f10b` |

**The data may not change after this commit.** The run stops unless the rebuilt test set's
fingerprint equals the one in `docs/experiments/results/split_views.json`.

The model is the published random-split full fine-tune, `krimits/distilbert-hotel-reviews` at
Hub revision `7306aebcaaebc00d579f5d0a91001ae376f18158`, whose `model.safetensors` has sha256
`f2a2bbc9a31c8f3b1d9bee46ec11bb3d8942e7e59c0ef4711c3cb2a6c1cf81c2`. Its test logits from
training are committed in `docs/experiments/results/distilbert_v2/random/distilbert/`
(macro-F1 0.9642). **Integrity gate:** the FP32 model, run on the CPU in the session, must give
the argmax of those committed logits on at least 99% of the test reviews, or the run stops.

## 3. One primary comparison

**INT8 versus FP32, macro-F1, on all 13,263 test reviews.** Both are the same checkpoint run
on the CPU in the same session: FP32 as loaded, INT8 after
`torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)`.
`d` = macro-F1(INT8) − macro-F1(FP32), with its paired bootstrap interval `[lo, hi]` over the
same test reviews, 2000 resamples, seed 0 (`paired_bootstrap_diff`).

Speed is the median time per review at batch size 1, text to label with tokenization
included, on one PyTorch thread, after 3 warm-up calls, on the 500 test reviews drawn with
seed 0: the method of `latency.json` and of the BiLSTM frontier. `lat_int8` and `lat_fp32`
are those medians.

Reported but **unable to overturn the rule**:
- the serialized size of each model (`torch.save` of its `state_dict`);
- the share of test reviews on which INT8 and FP32 agree, and each class's recall;
- **batching**, for FP32 and INT8 on the same 500 reviews, in ms per review: batch 1, batches
  of 32 in test order, and batches of 32 after sorting the 500 by token length. A sorted
  batching is called faster only if its time per review is under 0.9 times batch 1's, and its
  labels must agree with batch 1's on at least 99% of the reviews.

## 4. Decision rule

| Condition | Conclusion |
| --- | --- |
| `lo > -0.01 and lat_int8 < 0.9 * lat_fp32` | INT8 keeps macro-F1 within one point and is meaningfully faster: recommended for single-review CPU serving. |
| `lo > -0.01 and lat_int8 >= 0.9 * lat_fp32` | INT8 keeps macro-F1 within one point but is not meaningfully faster: no reason to switch. |
| `lo <= -0.01` | INT8 may cost a meaningful point of macro-F1: not recommended. |

The one-point margin is the repository's own threshold for a meaningful difference
([phase 2 plan](../phase2_analysis_plan.md)). The 10% speed margin is the tie of the
[BiLSTM protocol](../bilstm_clean_splits/PROTOCOL.md), because timings on a shared cloud CPU
vary. This is applied exactly as written once results exist, with no discretion. If the rule
turns out to be the wrong question, the decision note says so in those words.

## 5. Limits

- Binary positive/negative sentiment over Booking.com review text. **No production
  accuracy claim**: a development split of a public dataset.
- **One CPU and one thread.** Timings depend on the processor and its load; the session's CPU
  is recorded. Batch 1 on one thread is the single-request case, not a throughput benchmark.
- **Dynamic quantization of the linear layers only.** No static quantization, no ONNX Runtime,
  no quantization-aware training.
- **The random split.** The out-of-time view has no published checkpoint; its models were not
  kept.
- **One checkpoint, seed 42.** The quantization effect could differ for another seed.
- No selection is frozen and no candidate is promoted. `selection_frozen` and `promoted`
  stay false.

## Execution

At a **human gate**, in Colab on a T4 runtime, whose CPU is the one the existing timings used.
The notebook rebuilds the random view from the hash-checked raw file and downloads the model at
the pinned revision. It writes, under `docs/experiments/results/int8_v1/`:
`int8_benchmark.json` (fingerprints, hardware, timings, sizes, the comparison and the rule's
conclusion), `fp32_test_logits.npy`, `int8_test_logits.npy`, `test_labels.npy` and
`environment.json`, with a sha256 manifest and no model weights.

## Version history

| Version | Commit | Change |
| --- | --- | --- |
| 1 | this commit | Locked before any run. |
