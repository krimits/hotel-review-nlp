# Decision: dynamic INT8 and CPU batching for the published DistilBERT

**15 October 2026. The rule of the [INT8 and batching protocol, version 1](int8_and_batching/PROTOCOL.md),
applied as written to a run made after it was locked. Scope: one CPU, development evidence on a
public dataset; no selection is frozen and no model is promoted.**

## Question

On one CPU thread, does dynamic INT8 quantization keep the published DistilBERT's macro-F1
within one point, and is it meaningfully faster?

## Answer

**Yes to both. INT8 keeps macro-F1 within one point and is meaningfully faster: recommended for
single-review CPU serving.**

| Random test, 13,263 reviews | FP32 | INT8 |
| :--- | ---: | ---: |
| Macro-F1 | 0.9642 | 0.9637 |
| Negative recall | 0.9531 | 0.9510 |
| Median time per review, batch 1, one thread | 51.5 ms (p95 84.7) | **17.2 ms** (p95 50.5) |
| Serialized size | 255.5 MB | 132.3 MB |

- **Accuracy.** d = macro-F1(INT8) − macro-F1(FP32) = −0.0005, paired 95% interval
  [−0.0024, 0.0013]. The interval includes zero and lies far inside the one-point margin.
- **Agreement.** The two models give the same label on 13,182 of 13,263 reviews (99.4%).
- **Speed.** INT8 is 3.0 times faster at batch 1, well past the protocol's 10% tie.
- **Branch.** `lo > -0.01 and lat_int8 < 0.9 * lat_fp32`.

## Secondary results (cannot change the answer)

**Batching, on the same 500 reviews and one thread, in ms per review:**

| | Batch 1 (median) | Batch 32, test order | Batch 32, sorted by length |
| :--- | ---: | ---: | ---: |
| FP32 | 51.5 | 93.0 | **25.1** — faster |
| INT8 | 17.2 | 82.4 | 21.3 — not faster |

- **Unsorted batches are slower than single reviews** for both models: padding every review to
  the longest in its batch costs more than batching saves. This is the README's old observation,
  now measured.
- **Sorting by length fixes it for FP32**, which runs twice as fast per review as at batch 1.
- **For INT8 it does not:** sorted batches cost about the same per review as single reviews.
  The quantized model's per-review cost is already low, so batching has little left to save.
- The verdicts are the same whether batch 1 is taken as its median or its mean per review
  (FP32 55.6, INT8 21.9 ms).
- **Sorting assumes the reviews are available together**, as in offline scoring. An API
  answering one request at a time cannot sort; there, batch 1 is the relevant number.

**Consistency with earlier figures.** The phase 2 latency of 57 ms per review was measured on
the out-of-time checkpoint, which has the same architecture, in another session; 51.5 ms here is
in line with it. The old 32-review estimate (1.38 times faster) timed batches of 32 with a
different method and is superseded.

## What this decision does not claim

- **Production readiness or accuracy.** One development split of a public dataset.
- **Other hardware.** One Colab CPU (Intel Xeon @ 2.00 GHz) and one thread. More threads, other
  processors or other quantization backends can change both speed and ratio.
- **Other quantization methods.** Only dynamic quantization of the linear layers. The embeddings
  stay in FP32, which is why the size halves rather than quarters. No static quantization, no
  ONNX Runtime, no quantization-aware training.
- **Other checkpoints.** One seed-42 checkpoint, on the random split.
- **That the Space or API now uses INT8.** Nothing was changed in serving.

## How it was run

- Notebook 28, one Colab T4 runtime (only its CPU was used), pinned to commit `93dfdff`.
  - It rebuilt the random view from the hash-checked raw file and checked its fingerprint.
  - It downloaded the model at `7306aeb` and checked the weights hash.
- **Integrity gate.** The FP32 model on the CPU gave the committed test logits' label on all
  13,263 reviews (the protocol requires at least 99%).
- **The bundle.** It came back as `int8_and_batching_results.zip`.
  - All 5 files matched their manifest.
  - The comparison, both macro-F1 values, the agreement and the rule were recomputed locally
    from the logits and matched.
  - CI recomputes them on every push ([`test_int8_results.py`](../../tests/test_int8_results.py)).
- **One detail of the timings.** The INT8 model timed on one thread gave the full-pass label on
  498 of the 500 sampled reviews: the full pass ran on two threads, and quantized kernels can round
  differently across thread counts. Both are above the 99% floor.

## Artifacts

- [`results/int8_v1/`](results/int8_v1/): `int8_benchmark.json`, `fp32_test_logits.npy`,
  `int8_test_logits.npy`, `test_labels.npy`, `environment.json`, `artifact_manifest.json`.
- [`docs/RUNS.md`](../RUNS.md), entry `20261015-int8-and-batching-results`.
