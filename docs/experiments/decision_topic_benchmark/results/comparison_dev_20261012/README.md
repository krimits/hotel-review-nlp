# Three-arm primary-topic comparison: development only

**Actual Colab execution, reviewed 12 October 2026.** All three arms completed
the same 100 development reviews, with 300/300 successful records, no setup
errors and no retries. No test prediction, model promotion or Space change
was performed. Microsoft is a promising candidate for further evaluation,
not a selected production model.

Source commit `fe51104b73e0ad8c96ffe903a4ef5c63d1f16f39`; notebook 23 from
`83249852268507da66bc169422d5c3697dec8999`. Python 3.13.15; isolated Colab
environment, pinned package versions, Tesla T4; Qwen actually loaded in
`torch.bfloat16`. See the precision caveat below before using this freeze.

The [annotation record](../../annotation_record.json) predates these
predictions. The submitted input package passed the original identity and
completeness validator again. [verification.json](verification.json) records
an independent recomputation from the locked human labels and captured
probabilities: macro-F1, per-class F1/support, accuracy, ECE, Brier, log loss,
referrals, retained errors, p95 and provider-reported spend all agree.
It also checks source/configuration, capture/freeze hashes, exact dev pairs,
review fingerprints, model identities and the calibrated human queue.

| Measurement on 100 dev reviews | Logistic regression | Qwen2.5-1.5B | Microsoft Decision-1 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.6400 | 0.3300 | 0.6500 |
| Macro-F1, five classes | 0.1561 | 0.2299 | 0.6524 |
| Raw ECE, 10 fixed bins | 0.1528 | 0.2388 | 0.1483 |
| Dev-fitted ECE, in-sample | 0.1432 | 0.1188 | 0.0619 |
| Raw retained at max probability >=0.7 | 42 | 19 | 69 |
| Raw errors among retained | 6/42 | 9/19 | 20/69 |
| Dev-calibrated retained at >=0.7 | 47 | 0 | 52 |
| Dev-calibrated errors among retained | 8/47 | undefined, no retained cases | 12/52 |
| Dev-calibrated human referrals | 53 | 100 | 48 |
| Request p95, ms | 2.770 | 1279.553 | 668.567 |
| Provider-reported USD / 1,000 attempts | unknown local cost | unknown GPU cost | 0.0207564 |

The provider reported **USD 0.00207564 for these 100 Microsoft requests**;
all attempts have cost observations. USD/1,000 is arithmetic scaling of this
sample, not a forecast or a current tariff. The observed software budget was
USD 0.30. GPU/CPU billing and training costs remain unknown, not zero.

Dev class support is **cleanliness 1, service 18, location 14, value 3,
other 64**. Per-class F1:

| Class | Support | Logistic | Qwen | Microsoft |
| --- | ---: | ---: | ---: | ---: |
| cleanliness | 1 | 0 | 0 | 1.0000 |
| service | 18 | 0 | 0.3333 | 0.7442 |
| location | 14 | 0 | 0.4828 | 0.5652 |
| value | 3 | 0 | 0 | 0.2857 |
| other | 64 | 0.7805 | 0.3333 | 0.6667 |

The perfect cleanliness F1 comes from just one example; it is not robust
class-level evidence. All 100 logistic predictions are `other`, so its 64%
accuracy equals the majority-class baseline. Microsoft recovers 31 cases
that logistic misses but misses 30 that logistic gets right: the +1 accuracy
point does not establish a meaningful accuracy advantage. Its main dev
advantage is recognizing the named topics, reflected in macro-F1.
Microsoft labels 30/64 human-`other` reviews as one of the named topics;
Qwen labels 35/64 as `service`. The wide `other` category and main-topic
tie rule deserve an error review, without changing locked human labels to
agree with model predictions.

Temperatures 0.96662, 3.03176 and 1.30650 were fitted on these same dev
examples. Lower dev ECE is **in-sample calibration diagnostics**. It does
not validate confidence on new reviews or establish a safe 0.7 policy.
Microsoft still makes 12/52 errors (23.08%) among calibrated retained cases.
Qwen retains none; its undefined selective error must not be called zero.
Its full-vocabulary A-E mass is 0.99443--0.99987, so this run is not simply
normalizing an almost-zero option mass. This prompted next-token classifier
result does not measure Qwen's ability to generate hotel improvement actions.

## Precision and timing caveat

The historical [runtime_preflight.json](runtime_preflight.json) says
`native_bf16: true`, but the source calls `torch.cuda.is_bf16_supported()`.
That function defaults to `including_emulation=True`, which does not establish
native hardware support. T4 is Turing: its Tensor Core matrix input support
is FP16; native BF16 Tensor Core support was added with Ampere. The actual
Qwen capture is a bfloat16 path on T4, not the protocol's intended FP16
fallback for hardware without native BF16. The original files are preserved
unchanged and the mislabel is explicitly recorded here and in verification.

- [PyTorch BF16 support API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html)
- [NVIDIA's architecture precision table](https://docs.nvidia.com/cuda/archive/11.5.2/ampere-tuning-guide/index.html)

The measured p95 is valid for the path that actually ran. It is not a native
BF16 or corrected FP16 benchmark. Batch 1 timings include text processing and
probability validation, excluding model loading/warm-up. The Microsoft API
also includes network/provider time. These different deployment paths do not
establish an intrinsic architecture speed ranking.

The [freeze.json](freeze.json) matches the exact original source and captured
runtime. **Resolve precision policy before the final test.** Any corrected
source/precision requires a separately recorded dev run and new freeze;
never edit the historical capture or reuse its temperature as proof that a
different precision path was calibrated. The test remains unexecuted.

## Scope and next decision

This is one primary topic irrespective of sentiment, on reviews from two
NYC hotels in 2008--2009. Reviewer independence remains unverified. It is not
a current-traffic, unseen-hotel, negative-complaint, evidence-extraction or
action-generation evaluation. In particular, `other` includes maintenance,
noise, food and amenities; it must not mean “no hotel action needed”.

1. Preserve this dev result and inspect the topic/tie-rule errors on dev only.
2. Resolve the native BF16 check and validate the intended Qwen precision
   in a new dev configuration before touching the held-out test.
3. Freeze the configuration and run the 100-review test once, with explicit
   owner API opt-in/budget; report support and uncertainty with its scores.
4. Consider optional Space integration only after that decision. Complaint
   coverage, exact evidence, pending/resolved status, department assignment
   and usefulness of hotel measures still need their own human evaluation.

Archived captures contain IDs, model probabilities, timings and usage;
no review text, annotation sheets, per-review human gold or credentials.
`human_queue.csv` is a model-referral list, not another labeling task.
