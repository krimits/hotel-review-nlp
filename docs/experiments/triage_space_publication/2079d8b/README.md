# Local lexicon primary Space update — 13 October 2026

Public demo: [krimits/hotel-triage-demo](https://huggingface.co/spaces/krimits/hotel-triage-demo).
Space commit: [`2079d8bee12d63012896b6ea754910f169a945c2`](https://huggingface.co/spaces/krimits/hotel-triage-demo/tree/2079d8bee12d63012896b6ea754910f169a945c2).
Source: [`02d6e8c7b7297bd87b4060b8f9f504421e60244c`](https://github.com/krimits/hotel-review-nlp/commit/02d6e8c7b7297bd87b4060b8f9f504421e60244c),
published by the owner using [notebook 25](https://github.com/krimits/hotel-review-nlp/blob/d37491752b68c9708728905b6f3ef341586696ba/notebooks/25_update_triage_lexicon_space_colab.ipynb)
and their own Colab HF Secret.

**Experimental G-source-spans — not selected, not promoted, quality unvalidated.**
The complaint-topic primary is `complaint-lexicon-v1`; Jev enhancement is disabled.
Qwen still performs source-span issue extraction and suggested measures for every review.
Unknown topics require human review. The lexicon's precision, recall and routing accuracy
have not been measured on independent whole-review ground truth.

## Original execution evidence

The six ZIP members are preserved byte for byte, with `smoke.json` renamed to
[colab_smoke.json](colab_smoke.json). [record_manifest.json](record_manifest.json) binds
their individual hashes, the original ZIP hash and the independent verification record.
No review labels, private reviews, credentials or model weights were added to the Space.

- [source_manifest.json](source_manifest.json): 21 allowlisted source files, pinned model revisions,
  prompt/decoding/precision policy and lexicon fingerprint
  `e8311382893a280ffec11e87c0c922af20ffc77e141841a691b2e058a106445c`.
- [colab_smoke.json](colab_smoke.json) and [smoke_execution.txt](smoke_execution.txt): all 16 authored
  functional inputs completed, including seven required development regressions. Lexical primary,
  Jev opt-out and human referral were verified for each input. Real weights ran on CUDA/float32,
  Python 3.13.15. The eight known development inputs were executed without independent semantic ratings.
- [published.json](published.json) and [publish_execution.txt](publish_execution.txt): upload verified,
  exact loaded snapshot matched and all seven hosted regressions passed on ZeroGPU/bfloat16,
  Python 3.12.12. The two public Variables were set to
  `REVIEWNLP_TRIAGE_LEXICON_FALLBACK=1` and `REVIEWNLP_JEV_ENABLED=0`.

These are functional development checks, not a reliability estimate. Times are individual request
observations, not a latency benchmark. No reserved evaluation was opened. No provider request was made,
and no review was stored by the app. Hosting/GPU billing remains unknown; provider opt-out does not
establish zero infrastructure cost or absence of infrastructure processing.

## Setup failure and repair

[setup_execution.txt](setup_execution.txt) retains the initial failed attempts as well as the successful
repair. Colab's Python lacked `ensurepip`, leaving a venv without pip. After pip was bootstrapped,
the template's `--no-deps` editable installation omitted six declared core dependencies, and
`pip check` correctly rejected it. The owner completed a normal editable installation; the final
dependency check, GPU probe, real-weight smoke and publication then passed.

The current notebook template creates a venv with `--without-pip`, uses the parent pip's `--python`
option, and resolves the pinned Space requirements and project dependencies together. Offline regression
tests execute this installation path in fresh and partially created isolated environments using local
distribution metadata. This template correction does not alter the published source snapshot.
The successful publication does not need to be repeated solely for this correction.

## Independent verification and quota observation

[verification.json](verification.json) independently verifies all 22 deployed file hashes
(21 source files plus the manifest), their frozen Git provenance, the loaded source snapshot,
the lexicon fingerprint, Jev disabled state and ZeroGPU hardware at the same Space commit.
The final Hub read reported `RUNNING`.

The anonymous API repeat completed praise, the pending-lamp complaint and the mixed lamp/comfortable-bed
case. Each passed its evidence/action regression and lexical-primary policy checks, with zero Jev
attempts and no storage. The next short-lamp request returned `partial`, no issues/actions and the
visible stage failure `qwen_runtime: gpu_quota_exceeded`; Qwen generation did not run for that request.
The checker stopped there. The runtime reports quota exhaustion; account-specific quota, reset time
and authenticated-client availability were not inspected. The owner's seven successful hosted checks
and this later anonymous quota failure are separate observations.

Thus `snapshot_verified` and `local_primary_verified` are true, while the independent
`functional_checks_passed` and `passed` remain false. The failed request is retained rather than
converted into a semantic model failure or a successful regression. No paid hardware fallback,
secret change or repeated GPU request was attempted after the quota failure.

## Next step

Run the [offline whole-review operational-topic pilot](../../triage_lexicon_pilot/PROTOCOL.md):
freeze and sample private English whole reviews, obtain two independent annotation sheets,
record agreement before adjudication, then measure per-topic precision/recall and referral workload.
The historical five-way primary-topic benchmark and these authored smoke inputs do not replace those labels.
All reviews currently require human referral because unsupported `other` remains unknown.

Still pending: an independent real whole/mixed-review pilot for the complete chain and measure quality,
mobile/cold-start checks, optional Jev opt-in and systematic authenticated/anonymous quota behavior.
See the [deployment procedure](../../../TRIAGE_SPACE_DEPLOYMENT.md).
