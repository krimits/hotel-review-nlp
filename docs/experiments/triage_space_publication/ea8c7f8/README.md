# Source-span Space update — 5 October 2026

Public demo: [krimits/hotel-triage-demo](https://huggingface.co/spaces/krimits/hotel-triage-demo).
Space commit: [`ea8c7f86cf431585528226486b83f5872c87d7b8`](https://huggingface.co/spaces/krimits/hotel-triage-demo/tree/ea8c7f86cf431585528226486b83f5872c87d7b8).
Source: [`5fa1f5635f8df4d537a6d7adbdcb4b22e8ad3bac`](https://github.com/krimits/hotel-review-nlp/commit/5fa1f5635f8df4d537a6d7adbdcb4b22e8ad3bac),
workflow `actions-v6-source-spans`, published using corrected
[notebook 19](https://github.com/krimits/hotel-review-nlp/blob/44a944c68f13d8d9b6ba761af30f27a809d3bbd3/notebooks/19_update_triage_evidence_space_colab.ipynb).
The user ran it using their own HF write Secret. The [earlier publication](../a79717c/README.md) and
[local fault reproduction](../source-spans-v6/README.md) remain separate historical records.

**Experimental G-source-spans — not selected, not promoted, quality unvalidated.**
This update fixes quote copying by resolving model-selected integer ids to original review slices.
An authentic slice can still be irrelevant or misinterpreted; measures require human confirmation.

## Original execution evidence

The five input ZIP members are preserved byte for byte, with `smoke.json` renamed to
[colab_smoke.json](colab_smoke.json). The [record manifest](record_manifest.json) records their hashes,
the original ZIP hash and the independent verification hash. These records stay in Git; the Space
contains only source files and its source manifest, without labels, reference outputs, weights or keys.

- [source_manifest.json](source_manifest.json): 20 allowlisted source files, pinned model revisions,
  prompt fingerprint, category mapping, routing, limits, decoding and precision policy.
- [colab_smoke.json](colab_smoke.json) and [smoke_execution.txt](smoke_execution.txt): all 16 authored
  functional cases completed without stage errors on CUDA/float32, Python 3.13.15. Seven required
  development regressions passed, including both reported lamp inputs. The eight known development
  cases were executed but were not independently rated for semantic correctness.
- [published.json](published.json) and [publish_execution.txt](publish_execution.txt): upload verified,
  loaded snapshot matched, and all seven required hosted regressions passed on ZeroGPU/bfloat16,
  Python 3.12.12. The lamp cases passed the pending-issue, literal-evidence and maintenance-measure
  requirements; praise, resolved and hypothetical cases produced no measure.

Jev was disabled for all these checks. The server reports Jev enabled, but this does not establish a
successful provider call. No review was stored. Durations are per-request observations, not a latency
benchmark. The ignored sampling-parameter warning does not change the recorded `do_sample=False` setting.

## Independent verification and availability finding

[verification.json](verification.json) confirms the public Hub commit, all 21 deployed file hashes,
the exact source Git bytes, loaded snapshot and ZeroGPU hardware. The final Hub read was `RUNNING`.

The independent API repeat completed praise, the unfixed-lamp complaint and the mixed lamp/comfortable
bed case. Each passed its required regression assertions with Jev off and no storage. Their full model
outputs were not retained; their counts and completed assertions are recorded as observations.

The next short-lamp request failed the independent checker. A captured repeat returned `partial`,
`qwen_runtime: gpu_unavailable_or_timeout`, no issues or actions, and a visible GPU-stage failure in the
UI summary. The Qwen generation stage did not execute. The check stopped there rather than claiming
seven independent successes. During investigation the Hub briefly reported `RUNNING_APP_STARTING`;
the final read returned `RUNNING` at the same commit. The API groups GPU unavailability and timeout;
these observations do not establish whether quota, worker availability or another runtime condition
caused the failure. No paid fallback or hardware change was attempted.

Therefore the receipt establishes successful publication and the initial functional run, while the
independent record has `snapshot_verified: true`, `functional_checks_passed: false`, `passed: false`.
The availability failure must remain visible in the record; deployment success is not availability
reliability or semantic accuracy.

## Remaining work

GPU availability and quota/timeout behavior need further investigation through the owner's runtime
logs/session. Still pending: Jev opt-in provider check, mobile layout, first request after sleep,
recorded 15–20-case usability pilot and an independent real whole/mixed-review reliability pilot.
No reserved evaluation was opened, no candidate was promoted, and no production-quality claim is made.
GPU billing is unknown. See the [deployment procedure](../../../TRIAGE_SPACE_DEPLOYMENT.md).
