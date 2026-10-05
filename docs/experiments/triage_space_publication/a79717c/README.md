# Experimental G Space — publication on 5 October 2026

Public demo: [krimits/hotel-triage-demo](https://huggingface.co/spaces/krimits/hotel-triage-demo).
Uploaded Space commit:
[`a79717ceab2755658e01ce56aaf7a442ab5db82c`](https://huggingface.co/spaces/krimits/hotel-triage-demo/tree/a79717ceab2755658e01ce56aaf7a442ab5db82c).
Source: [`f2da186bd63be84a2debe9e6fb3047e14d846ea1`](https://github.com/krimits/hotel-review-nlp/commit/f2da186bd63be84a2debe9e6fb3047e14d846ea1),
published through [notebook 18](https://github.com/krimits/hotel-review-nlp/blob/b2110fc2d4bd3267bf635aab3665d607dbadb9bf/notebooks/18_publish_triage_space_colab.ipynb).

**Experimental snapshot of G — not selected, not promoted, quality unvalidated.**
The user executed the publication notebook with their own HF write Secret. These records contain
no credentials, human annotations, reference outputs or model weights. They are not uploaded to the Space.
Existing sentiment and ABSA Spaces remain separate.

## Original Colab evidence

The five original ZIP members are archived byte for byte; `smoke.json` is named `colab_smoke.json` here.
The [record manifest](record_manifest.json) records their SHA-256, the input ZIP's SHA-256 and the
independent verification file's SHA-256. The exact source-manifest SHA-256 is
`9c8c91eff7dde84e20005f1302f2a732684176acc1b7c26d2c385924a13a200a`.

- [source_manifest.json](source_manifest.json): 19 source files, prompts, category mapping, model
  revisions, routing, limits, decoding and precision policy. The Hub has these files plus this manifest
  and its `.gitattributes`, with no review dataset or annotation bundle.
- [colab_smoke.json](colab_smoke.json) and [smoke_execution.txt](smoke_execution.txt): real pinned weights
  on CUDA/float32, Python 3.13.15. All 14 authored synthetic cases exercised issue extraction with Jev off.
  The functional check passed; seven cases had visible quote/evidence/JSON failures and partial results.
- [published.json](published.json) and [publish_execution.txt](publish_execution.txt): verified upload,
  matching loaded snapshot and two live functional requests (praise and complaint) on ZeroGPU/bfloat16,
  Python 3.12.12. Both completed their stages and were not stored.

The different runtimes/precision are preserved rather than treated as one experiment. Per-request
durations are observations for these requests, not a latency benchmark or a quality score.

## Independent verification

[verification.json](verification.json) records the separate post-publication check of the public
Hub commit, file allowlist, all 20 file hashes, loaded snapshot and live API behavior.
The source hashes also match the archived source Git commit. The Space uses ZeroGPU and retains
the explicit experimental status; Jev is off and these checks make no external Jev requests.

The praise request completed with no issues or measures. The lamp complaint completed with one
issue and one proposed measure. The known `dev-07` input returned a partial result with
`evidence_quote_missing_or_invalid`, visibly named as a stage failure in the UI summary. All three
requests reported `stored: false` and zero Jev attempts. These are functional observations, not
independent human quality judgments or a replacement for the development comparison.

## What this establishes and what remains

This establishes a published source snapshot and functional deployment. It does not establish correct
interpretation, comprehensive issue coverage, correct departments or useful operational measures.
An exact excerpt's presence alone does not prove it supports a claim. The seven known development
failures remain; completed stages are not automatically correct outputs.

No candidate was selected or promoted, and no reserved evaluation was opened. Still pending:
mobile layout, first request after sleep, GPU quota/timeout behavior, Jev opt-in, a recorded 15–20-case
usability pilot and an independent real whole/mixed-review reliability pilot. GPU billing is unknown
and is not inferred from request duration. See the [deployment procedure](../../../TRIAGE_SPACE_DEPLOYMENT.md).
