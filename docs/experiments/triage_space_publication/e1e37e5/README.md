# Reliability update and fresh development capture — 6 October 2026

Space: [krimits/hotel-triage-demo](https://huggingface.co/spaces/krimits/hotel-triage-demo),
commit [`e1e37e58bc963cc8843ecdf4a0096266c16f6f7e`](https://huggingface.co/spaces/krimits/hotel-triage-demo/tree/e1e37e58bc963cc8843ecdf4a0096266c16f6f7e).
Source: [`321c20689418c45267122f537ea6451f3ef30417`](https://github.com/krimits/hotel-review-nlp/commit/321c20689418c45267122f537ea6451f3ef30417),
workflow `actions-v6-source-spans`, through [notebook 20](../../../../notebooks/20_triage_reliability_colab.ipynb).
**Experimental, not selected or promoted; development judgments reconciled, independent quality evaluation pending.**
The [previous release](../ea8c7f8/README.md), including its GPU failure, remains unchanged.

## Publication and integrity

The user-supplied ZIP members are preserved byte for byte; only `smoke.json` is renamed
[colab_smoke.json](colab_smoke.json). [record_manifest.json](record_manifest.json) binds the original
ZIP hash, all input-member hashes and the independent records. The [verification](verification.json)
compares the published source files with both the ZIP and the pinned Git bytes. No additional Qwen or
Jev inference request was made by this verification.

- All 16 Colab functional cases completed with real pinned weights on CUDA/float32.
- Upload and loaded-runtime checks passed, followed by seven required hosted regressions on
  ZeroGPU/bfloat16. [published.json](published.json) records these owner-run assertions.
- The [fresh development capture](review/run.json) returned all 24 authored cases, with Jev off.
  Source-manifest and capture-artifact hashes match. No reserved cases or real customer data were used.

A source match and these functional cases do not establish semantic quality or general availability.
`qwen_device: cpu` outside the GPU allocation is a runtime observation, not proof of CPU-only inference.
Per-request times are observations, not a comparable latency benchmark. GPU billing remains unknown.

## Machine findings and blocking questions

[Machine findings](machine_findings.json) count **20 complete and 4 partial workflows**, five accepted
measures and eleven UNCERTAIN issues without measures. No GPU-stage failure or Jev call occurred in the
captured development run. `execution_complete: true` means all planned requests returned; four returned
workflow errors. The error cases remain in the denominator and human sheets.

| Development case | Blind ID | Recorded issue-stage error |
| --- | --- | --- |
| dev-01 | 0014 | invalid_evidence_issue_schema |
| dev-05 | 0010 | invalid_evidence_span_id |
| dev-08 | 0017 | invalid_json |
| dev-11 | 0006 | invalid_evidence_issue_schema |

These cannot be attributed to a particular raw JSON defect: the Space intentionally clears raw Qwen
output. Reproduce them with the frozen prompts/model and recorded precision before proposing a parser
change; do not accept invalid IDs or invent missing evidence merely to make execution pass.

Developer inspection also found a measure for the successfully replaced bulb (dev-21 / 0020), and an
UNCERTAIN issue for breakfast praise (dev-18 / 0008). Several explicit complaints were left UNCERTAIN.
These are review priorities, **not independent human labels**. No automatic quality score or new
promotion decision was created. Old F/G ratings do not apply to these new outputs.

## Two human reviewers

The original [coverage_A.csv](review/coverage_A.csv) and [coverage_B.csv](review/coverage_B.csv) each
contain 24 rows with blank judgments. Keep them and this source bundle intact. Two people independently
save completed copies elsewhere; preserve IDs, review text, assessments, measures and execution flags.

Fill all eight count columns with nonnegative integers, including explicit `0`. Fill the four quality
columns with `1` or `0`; when no measure exists, `grounded`, `department_correct` and `no_invented_facts`
may be `na`. `useful=1` for an empty output requires a justified abstention, not a missed actual problem.
A workflow error cannot be rated fully useful. `useful_action_count` counts measures correct in all four
dimensions; an exact substring alone does not establish grounding. Explain decisions in `notes`.
Save UTF-8 CSV with commas as delimiters.

[Column definitions and release gates](../../../TRIAGE_RELIABILITY.md) remain the scoring protocol.

```bash
python scripts/score_triage_review.py \
  --run docs/experiments/triage_space_publication/e1e37e5/review \
  --ratings-a completed_A.csv --ratings-b completed_B.csv \
  --output runs/triage_e1e37e5_scored.json
```

The scorer rejects the original blank sheets. The separately archived
[completed v3 sheets and supplied adjudication](human_review_v3/README.md) now pass: all fifteen decisions
match both sheets and their model outputs are unchanged. Of seventeen actual problems, thirteen lack
an appropriate measure; four of five emitted measures are useful. Four issue-stage failures remain.
These are adjudicated synthetic development findings; reviewer independence is unverified.
The scorer checks immutable outputs, logical counts and every disagreement without averaging them away.
Correct development failures, select/freeze a supported workflow, and
then perform reserved evaluation and the independent real whole/mixed-review pilot before promotion.
Jev opt-in, mobile layout, long inputs and first-request-after-sleep checks remain pending.
