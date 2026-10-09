# Source-span formatting experiment — 6 October 2026

The [adjudicated development baseline](../../triage_space_publication/e1e37e5/human_review_v3/README.md)
has four issue-stage failures, five missed problems and thirteen problems without measures. This
experiment addresses the structural failures first. The deployed Space is unchanged.

## Reproduction with real pinned weights

[cpu_baseline_extraction.json](cpu_baseline_extraction.json) replays the four failed reviews with their
captured DistilBERT sentiment hints, the original v6 prompt and pinned Qwen revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. DistilBERT was not rerun and Jev was not called.
The raw stage-one output is retained because these inputs are verified authored synthetic development
reviews. Model execution used **CPU/float32, torch 2.14.1+cpu and transformers 4.56.2**;
this is not a reproduction of the hosted ZeroGPU/bfloat16 runtime or its discarded raw text.

| Case | Hosted error | Observed CPU output defect |
| --- | --- | --- |
| dev-01 | invalid_evidence_issue_schema | Unknown category `leaks_or_mechanical_noise` |
| dev-05 | invalid_evidence_span_id | Selects evidence span 2 although the review has only span 1 |
| dev-08 | invalid_json | Valid JSON with unknown category `noise`; the hosted syntax defect was not reproduced |
| dev-11 | invalid_evidence_issue_schema | Unknown category `leaks_or_mechanical_noise` |

All four inputs failed extraction on CPU; three had the same error code as the capture. A matching
error code is a local reproduction, not recovery of the original hosted cause.

## Controlled change

The new **development-only** `actions-v6-source-spans-json-v1` extractor restricts generation with
[`lm-format-enforcer 0.11.3`](https://github.com/noamgat/lm-format-enforcer). It permits only the
required JSON fields, existing status/category enums and integer IDs belonging to that review's source
spans. Null evidence and empty issue lists remain allowed. The **prompt, evidence parser, category to
department mapping, greedy decoding, 400-token budget and measures stage are unchanged**.
There is no repair, extra model call, automatic retry or keyword-based inference of facts.

[cpu_structured_extraction.json](cpu_structured_extraction.json) records the four paired stage-one
probes with the same model/runtime/signals. All four constrained outputs passed the existing strict
parser. The [manifest](probe_manifest.json) binds raw traces, executed diagnostic drivers and source
hashes; [structured_span_source.py.txt](structured_span_source.py.txt) preserves the experimental code
used by the probe. The recorded base commit precedes this addition; file hashes bind the new code.
The diagnostic drivers retain their original execution paths; use the portable runner below to replay.

**This is a formatting result, not a semantic fix or an action-quality score.** The latch remains
uncertain and is assigned to access/management; the cooling leak remains uncertain and is assigned to
cleanliness/housekeeping. The pump and shower are pending. Grammar-compliant IDs do not prove the
chosen span supports the status. The unchanged parser still rejects duplicates and downgrades
conflicting evidence. Truncation remains a visible workflow failure.

The subsequent [full CPU workflow capture](cpu_workflow/run.json), including the unchanged measures
stage, also completed: the plain arm had four workflow errors and no accepted actions; the constrained
arm had no workflow errors, two accepted actions (shower and pump) and two uncertain issues without
actions (latch and cooling leak). Its [results](cpu_workflow/results.jsonl) retain both stages and raw
synthetic-only text; [the executed runner](cpu_workflow_driver.py.txt) is preserved byte for byte. These
are execution counts, not human usefulness ratings. The eight-row sheets remain blank. This diagnostic
runner predates the portable runner's `failures` field for the scorer; the historical capture is
preserved unchanged and is not the new 48-row evaluation handoff.

## Portable paired workflow comparison

The [received Colab run](colab_incomplete_20261009/README.md) stopped on an ImportError with
transformers 5.18.0 instead of the specified 4.56.2. Its hashes pass, but 46 outputs were unexecuted;
the supplied sheets cannot be scored. The new runtime preflight verifies pinned distributions,
actual imports and library integration before downloading weights. When an isolated prefix is
specified it also verifies the interpreter and package origins. The corrected GPU run remains pending.

[Notebook 21](../../../../notebooks/21_triage_format_comparison_colab.ipynb) pins the tested source at
`139cdc52c09d80e87dcbfa73d67ef0fc7663df4d`, verifies source hashes and installs its dependencies in an
isolated Colab environment. It defaults to all 24 reviews, requires GPU, and downloads the diagnostic
ZIP even when the model capture fails. No token, provider key, Drive or previous ZIP is needed. Its
corrected GPU execution and new human ratings are pending; the Space remains unchanged. The notebook
uses isolated Python/pip, checks a model-free runtime receipt before generation and exports that receipt
even on failure. It cannot start with the observed incompatible/global environment.

```bash
pip install -r configs/triage_format_requirements.txt
python scripts/compare_triage_formats.py --cases all --device cuda \
  --output runs/triage_format_dev_NEW
```

Use a fresh output directory. For an explicitly requested CPU replay, use `--device cpu`; the default
requires CUDA and does not silently fall back. `--cases failures` targets the four captured failures.
Native precision uses bfloat16 only on a GPU with native support, otherwise float32; `--precision fp32`
requests float32. The actual hardware, precision, versions and attention implementation are recorded.
Matching precision alone does not reproduce the hosted runtime. GPU billing remains unknown.

The runner verifies runtime compatibility, captured-file hashes, the original authored dev corpus and the unchanged deployed
backend before downloading weights. Both arms share the same loaded weights and replayed hints, and
each review reaches extraction. Only issue-stage constrained decoding differs. Both arms use the
unchanged strict measures assembler, with at most two calls per review/arm. A generation exception stops
remaining calls, preserving explicit `not_executed` rows; parse/evidence failures remain visible outputs.
The runner refuses to overwrite evidence. It never reads reserved cases, human ratings or provider keys.

For the full comparison, 24 reviews produce 48 blinded rows in each fresh `coverage_A.csv` and
`coverage_B.csv`. Two reviewers fill them independently. Actual-problem counts must agree for the same
review across both arms. Do not reuse action/coverage judgments from different outputs or treat the
number of accepted actions as usefulness. Score with `scripts/score_triage_review.py` against this new
run, reconcile disagreements, then address semantic defects. There is no automatic winner or Space
update. [Release gates](../../../TRIAGE_RELIABILITY.md) still apply.
