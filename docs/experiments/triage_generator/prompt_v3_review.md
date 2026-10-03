# Recorded actions-v3 diagnosis

The user supplied the notebook 15 archive `generator_prompt_v3_dev_20261003T121429Z_876f0781.zip`.
Its nine original files are preserved [byte for byte](runs/20261003T121429Z_876f0781/manifest.json).
Run SHA-256: `f22c0dcd6da69e27aa112497ae8305d4fac5d5e2bc79c7e8d0c42fbca358640e`. The supplied rating-sheet SHA-256 is
`d56048e1d65fd62de3d0abb59ef9a52639424bfffb6414a4dab69f8f7f5a353c`; that externally supplied sheet is not an independent gold set.

The archive verifies 24 synthetic development reviews, C/E on the same Qwen
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, code `0057baaa0d6d2d4941cb1e97a8f8e43e9e6cf747`,
400-token greedy generations, identical reused upstream and zero new Jev attempts.
Every exported review and accepted-action list matches the supplied 48-row ratings.
The actual annotation key confirms the earlier reconstructed C/E mapping.

## Structural results are not usefulness

| Candidate | Reviews with historical accepted actions | Entire JSON object, unique keys | Outputs with repeated keys | Budget hits | Runtime errors |
|---|---:|---:|---:|---:|---:|
| C / actions-v2 | 18 / 24 | 17 / 24 | 3 / 24 | 0 | 0 |
| E / actions-v3 | 5 / 24 | 12 / 24 | 12 / 24 | 0 | 0 |

The production parser of that recorded revision interpreted repeated JSON keys
with last-value-wins semantics. This is why parser_json_valid was 24/24 for
both candidates despite the duplicate keys. Whole-document validity is
structural only: it does not establish the requested schema or useful advice.

Of the 12 E outputs graded as empty missed complaints, nine contain a proposed
measure followed by a second `actions: []`, and two contain a proposed measure
followed by an unrelated/unsupported second actions list that was dropped.
One truly outputs an empty actions list: it explicitly discounts blocked
accessible entry because the review is otherwise positive. Thus 11/12 empty
results involve malformed ambiguous output and last-value-wins parsing,
not simply failure to recognize any issue. Some first lists also contain wrong
departments or poor measures; retaining the first list cannot be assumed correct.

The supplied human ratings count five fully correct nonempty C outputs and two
E outputs. These are judgments about **what the old pipeline exposed**, not
ratings of the hidden first lists and not an issue-level recall estimate.
Overall totals pooled over A–D in the earlier experiment should not be compared
as if they came from one model or a new independent review sample.

## Remediation and unresolved judgments

The current action parser rejects duplicate keys in decoded objects, including
objects recovered from prose, rather than salvaging one competing value. The
triage pipeline therefore reports invalid_output, partial and actions_failed.
[Regression tests](../../../tests/test_recorded_prompt_v3.py) replay all 15
ambiguous C/E outputs through that guard; historical files and counts stay intact.
The usual action quote normalization and production prompt/model stay as before.

The supplied sheet overwrote all 48 execution_issue cells with no. That does
not establish valid JSON. The new review audit restores machine columns from
the intact original in a separate sheet. It converts yes/no/n/a unambiguously
but leaves 18 partial judgments across
13 rows for adjudication.
It makes no semantic regrading, winner selection or automatic deployment.

Next, [notebook 16](../../../notebooks/16_triage_staged_comparison_colab.ipynb)
compares the updated-parser C baseline with the same-weight two-stage F workflow.
The extraction and measure stages have separate strict schemas, literal issue
quotes and issue-ID linkage, with actual raw generations and timings retained.
This is a hypothesis with up to twice the generation budget, not a verified
improvement. New ratings, a later reserved evaluation and an independently
annotated real-review pilot remain necessary before a reliable Space release.
