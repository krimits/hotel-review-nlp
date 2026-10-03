# Staged generator: verified run and offline parser replay

The [original notebook 16 run](runs/20261003T145901Z_a29e8d68/run.json)
used code 7aa6bf14afb419add1953afa256ccc2a18a9c3f1, a T4 and the same pinned
Qwen 1.5B checkpoint for C and F. All 48 candidate/review pairs completed,
without runtime failures or exhausted generation budgets. The 24 synthetic
development texts and cached upstream results were unchanged. There were no
new Jev calls. Human quality has not been evaluated.

The ZIP SHA-256 is cd72112990aaaa205aa39cad25cca8db6b85a899c19221ecb6fa690e372b022d.
[manifest.json](runs/20261003T145901Z_a29e8d68/manifest.json) hashes all nine
original files. Every byte is preserved, including the historical summary's errors.

| Original recorded result | C | F |
|---|---:|---:|
| Reviews attempted | 24 | 24 |
| Reviews with accepted actions | 16 | 0 |
| Accepted actions | 22 | 0 |
| Workflow failures | 3 | 9 |
| Issue / measure calls | n/a | 24 / 5 |

These are execution and structural counts, not precision, recall or usefulness.
C has one 400-token call; F has up to two. This is a workflow comparison, not an
equal-budget prompt experiment.

## Two demonstrated implementation defects

1. **Five complete JSON fences were rejected.** F's second stage ran on
   dev-01, dev-02, dev-03, dev-10 and dev-14. Every response contained one complete
   JSON object inside a Markdown JSON fence. Its action had a valid issue ID,
   measure and to_confirm list, but the staged parser attempted to decode the
   opening fence as JSON.
2. **Rejected outputs vanished from syntax diagnostics.** C has three
   duplicate-action-key outputs, dev-04, dev-15 and dev-23. They were correctly
   rejected, but the historical summary counted duplicate keys only among
   successful workflows and reported zero. It similarly counted 15 literal
   whole-JSON responses for F instead of 19, excluding four syntactically valid
   issue objects rejected by schema or quote checks.

The parser now accepts a bare whole object or one complete JSON/unlabelled
Markdown fence. It never scans for an arbitrary JSON fragment. Prose outside the
fence, multiple objects, incomplete fences, duplicate keys, non-JSON constants,
unsupported fields, invalid issue IDs and missing literal quotes remain failures.
The actual model text is kept unchanged in stages[].raw. Literal whole-JSON
diagnostics still describe the original text; accepting a wrapper does not make
that text a bare JSON object. Syntax diagnostics include rejected workflows.

## Replay is not a new GPU run

The tests replay **all 24 saved F stage sequences**, without model inference,
new API requests, prompt changes or changes to model classifications.
They recover exactly the five fenced measures and retain the four issue-stage
failures: paraphrased quotes in dev-09 and dev-13, contradictory duplicate
excerpts in dev-11, and an invalid status in dev-12.

This proves format compatibility and linkage, not that five measures are good.
For example dev-01 assigns a water/leak issue to housekeeping, and its measure
assumes checking water valves is an appropriate response. Both need operational
review. The original run still has zero accepted F actions; the five recovered
actions belong only to the labelled offline replay.

## Semantic problems remain

Among the valid first-stage results, seven clearly reported complaints were
excluded: dev-04, dev-05, dev-08, dev-15 and dev-16 were called hypothetical;
dev-06 and dev-07 were called resolved. Ants appearing during the stay are not
imagined; carrying suitcases upstairs does not establish that staff repaired
the lift; replacing cold eggs with cold eggs does not resolve the complaint.
The issue-stage failures contain additional status and department mistakes.

F should not become the demo's default on these results. C remains a baseline,
also requiring fresh human review; more accepted actions do not establish better
quality. Continue the Space UI and integration work with explicit error,
abstention and human-review states. Keep selection, reserved evaluation and the
real-review pilot separate.

To reproduce the archive checks, wrapper replay and diagnostic correction:

    PYTHONPATH=src pytest tests/test_recorded_staged_run.py tests/test_staged_generator.py -q

The [original blank rating sheet](runs/20261003T145901Z_a29e8d68/human_review.csv)
must not be silently rewritten with replayed actions or historical ratings.
