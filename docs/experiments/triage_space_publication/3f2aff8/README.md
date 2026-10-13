# Cached-examples Space update — 13 October 2026

Public demo: [krimits/hotel-triage-demo](https://huggingface.co/spaces/krimits/hotel-triage-demo).
Space commit: [`3f2aff8c1e0920ac0c052937d69dac9920a3d380`](https://huggingface.co/spaces/krimits/hotel-triage-demo/tree/3f2aff8c1e0920ac0c052937d69dac9920a3d380).
Source: [`c65e638c0eccb7979c3d2a1fa48d0ce5e06735fc`](https://github.com/krimits/hotel-review-nlp/commit/c65e638c0eccb7979c3d2a1fa48d0ce5e06735fc)
(merge of #46), published by the owner using
[notebook 26](https://github.com/krimits/hotel-review-nlp/blob/2dab9bdc11c104b3eeb85893910fc13b0db53711/notebooks/26_update_triage_examples_space_colab.ipynb)
and their own Colab HF Secret.

**Experimental G-source-spans — not selected, not promoted, quality unvalidated.**
Relative to the [previous publication](../2079d8b/README.md), only two of the 21 source files changed:

- `app.py`: the two authored examples are lazily cached. The first click on an example runs the models once,
  and later clicks replay the stored answer without GPU. The label and the answer say so. A GPU or runtime
  failure is raised and never cached.
- `requirements.txt`: `spaces==0.51.3`, the version every recorded publication ran.

The lexicon (`complaint-lexicon-v1`, `e8311382…`), prompts, model revisions, decoding and the local-primary,
Jev-off policy are unchanged.

## Original execution evidence

The six ZIP members are preserved byte for byte, with `smoke.json` renamed to
[colab_smoke.json](colab_smoke.json). [record_manifest.json](record_manifest.json) binds their individual
hashes, the original ZIP hash and the verification record below.

- [setup_execution.txt](setup_execution.txt): all nine setup commands exited 0. This is the first publication
  with the corrected notebook template, which needed no repair.
- [colab_smoke.json](colab_smoke.json) and [smoke_execution.txt](smoke_execution.txt): all 16 authored or
  development functional inputs completed, including the seven required regressions. Real weights ran on
  CUDA/float32 with Python 3.13.15 and `spaces` 0.51.3. Lexical primary and Jev opt-out were verified
  for every input.
- [published.json](published.json) and [publish_execution.txt](publish_execution.txt) record:
  - the upload was verified;
  - the loaded source snapshot equals the archived manifest;
  - all seven hosted regressions passed on ZeroGPU;
  - the two public Variables are unchanged (`REVIEWNLP_TRIAGE_LEXICON_FALLBACK=1`, `REVIEWNLP_JEV_ENABLED=0`).

These are functional development checks, not a reliability estimate. No reserved evaluation was opened,
no provider request was made and no review was stored.

## Independent verification and its limits

[verification.json](verification.json) records what was checked independently of the owner's run.

**Checked:**
- **Source.** The package rebuilt from a clean checkout of the source commit has exactly the archived
  manifest's file hashes and runtime snapshot.
- **Hub.** At the Space commit:
  - the file set equals the package (21 source files plus the manifest);
  - every file's size equals the source;
  - `requirements.txt`, read in full, has the source hash.

**Not checked:**
- **File hashes on the Hub** beyond `requirements.txt`. The Hub tool available to the verifier returns
  sizes, not hashes.
- **The live API.** The verifier's network proxy refuses `huggingface.co`. The loaded snapshot and the
  hosted regressions therefore rest on the owner's receipt alone.
- **Cached examples on the live Space.** The publisher calls `/analyze`, not the example path. The first
  example click after this rebuild is the first live use of the cache. The cache is empty after every
  restart or rebuild, so it needs warming before a demo.

## Owner's check of the cached examples

After the rebuild, the owner clicked the examples on the live Space while signed in, and reported the
following. This is an owner observation, not part of the independent verification above, so
`example_cache_checked_live` in [verification.json](verification.json) stays false.

- **Praise example.** The first example click showed the stored-answer note with a recorded time of 11.9 s.
  The result was positive (99.8%), with no lexicon match, no issue and no measure; every review is referred to a
  person under the current policy. A later run without the note came from the «Ανάλυση κριτικής» button,
  which is a fresh GPU analysis by design (2.2 s), not from the example cache.
- **Lamp example.**
  - The first click showed the note and a recorded time of 4.5 s.
  - The issue was "Flickering reading lamp", reported, with a verbatim excerpt.
  - The department was maintenance, and the measure was a check of the room's circuit, with points to confirm.
  - Clicking it again returned the same answer immediately, with the same 4.5 s. This is the cache replay
    working on the live Space.
  - The lexicon had no match: none of its six rules covers lamps. Qwen found the issue from the whole review,
    which is what the UI's "absence of a match is not absence of a problem" wording is for.

One existing wording issue was seen: "Η θετική ετικέτα δεν αποκλείει παράπονο" also appears when the sentiment
is negative. It predates this update and is left for the next change that needs a publication anyway.

Also still pending: mobile and cold-start checks, Jev opt-in, systematic quota behaviour, and the
[independent whole-review pilot](../../triage_lexicon_pilot/PROTOCOL.md).
See the [deployment procedure](../../../TRIAGE_SPACE_DEPLOYMENT.md).
