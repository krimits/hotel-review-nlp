---
title: Hotel Review Actions
emoji: 🏨
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12.12"
app_file: app.py
license: mit
models:
  - krimits/distilbert-hotel-reviews
  - Qwen/Qwen2.5-1.5B-Instruct
---

# Hotel review actions — experimental

Greek UI, English input. One review (3–4,000 characters) per request. DistilBERT gives overall sentiment;
optional Jev supplies complaint hints; Qwen extracts evidence and then proposes measures for reported
problems. **Experimental G-source-spans workflow — not selected, not promoted for production.**
Independent quality evaluation is pending. This workflow has no validated hotel-action accuracy.

Notebook 17 provides development evidence from 24 authored synthetic cases. It found missed problems,
invented issues, wrong departments and extraction failures. Development ratings are not production
accuracy, and this Space does not use them as a headline score. No reserved evaluation was opened for
this deployment. The comparison's historical decision record remains unchanged. The current workflow
uses numbered source spans to avoid model-generated quotes; it is a new version, not the evaluated G.

Every review reaches issue extraction, even with confidently positive sentiment. This is a diagnostic
demo policy, separate from the existing API's default routing. Uncertain classifications, unsuccessful
generation and valid empty suggestions are displayed separately. Excluded issues are also visible.
Suggested measures require a person's confirmation; they do not create work orders automatically.

## Versions and limits

- DistilBERT: `7306aebcaaebc00d579f5d0a91001ae376f18158`, CPU, input truncated to 256 tokens.
- Qwen2.5-1.5B-Instruct: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Prompt: `actions-v6-source-spans`, up to two calls of 400 new tokens each, deterministic decoding.
- The software numbers literal source slices (sentence boundaries, then at most 400 characters).
  Qwen selects integer span ids; software resolves every excerpt and evidence value from the original
  review. Quotes are never generated, punctuation-repaired, normalized or supplied by Jev.
  Invalid ids or JSON remain visible stage failures. Distinct issues can share a source span.
- The whole supplied review is available to Qwen (within the 4,000-character input limit).
- At most five extracted items and two proposed measures. Missing measures are flagged for review.
- `REAL_PENDING` means an actual guest-reported problem without a stated successful fix; its current
  state must be verified. `UNCERTAIN` stays visible and is not sent for corrective measures.
- Literal quote presence and evidence consistency checks do not establish semantic truth. A copied
  workaround can still be misread as resolution. Human review is necessary.
- No constrained decoding or new training. No Space latency benchmark; elapsed time is per request.
- Target hardware: ZeroGPU, with a 60-second maximum GPU function duration. Models load during startup;
  both Qwen calls run inside one GPU function. DistilBERT and optional Jev run outside GPU allocation.
- ZeroGPU uses bfloat16; local CPU/T4 uses float32. This differs from the T4 development experiment.
  Deterministic decoding is not a promise of identical outputs across precision or hardware changes.
- Python 3.12.12, torch 2.11.0, Gradio 5.49.1 and model-library pins are recorded in this package.
  `spaces` is managed by the HF SDK image; its actual version is reported at runtime.
- No automatic paid-hardware request or GPU billing estimate. Per-request time includes waiting inside
  the analysis handler, including GPU acquisition; Gradio queue waiting before the handler is excluded.
- Invalid source selection or JSON names the failed extraction/measures stage. A GPU failure is visible;
  an empty suggestion is not interpreted as absence of a complaint.
- Known limitations: selecting a source span can still attach irrelevant evidence or misclassify it.
  Long sentences are split into bounded slices; all slices are shown to Qwen together, but it can miss
  context across them. UNCERTAIN issues receive no measures, at most two measures may leave more issues
  uncovered, and literal evidence can still be misinterpreted. Every displayed issue needs review.

## Run locally

From the main GitHub repository checkout, install this folder's requirements, then:

```bash
pip install -e . --no-deps
python spaces/hotel-triage-demo/app.py
```

## Jev (optional)

The checkbox is off by default. To enable it, configure Space Variables:
`REVIEWNLP_JEV_ENABLED=1`, `REVIEWNLP_JEV_ROUTE=openrouter`, and Secret `OPENROUTER_API_KEY`.
There is no key input in the UI. Selecting Jev sends the review to that provider. Provider-reported usage
and cost are returned for this request only; missing cost is unknown, incomplete sums are marked partial,
and no currency is inferred from `usage.cost`. The Jev route can resolve a newer
provider model and is recorded per request rather than claimed immutable. There is no review database.

## Build a reviewable Space package

The app depends on repository modules. Do not upload only `app.py`. Build the small self-contained
package from the repository root:

```bash
python scripts/prepare_triage_space.py --output runs/hotel-triage-space
```

This copies an explicit module allowlist and adds SHA-256 manifest/provenance. It performs no Hub writes.
The manifest records model revisions, prompt fingerprint, category mapping, Jev questions, routing,
decoding, limits and precision policy. The `/model_info` endpoint reports this snapshot and actual runtime.
No human annotations, evaluation key, review datasets, run outputs or credentials are uploaded.
Review the package and run `scripts/check_triage_space.py` with real weights before deployment.
An experimental deployment does not select a final generator or establish hotel-use reliability.
The existing ABSA demo and published sentiment Space are separate applications and are not replaced.
