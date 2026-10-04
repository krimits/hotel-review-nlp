---
title: Hotel Review Actions
emoji: 🏨
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
license: mit
models:
  - krimits/distilbert-hotel-reviews
  - Qwen/Qwen2.5-1.5B-Instruct
---

# Hotel review actions — experimental

Greek UI, English input. One review (3–4,000 characters) per request. DistilBERT gives overall sentiment;
optional Jev supplies complaint hints; Qwen extracts evidence and then proposes measures for reported
problems. **Quality evaluation is pending.** This workflow has no validated hotel-action accuracy.

Every review reaches issue extraction, even with confidently positive sentiment. This is a diagnostic
demo policy, separate from the existing API's default routing. Uncertain classifications, unsuccessful
generation and valid empty suggestions are displayed separately. Excluded issues are also visible.
Suggested measures require a person's confirmation; they do not create work orders automatically.

## Versions and limits

- DistilBERT: `7306aebcaaebc00d579f5d0a91001ae376f18158`, CPU, input truncated to 256 tokens.
- Qwen2.5-1.5B-Instruct: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Prompt: `actions-v5-evidence`, up to two calls of 400 new tokens each, deterministic decoding.
- The whole supplied review is available to Qwen (within the 4,000-character input limit).
- At most five extracted items and two proposed measures. Missing measures are flagged for review.
- `REAL_PENDING` means an actual guest-reported problem without a stated successful fix; its current
  state must be verified. `UNCERTAIN` stays visible and is not sent for corrective measures.
- Literal quote presence and evidence consistency checks do not establish semantic truth. A copied
  workaround can still be misread as resolution. Human review is necessary.
- No constrained decoding or new training. No Space latency benchmark; elapsed time is per request.
- GPU is used by Qwen when available; CPU is supported with no response-time promise. No automatic
  paid-hardware request. The existing loader uses float32 on T4/CPU and bfloat16 on supported GPUs.

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
and cost are returned for this request only; missing cost is unknown. The Jev route can resolve a newer
provider model and is recorded per request rather than claimed immutable. There is no review database.

## Build a reviewable Space package

The app depends on repository modules. Do not upload only `app.py`. Build the small self-contained
package from the repository root:

```bash
python scripts/prepare_triage_space.py --output runs/hotel-triage-space
```

This copies an explicit module allowlist and adds SHA-256 manifest/provenance. It performs no Hub writes.
Review this package and complete the notebook 17 comparison before selecting the final generator.
The existing ABSA demo and published sentiment Space are separate applications and are not replaced.
