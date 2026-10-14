# Decision: Qwen2.5-0.5B QLoRA is not carried forward for binary scoring

**14 October 2026. A scoping decision taken on existing evidence, after the clean splits were
built. It is not the result of a locked protocol, and it was not a gate set in advance.**

## Question

Should Qwen2.5-0.5B + QLoRA be trained and scored again on the clean v2 splits, for binary
positive/negative sentiment?

## Decision

No.
- Its legacy row stays in the README, labelled legacy.
- It stays in the [sentiment demo](https://huggingface.co/spaces/krimits/hotel-review-demo), to
  show an encoder beside a decoder on the same review.
- The base Qwen2.5-0.5B-Instruct stays in use, zero-shot, for aspect extraction (`/absa`),
  where generating structured text is the point.

## Why

| Consideration | Evidence | Source |
| :--- | :--- | :--- |
| Accuracy, legacy split | Its first-step label scores give 0.9571. That is indistinguishable from the scratch LoRA (0.9573, McNemar p = 0.53) and below the full fine-tune (0.9634, p = 4.2 × 10⁻⁵). The McNemar table is transcribed, not regenerated. | README, historical results |
| How the label is read | Its generated answers failed a strict parser (macro-F1 0.0). The reported score reads the two label tokens at the first generated position. | [`qwen_legacy_v1`](results/qwen_legacy_v1/README.md) |
| Size | 0.5B parameters (8.8M trainable adapter parameters), against DistilBERT's 67M | [`qwen_legacy_v1`](results/qwen_legacy_v1/README.md) |
| Output | A generated word, with no class probabilities to threshold or route on | [`prompt_format.py`](../../src/reviewnlp/llm/prompt_format.py) |
| Cost of a re-run | About 2.7 hours of T4 training on 20,000 rows, plus 15 minutes to score a test set | [`qwen_legacy_v1`](results/qwen_legacy_v1/README.md) |

## What this decision does not claim

- **That Qwen would score lower on the clean splits.** That is unmeasured.
- **A selection fixed in advance.** It was taken after the clean splits existed, on legacy
  evidence from a test set that shares 170 texts with training.
- **Anything about speed.** Qwen's speed against DistilBERT has not been measured.

## Revisit if

- Binary scoring needs instruction-following, for example a reason given with the label.
- A clean-split run becomes worth its GPU time. Its protocol would then fix, before any run,
  how the label is read: first-step scores or the lenient parser.

## Artifacts

- [`docs/experiments/results/qwen_legacy_v1/`](results/qwen_legacy_v1/README.md):
  `metrics.json`, SHA-256 `f48fc006…`.
- The README's historical table and Known limitations.
- [`docs/RUNS.md`](../RUNS.md), entry `20261014-seed-protocols-and-qwen-scope`.
