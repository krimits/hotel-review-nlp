# Incomplete owner-run Colab comparison — received 9 October 2026

The uploaded `triage_format_comparison.zip` has SHA-256
`82ae7378df5cead5c7e211c10f2735cf45e515b28282c169b5c6c99ae3221a49`.
[receipt_manifest.json](receipt_manifest.json) binds its ten original files, preserved byte for byte.
The runner is the expected source at `1215e4008225c7c1b766af37d609dbd63535ef2c`;
the corpus, prompt, original captured hints, schema and experimental-source hashes match the repo.
All six artifact hashes in [comparison/run.json](comparison/run.json) pass. There are 48 unique
review/arm pairs and blind IDs. The original CSVs remain blank and agree exactly with captured outputs.

## What actually ran

The [launcher](launcher.json) reports exit 1 and incomplete execution. One plain-arm extraction ran
on `dev-01`, failing with `invalid_evidence_issue_schema`; the schema arm then failed with `ImportError`.
The other **46 planned outputs were not executed**. No measures were accepted. Both 48-row sheets
contain these unavailable outputs explicitly; row count does not establish completed execution.
The old report's `execution_errors` also counts unexecuted rows; its reported 48 `failures` do not
mean that 48 model calls failed. The updated runner separates execution failures from unexecuted cases.

Actual recorded runtime: Tesla T4, float32, Python 3.13.15, torch 2.11.0+cu130,
**transformers 5.18.0**, lm-format-enforcer 0.11.3, accelerate 1.15.0 and pydantic 2.13.5.
These differ from the specified dependency pins (transformers 4.56.2, accelerate 1.10.1,
pydantic 2.11.7). The supplied freeze lists 724 packages, including preinstalled Gradio and diffusers.
The old receipt contains no interpreter/module paths, so the original reason the environment diverged
cannot be established. Do not attribute it to an owner edit, specific Colab mechanism or out-of-order cell.

[local_import_reproduction.json](local_import_reproduction.json) records an independent CPU dependency
probe: transformers 5.18.0 with lm-format-enforcer 0.11.3 raises ImportError when importing the format
integration. Its wrapper says transformers is not installed although distribution metadata proves it is.
This reproduces an import failure under the observed version pair; the original capture discarded the
exception message and cannot establish that it was the identical underlying import. No model was loaded
for this diagnostic. After installing the specified pins,
[local_pinned_preflight.json](local_pinned_preflight.json) verifies actual library imports and module
origins in the isolated local CPU environment. This is not a Colab/GPU validation.
[local_tiny_generation.json](local_tiny_generation.json) also records a real Transformers generation
call with a tiny randomly initialized Qwen2, a local byte tokenizer and a simple finite JSON schema:
78 prefix-filter calls, valid JSON and EOS. It tests library wiring, not the hotel model, source-span
issue semantics or the hosted runtime. No Hub model download was used for this probe.

## Disposition

**Do not fill or score these sheets.** The existing scorer rejects the incomplete dev run before
reading human judgments. Do not compare candidate quality, select a winner or infer a GPU reliability
score from this capture. Jev was not requested; there were no Hub writes or reserved-set evaluations.
The Space has not been changed.

Notebook 21 is being corrected to check actual distribution versions, imported versions, interpreter
prefix and package origins, plus the real format-integration import, before loading model weights.
The same preflight also protects the comparison script. A fresh corrected Colab GPU run is still
required; it will use new blank sheets only after a complete capture.
