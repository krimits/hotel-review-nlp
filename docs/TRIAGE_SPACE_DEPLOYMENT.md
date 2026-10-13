# Experimental triage Space publication

**Latest publication (13 October 2026):** Space commit `3f2aff8`, from source `c65e638`, using
[notebook 26](../notebooks/26_update_triage_examples_space_colab.ipynb). It adds lazily cached, labelled
examples and pins `spaces==0.51.3`. The lexicon, prompts, models and local-primary, Jev-off policy are those of
the `2079d8b` publication described below. The [publication record](experiments/triage_space_publication/3f2aff8/README.md)
preserves the owner's receipt (16 real-weight checks, seven hosted regressions) and states what independent
verification could not reach: the live API and Hub file hashes beyond `requirements.txt`. **Before a demo,
warm the example cache.** Open the Space signed in and click each example once. The cache is empty after
every restart or rebuild.

Current source adds local lexical topic hints as the default primary (`REVIEWNLP_TRIAGE_LEXICON_FALLBACK=1`),
with Jev an opt-in enhancement (`REVIEWNLP_JEV_ENABLED=0` by default). It was published on 13 October at
Space commit `2079d8b`; the [publication record](experiments/triage_space_publication/2079d8b/README.md)
preserves the owner-run real-weight/live receipt and independent verification. For future updates,
build a fresh package from its commit, run a new real-weight check and use the existing `--resume`
procedure. Old pinned publication notebooks do not include this source change.
The package manifest binds the lexicon version/SHA; diagnostics report actual configuration.
Use [notebook 25](../notebooks/25_update_triage_lexicon_space_colab.ipynb) in a fresh Colab GPU session.
It pins source `02d6e8c7b7297bd87b4060b8f9f504421e60244c`, uses `HF_TOKEN` only for the owner-run
update, and exports `triage_lexicon_space_update.zip` including setup/model/publication failures.
The owner-run receipt has 16 completed Colab checks and seven successful hosted regressions.
Independent verification matched the source files, loaded snapshot and local-primary policy;
three anonymous regressions passed before a visible `gpu_quota_exceeded` stopped the repeat.
The template now handles Colab without `ensurepip` and installs all core dependencies before `pip check`.
The successful deployment does not need a rerun for this setup-only correction. The old notebooks keep
their historical pins.
For this update, the publisher's `--local-primary` option requires a real-weight smoke receipt
with verified lexical provenance and opt-out for every case. It sets exactly the two public Variables
above, preserving all secrets/hardware, and checks the loaded source snapshot, actual lexicon hash,
Jev disabled state and human-review policy after the rebuild. A mismatched setting cannot pass as a
successful local-primary deployment. A dry run validates these requirements without changing the Hub.
See [local-primary behavior](TRIAGE.md#local-primary-optional-external-enhancement) and
the [whole-review pilot](experiments/triage_lexicon_pilot/PROTOCOL.md). No provider benchmark is required now.

Target: `krimits/hotel-triage-demo`, public Gradio on **ZeroGPU**. The update package uses G-source-spans,
not a selected generator, production promotion or validated hotel system. Existing Spaces are unchanged.
Reserved evaluation stays closed. A live publication is established only by a successful deployment
receipt and a matching loaded snapshot; preparing a package or passing CI does not establish it.

## Recorded experimental publication

The separate public Space was published on **5 October 2026** at
[`a79717ceab2755658e01ce56aaf7a442ab5db82c`](https://huggingface.co/spaces/krimits/hotel-triage-demo/tree/a79717ceab2755658e01ce56aaf7a442ab5db82c),
from source `f2da186bd63be84a2debe9e6fb3047e14d846ea1` using notebook 18.
The [publication record](experiments/triage_space_publication/a79717c/README.md) preserves the Colab
receipt and real-weight functional check, plus independent Hub-file and live-API verification.
That receipt documents the original G runtime, not a deployment of the source-span fix.
The source-span update was published later the same day at
[`ea8c7f86cf431585528226486b83f5872c87d7b8`](https://huggingface.co/spaces/krimits/hotel-triage-demo/tree/ea8c7f86cf431585528226486b83f5872c87d7b8),
from source `5fa1f5635f8df4d537a6d7adbdcb4b22e8ad3bac` using notebook 19. The
[update record](experiments/triage_space_publication/ea8c7f8/README.md) preserves 16 completed Colab
checks, seven required hosted regressions and independent verification of all 21 files and the
loaded snapshot. Three independent repeated regressions passed; the short-lamp repeat returned
`gpu_unavailable_or_timeout`, visibly reported by the UI. The independent repeat was stopped there;
GPU availability is not established by the successful publication run. The remaining checks below still apply.

## Evidence extraction update

Use [notebook 20](../notebooks/20_triage_reliability_colab.ipynb) in a fresh Colab GPU session,
with the existing `HF_TOKEN` write Secret. It has `RESUME=True` for the existing Space. The
[regression record](experiments/triage_space_publication/source-spans-v6/README.md) documents the local
Qwen reproduction; the hosted verification is in the update record above.

`actions-v6-source-spans` fixes the free-text quote-copying failure by asking Qwen to select integer
source ids. Software resolves them to literal slices of the supplied review. The historical G generator,
notebook 17, original notebook 18 and archived publication remain reproducible at their original pins.
This is a new experimental workflow; the F/G ratings do not evaluate it.

The two reported lamp inputs must now yield a pending issue, a literal source excerpt and a maintenance
measure. These are named development regression checks, not independent quality estimates. The current
publisher requires successful regression records before upload and repeats the core checks on the Space.
An existing Space update preserves its Variables, Secrets and hardware; the upload replaces source files
under a parent-commit guard. The successful live receipt is now archived in the update record.

## Review and check

Use a clean checkout of the pinned update source commit. Install the Space requirements in an
isolated environment to avoid conflicts with a preinstalled newer Gradio/diffusers. The `spaces` library
is supplied by the HF SDK; its runtime version is reported. Never put a key in a notebook cell or output.

```bash
python scripts/prepare_triage_space.py --output runs/triage-space-package
python scripts/check_triage_space.py --package runs/triage-space-package \
  --output runs/triage-space-smoke.json
python scripts/publish_triage_space.py --package runs/triage-space-package \
  --smoke-report runs/triage-space-smoke.json --output runs/triage-space-published.json --dry-run --local-primary
```

The real-weight check uses eight explicitly synthetic edge cases and the eight known G development
failure cases. No new Jev calls or held-out cases. It checks execution, opt-out, no storage and visible
failures. Seven core cases must pass their explicit regression expectations; their stage failures or
missing lamp measures block publication. Other development failures remain visible observations.
These checks do **not** estimate general semantic quality. Local CPU/T4 float32 differs from ZeroGPU
bfloat16, so on-Space functional checks are still required.

The package allowlist contains source only. `source_manifest.json` records SHA-256, source commit,
model revisions, prompts, mapping, routing, input limits, Jev questions, decoding and precision policy.
No CSV annotations, annotation key, dataset, reference outputs, weights or secret is uploaded.

## Publish

Use an existing HF CLI login or the `HF_TOKEN` Colab Secret with write permission for the target Space.
The account must be `krimits`. Publication is already authorized for this experimental Space.

```bash
python scripts/publish_triage_space.py --package runs/triage-space-package \
  --smoke-report runs/triage-space-smoke.json --output runs/triage-space-published.json --resume --local-primary
```

The script creates a **new** public Space, requesting only `zero-a10g` (the HF API name for ZeroGPU).
It refuses an existing name by default. `--resume` applies only to this experimental Space with its
manifest and existing ZeroGPU hardware. Upload uses the current parent commit, then downloads every
allowlisted source file at the returned commit to verify its bytes. It never selects a paid fallback.

The upload receipt is saved before waiting for startup, so a failed build is not reported as a failed
upload or a working demo. The live check waits up to 25 minutes, verifies the loaded snapshot/ZeroGPU,
and exercises the seven required regression inputs with Jev off. It waits for the new loaded snapshot
even if the old process remains RUNNING during the rebuild. Quote/schema failures must be visible. Runtime
and GPU execution failure fail the functional check. No accuracy is computed.

## Reliability development

[Notebook 20](../notebooks/20_triage_reliability_colab.ipynb) published the diagnostic update at
[`e1e37e5`](experiments/triage_space_publication/e1e37e5/README.md) on 6 October and captured
24 authored-development outputs. Twenty workflows were complete and four partial; two blank
human-review sheets are preserved with the outputs. No quality score or promotion was produced. See
[TRIAGE_RELIABILITY.md](TRIAGE_RELIABILITY.md) for two-reviewer scoring, interruption handling and
the remaining release gates. A failed hosted check now preserves its attempt and prior successes
in the upload receipt; it never reports a successful runtime.

## Remaining checks

- On-Space first request after sleep, repeat request, quota/timeout message and mobile layout.
- The server reports Jev enabled; opt-in requests remain untested in this update verification.
  It uses server-side Variables and a fresh Secret. Provider-reported
  `usage.cost` is shown without inferring currency; missing/incomplete cost stays unknown/partial.
- A 15–20-case usability pilot with recorded findings. This is separate from reliability evaluation.
- Further development review before final configuration selection, then a reserved evaluation once.
- A real whole/mixed-review pilot with independent human labels before hotel-use reliability claims.

The app stores no review database, sends no work orders and requests human confirmation for every
measure. The platform's GPU billing is not recorded or inferred by this application.

ZeroGPU loading/worker rules: <https://huggingface.co/docs/hub/spaces-zerogpu>.
