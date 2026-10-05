# Experimental triage Space publication

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
The remaining checks below still apply.

## Evidence extraction update

`actions-v6-source-spans` fixes the free-text quote-copying failure by asking Qwen to select integer
source ids. Software resolves them to literal slices of the supplied review. The historical G generator,
notebook 17, original notebook 18 and archived publication remain reproducible at their original pins.
This is a new experimental workflow; the F/G ratings do not evaluate it.

The two reported lamp inputs must now yield a pending issue, a literal source excerpt and a maintenance
measure. These are named development regression checks, not independent quality estimates. The current
publisher requires successful regression records before upload and repeats the core checks on the Space.
An existing Space update preserves its Variables, Secrets and hardware; the upload replaces source files
under a parent-commit guard. Publication of the fix is pending until a new live receipt is recorded.

## Review and check

Use a clean checkout of the pinned update source commit. Install the Space requirements in an
isolated environment to avoid conflicts with a preinstalled newer Gradio/diffusers. The `spaces` library
is supplied by the HF SDK; its runtime version is reported. Never put a key in a notebook cell or output.

```bash
python scripts/prepare_triage_space.py --output runs/triage-space-package
python scripts/check_triage_space.py --package runs/triage-space-package \
  --output runs/triage-space-smoke.json
python scripts/publish_triage_space.py --package runs/triage-space-package \
  --smoke-report runs/triage-space-smoke.json --output runs/triage-space-published.json --dry-run
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
  --smoke-report runs/triage-space-smoke.json --output runs/triage-space-published.json --resume
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

## Remaining checks

- On-Space first request after sleep, repeat request, quota/timeout message and mobile layout.
- Opt-in Jev can be enabled later using server-side Variables and a fresh Secret. Provider-reported
  `usage.cost` is shown without inferring currency; missing/incomplete cost stays unknown/partial.
- A 15–20-case usability pilot with recorded findings. This is separate from reliability evaluation.
- Further development review before final configuration selection, then a reserved evaluation once.
- A real whole/mixed-review pilot with independent human labels before hotel-use reliability claims.

The app stores no review database, sends no work orders and requests human confirmation for every
measure. The platform's GPU billing is not recorded or inferred by this application.

ZeroGPU loading/worker rules: <https://huggingface.co/docs/hub/spaces-zerogpu>.
