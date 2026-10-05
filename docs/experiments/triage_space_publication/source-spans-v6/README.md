# Source-span extraction fix — experimental, deployment recorded

Code: `5fa1f5635f8df4d537a6d7adbdcb4b22e8ad3bac`. Workflow: `actions-v6-source-spans`.
The [update notebook](../../../../notebooks/19_update_triage_evidence_space_colab.ipynb) pins that code
and updates the existing ZeroGPU Space after real-weight regression checks. The
[live update record](../ea8c7f8/README.md) now confirms publication at `ea8c7f8`, the matching loaded
snapshot and seven hosted regressions in the publication run. A later independent repeat passed
three cases before a visible GPU-unavailable/timeout failure on the short lamp input. The [original publication](../a79717c/README.md) remains an
archival record of G.

## Reproduced fault

The pinned Qwen 1.5B, on CPU in float32, reproduced both screenshot failure codes.
In the mixed lamp review it added a period to the main excerpt; in the short lamp review it preserved
the excerpt but added a period to `evidence.reported`. Neither altered quote appeared in the review.
The atomic parser rejected extraction, so the measures stage never ran.
[historical_failures.json](historical_failures.json) preserves both synthetic inputs, explicit tool
hints, raw model outputs and failure codes. This is a local reproduction, not recovery of the discarded
raw output from the original hosted requests.

## Changed behavior

The model selects integer ids from numbered source slices. Software resolves every excerpt and evidence
value from the original review and still validates the evidence/status policy. Invalid ids are visible
failures; quotes are not repaired or accepted through approximate matching. Distinct issues may share
a source slice. The original G prompt, parser and historical comparison artifacts are unchanged.

The prompt also distinguishes Jev's `other` topic from the generator's category taxonomy. An exploratory
version fixed the quotes but assigned some lamp cases to access/other. That version was revised before
the final probe; the final lamp regressions require maintenance, not just a nonempty action list.

## Real-weight Qwen probe

[qwen_regression_probe.json](qwen_regression_probe.json) records the final seven synthetic cases,
raw stage outputs and source SHA-256. Qwen revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Actual environment: torch 2.11.0+cpu, float32. Sentiment and topic hints were explicit synthetic inputs;
this probe did not invoke DistilBERT, the Jev provider, the full Space or a held-out evaluation.

| Input | Observed final behavior |
| --- | --- |
| Mixed lamp/comfortable-bed review | Pending issue, literal evidence, one maintenance measure |
| Short lamp review without punctuation | Pending issue, literal evidence, one maintenance measure |
| Lamp with nobody fixing it | Pending issue, literal evidence, one maintenance measure |
| Lamp successfully replaced | Resolved issue retained; no measure |
| Hypothetical lift breakdown | UNCERTAIN retained for human review; no measure |
| Breakfast/welcome-drink praise | No extracted issue or measure |
| Heating repair attempted but unsuccessful | Pending issue, literal evidence, one maintenance measure |

All seven completed without a workflow error. These are development regressions, not independent
accuracy estimates or human ratings of action usefulness. The hypothetical case still demonstrates a
classification/evidence limitation. A selected span can be irrelevant or misread, and measures still
need human confirmation. CPU timing is not a Space benchmark.

## Release checks

Notebook 19 runs the packaged app with real DistilBERT/Qwen weights on eight synthetic cases plus eight
known development failures. Seven named regressions must pass before any upload. The publisher repeats
those seven checks on the loaded Space snapshot, with Jev off, after the update. Variables and Secrets
are not rewritten. On-Space bfloat16 can differ from local/T4 float32, so the hosted checks remain
necessary. Jev opt-in, mobile/cold-start checks and independent quality evaluation remain separate.
