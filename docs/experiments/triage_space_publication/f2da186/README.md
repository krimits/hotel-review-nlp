# Real-weight package check — experimental G

Source: `f2da186bd63be84a2debe9e6fb3047e14d846ea1`.
The allowlisted Space package ran locally with the real pinned DistilBERT and Qwen 1.5B weights,
on CPU in float32. All 14 synthetic inputs exercised Qwen issue generation; Jev was off throughout.
The functional check passed: no model execution failure, no storage, no external Jev call and all
stage failures were visible. The publisher dry run accepted the package and this report.

**This is not a quality evaluation, a ZeroGPU test or evidence of a published Space.**
Seven inputs had issue extraction rejected by quote/evidence/JSON checks. They produced visible
partial results. The other seven completed their stages; that does not establish correct semantic
interpretation or useful measures. No reserved cases were opened, and G was not selected/promoted.

- [cpu_smoke.json](cpu_smoke.json): per-case execution states, durations, runtime and package binding.
- [source_manifest.json](source_manifest.json): exact source hashes and runtime configuration snapshot.
- [Deployment procedure](../../../TRIAGE_SPACE_DEPLOYMENT.md): publication and remaining checks.
- [Notebook 18](../../../../notebooks/18_publish_triage_space_colab.ipynb): isolated Colab execution
  and publication using a server-side `HF_TOKEN` Secret.

HF upload and live runtime verification still require the authenticated publication step.
