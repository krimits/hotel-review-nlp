# RUNS — append-only run log

Cross-session state for agent-driven work. One row per run or recorded task; append
only (corrections are added as new rows that reference the old row). For gated runs,
the protocol path and artifacts are mandatory before the row may be closed.

| date | run id | stage | question / task | protocol | status | artifacts / notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-13 | 20261013-harness-smoke | record | Smoke test: fix `95.09%%` → `95.09%` style typos in README and WEEK1 README (trivial edit path, no gate) | none (trivial edit, no claims) | done | commit on `feat/agent-harness`; verified by eye + diff |
| 2026-10-13 | 20261013-adjudication-manifest-inputs | record | Review of the harness and adjudication PRs: pin the adjudication's inputs in the manifest, measure the flags it used to assert, correct the lint target and the protocol path | none (tooling correctness + docs, no new claims) | done | `claude/repo-analysis-irj6a7`; replay of `e1e37e5` unchanged (B sheet, disagreement record and `scored_v2.json` byte-identical); 1120 passed, ruff clean |
| 2026-10-13 | 20261013-bilstm-clean-splits | lock protocol | Close the BiLSTM gap: its README comparisons still rest on the legacy split with 170 overlapping texts | [`docs/experiments/bilstm_clean_splits/PROTOCOL.md`](experiments/bilstm_clean_splits/PROTOCOL.md) | blocked: human gate | Protocol locked with 6 lock tests. Primary: BiLSTM vs DistilBERT full FT on the `time` view, paired bootstrap CI excluding zero. Qwen out of scope (needs GPU + HF Hub). Training cannot run in an agent container: the 238 MB source CSV is uncommitted and kaggle.com is refused by the network policy — needs a machine holding the CSV, or Colab |
| 2026-10-13 | 20261013-reviewer-entry-path | record | Hiring-review legibility: add a ten-minute reviewer path and move the agent harness out of the repository root listing | none (docs and repository layout, no claims changed) | done | `docs/START_HERE.md`; `_agents_workspace/README.md` -> `docs/AGENT_HARNESS.md` and the workspace fully gitignored, so nothing of it is committed; no number in README or EVIDENCE_MAP was touched |
