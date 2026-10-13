# RUNS — append-only run log

Cross-session state for agent-driven work. One row per run or recorded task; append
only (corrections are added as new rows that reference the old row). For gated runs,
the protocol path and artifacts are mandatory before the row may be closed.

| date | run id | stage | question / task | protocol | status | artifacts / notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-13 | 20261013-harness-smoke | record | Smoke test: fix `95.09%%` → `95.09%` style typos in README and WEEK1 README (trivial edit path, no gate) | none (trivial edit, no claims) | done | commit on `feat/agent-harness`; verified by eye + diff |
| 2026-10-13 | 20261013-adjudication-manifest-inputs | record | Review of the harness and adjudication PRs: pin the adjudication's inputs in the manifest, measure the flags it used to assert, correct the lint target and the protocol path | none (tooling correctness + docs, no new claims) | done | `claude/repo-analysis-irj6a7`; replay of `e1e37e5` unchanged (B sheet, disagreement record and `scored_v2.json` byte-identical); 1120 passed, ruff clean |
