# RUNS — append-only run log

Cross-session state for agent-driven work. One row per run or recorded task; append
only (corrections are added as new rows that reference the old row). For gated runs,
the protocol path and artifacts are mandatory before the row may be closed.

| date | run id | stage | question / task | protocol | status | artifacts / notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-13 | 20261013-harness-smoke | record | Smoke test: fix `95.09%%` → `95.09%` style typos in README and WEEK1 README (trivial edit path, no gate) | none (trivial edit, no claims) | done | commit on `feat/agent-harness`; verified by eye + diff |
