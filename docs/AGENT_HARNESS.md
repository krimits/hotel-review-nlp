# Agent harness — how the experiment process is run

This project's experiment rules live in [`CLAUDE.md`](../CLAUDE.md) at the repository
root: nothing that produces evidence or claims starts without a protocol locked before
the run, and a lock test that fails if the protocol changes after results exist. That
discipline predates the harness — see the `protocol_sha256` assertions in
`tests/test_pilot_results.py` and `tests/test_annotation_pilot.py` — which only
mechanised a practice the repository already followed.

This page documents the mechanics. It is process engineering, not part of the ML work;
the evidence a reviewer cares about is in [`START_HERE.md`](START_HERE.md).

## Roles

Six skills under `.claude/skills/` split the work so that the agent producing results
is never the one judging them:

| Skill | Role |
| --- | --- |
| `reviewnlp-orchestrator` | triage, routing, human-gate handoffs, run state |
| `methodologist` | writes and locks protocols plus lock tests; applies decision rules |
| `reviewnlp-engineering` | code, scripts, configs, the Space app; offline tests with fakes |
| `colab-handoff` | notebook templates and bundles verified by `scripts/verify_distilbert_handoff.py` |
| `evidence-audit` | CI checks plus the protocol ↔ config ↔ code ↔ notebook ↔ hashes ↔ claims comparison; at most 2 rounds, then escalate |
| `evidence-reporting` | decision notes and reports with explicit scope and limits |

## The handoff workspace

`_agents_workspace/` is **local and gitignored** — nothing in it is committed. It is the
only place agents exchange state. Everything that must survive across sessions is
mirrored to the committed [`RUNS.md`](RUNS.md) run log.

## Layout

```
_agents_workspace/
  tasks/
    <task-id>/         <- one folder per dispatch
      task.md          <- the envelope (see below)
      status.json      <- machine-readable state
      artifacts/       <- files the producing agent hands onward
```

`<task-id>` is `<date>-<slug>` (e.g. `20261013-latch-discreteness`). Run-level folders
use the run id from `docs/RUNS.md`.

## Envelope (`task.md`)

```markdown
# Task <task-id>

- from: <agent role>        # who dispatched
- to: <agent role>          # who executes
- stage: <lock|build|verify|bundle|record|audit>
- run: <run-id or "none">
- protocol: <path or "none">          # required when stage produces evidence
- task: <what to do, precise>
- inputs: <paths the executor may read>
- outputs: <paths the executor must write>
- constraints: <forbidden actions, e.g. "do not touch src/">
```

## status.json

```json
{"task": "<task-id>", "state": "open|done|blocked|failed",
 "rounds_used": 0, "verdict": "pass|fail|null",
 "blocked_reason": "human gate: ... | null",
 "notes": "<one or two lines>"}
```

- `blocked: human gate` requires `blocked_reason` with exact operator instructions.
- The evidence-audit sets `rounds_used` (max 2) and `verdict`; after a second failing
  round it sets `state: "failed"` and the orchestrator escalates instead of retrying.

## Write boundaries (who may write where)

| Role | May write |
| --- | --- |
| orchestrator | `_agents_workspace/**`, `docs/RUNS.md` |
| methodologist | `docs/experiments/**` (protocols live at `docs/experiments/<slug>/PROTOCOL.md`), `docs/annotation/**`, `docs/RUNS.md`, workspace tasks; **never `src/`, `configs/`, `scripts/`** |
| engineering | `src/**`, `configs/**`, `scripts/**`, `docker/**`, `spaces/**`, `tests/**`, workspace tasks |
| evidence-audit | workspace tasks only (verdicts) — never source, docs claims, or configs |

No agent ever writes to `data/raw/*`, `runs/*` (outside the whitelisted benchmark
artifacts), or any `.env` / secret store.

## Post-adjudication convention

After any adjudication step produces reconciled sheets, the run's `manifest.json` is
regenerated in the same task (so its hashes cover the final sheets) — stale manifests
are treated as failed verification.
