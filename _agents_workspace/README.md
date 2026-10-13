# _agents_workspace — agent handoff contract

This directory is **local and gitignored**. It is the only place agents exchange state.
Everything that must survive across sessions is mirrored to the committed
`docs/RUNS.md` run log.

## Layout

```
_agents_workspace/
  README.md            <- this contract (committed, so the contract itself is versioned)
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
| methodologist | `docs/protocols/**`, `docs/experiments/**`, `docs/RUNS.md`, workspace tasks; **never `src/`, `configs/`, `scripts/`** |
| engineering | `src/**`, `configs/**`, `scripts/**`, `docker/**`, `spaces/**`, `tests/**`, workspace tasks |
| evidence-audit | workspace tasks only (verdicts) — never source, docs claims, or configs |

No agent ever writes to `data/raw/*`, `runs/*` (outside the whitelisted benchmark
artifacts), or any `.env` / secret store.

## Post-adjudication convention

After any adjudication step produces reconciled sheets, the run's `manifest.json` is
regenerated in the same task (so its hashes cover the final sheets) — stale manifests
are treated as failed verification.
