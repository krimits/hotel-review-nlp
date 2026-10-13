# CLAUDE.md — hotel-review-nlp

Entry point for every agent prompt in this repository. These rules apply in every
session (local or Claude Code on the web) because they are written here, inline,
not only in skills.

## The gate (read this first)

**Hard gate: nothing that produces evidence, claims, executions, or publishes
anything may start without a locked protocol.**

A run needs the gate if it does any of the following:

- executes an experiment, benchmark, or evaluation that will produce numbers;
- produces any claim that could end up in the README, EVIDENCE_MAP, or a decision note;
- captures or labels data that will be scored;
- publishes (HF Hub, Space), pushes, or opens a PR.

Before such a run there must be a `PROTOCOL.md` (or a protocol section inside the run
folder) that states: the question, the fingerprinted data, the one primary comparison,
the decision rule, and the limits — and a lock test that fails if the protocol changes
after results exist. The methodologist skill owns writing and locking it.

**Explicitly exempt (no protocol needed):** answering questions, reading code or data,
trivial fixes (typos, wording, formatting), pure refactors covered by the existing test
suite, and dependency upgrades that do not change any recorded number. If you are unsure,
say in one line why the task does not produce claims and proceed without the gate.

Secrets are never touched by agents: OpenRouter keys, Microsoft credentials, HF write
tokens live in the environment or `.env` (gitignored) and are used by the human operator
at a human gate.

## Pipeline

```
intake/triage → lock protocol → build → verify → ⏸ human gate → verify bundle
             → record results → audit claims
```

Intake sends each request down the shortest path that still covers it:

| Request shape | Path |
| --- | --- |
| Trivial question about the repo, data, or results | answer directly, no protocol |
| Trivial edit (typo, wording, formatting) | build → verify → record (e.g. README line fix) |
| Space bug fix | build → verify → record (no protocol stage) |
| Anything producing numbers or claims | **gate** → lock protocol → build → verify → ⏸ human gate (Colab / label / publish / push) → verify bundle → record → audit |
| New evaluation tooling | gate → lock protocol → build (with tests) → verify → record |

Human gates — the orchestrator stops and gives exact instructions when a step needs a
human: running Colab notebooks, labelling, secrets, publishing to the HF Hub or a Space,
git push and PRs.

## Agents and handoff

Agents are subagents that coordinate through `_agents_workspace/` (gitignored and never
committed — schema and write boundaries are in `docs/AGENT_HARNESS.md`). Cross-session state lives in
`docs/RUNS.md` (append-only run log, committed). Skills live in `.claude/skills/`:

| Skill | Role |
| --- | --- |
| `reviewnlp-orchestrator` | triage, routing, human-gate handoffs, run state |
| `methodologist` | writes and locks protocols + lock tests; applies decision rules; updates README/EVIDENCE_MAP |
| `reviewnlp-engineering` | code, scripts, configs, Space app; offline tests with fakes and tiny fixtures; Colab templates pinned to the code's commit |
| `colab-handoff` | notebook templates and handoff bundles verified by `scripts/verify_distilbert_handoff.py` |
| `evidence-audit` | CI checks plus protocol ↔ config ↔ code ↔ notebook ↔ bundle hashes ↔ claims comparison; at most 2 review rounds, then escalate |
| `evidence-reporting` | decision notes and reports with explicit scope and limits |

## Non-negotiables carried over from the project's own docs

- Do not commit raw data (`data/raw/*` except `README.md`/`hotel_reviews.csv`), model
  weights, `.env`, or anything under `runs/` outside the whitelisted benchmark artifacts.
- Claims about quality must state scope: the triage capture is an authored development
  snapshot, not production accuracy (see `docs/annotation/pilot_protocol.md`).
- Judge the final `status`, not `model_status` (the protocol's own wording).
- Every evaluation artifact that a protocol references must be versioned in this repo —
  no scorer, capture, or adjudication script may live only in a Downloads folder.
- Run `ruff check src tests scripts spaces` and `pytest tests -q` before any PR — the
  same two commands CI runs, plus `python scripts/verify_distilbert_handoff.py` there.
  `spaces` is part of the lint target: leave it out and CI can still fail after your
  own check passed.
