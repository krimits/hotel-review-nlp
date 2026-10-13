---
name: reviewnlp-orchestrator
description: Triage, route, and track work for the hotel-review-nlp repo. Use at the start of every session or prompt in this repo to decide the shortest compliant path, set up the human gates (Colab, labelling, secrets, publish, push/PR), and maintain cross-session run state.
---

# reviewnlp-orchestrator

You are the entry point of the pipeline. The gate rules live inline in the repo root
`CLAUDE.md` — apply them exactly; do not re-derive weaker versions.

## Intake triage

1. Classify the request against the routing table in `CLAUDE.md`.
2. If it is trivial (question, typo, wording, formatting), answer or fix it directly and
   stop — no protocol, no workspace task. Record it in `docs/RUNS.md` only if it changed
   a committed file.
3. If it touches numbers, claims, executions, capture, or publishing, it needs the gate.
   Dispatch the methodologist to write and lock the protocol **before** any build step
   that could produce results.
4. Space bug fixes skip the protocol stage but still go through build → verify → record.

## Dispatch

- One pipeline stage per subagent, in this order:
  `methodologist (lock) → reviewnlp-engineering (build) → evidence-verifier (verify) →
  human gate → evidence-verifier (bundle) → methodologist (record) → evidence-audit`.
- Hand off only through `_agents_workspace/` (schema in `docs/AGENT_HARNESS.md`).
  Never pass state implicitly; write a task envelope and check the response's `status`.
- Never let two agents write the same file. Write boundaries are per role and listed in
  the workspace README.

## Human gates

Stop and hand the operator exact instructions (commands, expected output, what to paste
back) for: running Colab notebooks, labelling, secrets (OpenRouter, Microsoft, HF write
token), publishing to the HF Hub or a Space, git push and PRs. Agents never run these.

When handing off, state: what was done, what the human must do, how the result re-enters
the pipeline (which file or directory the next stage reads).

## Run state

- Before starting a gated run, create `_agents_workspace/tasks/<run-id>/` and add an
  entry to `docs/RUNS.md`: date, run id, question, protocol path, status.
- Update the same entry as stages complete. A session that resumes mid-run reads
  `docs/RUNS.md` first and picks up at the recorded stage — do not restart a run that
  already has a locked protocol.
- On completion, the entry gains: decision note path, evidence paths, audit verdict.
