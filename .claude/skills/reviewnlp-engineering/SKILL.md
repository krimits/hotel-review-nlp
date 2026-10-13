---
name: reviewnlp-engineering
description: Build and change code, scripts, configs, SQL, and the Space app for hotel-review-nlp with offline tests (fakes, tiny fixtures) and Colab notebook templates pinned to the code's commit. Use for any implementation task in this repo.
---

# reviewnlp-engineering

## Ground rules

- Follow the existing style: src-layout package `src/reviewnlp/`, module docstrings that
  say why the file exists, full type hints, `from __future__ import annotations`,
  4-space indent. Ruff rules E,F,I,W,UP,B are enforced by CI.
- Tests live in `tests/test_<topic>.py`, run offline (no network, no model downloads),
  use fakes and tiny fixtures (see `tests/triage_fakes.py` and `test_workflow_review.py`
  for the established pattern), and must pass under `pytest tests -q` before handoff.
- Scripts in `scripts/` are argparse CLIs whose docstring states their scope limit
  (see `capture_triage_review.py` for the tone: "No model download, Hub write, reserved
  cases, provider call by default, or automatic promotion").
- Never write to `data/raw/*`, `runs/*`, or `artifacts/` from tests. Tests work in
  `tmp_path` only.

## Colab notebook templates

- Every notebook intended for a Colab run pins the code commit it expects: assert the
  repo commit in the notebook's first code cell (the capture scripts already do this via
  `source_commit` checks — keep that convention, full 40-char SHA).
- Notebooks must be committed without outputs and stay small; executed copies for audit
  belong under `docs/experiments/notebooks/`, not `notebooks/`.
- Handoff bundles produced by a notebook are verified by
  `scripts/verify_distilbert_handoff.py` — never invent a parallel verifier.

## Working with other agents

- You receive task envelopes from `_agents_workspace/tasks/<id>/task.md` and write
  results back to `_agents_workspace/tasks/<id>/` (schema in `_agents_workspace/README.md`).
- You own `src/`, `configs/`, `scripts/`, `docker/`, `spaces/`. You do not edit protocols
  or decision notes (methodologist territory) and do not write audit verdicts
  (evidence-audit territory).
- If a task cannot be completed offline (needs Colab, a model download, or a live Space),
  stop at the boundary, mark the envelope `blocked: human gate`, and write the exact
  operator instructions into the envelope.
