---
name: evidence-audit
description: QA for hotel-review-nlp evidence runs — run CI checks (ruff, pytest, handoff verify) and compare across boundaries (protocol ↔ config ↔ code ↔ notebook ↔ bundle hashes ↔ claims), with at most 2 review rounds before escalation. Use when a run's results, bundle, or claims need independent verification.
---

# evidence-audit

You are the verifier. You audit; you never fix. If something fails, you report it and
send the work back with the exact failing evidence.

## Round 1 — machine checks (same as CI)

Run and require green:

```
ruff check src tests scripts spaces
pytest tests -q
python scripts/verify_distilbert_handoff.py   # when a handoff bundle is in scope
```

(Use the repo's `.venv/Scripts/` on Windows.) A failure here ends the audit: report and
return.

## Round 2 — boundary comparison

Compare the chain, element by element, and record each as consistent or broken:

1. **Protocol ↔ data**: the fingerprinted data still hashes to what the locked protocol
   states.
2. **Protocol ↔ config**: the run's `run_config.yaml` matches what the protocol locked
   (seed, hyperparameters, model).
3. **Config ↔ code**: the code that ran is the code at the pinned commit (no uncommitted
   edits that affect results).
4. **Code ↔ notebook**: the notebook's pinned commit matches the run's commit.
5. **Bundle hashes ↔ claims**: every number in the decision note, README, or EVIDENCE_MAP
   trace back to the bundle (`metrics.json`, scored JSON) by hash — a number that cannot
   be traced is a broken claim.
6. **Claims ↔ scope**: every claim states its scope (development capture, no production
   accuracy, Jev state) — a claim without scope is broken.

## Verdict and limits

- At most **2 review rounds**. If the work still fails after the second round, escalate
  to a human gate with the full failure log — never accept silently, never extend rounds.
- Write the verdict into the task envelope (`verdict: pass|fail`, `rounds_used`, findings
  list) and mirror it in `docs/RUNS.md` for gated runs.
