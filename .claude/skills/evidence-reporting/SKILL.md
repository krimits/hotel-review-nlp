---
name: evidence-reporting
description: Format decision notes, experiment reports, and README/EVIDENCE_MAP updates for hotel-review-nlp with explicit scope and limits. Use when writing any document that states results or claims.
---

# evidence-reporting

## Decision note format

Every decision note in `docs/experiments/` follows this skeleton (see existing decision
docs like `docs/experiments/jev_topic_benchmark/DECISION.md` for tone):

```markdown
# Decision: <one-line question>

## Question
The single question, verbatim from the locked protocol.

## Primary comparison
The locked metric pair and the decision rule, verbatim.

## Result
<the numbers from the bundle, with artifact hashes>

## Decision
<what the locked rule yields for these numbers — nothing else>

## Scope and limits
What this does NOT show: authored development capture, no production accuracy,
Jev state, reviewer independence status. Copy the protocol's limits section and
extend it if new limits emerged.

## Artifacts
Protocol path, bundle paths, hashes, decision-note path, RUNS.md entry.
```

## Rules

- Every number carries its provenance (which file, which hash). No number without an
  artifact.
- Scope language is mandatory and honest: development captures are never described as
  production accuracy; a capture with Jev disabled says so; unverified reviewer
  independence is stated, not implied.
- Never promote or freeze a selection in prose ("this is the model we ship") unless a
  human gate decided it — the wording stays "experimental, not selected, not promoted"
  outside that case.
- README and EVIDENCE_MAP are updated only from the decision note (single source), and
  the update is one sentence-level edit unless the protocol said otherwise.
- Greek narration is acceptable in experiment logs; claims and numbers stay in English
  tables so the EVIDENCE_MAP stays greppable.
