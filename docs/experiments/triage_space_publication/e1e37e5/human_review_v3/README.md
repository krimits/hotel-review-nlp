# Reconciled development review — 6 October 2026

The submitted [A v3](completed_A_v3.csv) and [B v3](completed_B_v3.csv) each contain 24 complete
ratings of the unchanged [development capture](../review/run.json). The
[scorer report](scored_v3.json) passed with exit 0, no disagreements and `judgments_reconciled: true`.
All fifteen supplied [adjudication decisions](triage_review_disagreements.csv) match both sheets.

These are **adjudicated development judgments, not independent agreement or production accuracy**.
Reviewer independence has not been verified. The reserved set was not opened; no workflow was selected,
frozen, promoted or republished. Jev was off in every captured request, so this review does not evaluate
Jev or the combined route.

## Provenance

[manifest.json](manifest.json) binds every archived CSV/report by SHA-256 and the original `run.json`.
CSV bytes, including their line endings, are preserved exactly as supplied. The earlier
[A v2](completed_A_v2.csv), [B v2](completed_B_v2.csv) and [unreconciled report](scored_v2.json)
remain available: fifteen differing fields across eleven reviews preceded the supplied adjudication.
Only documented judgment fields and notes changed in v3; reviews, model outputs and execution flags
are unchanged. The original blank capture sheets and publication receipts remain historical evidence.
Do not interpret equality after adjudication as a measure of initial inter-reviewer agreement.

## Common findings and denominators

| Finding | Count |
| --- | --- |
| Authored synthetic development reviews | 24 |
| Reviews with actual unresolved problems | 16 |
| Distinct actual unresolved problems | 17 |
| Actual problems not represented as pending or uncertain | 5 |
| Actual problems without an appropriate grounded measure | 13 |
| Invented problems in assessments/proposals | 3 |
| Accepted measures | 5 |
| Measures useful in all four dimensions | 4 |
| Unnecessary measures | 1 |
| Incorrect departments among accepted measures | 0 |
| Issue-stage workflow failures, retained in the denominator | 4 |
| Reviews rated fully useful | 8 |

Issue representation is **12/17** under the protocol, which includes correctly identified `UNCERTAIN`
issues. Appropriate action coverage is **4/17**. Useful measures are **4/5** among the five emitted
measures; that fraction does not describe performance on all reviews or all actual issues. Four
workflow failures accounted for five missed problems after the two distinct problems in blind `0010`
were agreed. Zero false exclusions does not imply correct classification: uncertain issues remain
visible but can receive no measure.

## Adjudication and next engineering priorities

- Eight recognized `UNCERTAIN` complaints have `missed=0` and `unaddressed=1`. Their weak classification
  remains a development defect; changing a count definition does not improve the model.
- Blind `0010` contains a broken latch and absent follow-up: two independently actionable problems.
  The same distinctness rule applies throughout the review.
- Blind `0020` reports a bulb successfully replaced for the rest of the stay. Its pending assessment
  and unnecessary measure have `grounded=0` and `no_invented_facts=0`; a confirmation question alone
  was not treated as an invented fact.
- Blind `0022` reports an imagined lift failure while stating that it always worked. The supplied
  decision counts the named uncertain failure as invented and rates the workflow `useful=0`.

Prioritize reproducing the four issue-stage failures without weakening span/schema validation,
then the explicit complaints stranded as uncertain, resolved-context omissions and hypothetical/praise
inversions. Every new prompt/model/policy version needs fresh outputs and fresh ratings; the present
sheets must not be reused as judgments of a changed workflow. Follow the
[release gates](../../../../TRIAGE_RELIABILITY.md) before changing the experimental label.

```bash
python scripts/score_triage_review.py \
  --run docs/experiments/triage_space_publication/e1e37e5/review \
  --ratings-a docs/experiments/triage_space_publication/e1e37e5/human_review_v3/completed_A_v3.csv \
  --ratings-b docs/experiments/triage_space_publication/e1e37e5/human_review_v3/completed_B_v3.csv \
  --output runs/triage_e1e37e5_scored_v3_recheck.json
```

Use a new output path; the scorer refuses to overwrite an existing report.
