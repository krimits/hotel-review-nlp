# Triage diagnosis: how to fill the labelling sheet

For the owner, step 3 of [PROTOCOL.md](PROTOCOL.md). The sheet is `runs/triage_diagnosis/sheet.csv`
(800 reviews, git-ignored because it holds review text). It was drawn by
`scripts/triage_diagnosis_sample.py` at commit `a61b51f`; its SHA-256 is in
[`samples/manifest.json`](samples/manifest.json).

## What to do per row

1. Read the whole `text`: the `Disliked:` and the `Liked:` part. Complaints written under `Liked:` count.
2. Put `1` in the column of every problem type present. Leave the other type columns empty.
3. If the review reports no problem at all, put `1` in `none` instead.
4. Put `1` in `done` when the row is finished. A row without `done` counts as not labelled.
5. `note` is free text for yourself; it is never committed.

The types, their definitions and what each excludes are in [`catalogue_v1.yaml`](catalogue_v1.yaml), and its
`labelling_rules` apply:

- Label a problem the guest reports as having happened during the stay, even if the hotel fixed it.
- Do not label praise, hypotheticals or wishes.
- Label each distinct problem once. When two types fit, choose the more specific one.
- Use `other_problem` for a real operational problem that fits no type.

## Keep it blind

- Do not open `samples/key.csv` while labelling. It says which set and stratum each item belongs to.
- No model output is on the sheet. Do not run the Space or any model on these reviews before the labels are committed.
- Keep the text column unchanged. The sheet is checked against the text hashes when it comes back, and a
  changed text is refused.
- Work in the item order or any order you like. About 80 a day means about two weeks.

When it is finished, send the sheet back as it is. It becomes `owner_initial`. The 80 reviews for the blind
self-relabelling (30 development, 50 test) come later, at least 7 days after you label them.
