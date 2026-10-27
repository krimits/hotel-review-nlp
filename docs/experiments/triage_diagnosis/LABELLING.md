# Triage diagnosis: how to fill the check sheet

For the owner, step 3 of [PROTOCOL.md](PROTOCOL.md) (version 2). The sheet is
`runs/triage_diagnosis/check_sheet.csv`: 100 reviews, git-ignored because it holds review text. It was drawn by
`scripts/triage_diagnosis_sample.py draw-check` at commit `8e41e32`; its SHA-256 is in
[`samples/check_manifest.json`](samples/check_manifest.json).

These are your own labels, `owner_check`. They are the only human labels in version 2: the only measure of how
the student, and the Gemini second rater, compare with a person's judgement.

## Label by hand only

- **No ChatGPT, Gemini, Claude or any other AI tool**, for any row, not even to check or to suggest.
- **No model output** on these reviews before the labels are committed: not the Space, not the ChatGPT batches
  (they hold other reviews), not any script.
- If a row is hard, decide it yourself or write in `note`. A guess of your own is worth more here than a model's
  answer.

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

- Do not open `samples/check_key.csv` while labelling. It says which stratum each item belongs to.
- Keep the `text` column unchanged. The sheet is checked against the text hashes when it comes back, and a
  changed text is refused.
- Save as "CSV UTF-8 (comma delimited)", not .xlsx. Work on a copy and keep the blank sheet.
- About 40 seconds a review: roughly an hour and a half in all, in one sitting or several.

When it is finished, send the sheet back as it is, and say that it was labelled by hand. It is recorded with
`scripts/triage_diagnosis_labels.py finish --kind check`.
