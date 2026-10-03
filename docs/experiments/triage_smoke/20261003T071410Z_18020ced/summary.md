# Triage smoke test

Invented reviews, no labels and no metrics. The last column of the author's reading is for looking at, not for scoring.

- Run: 2026-10-03T07:14:10+00:00 to 2026-10-03T07:15:38+00:00
- Arms: with-jev, without-jev; reviews: 14
- Jev: openrouter, asked for jev-latest, answered by typesafe/jev-1.13-20260917

```
id      kind      arm          sentiment      jev yes (author's reading)             qwen                  needs a look                            ms sent/jev/qwen
------  --------  -----------  -------------  -------------------------------------  --------------------  --------------------------------------  ----------------
pos-1   positive  without-jev  positive 1.00  - (author: none)                       not_triggered         -                                       7966/-/-
pos-1   positive  with-jev     positive 1.00  none (author: none)                    not_triggered         -                                       8/287/-
pos-2   positive  without-jev  positive 1.00  - (author: none)                       not_triggered         -                                       14/-/-
pos-2   positive  with-jev     positive 1.00  none (author: none)                    not_triggered         -                                       10/250/-
pos-3   positive  without-jev  positive 0.99  - (author: none)                       not_triggered         -                                       12/-/-
pos-3   positive  with-jev     positive 0.99  none (author: none)                    not_triggered         -                                       10/295/-
pos-4   positive  without-jev  positive 1.00  - (author: none)                       not_triggered         -                                       13/-/-
pos-4   positive  with-jev     positive 1.00  none (author: none)                    not_triggered         -                                       9/231/-
neg-1   negative  without-jev  negative 0.98  - (author: bath,resp)                  ok x0                 no_actions_suggested                    13/-/15950
neg-1   negative  with-jev     negative 0.98  bath,resp,other (author: bath,resp)    error:invalid_output  actions_failed                          7/260/3867
neg-2   negative  without-jev  negative 0.98  - (author: clean,resp)                 error:invalid_output  actions_failed                          8/-/4341
neg-2   negative  with-jev     negative 0.98  clean,resp (author: clean,resp)        error:invalid_output  actions_failed                          8/249/3896
neg-3   negative  without-jev  negative 0.98  - (author: ac)                         error:invalid_output  actions_failed                          7/-/2936
neg-3   negative  with-jev     negative 0.98  ac,other (author: ac)                  ok x0                 no_actions_suggested                    12/296/3755
neg-4   negative  without-jev  negative 0.98  - (author: pests,resp)                 error:invalid_output  actions_failed                          8/-/2139
neg-4   negative  with-jev     negative 0.98  clean,pests,resp (author: pests,resp)  no_grounded_actions   actions_ungrounded                      10/268/6064
mix-1   mixed     without-jev  positive 0.82  - (author: ac,resp)                    not_triggered         -                                       7/-/-
mix-1   mixed     with-jev     positive 0.82  ac,resp (author: ac,resp)              ok x0                 no_actions_suggested                    6/241/2327
mix-2   mixed     without-jev  positive 0.73  - (author: bath)                       no_grounded_actions   uncertain_sentiment,actions_ungrounded  8/-/4660
mix-2   mixed     with-jev     positive 0.73  bath (author: bath)                    no_grounded_actions   uncertain_sentiment,actions_ungrounded  10/274/4335
mix-3   mixed     without-jev  positive 1.00  - (author: clean,resp)                 not_triggered         -                                       9/-/-
mix-3   mixed     with-jev     positive 1.00  clean,resp (author: clean,resp)        no_grounded_actions   actions_ungrounded                      8/225/1889
mix-4   mixed     without-jev  positive 0.94  - (author: other)                      not_triggered         -                                       8/-/-
mix-4   mixed     with-jev     positive 0.94  other (author: other)                  no_grounded_actions   actions_ungrounded                      6/227/2963
mix-5   mixed     without-jev  negative 0.88  - (author: other)                      error:invalid_output  actions_failed                          7/-/1722
mix-5   mixed     with-jev     negative 0.88  other (author: other)                  error:invalid_output  actions_failed                          7/348/2101
edge-1  edge      without-jev  positive 0.98  - (author: none)                       not_triggered         -                                       38/-/-
edge-1  edge      with-jev     positive 0.98  none (author: none)                    not_triggered         -                                       6/221/-
```

## Stages

| arm | reviews | complaint stage | suggestion stage | suggestion model wrote text for |
|---|---|---|---|---|
| with-jev | 14 | ok 14 | error:invalid_output 3, no_grounded_actions 4, not_triggered 5, ok 2 | 9 |
| without-jev | 14 | disabled 14 | error:invalid_output 4, no_grounded_actions 1, not_triggered 8, ok 1 | 6 |

## Problems

None.

## Notes

- neg-1 (without-jev): the suggestion stage was asked and gave an empty list
- neg-1 (with-jev): the suggestion stage gave error (invalid_output)
- neg-1 (with-jev): found ['other'], not expected
- neg-2 (without-jev): the suggestion stage gave error (invalid_output)
- neg-2 (with-jev): the suggestion stage gave error (invalid_output)
- neg-3 (without-jev): the suggestion stage gave error (invalid_output)
- neg-3 (with-jev): the suggestion stage was asked and gave an empty list
- neg-3 (with-jev): found ['other'], not expected
- neg-4 (without-jev): the suggestion stage gave error (invalid_output)
- neg-4 (with-jev): the suggestion stage gave no_grounded_actions, 3 rejected
- neg-4 (with-jev): found ['cleanliness'], not expected
- mix-1 (with-jev): the suggestion stage was asked and gave an empty list
- mix-2 (without-jev): the suggestion stage gave no_grounded_actions, 3 rejected
- mix-2 (with-jev): the suggestion stage gave no_grounded_actions, 3 rejected
- mix-3 (with-jev): the suggestion stage gave no_grounded_actions, 1 rejected
- mix-4 (with-jev): the suggestion stage gave no_grounded_actions, 2 rejected
- mix-5 (without-jev): the suggestion stage gave error (invalid_output)
- mix-5 (with-jev): the suggestion stage gave error (invalid_output)

## What the suggestion model wrote where it gave no usable actions

The full text of every generation is in results.jsonl.

### neg-1 (without-jev): ok

```
{"actions": [], "department": ["other"], "to_confirm": []}
```

### neg-1 (with-jev): error, invalid_output

```
{"action": [{"problem": "shower was freezing cold every morning and the water pressure was almost non-existent.", "excerpt": "The shower was freezing cold every morning and the water pressure was almost non-existent.", "measure": "", "department": "other"}, {"problem": "I told reception twice and nobody ever came to fix it.", "excerpt": "I told reception twice and nobody ever came to fix it.", "measure": "", "department": "reception"}, {"problem": "Very disappointing for the price.", "excerpt": "Very disappointing for the price.", "measure": "", "department": "management"}] }
```

### neg-2 (without-jev): error, invalid_output

```
{"action": [{"problem": "The room smelled of smoke and there was dust on every shelf.", "excerpt": "The room smelled of smoke and there was dust on every shelf.", "measure": "", "department": "reception", "to_confirm": []}, {"problem": "The sheets had stains on them.", "excerpt": "The sheets had stains on them.", "measure": "", "department": "housekeeping", "to_confirm": []}, {"problem": "We asked for the room to be cleaned again and nothing happened.", "excerpt": "We asked for the room to be cleaned again and nothing happened.", "measure": "", "department": "maintenance", "to_confirm": []}]}
```

### neg-2 (with-jev): error, invalid_output

```
{"action": [{"problem": "smoke smell", "excerpt": "The room smelled of smoke", "measure": "", "department": "general", "to_confirm": ""}], "action": [{"problem": "dust on shelves", "excerpt": "there was dust on every shelf", "measure": "", "department": "general", "to_confirm": ""}, {"problem": "sheets have stains", "excerpt": "the sheets had stains on them", "measure": "", "department": "general", "to_confirm": ""}]}
```

### neg-3 (without-jev): error, invalid_output

```
{"problems": [{"problem": "Air conditioning broke", "excerpt": "The air conditioning was broken and the room was unbearably hot.", "measure": "", "department": "", "to_confirm": []}, {"problem": "Room temperature was unbearable", "excerpt": "We could not sleep, and the fan made a horrible noise all night.", "measure": "No specific measures taken by the hotel", "department": ["maintenance"], "to_confirm": []}]}
```

### neg-3 (with-jev): ok

```
{"action": [{"problem": "Air conditioning broke", "excerpt": "The air conditioning was broken and the room was unbearably hot.", "measure": "", "department": "maintenance", "to_confirm": []}, {"problem": "Room temperature was unbearable", "excerpt": "We could not sleep, and the fan made a horrible noise all night.", "measure": "", "department": "food_and_beverage", "to_confirm": ""}], "actions": [], "sentiment": 0.98}
```

9 more are in results.jsonl.
