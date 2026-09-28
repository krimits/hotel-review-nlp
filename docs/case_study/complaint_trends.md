# Which complaints are rising, and which need a look?

**Question.** A hotel group reads thousands of reviews a year. Which complaint
topics became more frequent? Does a rise hold within the same hotels, or does
it come from a change in which hotels and which guests were reviewed?

**Status.** Two runs on all 515K reviews are done. People have not yet measured
the lexicon's precision and recall; that is the next step.
- **First run.** It compares September 2015–July 2016 with September 2016–July
  2017. It found a step in February 2016 that lifts every topic, possibly a
  change in how reviews were collected or recorded.
- **Second run.** It compares only months after that step.

The notebook
[`notebooks/08_phase1_data_and_trends_colab.ipynb`](../../notebooks/08_phase1_data_and_trends_colab.ipynb)
reruns both.

## Answer

The comparison is February–July 2016 against February–July 2017, in the 863
hotels with at least 30 reviews in both periods. Five topics pass the rule fixed
before the first run:
- they rise within the same hotels;
- the stricter interval for 29 topics excludes zero;
- the rise is at least 10% of the base rate.

How the columns below are measured:
- **Change within the same hotels.** The change in the rate, holding the hotel
  mix at the base period's.
- **Against other topics.** The topic's gain relative to all topic mentions, in
  the same hotels.
- **Quotes about the topic.** 20 quotes the lexicon matched in each period, read
  by Claude. This is precision only.
- **Precision fall that erases the rise.** How far precision would have to fall
  between the periods for the genuine change to be zero.

**1. Worth a look, not confirmed: bathroom & shower, air conditioning, pests.**

| Topic | Reviews complaining, 2016 → 2017 | Change within the same hotels | Against other topics | Quotes about the topic, 2016 → 2017 | Precision fall that erases the rise |
|---|---:|---:|---:|---:|---:|
| Bathroom & shower | 8.8% → 10.0% | +1.20 points (+14%) | +8% | 17/20 → 19/20 | 12% |
| Air conditioning & ventilation | 4.2% → 5.0% | +0.73 points (+17%) | +11% | 18/20 → 19/20 | 15% |
| Pests | 0.22% → 0.33% | +0.09 points (+40%) | +33% | 18/20 → 19/20 | 28% |

- **Precision.** In the sampled quotes it is high in both periods.
- **Bathroom & shower.** It has the largest rise in points. Of its +14%, about
  5% is the general growth in topic mentions, which lifts every topic.
- **Pests.** They are rare, about 1 review in 300 in 2017, but serious: bed bugs,
  mice, cockroaches.

**2. Not yet findings: cleanliness and responsiveness.**

| Topic | Reviews complaining, 2016 → 2017 | Change within the same hotels | Against other topics | Quotes about the topic, 2016 → 2017 | Precision fall that erases the rise |
|---|---:|---:|---:|---:|---:|
| Cleanliness | 5.0% → 6.0% | +0.80 points (+16%) | +10% | 10/20 → 12/20 | 14% |
| Responsiveness | 0.92% → 1.12% | +0.19 points (+20%) | +14% | 11/20 → 10/20 | 17% |

- **Precision.** It is about half: 22 of 40 quotes for cleanliness, 21 of 40 for
  responsiveness.
- **Cleanliness errors.** Mostly housekeeping knocking early, or praise such as
  "very clean".
- **Responsiveness errors.** Phrases such as "the answer was yes".
- **Why not yet findings.** A relative fall in precision of 14% or 17% would
  erase these rises, and 20 quotes per period cannot rule that out.

**3. Not rising faster than other topics.**
- **Bed.** +8% within the same hotels, under the 10% bar, and +2.5% against other
  topics. Most of the first run's +14% came from the February 2016 step.
- **Restaurant, bar & coffee.** +0.5%.
- **Breakfast.** −3%.

**4. Location.** It rose by 0.91 points, but 0.39 of that came from which hotels
were reviewed. Within the same hotels the rise is +0.52 points (+7%), under the
bar.

**Limits.**
- **Topics come from a lexicon, not a model.** Only its precision was checked,
  by Claude, on 20 quotes per topic and period. About three quarters of the 480
  quotes read are about the named topic.
- **Recall is not measured.** A rise could also come from guests using words the
  lexicon catches more often.
- **The step in February 2016.** Its cause is unknown. The second run avoids it,
  whatever caused it.
- **Short periods.** Each period is six months, and the data end in August 2017.
- **Guests' nationality is not adjusted for.** For nationality there is only a
  sensitivity bound, not a test (see
  [the second run](#second-run-after-the-february-2016-step)).

**Next step in real work.**
- **Measure the lexicon.** Two people label a pilot sample for the five topics:
  precision and recall in both periods
  ([protocol](../annotation/pilot_protocol.md), locked before the sample was
  drawn).
- **By hotel and city.** Break the rising topics down by hotel and city to find
  whether a few hotels drive each rise. Query 03 already holds the per-hotel
  counts.

## Data

- **Source.** Booking.com reviews of 1,493 hotels in six European cities, from
  August 2015 to August 2017 (Kaggle, "515K Hotel Reviews Data in Europe").
- **Provenance.** [`scripts/fetch_booking_515k.py`](../../scripts/fetch_booking_515k.py)
  downloads a byte-identical copy from a pinned revision and refuses it unless
  its size and SHA-256 match the Kaggle download.
- **What a guest writes.** Each review has a "positive" and a "negative" field,
  as well as the date, the hotel, the reviewer's nationality and Booking's tags
  (trip type, traveller type).
- **Duplicates.** Exact duplicate rows are dropped: same hotel, date,
  nationality, texts and score.

## Definitions

- **Complaint.** A topic named in the guest's own negative field.
  - The guest supplies the polarity.
  - The topic is named without a model, by the lexicon of the hotel-ops demo
    ([`triage.py`](../../spaces/hotel-ops-demo/triage.py)): 29 topics such as
    bed, air conditioning or check-in. So no model update can create a trend.
  - The demo's 30th topic, problem resolution, comes from a separate sentence
    rule and is not used here.
- **Not a complaint.** Booking's placeholder "No Negative", and answers that
  open with "nothing", "none" or "all good". The exception is when the guest
  goes on with "but", "except" or "however".
- **Counting.** A review counts once per topic, however often it repeats the
  topic. This is enforced by the primary key of the `complaints` table.
- **Rate.** Reviews complaining about a topic, divided by **all** reviews in the
  same period. A busier month does not look worse just because it has more
  reviews.
- **Periods.** Only full calendar months. The comparison uses the same months
  one year apart, so the seasons match:
  - first run: September 2015 to July 2016 against September 2016 to July 2017;
  - second run (`--since 2016-02`): February to July 2016 against February to
    July 2017, both after the step in February 2016.

## Method

| Step | Where | What it answers |
|---|---|---|
| Schema | [`00_schema.sql`](../../analysis/complaint_trends/sql/00_schema.sql) | hotels, reviews, complaints; windows in `params` |
| Monthly rates | [`01_monthly_rates.sql`](../../analysis/complaint_trends/sql/01_monthly_rates.sql) | the rate per topic and month, with a 3-month rolling mean (window function) |
| Year over year | [`02_year_over_year.sql`](../../analysis/complaint_trends/sql/02_year_over_year.sql) | the raw change per topic between the two periods, and the topic's share of the reviews that complain at all |
| Per hotel | [`03_hotel_counts.sql`](../../analysis/complaint_trends/sql/03_hotel_counts.sql) | counts for the hotels with 30+ reviews in both periods, zeros included |
| Within hotels | [`04_within_hotel.sql`](../../analysis/complaint_trends/sql/04_within_hotel.sql) | the recent rate at the base period's hotel mix; the raw change splits into a within-hotel change plus a mix effect |
| Guest mix | [`05_guest_mix.sql`](../../analysis/complaint_trends/sql/05_guest_mix.sql) | shares of trip type, traveller type and nationality in each period |
| Checks | [`06_checks.sql`](../../analysis/complaint_trends/sql/06_checks.sql) | no double counting, no orphan complaint, denominators add up, periods do not overlap |
| Steps in the data | [`07_complaint_text_share.sql`](../../analysis/complaint_trends/sql/07_complaint_text_share.sql) | share of reviews with any complaint; a step would move every topic at once |

The runner [`scripts/complaint_trends.py`](../../scripts/complaint_trends.py)
adds four things to the SQL:

1. **Intervals.** It resamples whole hotels 2,000 times, because reviews of one
   hotel are not independent. This gives a 95% interval for each topic's
   within-hotel change.
2. **Many topics at once.** It also computes a stricter interval, at
   1 − 0.05/k for the k topics tested together.
   - Here k = 29, so the interval is at 99.83%.
   - A topic is flagged as **rising** or **falling** only when that interval
     excludes zero **and** the change is at least 10% of the base rate.
3. **Two implementations.** The Python and the SQL within-hotel changes must
   agree, or nothing is reported.
4. **Reading.** For every flagged topic it samples 20 quotes from each period.
   - Reading them checks the lexicon's precision on what it matched.
   - It does not measure what the lexicon misses.

## What would make a rise not real

| Explanation | How it is checked |
|---|---|
| More reviews overall | Rates use all reviews of the period as the denominator. |
| Different hotels reviewed | The within-hotel change and the mix effect (query 04). |
| Different guests | The shares of trip type, traveller type and nationality (query 05). For nationality, only a sensitivity bound, not a test. |
| A change in how reviews were collected or recorded | The monthly share of reviews with any complaint (query 07). The second run uses only months after the February 2016 step. |
| Complaints that name more topics | Each topic's share of all topic mentions, in the same hotels at the base period's mix (from query 04). |
| A change in Booking's tags | Tag shares that swap between two labels (query 05). |
| The lexicon misreading words | 20 quotes per period read for precision. Recall is not measured. |
| Chance, with 29 topics tested | The stricter interval and the 10% minimum. |

## Results

### Data and checks

- **Reviews.** 515,738 rows were read and 565 exact duplicates dropped, leaving
  515,173 reviews of 1,493 hotels.
- **Complaints.** The negative fields gave 771,882 (review, topic) complaints.
- **Checks.** In both runs, all five checks in query 06 are zero:
  - no review is counted twice for one topic;
  - every complaint has a review;
  - no complaint sits on a review whose negative field says nothing is wrong;
  - the monthly denominators add up;
  - the two periods do not overlap.
- **Hotels compared.** Hotels with at least 30 reviews in each period: 1,130 in
  the first run, 863 in the second, whose periods are shorter.

### First run: September 2015 to July 2016 against September 2016 to July 2017

Seven topics were flagged as rising. None was flagged as falling.

| Topic | Base rate | Recent rate | Change within the same hotels (95% CI) | Relative |
|---|---:|---:|---:|---:|
| Bathroom & shower | 8.47% | 9.64% | +1.23 pp (+1.01 to +1.44) | +15% |
| Bed | 6.26% | 7.12% | +0.86 pp (+0.69 to +1.02) | +14% |
| Cleanliness | 5.03% | 5.89% | +0.78 pp (+0.61 to +0.95) | +16% |
| Air conditioning & ventilation | 3.94% | 4.70% | +0.76 pp (+0.59 to +0.94) | +19% |
| Linen & towels | 1.68% | 1.91% | +0.21 pp (+0.12 to +0.29) | +12% |
| Responsiveness | 0.91% | 1.04% | +0.12 pp (+0.06 to +0.18) | +13% |
| Pests | 0.22% | 0.30% | +0.08 pp (+0.04 to +0.12) | +35% |

The change in hotel mix explains almost nothing: the mix effect is under
0.25 percentage points for every topic.
- **Charts:** [monthly rates](results/full_window/monthly_rates.png) and
  [changes with intervals](results/full_window/within_hotel_change.png).
- **Data:** [`results/full_window/`](results/full_window/).

### Why the first run overstates the rises

**Every topic went up together.** 25 of the 29 topics rose within the same
hotels. That points to one common cause rather than 25 separate problems.

**A step in February 2016.** Query 07 shows the share of reviews whose negative
field holds a complaint:

| Month | Share with a complaint |
|---|---:|
| December 2015 | 61.7% |
| January 2016 | 61.3% |
| **February 2016** | **67.2%** |
| March 2016 | 67.3% |
| January 2017 | 67.8% |
| February 2017 | 68.9% |

- **A step, not a season.** The share jumps six points from January to February
  2016. A year later the same months differ by 1.1 points.
- **Its cause is unknown.** It is possibly a change in how reviews were collected
  or recorded, for example in Booking's review form. No outside source confirms
  the cause, and the method does not depend on it.
- **What it does to the first run.**
  - Across the two periods the share went from 66.1% to 69.4%.
  - The first run's base period mixes five months before the step and six after.
  - So part of every rise is the step, not the hotels.

**The family tags swap.**
- **What changed.** "Family with older children" fell from 8.2% of reviews to
  0.5%, while "Family with young children" rose from 5.8% to 17.4%.
- **What it means.** This looks like a relabelling rather than a change in
  guests, but the data cannot confirm it. Either way, traveller type cannot be
  compared across the two periods.
- **The other guest fields.** Trip type and nationality moved by at most 2.4
  points (United Kingdom 49.3% → 46.9%).

### Second run, after the February 2016 step

**Setup.**
- **Periods.** February to July 2016 against February to July 2017. Both periods
  follow the step, and they hold 130,860 and 129,968 reviews.
- **Share with a complaint.** It still grows a little, from 68.0% to 69.6% (all
  hotels).
- **Result.** Five topics were flagged as rising. None was flagged as falling.

| Topic | Base rate | Recent rate | Change within the same hotels (95% CI) | Stricter interval | Relative | Against other topics, same hotels |
|---|---:|---:|---:|---:|---:|---:|
| Bathroom & shower | 8.77% | 9.98% | +1.20 pp (+0.91 to +1.51) | +0.64 to +1.66 | +14% | +8% |
| Cleanliness | 5.04% | 5.99% | +0.80 pp (+0.58 to +1.02) | +0.47 to +1.13 | +16% | +10% |
| Air conditioning & ventilation | 4.24% | 5.02% | +0.73 pp (+0.49 to +0.99) | +0.36 to +1.18 | +17% | +11% |
| Responsiveness | 0.92% | 1.12% | +0.19 pp (+0.10 to +0.28) | +0.06 to +0.34 | +20% | +14% |
| Pests | 0.22% | 0.33% | +0.09 pp (+0.04 to +0.14) | +0.02 to +0.18 | +40% | +33% |

- **Population.** All columns come from the 863 hotels. The mix effect is at most
  0.15 points for these five topics (cleanliness).
- **Charts:** [monthly rates](results/since_2016_02/monthly_rates.png) and
  [changes with intervals](results/since_2016_02/within_hotel_change.png).
- **Data:** [`results/since_2016_02/`](results/since_2016_02/).

**Topic mentions grew, and that lifts every topic.**
- **In the same 863 hotels,** at the base period's mix, topic mentions per review
  rose 5.3%, from 1.49 to 1.57.
- **Across all 1,493 hotels,** the same growth comes from two places:
  - more reviews with a complaint (68.0% → 69.6%);
  - more topics per complaining review (2.18 → 2.26).

The last column of the table removes this growth.
- **What it measures.** Each topic's share of all topic mentions, from the rates
  in [`04_within_hotel.csv`](results/since_2016_02/csv/04_within_hotel.csv). It
  uses the 863 hotels at the base period's mix.
- **How to read it.** A topic that only kept pace with the rest scores 0%.
- **When it was added.** After the run.

For bathroom, the rise of 14% within the same hotels is about 5% more mentions
per review, shared by all topics, times 8% gained on the other topics.

**Topics that do not rise faster than the rest.**
- **Bed.**
  - Change within the same hotels: +0.52 pp (+0.29 to +0.75), +8%.
  - The stricter interval excludes zero. But the rise is under the 10% bar, and
    it gains only 2.5% on the other topics.
  - Monthly rates: bed complaints peaked in the winter, at 7.8% of reviews in
    February 2017, and were back to 6.2% by June–July 2017.
- **Restaurant, bar & coffee.** +0.05 pp (−0.25 to +0.35), +0.5%.
- **Breakfast.** −0.35 pp (−0.65 to −0.02), −3%.
  - The 95% interval is below zero, but the stricter one is not (−0.85 to +0.18).
  - So it is not flagged as falling.
- **Location.**
  - In the 863 hotels: from 7.21% to 8.12% (+0.91 pp).
  - Query 04 splits this into +0.52 pp within the same hotels and +0.39 pp from
    the hotel mix: in 2017 more reviews came from hotels whose guests complain
    more about the location.
  - Within the same hotels the rise is +7%, under the bar.

**Different guests** (all hotels).
- **Trip type** moved by less than half a point.
- **Nationality.**
  - The United Kingdom's share of reviews fell from 51.0% to 46.6%, and the
    United States' rose from 6.4% to 7.7%.
  - Over the ten largest nationalities and the rest, the mix moved by 4.6%: half
    the sum of the absolute changes in share.
- **Sensitivity, not a test.**
  - A shift of 4.6% can move a topic's rate by at most 4.6% of the gap between
    the nationalities with the highest and the lowest rate.
  - For nationality alone to produce the bathroom rise of 1.2 points, some
    nationalities would have to complain about bathrooms at least 26 points more
    often than others. Cleanliness would need a gap of 17 points, and air
    conditioning 16.
  - The rates per nationality were not measured. So this shows how large the
    differences would have to be, not that they are smaller.
  - The bound does not cover shifts within the smaller nationalities.
- **Traveller type** cannot be compared: the family tags swap again ("older
  children" 8.5% → 0.1%, "young children" 6.0% → 18.2%).

### Reading the quotes

**How they were read.**
- **Who.** Claude, the AI assistant, not an independent annotator.
- **What.** For every flagged topic, 20 quotes per period were sampled from all
  hotels of the period and read in full: 480 quotes over both runs.
- **When a quote counts.** It complains about the named topic. This includes a
  missing item ("no bath") and a problem anywhere in the hotel (a mouse in the
  bar).
- **When it does not.** The word means something else, the topic is praised, or
  the complaint is about another topic (hearing the neighbours' toilets flush is
  noise).
- **Record.** The counts and the positions of the quotes judged not about the
  topic are in `reading_counts.json`, one per run
  ([first](results/full_window/reading_counts.json),
  [second](results/since_2016_02/reading_counts.json)). The quotes themselves
  hold review text and are not committed.

| Topic | Run 1, base | Run 1, recent | Run 2, base | Run 2, recent |
|---|---:|---:|---:|---:|
| Bathroom & shower | 17/20 | 20/20 | 17/20 | 19/20 |
| Bed | 16/20 | 18/20 | | |
| Cleanliness | 12/20 | 13/20 | 10/20 | 12/20 |
| Air conditioning & ventilation | 18/20 | 16/20 | 18/20 | 19/20 |
| Linen & towels | 15/20 | 15/20 | | |
| Responsiveness | 8/20 | 11/20 | 11/20 | 10/20 |
| Pests | 15/20 | 16/20 | 18/20 | 19/20 |
| **All** | **101/140** | **109/140** | **74/100** | **79/100** |

Bed and linen were flagged only in the first run.

**Correction.** An earlier version of this table was read from quotes cut at 120
characters.
- **What it missed.** Topics named late in long reviews, such as bed bug bites at
  the end of a list.
- **Effect.** The counts were lower, for example 12/20 for pests in both periods
  of the first run.
- **Now.** The counts above come from the full texts.

**Precision in the sample.** About three quarters of the quotes are about the
named topic.
- **By topic.** Highest for bathroom, air conditioning and pests (85–91%), and
  lowest for cleanliness (about 60%) and responsiveness (about half).
- **Errors.** They appear in both periods:
  - **another meaning of the word:** "bite-size" food, a "flea market", a
    restaurant called "The Five Flies" or a computer "mouse" read as pests; a
    tablet that is "unresponsive" read as staff responsiveness; "freezing
    weather" outside read as air conditioning;
  - **praise in the negative field:** "Clean tidy comfortable nice breakfast";
  - **a neighbouring topic:** housekeeping knocking early read as cleanliness.

**What the reading shows, and what it does not.**
- **Shows.** In this small sample, read by Claude, precision did not fall in the
  recent period:
  - run 1: 109/140 against 101/140;
  - run 2: 79/100 against 74/100.
- **Does not show:**
  - **Recall.** Only quotes the lexicon matched were read. So complaints it
    missed, and any change in how many it misses, are not measured.
  - **Small changes in precision.** With 20 quotes per cell, the counts cannot
    detect them.

**How far precision would have to fall.**
- **The calculation.** Within the same hotels, the lexicon's rate goes from L1
  to L2 (the recent rate at the base period's mix). If precision went from p1 to
  p2, the genuine change is zero when p2 / p1 = L1 / L2. So precision would have
  to fall by 1 − L1 / L2.
- **No labels needed.** The fall is computed from the rates alone.
- **Source.**
  [`scripts/precision_sensitivity.py`](../../scripts/precision_sensitivity.py)
  writes it to
  [`precision_sensitivity.json`](results/since_2016_02/precision_sensitivity.json).

| Topic | Read, 2016 → 2017 | Precision fall that erases the rise |
|---|---:|---:|
| Bathroom & shower | 17/20 → 19/20 | 12.0% |
| Air conditioning & ventilation | 18/20 → 19/20 | 14.8% |
| Pests | 18/20 → 19/20 | 28.4% |
| Cleanliness | 10/20 → 12/20 | 13.6% |
| Responsiveness | 11/20 → 10/20 | 16.8% |

## Supplementary: a hypothetical sensitivity analysis

This section asks how the reading bears on the falls above. It rests on
assumptions that may not hold. The numbers do not confirm that any trend is
real, and the answer at the top does not use them.

**Assumptions.**
- The quotes were read by Claude, not by an independent annotator.
- The quotes come from all hotels in each period. The within-hotel rates come
  from the 863 compared hotels at the base period's mix. Precision may differ
  between the two.
- Recall is unknown and taken as unchanged between the periods.
- Each count is binomial, with a uniform prior on each precision.

**Method.**
- **Beta.** The two counts give Beta posteriors for p1 and p2. The chance is the
  share of 200,000 draws in which p2 / p1 falls to the needed ratio or below
  (seed 0).
- **Katz.** The Katz log interval is a frequentist 95% interval for p2 / p1.

| Topic | Chance precision fell that far | Katz 95% interval for p2 / p1 | Needed ratio |
|---|---:|---:|---:|
| Bathroom & shower | 0.023 | 0.91 to 1.38 | 0.88 |
| Air conditioning & ventilation | 0.026 | 0.88 to 1.26 | 0.85 |
| Pests | 0.001 | 0.88 to 1.26 | 0.72 |
| Cleanliness | 0.124 | 0.68 to 2.11 | 0.86 |
| Responsiveness | 0.371 | 0.50 to 1.64 | 0.83 |

**Reading the table.**
- **Under these assumptions,** a fall large enough to erase the rise is unlikely
  for bathroom, air conditioning and pests. For cleanliness and responsiveness it
  is quite possible.
- **Only the pilot can test it.** The pilot evaluation, with two people and
  recall included, can test this. This section cannot.
