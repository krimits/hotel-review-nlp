# Which complaints are rising, and which need a look?

**Question.** A hotel group reads thousands of reviews a year. Which complaint
topics became more frequent, and does the rise hold within the same hotels,
or does it come from a change in which hotels and which guests were reviewed?

**Status.**
- **First run done**, on all 515K reviews (September 2015 to July 2017).
- **What it showed.** The review form changed in February 2016, and that
  change lifts every topic at once.
- **Pending.** A second run that compares only months after the change, with
  intervals. The notebook for both runs is
  [`notebooks/08_phase1_data_and_trends_colab.ipynb`](../../notebooks/08_phase1_data_and_trends_colab.ipynb).

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

- **Complaint.** A topic named in the guest's own negative field. The guest
  supplies the polarity. The topic is named by the lexicon of the hotel-ops demo
  ([`triage.py`](../../spaces/hotel-ops-demo/triage.py), 30 topics such as bed,
  air conditioning or check-in), without a model. So no model update can create
  a trend.
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
    July 2017, both after the change in the review form.

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
| Form changes | [`07_complaint_text_share.sql`](../../analysis/complaint_trends/sql/07_complaint_text_share.sql) | share of reviews with any complaint; a step would move every topic at once |

The runner [`scripts/complaint_trends.py`](../../scripts/complaint_trends.py)
adds four things to the SQL:

1. **Intervals.** It resamples whole hotels 2,000 times, because reviews of one
   hotel are not independent. This gives a 95% interval for each topic's
   within-hotel change.
2. **Many topics at once.** It also computes a stricter interval, at
   1 − 0.05/30 for 30 topics tested together. A topic is flagged as **rising** or
   **falling** only when that interval excludes zero **and** the change is at
   least 10% of the base rate.
3. **Two implementations.** The Python and the SQL within-hotel changes must
   agree, or nothing is reported.
4. **Reading.** For every flagged topic it samples 20 quotes from each period.
   Reading them checks that the trend is in what guests wrote, not in a word the
   lexicon misreads.

## What would make a rise not real

| Explanation | How it is checked |
|---|---|
| More reviews overall | Rates use all reviews of the period as the denominator. |
| Different hotels reviewed | The within-hotel change and the mix effect (query 04). |
| Different guests | The shares of trip type, traveller type and nationality (query 05). |
| A change in the review form | The monthly share of reviews with any complaint (query 07), and each topic's share of the reviews that complain at all (query 02). |
| A change in Booking's tags | Tag shares that swap between two labels (query 05). |
| The lexicon misreading a new phrasing | 20 quotes per period for each flagged topic. |
| Chance, with 30 topics tested | The stricter interval and the 10% minimum. |

## Results

### Data and checks

- **Reviews.** 515,738 rows were read and 565 exact duplicates dropped, leaving
  515,173 reviews of 1,493 hotels.
- **Complaints.** The negative fields gave 771,882 (review, topic) complaints.
- **Checks.** All five checks in query 06 are zero:
  - no review is counted twice for one topic;
  - every complaint has a review;
  - no complaint sits on a review whose negative field says nothing is wrong;
  - the monthly denominators add up;
  - the two periods do not overlap.
- **Hotels compared.** 1,130 hotels have at least 30 reviews in each period.

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
0.25 percentage points for every topic. Charts:
[monthly rates](results/full_window/monthly_rates.png) and
[changes with intervals](results/full_window/within_hotel_change.png). Data:
[`results/full_window/`](results/full_window/).

### Why the first run overstates the rises

**Every topic went up together.** 25 of the 29 topics rose within the same
hotels. That points to one common cause rather than 25 separate problems.

**The review form changed in February 2016.** Query 07 shows the share of
reviews whose negative field holds a complaint:

| Month | Share with a complaint |
|---|---:|
| December 2015 | 61.7% |
| January 2016 | 61.3% |
| **February 2016** | **67.2%** |
| March 2016 | 67.3% |
| January 2017 | 67.8% |
| February 2017 | 68.9% |

The six-point jump from January to February 2016 is a step, not a season: a
year later the same months differ by 1.1 points. Across the two periods the
share went from 66.1% to 69.4%. The first run's base period mixes five months
before the step and six after, so part of every rise is the form, not the
hotels.

**Booking's tags changed too.** "Family with older children" fell from 8.2% of
reviews to 0.5%, while "Family with young children" rose from 5.8% to 17.4%.
That is a relabelling, not a change in guests. So traveller type cannot be
compared across the two periods. Trip type and nationality moved by at most 2.4
points (United Kingdom 49.3% → 46.9%).

### Reading the quotes

I (Claude, not an independent annotator) read the 20 sampled quotes per period
for each flagged topic, 280 quotes in all. For each one I asked whether it is a
complaint about the named topic.

| Topic | Base | Recent |
|---|---:|---:|
| Bathroom & shower | 13/20 | 18/20 |
| Bed | 15/20 | 17/20 |
| Cleanliness | 9/20 | 13/20 |
| Air conditioning & ventilation | 17/20 | 16/20 |
| Linen & towels | 13/20 | 12/20 |
| Responsiveness | 9/20 | 12/20 |
| Pests | 12/20 | 12/20 |

**Precision.** The lexicon's precision in guests' negative fields is moderate:
about two thirds overall, and lowest for cleanliness and responsiveness. The
errors appear in both periods:
- "bite-size" food and a "flea market" read as pests;
- an "unresponsive" tablet or air conditioner read as staff responsiveness;
- praise written into the negative field ("Clean tidy comfortable nice breakfast").

**What this means for the rises.** The recent quotes are not less often
genuine (100/140 against 88/140), so false readings do not explain the rises.
With 20 quotes per cell, these counts cannot measure a difference in precision.

### A first look after the change in the form

These rates come from the monthly counts of the first run
([`01_monthly_rates.csv`](results/full_window/csv/01_monthly_rates.csv)). They
compare February to July 2016 with February to July 2017, over all hotels,
**without intervals**. The second run replaces them.

| Topic | Feb–Jul 2016 | Feb–Jul 2017 | Relative | As a share of all complaints |
|---|---:|---:|---:|---:|
| Cleanliness | 5.00% | 5.85% | +17% | +14% |
| Air conditioning & ventilation | 4.24% | 4.93% | +16% | +14% |
| Location | 7.18% | 8.09% | +13% | +10% |
| Bathroom & shower | 8.77% | 9.86% | +12% | +10% |
| Odours | 1.74% | 1.96% | +12% | +10% |
| Responsiveness | 0.91% | 1.11% | +22% | +19% |
| Bed | 6.48% | 6.86% | +6% | +4% |
| Restaurant, bar & coffee | 10.25% | 10.49% | +2% | +0% |
| Breakfast | 10.24% | 9.97% | −3% | −5% |

**Rises that hold.** Cleanliness, air conditioning, bathroom and
responsiveness keep rising after the change, and faster than complaints
overall. Location joins them.

**Rises that shrink.** Most of the first run's rise in bed complaints came from
the form change. The monthly chart also shows it peaking in February–March 2017
and falling back after.

**Still to do.** The second run gives the within-hotel changes and their
intervals for these windows. The answer to the question will rest on it.
