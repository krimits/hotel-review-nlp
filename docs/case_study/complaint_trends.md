# Which complaints are rising, and which need a look?

**Question.** A hotel group reads thousands of reviews a year. Which complaint
topics became more frequent, and does the rise hold within the same hotels,
or does it come from a change in which hotels and which guests were reviewed?

**Status.** The method, the SQL and the checks are done and tested. The results
on all 515K reviews are pending the run of
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
  one year apart, so the seasons match: September 2015 to July 2016 against
  September 2016 to July 2017.

## Method

| Step | Where | What it answers |
|---|---|---|
| Schema | [`00_schema.sql`](../../analysis/complaint_trends/sql/00_schema.sql) | hotels, reviews, complaints; windows in `params` |
| Monthly rates | [`01_monthly_rates.sql`](../../analysis/complaint_trends/sql/01_monthly_rates.sql) | the rate per topic and month, with a 3-month rolling mean (window function) |
| Year over year | [`02_year_over_year.sql`](../../analysis/complaint_trends/sql/02_year_over_year.sql) | the raw change per topic between the two periods |
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
| A change in the review form | The monthly share of reviews with any complaint (query 07). |
| The lexicon misreading a new phrasing | 20 quotes per period for each flagged topic. |
| Chance, with 30 topics tested | The stricter interval and the 10% minimum. |

## Results

Pending the full run.
