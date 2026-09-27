# Complaint trends

SQL for the question "which complaint topics are rising, and does the rise hold
within the same hotels?". The method, the checks and the results are in
[`docs/case_study/complaint_trends.md`](../../docs/case_study/complaint_trends.md).

Run it with `python scripts/complaint_trends.py` once `make data` has fetched
the raw CSV, or all in one go with
[`notebooks/08_phase1_data_and_trends_colab.ipynb`](../../notebooks/08_phase1_data_and_trends_colab.ipynb).
Any SQLite client can run the queries in `sql/` against the database it builds
(`runs/complaint_trends/reviews.sqlite`).
