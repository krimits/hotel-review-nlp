"""Locust load test for the sentiment API.

Run (after `make serve` in another terminal):
    locust -f scripts/load_test.py --host http://127.0.0.1:8000 --headless \
           -u 20 -r 2 -t 60s --only-summary

Reports RPS and p50/p95/p99 latency - the numbers to quote in the README's
"serving" section. The CSV dataset provides realistic review payloads.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

_SAMPLE_REVIEWS = [
    "Absolutely fantastic hotel, the staff went above and beyond and the breakfast was superb.",
    "Terrible experience. The room was dirty, noisy and nothing like the photos.",
    "Great location, walking distance to everything, but the rooms are really tiny.",
    "The air conditioning was broken for three days and nobody cared. Never again.",
    "Beautiful lobby, comfortable bed, would definitely come back next year!",
]


class SentimentUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        # optionally load real review texts from the raw CSV
        self.texts = _SAMPLE_REVIEWS
        csv_path = os.environ.get("LOADTEST_CSV", "data/raw/hotel_reviews.csv")
        if os.path.exists(csv_path):
            import pandas as pd

            df = pd.read_csv(csv_path)
            texts = df["Review"].dropna().astype(str).tolist()
            if texts:
                self.texts = texts

    @task(8)
    def predict_single(self) -> None:
        self.client.post("/predict", json={"text": random.choice(self.texts)})

    @task(2)
    def predict_batch(self) -> None:
        batch = random.sample(self.texts, k=min(8, len(self.texts)))
        self.client.post("/predict/batch", json={"texts": batch})

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
