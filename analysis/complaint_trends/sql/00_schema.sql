-- One row per hotel, per review and per (review, topic) complaint.
-- The primary key on complaints makes a review count once per topic, however
-- many times the guest mentions it.
CREATE TABLE hotels (
    hotel_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT NOT NULL UNIQUE,        -- two hotels can share a name, not an address
    city TEXT NOT NULL
);

CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY,
    hotel_id INTEGER NOT NULL REFERENCES hotels (hotel_id),
    review_date TEXT NOT NULL,           -- yyyy-mm-dd
    month TEXT NOT NULL,                 -- yyyy-mm
    nationality TEXT NOT NULL,
    trip_type TEXT NOT NULL,             -- Leisure trip, Business trip or Unknown
    traveller_type TEXT NOT NULL,        -- Couple, Solo traveler, Group, ...
    score REAL NOT NULL,
    has_complaint_text INTEGER NOT NULL  -- 1 when the guest's "negative" field holds a complaint
);

CREATE TABLE complaints (
    review_id INTEGER NOT NULL REFERENCES reviews (review_id),
    topic TEXT NOT NULL,                 -- one of the 30 topics, e.g. room.climate
    quote TEXT NOT NULL,                 -- the first clause that named the topic
    PRIMARY KEY (review_id, topic)
);

-- The analysis windows, filled from the data by scripts/complaint_trends.py:
-- first_month/last_month bound the full calendar months; base_* and recent_*
-- are the same calendar months one year apart; min_hotel_reviews is the
-- smallest number of reviews a hotel needs in each period for the
-- within-hotel comparison.
CREATE TABLE params (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX idx_reviews_month ON reviews (month);
CREATE INDEX idx_reviews_hotel ON reviews (hotel_id, month);
CREATE INDEX idx_complaints_topic ON complaints (topic);
