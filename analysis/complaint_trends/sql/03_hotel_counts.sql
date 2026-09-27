-- Per hotel, period and topic: reviews and complaining reviews, for the hotels
-- with at least min_hotel_reviews reviews in both periods. Every topic gets a
-- row for every such hotel and period, zero complaints included, so rates and
-- the hotel bootstrap in scripts/complaint_trends.py see the full picture.
WITH periods AS (
    SELECT 'base' AS period,
        (SELECT value FROM params WHERE name = 'base_start') AS start_month,
        (SELECT value FROM params WHERE name = 'base_end') AS end_month
    UNION ALL
    SELECT 'recent',
        (SELECT value FROM params WHERE name = 'recent_start'),
        (SELECT value FROM params WHERE name = 'recent_end')
),
review_period AS (
    SELECT r.review_id, r.hotel_id, p.period
    FROM reviews AS r
    JOIN periods AS p ON r.month BETWEEN p.start_month AND p.end_month
),
eligible AS (
    SELECT hotel_id
    FROM review_period
    GROUP BY hotel_id
    HAVING SUM(period = 'base') >= (SELECT CAST(value AS INTEGER) FROM params WHERE name = 'min_hotel_reviews')
       AND SUM(period = 'recent') >= (SELECT CAST(value AS INTEGER) FROM params WHERE name = 'min_hotel_reviews')
),
hotel_totals AS (
    SELECT rp.hotel_id, rp.period, COUNT(*) AS reviews
    FROM review_period AS rp
    JOIN eligible AS e ON e.hotel_id = rp.hotel_id
    GROUP BY rp.hotel_id, rp.period
),
hotel_complaints AS (
    SELECT rp.hotel_id, rp.period, c.topic, COUNT(*) AS complaining
    FROM complaints AS c
    JOIN review_period AS rp ON rp.review_id = c.review_id
    JOIN eligible AS e ON e.hotel_id = rp.hotel_id
    GROUP BY rp.hotel_id, rp.period, c.topic
),
topics AS (
    SELECT DISTINCT topic FROM complaints
)
SELECT
    t.topic,
    ht.hotel_id,
    ht.period,
    ht.reviews,
    COALESCE(hc.complaining, 0) AS complaining
FROM hotel_totals AS ht
CROSS JOIN topics AS t
LEFT JOIN hotel_complaints AS hc
    ON hc.hotel_id = ht.hotel_id AND hc.period = ht.period AND hc.topic = t.topic
ORDER BY t.topic, ht.hotel_id, ht.period;
