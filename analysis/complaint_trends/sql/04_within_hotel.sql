-- Separates a real change from a change in the hotel mix. For the hotels in
-- both periods (03_hotel_counts.sql), each hotel's recent rate is weighted by
-- its share of base-period reviews: the recent rate the base mix of hotels
-- would have had. The change within hotels is that rate minus the base rate;
-- the mix effect is what remains of the raw change.
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
base_weights AS (
    SELECT hotel_id, 1.0 * reviews / SUM(reviews) OVER () AS weight
    FROM hotel_totals
    WHERE period = 'base'
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
),
hotel_rates AS (
    SELECT
        t.topic,
        ht.hotel_id,
        ht.period,
        ht.reviews,
        COALESCE(hc.complaining, 0) AS complaining,
        1.0 * COALESCE(hc.complaining, 0) / ht.reviews AS rate
    FROM hotel_totals AS ht
    CROSS JOIN topics AS t
    LEFT JOIN hotel_complaints AS hc
        ON hc.hotel_id = ht.hotel_id AND hc.period = ht.period AND hc.topic = t.topic
),
summary AS (
    SELECT
        hr.topic,
        COUNT(DISTINCT hr.hotel_id) AS hotels,
        1.0 * SUM(CASE WHEN hr.period = 'base' THEN hr.complaining END)
            / SUM(CASE WHEN hr.period = 'base' THEN hr.reviews END) AS base_rate,
        1.0 * SUM(CASE WHEN hr.period = 'recent' THEN hr.complaining END)
            / SUM(CASE WHEN hr.period = 'recent' THEN hr.reviews END) AS recent_rate,
        SUM(CASE WHEN hr.period = 'recent' THEN w.weight * hr.rate END) AS recent_rate_base_mix
    FROM hotel_rates AS hr
    JOIN base_weights AS w ON w.hotel_id = hr.hotel_id
    GROUP BY hr.topic
)
SELECT
    topic,
    hotels,
    base_rate,
    recent_rate,
    recent_rate_base_mix,
    recent_rate - base_rate AS raw_change,
    recent_rate_base_mix - base_rate AS within_hotel_change,
    recent_rate - recent_rate_base_mix AS mix_effect
FROM summary
ORDER BY within_hotel_change DESC;
