-- Share of reviews that complain about each topic, per full month, and its
-- 3-month rolling mean. The denominator is every review of the month: a guest
-- who wrote nothing negative still counts, so a busier month does not look
-- worse just because it has more reviews.
WITH bounds AS (
    SELECT
        (SELECT value FROM params WHERE name = 'first_month') AS first_month,
        (SELECT value FROM params WHERE name = 'last_month') AS last_month
),
monthly_reviews AS (
    SELECT r.month, COUNT(*) AS reviews
    FROM reviews AS r
    JOIN bounds AS b ON r.month BETWEEN b.first_month AND b.last_month
    GROUP BY r.month
),
monthly_complaints AS (
    SELECT r.month, c.topic, COUNT(*) AS complaining_reviews
    FROM complaints AS c
    JOIN reviews AS r ON r.review_id = c.review_id
    JOIN bounds AS b ON r.month BETWEEN b.first_month AND b.last_month
    GROUP BY r.month, c.topic
),
topics AS (
    SELECT DISTINCT topic FROM complaints
),
grid AS (
    SELECT m.month, t.topic, m.reviews
    FROM monthly_reviews AS m
    CROSS JOIN topics AS t
)
SELECT
    g.month,
    g.topic,
    g.reviews,
    COALESCE(mc.complaining_reviews, 0) AS complaining_reviews,
    1.0 * COALESCE(mc.complaining_reviews, 0) / g.reviews AS rate,
    AVG(1.0 * COALESCE(mc.complaining_reviews, 0) / g.reviews) OVER (
        PARTITION BY g.topic ORDER BY g.month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rate_3m
FROM grid AS g
LEFT JOIN monthly_complaints AS mc ON mc.month = g.month AND mc.topic = g.topic
ORDER BY g.topic, g.month;
