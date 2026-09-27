-- The same calendar months one year apart, so the seasons match: the share of
-- reviews complaining about each topic in the base and the recent period, and
-- the topic's share of the reviews that complain at all. The second tells a
-- topic that rises faster than complaints overall from one that rises with them.
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
    SELECT r.review_id, r.has_complaint_text, p.period
    FROM reviews AS r
    JOIN periods AS p ON r.month BETWEEN p.start_month AND p.end_month
),
totals AS (
    SELECT
        SUM(period = 'base') AS base_reviews,
        SUM(period = 'recent') AS recent_reviews,
        SUM(period = 'base' AND has_complaint_text = 1) AS base_complaint_reviews,
        SUM(period = 'recent' AND has_complaint_text = 1) AS recent_complaint_reviews
    FROM review_period
),
counts AS (
    SELECT
        c.topic,
        SUM(rp.period = 'base') AS base_complaining,
        SUM(rp.period = 'recent') AS recent_complaining
    FROM complaints AS c
    JOIN review_period AS rp ON rp.review_id = c.review_id
    GROUP BY c.topic
)
SELECT
    c.topic,
    c.base_complaining,
    t.base_reviews,
    c.recent_complaining,
    t.recent_reviews,
    1.0 * c.base_complaining / t.base_reviews AS base_rate,
    1.0 * c.recent_complaining / t.recent_reviews AS recent_rate,
    1.0 * c.recent_complaining / t.recent_reviews - 1.0 * c.base_complaining / t.base_reviews AS change,
    1.0 * c.base_complaining / t.base_complaint_reviews AS base_share_of_complaints,
    1.0 * c.recent_complaining / t.recent_complaint_reviews AS recent_share_of_complaints
FROM counts AS c
CROSS JOIN totals AS t
ORDER BY change DESC;
