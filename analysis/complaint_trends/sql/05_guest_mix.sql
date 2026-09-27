-- Did the guests change? Share of reviews by trip type, traveller type and the
-- ten most common nationalities, in the base and the recent period.
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
    SELECT r.*, p.period
    FROM reviews AS r
    JOIN periods AS p ON r.month BETWEEN p.start_month AND p.end_month
),
top_nationalities AS (
    SELECT nationality
    FROM review_period
    GROUP BY nationality
    ORDER BY COUNT(*) DESC
    LIMIT 10
),
segments AS (
    SELECT period, 'trip_type' AS dimension, trip_type AS segment FROM review_period
    UNION ALL
    SELECT period, 'traveller_type', traveller_type FROM review_period
    UNION ALL
    SELECT period, 'nationality',
        CASE WHEN nationality IN (SELECT nationality FROM top_nationalities) THEN nationality ELSE 'Other' END
    FROM review_period
),
shares AS (
    SELECT
        dimension,
        segment,
        period,
        1.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dimension, period) AS share
    FROM segments
    GROUP BY dimension, segment, period
)
SELECT
    dimension,
    segment,
    MAX(CASE WHEN period = 'base' THEN share END) AS base_share,
    MAX(CASE WHEN period = 'recent' THEN share END) AS recent_share,
    COALESCE(MAX(CASE WHEN period = 'recent' THEN share END), 0)
        - COALESCE(MAX(CASE WHEN period = 'base' THEN share END), 0) AS change
FROM shares
GROUP BY dimension, segment
ORDER BY dimension, ABS(change) DESC;
