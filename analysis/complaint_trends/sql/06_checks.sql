-- Checks that must hold before any result is read. Each row is one check and
-- its value; scripts/complaint_trends.py refuses to report when a check that
-- must be zero is not.
SELECT 'complaints counted twice for one review and topic' AS check_name,
       COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT review_id, topic FROM complaints)) AS value
FROM complaints
UNION ALL
SELECT 'complaints without a review',
       COUNT(*)
FROM complaints AS c
LEFT JOIN reviews AS r ON r.review_id = c.review_id
WHERE r.review_id IS NULL
UNION ALL
SELECT 'complaints on a review whose negative field holds no complaint',
       COUNT(*)
FROM complaints AS c
JOIN reviews AS r ON r.review_id = c.review_id
WHERE r.has_complaint_text = 0
UNION ALL
SELECT 'reviews in full months minus the sum of monthly denominators',
       (SELECT COUNT(*) FROM reviews
        WHERE month BETWEEN (SELECT value FROM params WHERE name = 'first_month')
                        AND (SELECT value FROM params WHERE name = 'last_month'))
     - (SELECT SUM(n) FROM (SELECT COUNT(*) AS n FROM reviews
        WHERE month BETWEEN (SELECT value FROM params WHERE name = 'first_month')
                        AND (SELECT value FROM params WHERE name = 'last_month')
        GROUP BY month))
UNION ALL
SELECT 'overlap in months between the base and the recent period',
       (SELECT COUNT(DISTINCT month) FROM reviews
        WHERE month BETWEEN (SELECT value FROM params WHERE name = 'base_start')
                        AND (SELECT value FROM params WHERE name = 'base_end')
          AND month BETWEEN (SELECT value FROM params WHERE name = 'recent_start')
                        AND (SELECT value FROM params WHERE name = 'recent_end'));
