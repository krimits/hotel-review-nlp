-- Share of reviews whose negative field holds a complaint, per full month. A
-- step in this series points to a change in how reviews were collected or
-- recorded, or in the guests' habits; it would move every topic at once.
SELECT
    month,
    COUNT(*) AS reviews,
    AVG(has_complaint_text) AS complaint_text_share
FROM reviews
WHERE month BETWEEN (SELECT value FROM params WHERE name = 'first_month')
                AND (SELECT value FROM params WHERE name = 'last_month')
GROUP BY month
ORDER BY month;
