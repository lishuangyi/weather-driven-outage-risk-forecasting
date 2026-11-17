-- Where does the model beat persistence? Score of both per task and county, and the gain.
-- (Per-county scores are diagnostics; the official number is the pooled score.)
SELECT m.task, m.grp AS county, m.cases, round(100 * m.positive_rate, 1) AS event_rate_pct,
       round(m.score, 2) AS model, round(p.score, 2) AS persistence, round(m.score - p.score, 2) AS gain
FROM gold.metrics m
JOIN gold.metrics p ON p.model = 'persistence' AND (p.task, p.dimension, p.grp) = (m.task, m.dimension, m.grp)
WHERE m.model = 'model' AND m.dimension = 'county'
ORDER BY m.task, gain DESC;
