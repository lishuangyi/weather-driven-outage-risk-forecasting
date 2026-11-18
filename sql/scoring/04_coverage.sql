-- Scoring step 4: coverage of the scope, per task, county and issue month (UTC).
-- requested = scope requests; scored = cohort cases; the rest are excluded with the reason given.
CREATE OR REPLACE TABLE gold.coverage AS
SELECT s.task, s.county, strftime(s.issue_time, '%Y-%m') AS month,
       count(*)                                                                    AS requested,
       count(*) FILTER (WHERE l.truth_x IS NOT NULL AND isfinite(l.truth_x)
                          AND l.truth_x BETWEEN 0 AND 1)                           AS scored,
       count(*) FILTER (WHERE l.truth_x IS NULL)                                   AS excluded_no_truth,
       count(*) FILTER (WHERE l.truth_x IS NOT NULL
                          AND NOT (isfinite(l.truth_x) AND l.truth_x BETWEEN 0 AND 1)) AS excluded_invalid_truth
FROM gold.score_scope s
JOIN gold.labels l USING (row_id)
GROUP BY ALL
ORDER BY s.task, s.county, month;
