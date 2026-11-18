-- How much of the requested scope could be scored, and why the rest was not?
-- One row per task and county (all months); a case is scored when its truth is available and valid.
SELECT task, county,
       sum(requested)                                                     AS requested,
       sum(scored)                                                        AS scored,
       round(100.0 * sum(scored) / sum(requested), 1)                     AS scored_pct,
       sum(excluded_no_truth)                                             AS no_truth,
       sum(excluded_invalid_truth)                                        AS invalid_truth,
       min(round(100.0 * scored / requested, 1))                          AS worst_month_pct,
       arg_min(month, scored / requested)                                 AS worst_month
FROM gold.coverage
GROUP BY ALL
ORDER BY task, county;
