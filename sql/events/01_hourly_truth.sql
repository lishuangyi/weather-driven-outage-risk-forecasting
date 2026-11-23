-- Events step 1: hourly outage share in the definition of the Task A label.
-- The hour ending at H is the mean of the final values at H-45, H-30, H-15 and H min; it exists only when all four
-- quarter-hours are available and inside [0, 1] (gold.labels, Task A).
--
-- grain: one row per (county, hour end)
CREATE OR REPLACE TABLE gold.hourly_truth AS
SELECT county, (ceil(ts_s / 3600) * 3600)::BIGINT AS hour_s, sum(x ORDER BY ts_s) / 4 AS x
FROM gold.final_outages
WHERE ts_s % 900 = 0 AND isfinite(x) AND x BETWEEN 0 AND 1
GROUP BY ALL
HAVING count(*) = 4;
