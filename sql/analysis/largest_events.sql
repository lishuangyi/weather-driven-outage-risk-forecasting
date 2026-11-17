-- The largest outages of the forecast period and how early the model saw them.
-- For each county the five days with the highest observed hourly outage share, and the highest Task A risk (ARC01)
-- the model gave to that hour from issues 0-24 h and 24-48 h before it.
WITH period AS (
    SELECT min(target_s) AS first_s, max(target_s) AS last_s FROM gold.scored_cases WHERE task = 'A'
),
daily_peaks AS (
    SELECT o.county, o.ts_s, o.x
    FROM gold.final_outages o, period p
    WHERE o.x IS NOT NULL AND o.ts_s % 3600 = 0 AND o.ts_s BETWEEN p.first_s AND p.last_s
    QUALIFY row_number() OVER (PARTITION BY o.county, CAST(to_timestamp(o.ts_s) AS DATE) ORDER BY o.x DESC) = 1
),
top AS (
    SELECT * FROM daily_peaks QUALIFY row_number() OVER (PARTITION BY county ORDER BY x DESC) <= 5
)
SELECT t.county, strftime(to_timestamp(t.ts_s), '%Y-%m-%d %H:%M UTC') AS peak_time, round(100 * t.x, 3) AS peak_pct_out,
       round(max(s.predicted_risk) FILTER (WHERE s.lead_minutes <= 1440), 2) AS risk_0_24h_ahead,
       round(max(s.predicted_risk) FILTER (WHERE s.lead_minutes >  1440), 2) AS risk_24_48h_ahead
FROM top t
LEFT JOIN gold.scored_cases s
  ON s.model = 'model' AND s.task = 'A' AND s.arch = 'ARC01' AND s.county = t.county AND s.target_s = t.ts_s
GROUP BY ALL
ORDER BY t.county, peak_pct_out DESC;
