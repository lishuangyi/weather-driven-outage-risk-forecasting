-- Gold: one model-ready example per (task, county, issue, target): every feature known at the issue time,
-- and the label (truth_x; NULL when the truth is unavailable). This is the training set of the reference model
-- (outage_forecast.reference) and, for new inputs, its prediction input.
CREATE OR REPLACE TABLE gold.examples AS
WITH labels AS (
    SELECT task, county, issue_s, target_s, any_value(truth_x) AS truth_x
    FROM gold.labels GROUP BY ALL
)
SELECT k.task, k.county, k.issue_s, k.target_s,
       -- forecast set-up and calendar (UTC; the county index lets the trees learn local time and climate)
       (k.target_s - k.issue_s) / 60                                    AS lead_minutes,
       sin(2 * pi() * (k.target_s % 86400) / 86400)                     AS target_hour_sin,
       cos(2 * pi() * (k.target_s % 86400) / 86400)                     AS target_hour_cos,
       sin(2 * pi() * dayofyear(to_timestamp(k.target_s)) / 365.25)     AS day_sin,
       cos(2 * pi() * dayofyear(to_timestamp(k.target_s)) / 365.25)     AS day_cos,
       rc.county_index,
       ln(c.total_customers)                                            AS log_customers,
       h.* EXCLUDE (task, county, issue_s, issue_time),
       w.* EXCLUDE (county, issue_s, valid_s),
       r.* EXCLUDE (county, issue_s),
       l.truth_x
FROM gold.forecast_keys k
JOIN silver.customers c USING (county)
JOIN ref.counties rc USING (county)
JOIN gold.history_features h USING (task, county, issue_s)
LEFT JOIN gold.weather_as_of w USING (county, issue_s, valid_s)
LEFT JOIN gold.radar_as_of r USING (county, issue_s)
LEFT JOIN labels l USING (task, county, issue_s, target_s)
ORDER BY k.task, k.county, k.issue_s, k.target_s;
