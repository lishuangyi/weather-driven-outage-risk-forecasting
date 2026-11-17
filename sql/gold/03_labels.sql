-- Gold: the truth of every request (what actually happened), from the final version of each observation.
--
-- Final value of an observation time: the latest published version (ties: later delivery row), all records,
-- including records between 15-minute grid times; blank = unavailable. x = customers out / customers served.
--   Task A target T: mean of x at T-45, T-30, T-15 and T min; NULL if any of the four is unavailable,
--                    NaN (invalid, excluded from scoring) if any lies outside [0, 1].
--   Task B target T: x at T when the series has a record at T (native 5-minute data),
--                    otherwise the 15-minute value at floor(T / 15 min) (last value held within its interval).

-- grain: one row per (county, observation time)
CREATE OR REPLACE TABLE gold.final_outages AS
SELECT o.county, o.ts_s, o.customers_out / c.total_customers AS x
FROM silver.outage_versions o JOIN silver.customers c USING (county)
QUALIFY row_number() OVER (PARTITION BY o.county, o.ts_s ORDER BY o.src_s DESC, o.row_id DESC) = 1;

-- grain: one row per request
CREATE OR REPLACE TABLE gold.labels AS
WITH a AS (
    SELECT r.row_id, x45.x AS x45, x30.x AS x30, x15.x AS x15, x0.x AS x0
    FROM silver.requests r
    LEFT JOIN gold.final_outages x45 ON x45.county = r.county AND x45.ts_s = r.target_s - 2700
    LEFT JOIN gold.final_outages x30 ON x30.county = r.county AND x30.ts_s = r.target_s - 1800
    LEFT JOIN gold.final_outages x15 ON x15.county = r.county AND x15.ts_s = r.target_s - 900
    LEFT JOIN gold.final_outages x0  ON x0.county  = r.county AND x0.ts_s  = r.target_s
    WHERE r.task = 'A'
),
b AS (
    SELECT r.row_id, at_t.ts_s IS NOT NULL AS truth_at_t,
           CASE WHEN at_t.ts_s IS NOT NULL THEN at_t.x ELSE floor15.x END AS truth
    FROM silver.requests r
    LEFT JOIN gold.final_outages at_t    ON at_t.county = r.county    AND at_t.ts_s = r.target_s
    LEFT JOIN gold.final_outages floor15 ON floor15.county = r.county AND floor15.ts_s = floor(r.target_s / 900) * 900
    WHERE r.task = 'B'
)
SELECT r.row_id, r.task, r.arch, r.county, r.issue_s, r.target_s, r.issue_time, r.target_time,
       (r.target_s - r.issue_s) / 60 AS lead_minutes,
       CASE WHEN r.task = 'A' THEN
                CASE WHEN a.x45 IS NULL OR a.x30 IS NULL OR a.x15 IS NULL OR a.x0 IS NULL THEN NULL
                     WHEN a.x45 BETWEEN 0 AND 1 AND a.x30 BETWEEN 0 AND 1 AND a.x15 BETWEEN 0 AND 1 AND a.x0 BETWEEN 0 AND 1
                          THEN (a.x45 + a.x30 + a.x15 + a.x0) / 4
                     ELSE 'NaN'::DOUBLE END
            ELSE b.truth END                       AS truth_x,
       b.truth_at_t
FROM silver.requests r
LEFT JOIN a USING (row_id)
LEFT JOIN b USING (row_id);
