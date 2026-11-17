-- Gold: what a forecaster could see at each issue time T (point-in-time view of the outage records).
--
-- A record is visible at T when it was observed by T (ts_s <= T) and published by T (src_s <= T).
-- Of the visible versions of an observation time the latest publication wins (ties: the later delivery row);
-- if that version is blank the observation is unavailable at T (an older value is NOT reused).
-- Only the 15-minute grid feeds the models. Look-back: {{lookback_minutes}} minutes before T
-- (the longest lag/window is 24 h; older records cannot change any feature unless the series has a longer gap).
--
-- grain: one row per (county, issue time, observation time) with a usable value

CREATE OR REPLACE TABLE gold.visible_history AS
WITH issue_points AS (SELECT DISTINCT county, issue_s FROM silver.issues),
candidates AS (
    SELECT i.county, i.issue_s, o.ts_s, o.src_s, o.row_id, o.customers_out
    FROM issue_points i
    JOIN silver.outage_versions o
      ON o.county = i.county
     AND o.on_grid
     AND o.ts_s  <= i.issue_s                               -- observed by T
     AND o.src_s <= i.issue_s                               -- published by T
     AND o.ts_s  >= i.issue_s - {{lookback_minutes}} * 60
),
latest_version AS (
    SELECT * FROM candidates
    QUALIFY row_number() OVER (PARTITION BY county, issue_s, ts_s ORDER BY src_s DESC, row_id DESC) = 1
)
SELECT county, issue_s, ts_s, customers_out
FROM latest_version
WHERE customers_out IS NOT NULL;
