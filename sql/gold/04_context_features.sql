-- Gold: weather and radar context of each forecast, again strictly as of the issue time T.
--
-- Weather: the HRRR county statistics (mean and max of 14 variables) valid at the end of the target's hour,
--          from the latest model run with model_run_time <= T and availability_time <= T that covers that hour.
-- Radar:   MRMS reflectivity exposure of the county over the last 1 h, 6 h and 24 h before T,
--          from frames with availability_time <= T.

-- grain: one row per (county, model run, availability, valid hour); forecast counties only
CREATE OR REPLACE TABLE silver.weather_forecasts AS
SELECT lpad(trim(county_fips::VARCHAR), 5, '0')                       AS county,
       epoch(CAST(trim(model_run_time::VARCHAR) AS TIMESTAMPTZ))      AS run_s,
       epoch(CAST(trim(availability_time::VARCHAR) AS TIMESTAMPTZ))   AS available_s,
       epoch(CAST(trim(target_time::VARCHAR) AS TIMESTAMPTZ))         AS valid_s,
       {{weather_columns}}
FROM bronze.weather
WHERE lpad(trim(county_fips::VARCHAR), 5, '0') IN (SELECT county FROM silver.customers);

-- grain: one row per radar frame (county, frame time, availability)
CREATE OR REPLACE TABLE silver.radar_frames AS
SELECT lpad(trim(county_fips::VARCHAR), 5, '0')                       AS county,
       epoch(CAST(trim(grid_time::VARCHAR) AS TIMESTAMPTZ))           AS frame_s,
       epoch(CAST(trim(availability_time::VARCHAR) AS TIMESTAMPTZ))   AS available_s,
       TRY_CAST(avail AS DOUBLE) AS avail, TRY_CAST(f35 AS DOUBLE) AS f35, TRY_CAST(f50 AS DOUBLE) AS f50,
       TRY_CAST(max_dbz AS DOUBLE) AS max_dbz, TRY_CAST(ring35 AS DOUBLE) AS ring35
FROM bronze.radar;

-- grain: one row per (task, county, issue, target) that has a request
CREATE OR REPLACE TABLE gold.forecast_keys AS
SELECT DISTINCT task, county, issue_s, target_s, ceil(target_s / 3600) * 3600 AS valid_s
FROM silver.requests;

-- grain: (county, issue, valid hour): the latest run visible at the issue
CREATE OR REPLACE TABLE gold.weather_as_of AS
WITH needed AS (SELECT DISTINCT county, issue_s, valid_s FROM gold.forecast_keys)
SELECT n.county, n.issue_s, n.valid_s, (n.issue_s - w.run_s) / 3600 AS wx_run_age_h,
       w.* EXCLUDE (county, run_s, available_s, valid_s)
FROM needed n
JOIN silver.weather_forecasts w
  ON w.county = n.county AND w.valid_s = n.valid_s
 AND w.run_s <= n.issue_s AND w.available_s <= n.issue_s
QUALIFY row_number() OVER (PARTITION BY n.county, n.issue_s, n.valid_s ORDER BY w.run_s DESC, w.available_s DESC) = 1;

-- grain: (county, issue)
CREATE OR REPLACE TABLE gold.radar_as_of AS
WITH issue_points AS (SELECT DISTINCT county, issue_s FROM silver.issues),
frames AS (
    SELECT i.county, i.issue_s, r.frame_s, r.avail, r.f35, r.f50, r.max_dbz, r.ring35
    FROM issue_points i
    JOIN silver.radar_frames r
      ON r.county = i.county AND r.available_s <= i.issue_s AND r.frame_s >= i.issue_s - 86400
)
SELECT county, issue_s,
       count(*) FILTER (WHERE frame_s >= issue_s - 3600)           AS rx_frames_1h,
       max(f35) FILTER (WHERE frame_s >= issue_s - 3600)           AS rx_f35_max_1h,
       max(f50) FILTER (WHERE frame_s >= issue_s - 3600)           AS rx_f50_max_1h,
       max(max_dbz) FILTER (WHERE frame_s >= issue_s - 3600)       AS rx_dbz_max_1h,
       max(ring35) FILTER (WHERE frame_s >= issue_s - 3600)        AS rx_ring35_max_1h,
       max(f35) FILTER (WHERE frame_s >= issue_s - 21600)          AS rx_f35_max_6h,
       max(f50) FILTER (WHERE frame_s >= issue_s - 21600)          AS rx_f50_max_6h,
       max(max_dbz) FILTER (WHERE frame_s >= issue_s - 21600)      AS rx_dbz_max_6h,
       max(ring35) FILTER (WHERE frame_s >= issue_s - 21600)       AS rx_ring35_max_6h,
       max(f35)                                                    AS rx_f35_max_24h,
       max(f50)                                                    AS rx_f50_max_24h,
       max(max_dbz)                                                AS rx_dbz_max_24h,
       avg(f35 ORDER BY frame_s)                                   AS rx_f35_mean_24h
FROM frames
GROUP BY ALL;
