-- Gold: the 30 outage-history features of the models, one row per (task, county, issue time).
-- Same definitions as outage_forecast.features.history.history_features (tests/test_sql_parity.py checks them):
--   hist_latest_x, hist_latest_observation_age_minutes, hist_available, hist_missing_flags,
--   hist_lag_<L>m_x       value of the last observation at or before T - L,          L = 15 min .. 24 h
--   roll_<stat>_<W>       mean / max / min of the observations in [T - W, T],        W = 1 h .. 24 h
--   slope_<W>             Task A: (latest - lag W) / W per minute; Task B: least-squares slope per hour over [T - W, T]
-- All shares are counts / customers served. NULL = not available (the Python features use NaN).

-- Lags as an as-of lookup: for each (issue, lag) the last visible observation at or before T - lag.
CREATE OR REPLACE TABLE gold.history_lags AS
WITH probes AS (
    SELECT DISTINCT i.county, i.issue_s, l.lag_minutes, i.issue_s - l.lag_minutes * 60 AS probe_s
    FROM silver.issues i CROSS JOIN (SELECT unnest([15, 30, 60, 120, 180, 360, 720, 1440]) AS lag_minutes) l
)
SELECT p.county, p.issue_s, p.lag_minutes, v.ts_s AS observed_s, v.customers_out
FROM probes p
ASOF LEFT JOIN gold.visible_history v
  ON v.county = p.county AND v.issue_s = p.issue_s AND v.ts_s <= p.probe_s;

-- Windows: aggregates over [T - W, T] (both ends closed). The least-squares slope uses the two-pass formula
-- sum((t - mean t)(y - mean y)) / sum((t - mean t)^2) with t in hours before T, as the Python code does.
-- Sums run in time order (ordered aggregates): DuckDB aggregates in parallel, and without a fixed order the last
-- bits of a floating-point sum could differ between runs.
CREATE OR REPLACE TABLE gold.history_windows AS
WITH windows AS (SELECT unnest([30, 60, 180, 360, 720, 1440]) AS window_minutes),
points AS (
    SELECT v.county, v.issue_s, w.window_minutes, v.customers_out AS y, (v.ts_s - v.issue_s) / 3600 AS t
    FROM gold.visible_history v
    JOIN windows w ON v.ts_s >= v.issue_s - w.window_minutes * 60
),
stats AS (
    SELECT county, issue_s, window_minutes,
           count(*)             AS points,
           avg(y ORDER BY t)    AS mean_count,
           max(y)               AS max_count,
           min(y)               AS min_count,
           avg(t ORDER BY t)    AS mean_t
    FROM points GROUP BY ALL
),
ols AS (
    SELECT p.county, p.issue_s, p.window_minutes,
           sum((p.t - s.mean_t) * (p.y - s.mean_count) ORDER BY p.t) AS sxy,
           sum((p.t - s.mean_t) * (p.t - s.mean_t) ORDER BY p.t)     AS sxx
    FROM points p JOIN stats s USING (county, issue_s, window_minutes)
    GROUP BY ALL
)
SELECT s.county, s.issue_s, s.window_minutes, s.points, s.mean_count, s.max_count, s.min_count,
       CASE WHEN s.points < 2 THEN NULL WHEN o.sxx = 0 THEN 0 ELSE o.sxy / o.sxx END AS ols_slope_per_hour
FROM stats s JOIN ols o USING (county, issue_s, window_minutes);

CREATE OR REPLACE TABLE gold.history_features AS
WITH latest AS (
    SELECT county, issue_s, arg_max(customers_out, ts_s) AS latest_count, max(ts_s) AS latest_s
    FROM gold.visible_history GROUP BY ALL
),
lag AS (
    PIVOT gold.history_lags ON lag_minutes IN (15, 30, 60, 120, 180, 360, 720, 1440)
    USING any_value(customers_out) GROUP BY county, issue_s
),
win AS (
    PIVOT gold.history_windows ON window_minutes IN (30, 60, 180, 360, 720, 1440)
    USING any_value(points) AS n, any_value(mean_count) AS mean, any_value(max_count) AS max,
          any_value(min_count) AS min, any_value(ols_slope_per_hour) AS ols
    GROUP BY county, issue_s
),
base AS (
    SELECT i.task, i.county, i.issue_s, i.issue_time, c.total_customers::DOUBLE AS total,
           l.latest_count, l.latest_s,
           g."15" AS lag15, g."30" AS lag30, g."60" AS lag60, g."120" AS lag120,
           g."180" AS lag180, g."360" AS lag360, g."720" AS lag720, g."1440" AS lag1440,
           w.*  EXCLUDE (county, issue_s)
    FROM silver.issues i
    JOIN silver.customers c USING (county)
    LEFT JOIN latest l USING (county, issue_s)
    LEFT JOIN lag g USING (county, issue_s)
    LEFT JOIN win w USING (county, issue_s)
),
slopes AS (
    SELECT *,
           CASE WHEN task = 'A' THEN (latest_count - lag60)  / 60  ELSE "60_ols"  END AS s60,
           CASE WHEN task = 'A' THEN (latest_count - lag180) / 180 ELSE "180_ols" END AS s180,
           CASE WHEN task = 'A' THEN (latest_count - lag360) / 360 ELSE "360_ols" END AS s360,
           CASE WHEN task = 'A' THEN NULL                         ELSE "30_ols"  END AS s30
    FROM base
)
SELECT task, county, issue_s, issue_time,
       latest_count / total                                   AS hist_latest_x,
       (issue_s - latest_s) / 60                              AS hist_latest_observation_age_minutes,
       CASE WHEN latest_s IS NULL THEN 0.0 ELSE 1.0 END       AS hist_available,
       -- number of missing model inputs, counted as in training (different input sets per task)
       CASE WHEN task = 'A' THEN
            (lag60 IS NULL)::INT + (lag180 IS NULL)::INT + (lag360 IS NULL)::INT + (lag720 IS NULL)::INT + (lag1440 IS NULL)::INT
          + 2 * ((coalesce("180_n", 0) = 0)::INT + (coalesce("360_n", 0) = 0)::INT + (coalesce("720_n", 0) = 0)::INT + (coalesce("1440_n", 0) = 0)::INT)
          + (NOT isfinite(coalesce(s60, 'NaN'))) ::INT + (NOT isfinite(coalesce(s180, 'NaN')))::INT + (NOT isfinite(coalesce(s360, 'NaN')))::INT
       ELSE
            (lag15 IS NULL)::INT + (lag30 IS NULL)::INT + (lag60 IS NULL)::INT + (lag120 IS NULL)::INT
          + (lag180 IS NULL)::INT + (lag360 IS NULL)::INT + (lag720 IS NULL)::INT + (lag1440 IS NULL)::INT
          + 2 * ((coalesce("60_n", 0) = 0)::INT + (coalesce("180_n", 0) = 0)::INT + (coalesce("360_n", 0) = 0)::INT + (coalesce("1440_n", 0) = 0)::INT)
          + (NOT isfinite(coalesce(s30, 'NaN')))::INT + (NOT isfinite(coalesce(s60, 'NaN')))::INT + (NOT isfinite(coalesce(s180, 'NaN')))::INT
       END::DOUBLE                                            AS hist_missing_flags,
       lag15 / total AS hist_lag_15m_x,   lag30 / total AS hist_lag_30m_x,   lag60 / total AS hist_lag_60m_x,
       lag120 / total AS hist_lag_120m_x, lag180 / total AS hist_lag_180m_x, lag360 / total AS hist_lag_360m_x,
       lag720 / total AS hist_lag_720m_x, lag1440 / total AS hist_lag_1440m_x,
       "60_mean" / total AS roll_mean_60,     "60_max" / total AS roll_max_60,     "60_min" / total AS roll_min_60,
       "180_mean" / total AS roll_mean_180,   "180_max" / total AS roll_max_180,   "180_min" / total AS roll_min_180,
       "360_mean" / total AS roll_mean_360,   "360_max" / total AS roll_max_360,   "360_min" / total AS roll_min_360,
       "720_mean" / total AS roll_mean_720,   "720_max" / total AS roll_max_720,   "720_min" / total AS roll_min_720,
       "1440_mean" / total AS roll_mean_1440, "1440_max" / total AS roll_max_1440, "1440_min" / total AS roll_min_1440,
       s60 / total  AS slope_60,
       s180 / total AS slope_180,
       s360 / total AS slope_360
FROM slopes
ORDER BY task, county, issue_s;
