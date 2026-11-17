-- Silver: data-quality checks. One row per check with the number of violating rows and one example.
-- severity 'error'   : the inference run rejects such input, or a basic invariant of the data is broken
-- severity 'warning' : allowed, but worth knowing (e.g. revisions, blank records, records off the 15-minute grid)

CREATE OR REPLACE TABLE silver.quality_checks AS
WITH grid AS (SELECT * FROM ref.request_grid),
checks AS (
    -- requests
    SELECT 'requests: duplicate request key' AS check_name, 'error' AS severity, count(*) AS violations,
           any_value(task || ' ' || arch || ' ' || county || ' ' || issue_time || ' ' || target_time) AS example
    FROM (SELECT task, arch, county, issue_time, target_time FROM silver.requests
          GROUP BY ALL HAVING count(*) > 1)
    UNION ALL
    SELECT 'requests: unknown task, architecture or county', 'error', count(*), any_value(task || ' ' || arch || ' ' || county)
    FROM silver.requests r
    WHERE r.task NOT IN (SELECT task FROM grid)
       OR r.arch NOT IN (SELECT arch FROM ref.architectures)
       OR r.county NOT IN (SELECT county FROM silver.customers)
    UNION ALL
    SELECT 'requests: issue hour or lead off the task grid', 'error', count(*), any_value(task || ' ' || issue_time || ' -> ' || target_time)
    FROM silver.requests r JOIN grid g USING (task)
    WHERE r.issue_s % 3600 <> 0
       OR NOT list_contains(g.issue_hours_utc, CAST((r.issue_s // 3600) % 24 AS INTEGER))
       OR (r.target_s - r.issue_s) / 60 NOT BETWEEN g.lead_min_minutes AND g.lead_max_minutes
       OR ((r.target_s - r.issue_s) / 60) % g.lead_step_minutes <> 0
    UNION ALL
    -- customers
    SELECT 'customers: non-positive or duplicate county total', 'error', count(*), any_value(county)
    FROM (SELECT county FROM silver.customers GROUP BY county HAVING count(*) > 1 OR min(total_customers) <= 0)
    UNION ALL
    -- outages
    SELECT 'outages: negative or non-numeric customers_out', 'error', count(*), any_value(county || ' ' || ts_s)
    FROM silver.outage_versions o JOIN bronze.outages b ON b.file_row_number = o.row_id
    WHERE o.customers_out < 0 OR NOT isfinite(o.customers_out)
       OR (o.customers_out IS NULL AND b.customers_out IS NOT NULL AND trim(b.customers_out::VARCHAR) <> '')
    UNION ALL
    SELECT 'outages: published before observed (source_record_time < timestamp)', 'error', count(*), any_value(county || ' ' || ts_s)
    FROM silver.outage_versions WHERE src_s < ts_s
    UNION ALL
    SELECT 'outages: more customers out than served', 'warning', count(*), any_value(o.county || ' ' || o.ts_s)
    FROM silver.outage_versions o JOIN silver.customers c USING (county) WHERE o.customers_out > c.total_customers
    UNION ALL
    SELECT 'outages: revised observations (several versions)', 'warning', count(*), any_value(county || ' ' || ts_s)
    FROM (SELECT county, ts_s FROM silver.outage_versions GROUP BY ALL HAVING count(*) > 1)
    UNION ALL
    SELECT 'outages: blank records', 'warning', count(*), any_value(county || ' ' || ts_s)
    FROM silver.outage_versions WHERE customers_out IS NULL
    UNION ALL
    SELECT 'outages: records between 15-minute grid times (ignored by the models)', 'warning', count(*), any_value(county || ' ' || ts_s)
    FROM silver.outage_versions WHERE NOT on_grid
    UNION ALL
    -- weather
    SELECT 'weather: available before the model run', 'error', count(*), any_value(county_fips::VARCHAR || ' ' || model_run_time::VARCHAR)
    FROM bronze.weather
    WHERE CAST(trim(availability_time::VARCHAR) AS TIMESTAMPTZ) < CAST(trim(model_run_time::VARCHAR) AS TIMESTAMPTZ)
    UNION ALL
    SELECT 'weather: target before the model run', 'error', count(*), any_value(county_fips::VARCHAR || ' ' || model_run_time::VARCHAR)
    FROM bronze.weather
    WHERE CAST(trim(target_time::VARCHAR) AS TIMESTAMPTZ) < CAST(trim(model_run_time::VARCHAR) AS TIMESTAMPTZ)
    UNION ALL
    SELECT 'weather: duplicate (county, run, availability, target)', 'warning', count(*), any_value(county_fips::VARCHAR)
    FROM (SELECT county_fips, model_run_time, availability_time, target_time FROM bronze.weather GROUP BY ALL HAVING count(*) > 1)
    UNION ALL
    -- radar
    SELECT 'radar: available before the frame time', 'error', count(*), any_value(county_fips::VARCHAR || ' ' || grid_time::VARCHAR)
    FROM bronze.radar
    WHERE CAST(trim(availability_time::VARCHAR) AS TIMESTAMPTZ) < CAST(trim(grid_time::VARCHAR) AS TIMESTAMPTZ)
    UNION ALL
    SELECT 'radar: area fraction outside [0, 1]', 'error', count(*), any_value(county_fips::VARCHAR || ' ' || grid_time::VARCHAR)
    FROM bronze.radar
    WHERE TRY_CAST(f35 AS DOUBLE) NOT BETWEEN 0 AND 1 OR TRY_CAST(f50 AS DOUBLE) NOT BETWEEN 0 AND 1
       OR TRY_CAST(avail AS DOUBLE) NOT BETWEEN 0 AND 1
)
SELECT check_name, severity, violations, CASE WHEN violations > 0 THEN example END AS example,
       CASE WHEN violations = 0 THEN 'pass' WHEN severity = 'error' THEN 'FAIL' ELSE 'note' END AS status
FROM checks;
