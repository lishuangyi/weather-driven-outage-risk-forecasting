-- Scoring step 1: the predictions file, normalised and checked against the scoring scope.
-- Scope = the requests whose issue time lies in {{scope_filter}} (the whole request table unless a window is given).
-- Inside the scope every request needs exactly one prediction with finite values in [0, 1]; prediction rows outside
-- the scope are ignored (and counted). The runner stops when an 'error' row below has violations > 0, so the model
-- and the persistence baseline are always scored on the same, complete set of requests.

CREATE OR REPLACE VIEW bronze.predictions AS SELECT * FROM {{predictions}};

-- grain: one row per delivered prediction
CREATE OR REPLACE TABLE silver.predictions AS
SELECT file_row_number                                         AS row_id,
       trim(task_id::VARCHAR)                                  AS task,
       trim(architecture_id::VARCHAR)                          AS arch,
       lpad(trim(county_fips::VARCHAR), 5, '0')                AS county,
       epoch(CAST(trim(issue_time::VARCHAR) AS TIMESTAMPTZ))   AS issue_s,
       epoch(CAST(trim(target_time::VARCHAR) AS TIMESTAMPTZ))  AS target_s,
       TRY_CAST(predicted_x AS DOUBLE)                         AS predicted_x,
       TRY_CAST(predicted_risk AS DOUBLE)                      AS predicted_risk
FROM bronze.predictions;

-- grain: one row per request in the scoring scope
CREATE OR REPLACE TABLE gold.score_scope AS
SELECT * FROM silver.requests WHERE {{scope_filter}};

CREATE OR REPLACE TABLE gold.prediction_checks AS
WITH in_window AS (
    SELECT * FROM silver.predictions WHERE {{scope_filter}}
),
checks AS (
    SELECT 'duplicate prediction keys' AS check_name, 'error' AS severity, count(*) AS violations
    FROM (SELECT task, arch, county, issue_s, target_s FROM in_window GROUP BY ALL HAVING count(*) > 1)
    UNION ALL
    SELECT 'requests without a prediction', 'error', count(*)
    FROM gold.score_scope r ANTI JOIN in_window p USING (task, arch, county, issue_s, target_s)
    UNION ALL
    SELECT 'predictions without a request', 'error', count(*)
    FROM in_window p ANTI JOIN gold.score_scope r USING (task, arch, county, issue_s, target_s)
    UNION ALL
    SELECT 'missing, non-finite or out-of-range values', 'error', count(*)
    FROM in_window
    WHERE predicted_x IS NULL OR predicted_risk IS NULL OR NOT isfinite(predicted_x) OR NOT isfinite(predicted_risk)
       OR predicted_x NOT BETWEEN 0 AND 1 OR predicted_risk NOT BETWEEN 0 AND 1
    UNION ALL
    SELECT 'prediction rows outside the scope (ignored)', 'info', count(*)
    FROM silver.predictions WHERE NOT ({{scope_filter}})
)
SELECT check_name, severity, violations,
       CASE WHEN violations = 0 THEN 'pass' WHEN severity = 'error' THEN 'FAIL' ELSE 'note' END AS status
FROM checks;
