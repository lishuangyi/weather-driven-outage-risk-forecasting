-- Gold: every scorable forecast next to its truth, for the submitted model and a persistence baseline.
-- Requires the predictions file ({{predictions}}). Risk of architecture a: P = 1 / (1 + exp(-k (x - x0_a))).
-- A case is scored when its truth is available and inside [0, 1]; the event label is P_true >= 0.5.

CREATE OR REPLACE VIEW bronze.predictions AS SELECT * FROM {{predictions}};

-- grain: one row per delivered prediction
CREATE OR REPLACE TABLE silver.predictions AS
SELECT trim(task_id::VARCHAR)                                  AS task,
       trim(architecture_id::VARCHAR)                          AS arch,
       lpad(trim(county_fips::VARCHAR), 5, '0')                AS county,
       epoch(CAST(trim(issue_time::VARCHAR) AS TIMESTAMPTZ))   AS issue_s,
       epoch(CAST(trim(target_time::VARCHAR) AS TIMESTAMPTZ))  AS target_s,
       CAST(predicted_x AS DOUBLE)                             AS predicted_x,
       CAST(predicted_risk AS DOUBLE)                          AS predicted_risk
FROM bronze.predictions;

CREATE OR REPLACE MACRO risk(x, x0, k) AS 1 / (1 + exp(-greatest(least(k * (x - x0), 700), -700)));

-- grain: one row per (model, scored request)
CREATE OR REPLACE TABLE gold.scored_cases AS
WITH truth AS (
    SELECT l.*, risk(l.truth_x, a.x0, a.k) AS true_risk, a.x0, a.k
    FROM gold.labels l JOIN ref.architectures a USING (arch)
    WHERE l.truth_x IS NOT NULL AND isfinite(l.truth_x) AND l.truth_x BETWEEN 0 AND 1
)
SELECT 'model' AS model, t.task, t.arch, t.county, t.issue_s, t.target_s, t.lead_minutes, t.issue_time,
       t.truth_x, t.true_risk, p.predicted_x, p.predicted_risk
FROM truth t
JOIN silver.predictions p USING (task, arch, county, issue_s, target_s)
UNION ALL
-- persistence: the latest observation visible at the issue time, held for every lead (0 when there is none)
SELECT 'persistence', t.task, t.arch, t.county, t.issue_s, t.target_s, t.lead_minutes, t.issue_time,
       t.truth_x, t.true_risk, coalesce(h.hist_latest_x, 0), risk(coalesce(h.hist_latest_x, 0), t.x0, t.k)
FROM truth t
JOIN gold.history_features h USING (task, county, issue_s);
