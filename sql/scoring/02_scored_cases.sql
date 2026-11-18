-- Scoring step 2: every scorable request of the scope next to its truth, for the submitted model and persistence.
-- Cohort = requests in gold.score_scope whose truth is available and inside [0, 1]. Step 1 guarantees exactly one
-- prediction per scope request, so the model and the baseline are scored on the same cohort, row for row.
-- Risk of architecture a: P = 1 / (1 + exp(-k (x - x0_a))); the event label is P_true >= 0.5.

CREATE OR REPLACE MACRO risk(x, x0, k) AS 1 / (1 + exp(-greatest(least(k * (x - x0), 700), -700)));

-- grain: one row per (model, scored request)
CREATE OR REPLACE TABLE gold.score_cohort AS
SELECT l.*, risk(l.truth_x, a.x0, a.k) AS true_risk, a.x0, a.k
FROM gold.labels l
JOIN gold.score_scope s USING (row_id)
JOIN ref.architectures a ON a.arch = l.arch
WHERE l.truth_x IS NOT NULL AND isfinite(l.truth_x) AND l.truth_x BETWEEN 0 AND 1;

-- grain: one row per (model, cohort request)
CREATE OR REPLACE TABLE gold.scored_cases AS
SELECT 'model' AS model, t.task, t.arch, t.county, t.issue_s, t.target_s, t.lead_minutes, t.issue_time,
       t.truth_x, t.true_risk, p.predicted_x, p.predicted_risk
FROM gold.score_cohort t
JOIN silver.predictions p USING (task, arch, county, issue_s, target_s)
UNION ALL
-- persistence: the latest observation visible at the issue time, held for every lead (0 when there is none)
SELECT 'persistence', t.task, t.arch, t.county, t.issue_s, t.target_s, t.lead_minutes, t.issue_time,
       t.truth_x, t.true_risk, coalesce(h.hist_latest_x, 0), risk(coalesce(h.hist_latest_x, 0), t.x0, t.k)
FROM gold.score_cohort t
JOIN gold.history_features h USING (task, county, issue_s);
