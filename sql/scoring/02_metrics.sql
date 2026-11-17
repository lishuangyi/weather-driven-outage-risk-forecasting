-- Gold: the challenge metric in SQL, overall and broken down.
--   score = 100 * (0.75 * AUPRC_norm + 0.25 * (1 - MAE)),  on the risk scale
--   AUPRC = average precision of predicted_risk for the label (true risk >= 0.5); tied predictions form one
--           threshold (as scikit-learn); AUPRC_norm = clip((AP - pi) / (1 - pi), 0, 1), pi = positive rate;
--           0 when a group has only positives or only negatives.
--   MAE   = mean |predicted_risk - true risk|.
-- 'pooled' is the official number of each task. The other dimensions are diagnostics: their average is NOT the
-- pooled score, because AUPRC is not additive.

CREATE OR REPLACE TABLE gold.case_groups AS
WITH cases AS (
    SELECT model, task, predicted_risk AS pred, true_risk, (true_risk >= 0.5)::INT AS y,
           arch, county,
           CASE WHEN task = 'A' THEN
                CASE WHEN lead_minutes <= 360 THEN 'A 01-06 h' WHEN lead_minutes <= 720 THEN 'A 07-12 h'
                     WHEN lead_minutes <= 1440 THEN 'A 13-24 h' ELSE 'A 25-48 h' END
           ELSE CASE WHEN lead_minutes <= 15 THEN 'B 005-015 min' WHEN lead_minutes <= 60 THEN 'B 020-060 min'
                     WHEN lead_minutes <= 180 THEN 'B 065-180 min' ELSE 'B 185-360 min' END END AS lead_band,
           strftime(issue_time, '%Y-%m') AS month
    FROM gold.scored_cases
)
SELECT model, task, 'pooled' AS dimension, 'all' AS grp, pred, true_risk, y FROM cases
UNION ALL SELECT model, task, 'architecture', arch, pred, true_risk, y FROM cases
UNION ALL SELECT model, task, 'county', county, pred, true_risk, y FROM cases
UNION ALL SELECT model, task, 'lead', lead_band, pred, true_risk, y FROM cases
UNION ALL SELECT model, task, 'month', month, pred, true_risk, y FROM cases;

CREATE OR REPLACE TABLE gold.metrics AS
WITH thresholds AS (            -- one row per distinct predicted risk: cases and positives at that threshold
    SELECT model, task, dimension, grp, pred, count(*) AS n, sum(y) AS tp
    FROM gold.case_groups GROUP BY ALL
),
curve AS (                      -- precision at each threshold, walking from the highest predicted risk down
    SELECT *,
           sum(tp) OVER w / sum(n) OVER w                                AS precision_at,
           sum(tp) OVER (PARTITION BY model, task, dimension, grp)       AS positives
    FROM thresholds
    WINDOW w AS (PARTITION BY model, task, dimension, grp ORDER BY pred DESC ROWS UNBOUNDED PRECEDING)
),
ap AS (                         -- AP = sum over thresholds of (recall gain) x (precision)
    SELECT model, task, dimension, grp, sum(tp / positives * precision_at) AS ap
    FROM curve WHERE positives > 0 GROUP BY ALL
),
summary AS (
    SELECT model, task, dimension, grp, count(*) AS cases, sum(y) AS positives,
           avg(y) AS positive_rate, avg(abs(pred - true_risk)) AS mae_risk
    FROM gold.case_groups GROUP BY ALL
)
SELECT s.model, s.task, s.dimension, s.grp, s.cases, s.positives, s.positive_rate,
       CASE WHEN s.positives > 0 AND s.positives < s.cases THEN ap.ap ELSE 0 END AS auprc,
       CASE WHEN s.positives > 0 AND s.positives < s.cases
            THEN greatest(least((ap.ap - s.positive_rate) / (1 - s.positive_rate), 1), 0) ELSE 0 END AS auprc_norm,
       s.mae_risk,
       100 * (0.75 * auprc_norm + 0.25 * (1 - s.mae_risk)) AS score
FROM summary s LEFT JOIN ap USING (model, task, dimension, grp)
ORDER BY model, task, dimension, grp;

-- Headline: pooled task scores, generalisation term over architectures and the weighted technical score
--   g = 100 * clip(mean_a(s_a) - std_a(s_a), 0, 1),  s_a = (score_A,a + score_B,a) / 200
--   technical = 0.4 * A + 0.4 * B + 0.1 * g  (the remaining 0.1 of the challenge is compliance, not computed here)
CREATE OR REPLACE TABLE gold.scoreboard AS
WITH per_arch AS (
    SELECT model, grp AS arch, sum(score) / 200 AS s
    FROM gold.metrics WHERE dimension = 'architecture' GROUP BY ALL HAVING count(DISTINCT task) = 2
),
g AS (
    SELECT model, 100 * greatest(least(avg(s) - stddev_pop(s), 1), 0) AS generalization
    FROM per_arch GROUP BY model HAVING count(*) = (SELECT count(*) FROM ref.architectures)
),
pooled AS (
    PIVOT (SELECT model, task, score FROM gold.metrics WHERE dimension = 'pooled') ON task IN ('A', 'B') USING any_value(score)
)
SELECT p.model, p."A" AS task_a, p."B" AS task_b, g.generalization,
       0.4 * p."A" + 0.4 * p."B" + 0.1 * g.generalization AS technical_score
FROM pooled p LEFT JOIN g USING (model)
ORDER BY model;
