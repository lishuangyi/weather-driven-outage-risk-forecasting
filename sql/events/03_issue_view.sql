-- Events step 3: what an alerting rule could know at each Task A issue time T (one row per county and issue).
--
-- gate_ok      the county has been quiet: {{quiet_hours}} complete hourly shares below {{x0}}, computed from the
--              records visible at T (gold.visible_history), ending at T, or at T-1 h when the hour ending at T is
--              not complete yet
-- latest_x     the latest of those visible hourly shares (score of near-threshold persistence)
-- model_score  max predicted share over the targets T+1 h ... T+{{horizon_hours}} h (Task A, ARC01)
-- weather_score max HRRR county gust (wx_gust_max, run visible at T) over the same targets; NULL unless all present
-- verified     all hourly truths from T-{{quiet_hours}}+1 h to T+{{horizon_hours}} h exist, so every onset in the
--              window (T, T+{{horizon_hours}} h] is known (truth, used only to score the alerts)
WITH issues AS (
    SELECT DISTINCT county, issue_s FROM silver.issues WHERE task = 'A'
),
visible_hours AS (
    SELECT v.county, v.issue_s, (ceil(v.ts_s / 3600) * 3600)::BIGINT AS hour_s,
           sum(v.customers_out / c.total_customers ORDER BY v.ts_s) / 4 AS x   -- ordered: deterministic
    FROM gold.visible_history v JOIN silver.customers c USING (county)
    JOIN issues i USING (county, issue_s)
    WHERE v.ts_s % 900 = 0 AND v.ts_s > v.issue_s - ({{quiet_hours}} + 1) * 3600
    GROUP BY ALL
    HAVING count(*) = 4 AND bool_and(v.customers_out / c.total_customers BETWEEN 0 AND 1)
),
latest_end AS (            -- end of the most recent complete visible hour: T, else T - 1 h
    SELECT i.county, i.issue_s,
           CASE WHEN EXISTS (SELECT 1 FROM visible_hours h WHERE h.county = i.county AND h.issue_s = i.issue_s
                                                             AND h.hour_s = i.issue_s)
                THEN i.issue_s ELSE i.issue_s - 3600 END AS end_s
    FROM issues i
),
gate AS (
    SELECT e.county, e.issue_s,
           count(h.hour_s) FILTER (WHERE h.x < {{x0}}) = {{quiet_hours}} AS gate_ok,
           max(h.x) FILTER (WHERE h.hour_s = e.end_s) AS latest_x
    FROM latest_end e
    LEFT JOIN visible_hours h
      ON h.county = e.county AND h.issue_s = e.issue_s
     AND h.hour_s > e.end_s - {{quiet_hours}} * 3600 AND h.hour_s <= e.end_s
    GROUP BY ALL
),
model AS (
    SELECT county, issue_s, max(predicted_x) AS model_score
    FROM silver.predictions
    WHERE task = 'A' AND arch = 'ARC01' AND target_s > issue_s AND target_s <= issue_s + {{horizon_hours}} * 3600
    GROUP BY ALL
),
weather AS (
    SELECT county, issue_s,
           CASE WHEN count(wx_gust_max) = {{horizon_hours}} THEN max(wx_gust_max) END AS weather_score
    FROM gold.examples
    WHERE task = 'A' AND target_s > issue_s AND target_s <= issue_s + {{horizon_hours}} * 3600
    GROUP BY ALL
),
verification AS (
    SELECT i.county, i.issue_s, count(t.hour_s) = {{quiet_hours}} + {{horizon_hours}} AS verified
    FROM issues i
    LEFT JOIN gold.hourly_truth t
      ON t.county = i.county
     AND t.hour_s > i.issue_s - {{quiet_hours}} * 3600 AND t.hour_s <= i.issue_s + {{horizon_hours}} * 3600
    GROUP BY ALL
)
SELECT i.county, i.issue_s, coalesce(g.gate_ok, FALSE) AS gate_ok, g.latest_x, m.model_score, w.weather_score,
       v.verified
FROM issues i
LEFT JOIN gate g USING (county, issue_s)
LEFT JOIN model m USING (county, issue_s)
LEFT JOIN weather w USING (county, issue_s)
JOIN verification v USING (county, issue_s)
ORDER BY i.county, i.issue_s;
