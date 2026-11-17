-- Reliability (calibration) of the predicted risk: cases binned by predicted risk, against the share of real events.
-- A well-calibrated forecast has event_rate close to mean_predicted in every bin.
SELECT task, model,
       floor(least(predicted_risk, 0.9999) * 10) / 10 AS bin_from,
       count(*)                                        AS cases,
       round(avg(predicted_risk), 3)                   AS mean_predicted,
       round(avg((true_risk >= 0.5)::INT), 3)          AS event_rate
FROM gold.scored_cases
GROUP BY ALL
ORDER BY task, model, bin_from;
