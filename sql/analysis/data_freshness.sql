-- How fresh was the outage history at forecast time? Age of the latest usable observation at each issue,
-- and how often inputs were missing (hist_missing_flags counts missing lags, windows and slopes).
SELECT task, county,
       count(*)                                                   AS issues,
       round(avg(hist_latest_observation_age_minutes), 1)         AS mean_age_min,
       max(hist_latest_observation_age_minutes)                   AS max_age_min,
       sum((hist_available = 0)::INT)                             AS issues_without_history,
       sum((hist_missing_flags > 0)::INT)                         AS issues_with_missing_inputs
FROM gold.history_features
GROUP BY ALL
ORDER BY task, county;
