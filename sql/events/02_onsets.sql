-- Events step 2: outage onsets at threshold {{x0}}.
-- An onset is an hour H whose hourly share reaches the threshold (x >= x0) after {{quiet_hours}} complete hours
-- H-1 h ... H-{{quiet_hours}} h that were all below it. A gap breaks the quiet run, so a high value right after a gap
-- has an unknown onset status and is not an onset. One episode gives one onset: a new onset needs the quiet hours again.
--
-- grain: one row per onset
SELECT county, hour_s
FROM (
    SELECT county, hour_s, x,
           count(*) FILTER (WHERE x < {{x0}}) OVER (
               PARTITION BY county ORDER BY hour_s
               RANGE BETWEEN {{quiet_hours}} * 3600 PRECEDING AND 3600 PRECEDING) AS quiet_before
    FROM gold.hourly_truth
)
WHERE x >= {{x0}} AND quiet_before = {{quiet_hours}}
ORDER BY county, hour_s;
