"""Weather features from the forecast runs available at the issue time.

A forecast row is usable at issue T when model_run_time <= T and availability_time <= T. All expressions are
vectorised over every (county, issue, target) at once; each (issue, target) group is independent.

  select_weather      Task A: per county and target, the latest usable run ('latest'), or a 00Z run with the run of
                      T - 6 h preferred ('previous18'); plus the mean of the same variable over the other counties of
                      the county's region, run age and an availability flag; targets beyond the horizon are NaN
  task_b_weather      Task B: 00Z runs only, hourly endpoints interpolated to the 15-minute target when both endpoints
                      come from the same run
  with_lagavg         12 lagavg_* columns: average of the current and the previous-day 18Z run for six strong variables
  add_path_features   35 Task A columns along the forecast path (rolling max over leads, change, spatial gap, history
                      interactions)
  add_process_features 22 Task B columns (rolling max over 15-minute leads, change, history deviations, gust gap)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..inputs import VARIABLES, WEATHER_FIELDS

WEATHER_FEATURES = WEATHER_FIELDS + [f'neighbor_{v}' for v in VARIABLES] + ['weather_available', 'weather_run_age_hours']
KEY = ['county', 'issue', 'target']
STRONG = ['GUST', 'APCP_1hr_acc_fcst', 'CAPE', 'PRATE', 'TMP', 'PRES']
LAGAVG = [f'{v}__{s}' for v in STRONG for s in ('mean', 'max')]


def select_weather(weather: pl.DataFrame, keys: pl.DataFrame, max_lead_hours: int, policy: str,
                   counties: list[str], region_of: dict[str, str]) -> pl.DataFrame:
    """Weather columns (Float32) for each (county, issue, target) of keys.

    policy: 'latest' (latest usable run), 'previous18' (00Z or T - 6 h, the latter preferred) or 'b_00z' (00Z only,
    latest availability)."""
    requests = keys.select(KEY).unique()
    all_requests = keys.select('issue', 'target').unique().join(pl.DataFrame({'county': counties}), how='cross')
    cand = all_requests.join(weather.rename({'run': 'model_run_utc'}), on=['county', 'target'], how='inner').filter(
        (pl.col('availability') <= pl.col('issue')) & (pl.col('model_run_utc') <= pl.col('issue')))
    if policy == 'latest':
        cand = cand.sort(['model_run_utc', 'availability']).unique(KEY, keep='last')
    elif policy == 'b_00z':
        cand = cand.filter(pl.col('model_run_utc') % 86400 == 0)
        cand = cand.sort(['availability', 'model_run_utc']).unique(KEY, keep='last')
    elif policy == 'previous18':
        cand = cand.filter((pl.col('model_run_utc') % 86400 == 0) | (pl.col('model_run_utc') == pl.col('issue') - 6 * 3600))
        cand = cand.with_columns((pl.col('model_run_utc') == pl.col('issue') - 6 * 3600).alias('_prefer_previous18'))
        cand = cand.sort(['_prefer_previous18', 'model_run_utc', 'availability']).unique(KEY, keep='last')
    else:
        raise ValueError(policy)
    cand = cand.with_columns(pl.lit(1.).alias('weather_available'),
                             ((pl.col('issue') - pl.col('model_run_utc')) / 3600).alias('weather_run_age_hours'))
    cand = cand.with_columns(pl.col('county').replace_strict(region_of).alias('region'))
    for v in VARIABLES:  # mean over the other counties of the region with a finite value
        col = v + '__mean'
        finite = pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(None)
        cand = cand.with_columns(finite.sum().over('region', 'issue', 'target').alias('_sum'),
                                 finite.count().over('region', 'issue', 'target').alias('_n'))
        cand = cand.with_columns(((pl.col('_sum') - finite.fill_null(0)) / (pl.col('_n') - finite.is_not_null().cast(pl.Int64)))
                                 .alias('neighbor_' + v))
    out = requests.join(cand.select(*KEY, *WEATHER_FEATURES, 'model_run_utc', 'availability'), on=KEY, how='left')
    out = out.with_columns(pl.col('weather_available').fill_null(0.))
    beyond = (pl.col('target') - pl.col('issue')) > max_lead_hours * 3600
    return out.with_columns(*[pl.when(beyond).then(float('nan')).otherwise(pl.col(c)).cast(pl.Float32).alias(c)
                              for c in WEATHER_FEATURES])


def task_b_weather(frame: pl.DataFrame, weather: pl.DataFrame, counties, region_of) -> pl.DataFrame:
    """Replace the weather columns of the Task B frame (15-minute targets) by the 00Z-run interpolation."""
    keys = frame.select('county', 'issue', 'target', 'lead_minutes').with_columns(
        (pl.col('target') // 3600 * 3600).alias('left_hour'), ((pl.col('target') + 3599) // 3600 * 3600).alias('right_hour'))
    hourly = pl.concat([keys.select('county', 'issue', pl.col('left_hour').alias('target')),
                        keys.select('county', 'issue', pl.col('right_hour').alias('target'))]).unique()
    w = select_weather(weather, hourly, 48, 'b_00z', counties, region_of)
    a = w.rename({'target': 'left_hour', **{c: 'left_' + c for c in WEATHER_FEATURES + ['model_run_utc', 'availability']}})
    b = w.rename({'target': 'right_hour', **{c: 'right_' + c for c in WEATHER_FEATURES + ['model_run_utc', 'availability']}})
    z = keys.join(a, on=['county', 'issue', 'left_hour'], how='left').join(b, on=['county', 'issue', 'right_hour'], how='left')
    exact = pl.col('left_hour') == pl.col('right_hour')
    both = pl.col('left_model_run_utc').is_not_null() & pl.col('right_model_run_utc').is_not_null()
    incompatible = both & (pl.col('left_model_run_utc') != pl.col('right_model_run_utc'))
    run = pl.when(incompatible).then(None).otherwise(pl.coalesce('right_model_run_utc', 'left_model_run_utc'))
    fraction = (pl.col('target') - pl.col('left_hour')) / 3600
    values = []
    for c in WEATHER_FIELDS:
        interpolated = pl.when(both & ~incompatible).then(pl.col('left_' + c) + fraction * (pl.col('right_' + c) - pl.col('left_' + c))).otherwise(None)
        values.append(pl.when(exact).then(pl.col('left_' + c)).otherwise(interpolated).cast(pl.Float32).alias(c))
    values.extend(pl.col('right_' + c).cast(pl.Float32).alias(c) for c in WEATHER_FEATURES if c.startswith('neighbor_'))
    values.extend([run.is_not_null().cast(pl.Float32).alias('weather_available'),
                   ((pl.col('issue') - run) / 3600).cast(pl.Float32).alias('weather_run_age_hours')])
    z = z.with_columns(values).select('county', 'issue', 'target', 'lead_minutes', *WEATHER_FEATURES)
    return frame.drop(WEATHER_FEATURES).join(z, on=['county', 'issue', 'target', 'lead_minutes'], how='left',
                                             maintain_order='left').select(frame.columns)


def with_weather(frame: pl.DataFrame, selected: pl.DataFrame) -> pl.DataFrame:
    """Frame with its weather columns replaced by a select_weather() result."""
    return frame.drop(WEATHER_FEATURES).join(selected.select(*KEY, *WEATHER_FEATURES), on=KEY, maintain_order='left')


def with_lagavg(f00: pl.DataFrame, f18: pl.DataFrame) -> pl.DataFrame:
    """lagavg_* = mean of the current and the previous-day 18Z value where the 18Z run (age exactly 6 h) has a value,
    else the current value. f00 and f18 are row-aligned."""
    is18 = pl.Series(np.nan_to_num(f18['weather_run_age_hours'].to_numpy().astype(np.float64), nan=-1.) == 6)
    g = f00.with_columns(*[pl.when(is18).then(f18[c].fill_nan(None)).otherwise(None).alias(f'{c}_18z') for c in LAGAVG])
    return g.with_columns(*[pl.when(pl.col(f'{c}_18z').is_not_null()).then((pl.col(c).fill_nan(None) + pl.col(f'{c}_18z')) / 2)
                            .otherwise(pl.col(c)).alias(f'lagavg_{c}') for c in LAGAVG])


def add_path_features(frame: pl.DataFrame) -> pl.DataFrame:
    """The 35 Task A 'combined' columns, per (county, issue) along the hourly leads."""
    frame = frame.sort(['county', 'issue', 'lead_minutes'])
    keys = ['county', 'issue']
    expressions = []
    for variable in ['GUST', 'CAPE', 'PRATE', 'APCP_1hr_acc_fcst', 'TMP', 'PRES']:
        col = variable + '__mean'
        for hours in (6, 24):
            expressions.append(pl.col(col).fill_nan(None).rolling_max(hours, min_samples=1).over(keys).alias(f'path_{variable}_max{hours}'))
        expressions += [pl.col(col).fill_nan(None).cum_max().over(keys).alias(f'path_{variable}_max_since_issue'),
                        pl.col(col).diff().over(keys).alias(f'path_{variable}_change')]
    expressions.append(pl.col('APCP_1hr_acc_fcst__mean').fill_nan(None).rolling_sum(6, min_samples=1).over(keys).alias('path_precip_sum6'))
    for variable in ['GUST', 'CAPE', 'PRATE', 'APCP_1hr_acc_fcst']:
        expressions.append((pl.col(variable + '__max') - pl.col(variable + '__mean')).alias(f'{variable}_spatial_extreme_gap'))
    for hours in (1, 3, 6, 12, 24):
        expressions.append((pl.col('hist_latest_x') - pl.col(f'roll_mean_{hours * 60}')).alias(f'history_current_minus_mean{hours}h'))
    expressions.append((pl.col('hist_latest_x') * pl.col('weather_run_age_hours')).alias('history_weather_age_interaction'))
    return frame.with_columns(expressions)


def add_process_features(frame: pl.DataFrame) -> pl.DataFrame:
    """The 22 Task B process columns, per (county, issue) along the 15-minute leads."""
    frame = frame.sort('county', 'issue', 'lead_minutes')
    keys = ['county', 'issue']
    expressions = []
    for variable in ['GUST', 'CAPE', 'PRATE', 'APCP_1hr_acc_fcst', 'TMP', 'PRES']:
        col = pl.col(variable + '__mean').fill_nan(None)
        for minutes in (60, 180):
            expressions.append(col.rolling_max(minutes // 15, min_samples=1).over(keys).alias(f'b_{variable}_forecast_max{minutes}m'))
        expressions.append(col.diff().over(keys).alias(f'b_{variable}_forecast_change15m'))
    for minutes in (60, 180, 360):
        expressions.append((pl.col('hist_latest_x') - pl.col(f'roll_mean_{minutes}')).alias(f'b_history_current_minus_mean{minutes}m'))
    expressions.append((pl.col('GUST__max') - pl.col('GUST__mean')).alias('b_gust_spatial_gap'))
    return frame.with_columns(expressions)
