"""Assemble an input directory in the organizer layout (Parquet) from the aggregated public data.

    python data_pipeline/assemble_inputs.py --start 2025-06-01 --end 2025-09-01 --out data/inputs/input_2025 \
        --weather-00z <weather_YYYY_full_polygon_robust.parquet ...> --weather-18z-dir <hrrr18z/daily> \
        --snow-axis <snow_axis.parquet> --outages <eaglei_counts_YYYY.parquet ...> --radar <radar_series.parquet>

Inputs are the outputs of the earlier stages (see README.md): county-polygon HRRR statistics in long form (one row per
county, run, target and variable), the previous-day 18Z runs in daily files of the same form, the ASNOW lead axis of
each 00Z run, EAGLE-I counts per county and 15-minute time, and the MRMS county frame series.

Written tables (issues from --start 00Z to the last 18Z before --end):
  requests   Task A 00Z issues x 48 hourly leads, Task B 00/06/12/18Z issues x 72 five-minute leads, x counties x ARC01-04
  mcc        official customer totals of the scored counties
  weather    00Z runs from the day before --start, previous-day 18Z runs (run-time replay: availability = run time),
             14 variables x mean / std / max / p95 for the scored and neighbouring counties
  eaglei     counts of the scored counties before --end (observation-time replay: source_record_time = timestamp)
  radar      frame statistics of the scored counties (availability = grid time + 4 min)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

VARIABLES = ['TMP', 'DPT', 'UGRD', 'VGRD', 'GUST', 'CAPE', 'RH', 'PRES', 'MSLMA', 'TCDC', 'VIS', 'PRATE', 'APCP_1hr_acc_fcst', 'ASNOW_acc_fcst']
STATS = ['mean', 'std', 'max', 'p95']
WF = [f'{v}__{s}' for v in VARIABLES for s in STATS]
SNOW = [f'ASNOW_acc_fcst__{s}' for s in STATS]
BASIC = ['temperature_c', 'relative_humidity_pct', 'wind_speed_mps', 'precipitation_mm', 'pressure_hpa']
COUNTIES = {'06037': 3762600, '12086': 951391, '26163': 441528, '33015': 154048, '48201': 1827686}
ARCS = ['ARC01', 'ARC02', 'ARC03', 'ARC04']
RADAR_LATENCY_S = 240
UTC = timezone.utc


def epoch(day: str) -> int:
    return int(datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=UTC).timestamp())


def utc(seconds: pl.Expr) -> pl.Expr:
    return pl.from_epoch(seconds.cast(pl.Int64), time_unit='s').dt.replace_time_zone('UTC').dt.cast_time_unit('us')


def iso(seconds: pl.Expr) -> pl.Expr:
    return pl.from_epoch(seconds.cast(pl.Int64), time_unit='s').dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def requests(start: int, end: int, counties: list[str]) -> pl.DataFrame:
    rows = []
    for day in range(start, end, 86400):
        for task, slots, leads in (('A', [0], range(60, 2881, 60)), ('B', [0, 6, 12, 18], range(5, 361, 5))):
            for slot in slots:
                issue = day + slot * 3600
                for county in counties:
                    for lead in leads:
                        for arc in ARCS:
                            rows.append((task, arc, county, issue, issue + lead * 60))
    f = pl.DataFrame(rows, schema=['task_id', 'architecture_id', 'county_fips', 'issue', 'target'], orient='row')
    return f.select('task_id', 'architecture_id', 'county_fips', iso(pl.col('issue')).alias('issue_time'), iso(pl.col('target')).alias('target_time'))


def wide(long: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Long statistics -> one row per (county, run, target): VALID rows only, availability = run time (replay
    assumption), columns <VARIABLE>__<stat> (missing variables null)."""
    f = long.lazy().filter(pl.col('quality_status') == 'VALID').collect()
    values = f.pivot(on='variable', index=['fips_code', 'model_run_utc', 'valid_time_utc'], values=STATS)
    values = values.rename({f'{s}_{v}': f'{v}__{s}' for v in VARIABLES for s in STATS if f'{s}_{v}' in values.columns})
    return values.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(c) for c in WF if c not in values.columns]).select(
        'fips_code', 'model_run_utc', 'valid_time_utc', *WF)


def weather_00z(paths: list[Path], first_run: int, last_run: int, snow_axis: pl.DataFrame) -> pl.DataFrame:
    """00Z runs in [first_run, last_run] from the annual statistics files, with the ASNOW statistics moved to their
    verified lead axis (dropped for runs without a verified axis)."""
    lo, hi = (datetime.fromtimestamp(t, UTC).strftime('%Y-%m-%dT%H:%M:%S') for t in (first_run, last_run + 1))
    long = (pl.concat([pl.scan_parquet(p) for p in paths]).filter((pl.col('model_run_utc') >= lo) & (pl.col('model_run_utc') < hi))
            .select('fips_code', 'model_run_utc', 'valid_time_utc', 'variable', 'quality_status', *STATS)
            .with_columns(*[pl.col(c).str.to_datetime(time_zone='UTC').dt.epoch('s') for c in ('model_run_utc', 'valid_time_utc')]))
    f = wide(long)
    snow = (f.select('fips_code', 'model_run_utc', 'valid_time_utc', *SNOW)
            .join(snow_axis, on=['model_run_utc', 'valid_time_utc'], how='inner')
            .drop('valid_time_utc').rename({'corrected_time_utc': 'valid_time_utc'}))
    return f.drop(SNOW).join(snow, on=['fips_code', 'model_run_utc', 'valid_time_utc'], how='left').select(f.columns)


def weather_18z(directory: Path, first_day: int, last_day: int) -> pl.DataFrame:
    """Previous-day 18Z runs of the dates in [first_day, last_day]."""
    parts = []
    for day in range(first_day, last_day + 1, 86400):
        path = directory / f"{datetime.fromtimestamp(day, UTC).strftime('%Y%m%d')}_18z.parquet"
        if path.exists() and len(pl.read_parquet_schema(path)):
            parts.append(wide(pl.read_parquet(path).select('fips_code', 'model_run_utc', 'valid_time_utc', 'variable', 'quality_status', *STATS)))
    if not parts:  # no 18Z run available: Task A's lag columns fall back to the 00Z run at inference
        return pl.DataFrame(schema={'fips_code': pl.String, 'model_run_utc': pl.Int64, 'valid_time_utc': pl.Int64,
                                    **{c: pl.Float64 for c in WF}})
    return pl.concat(parts, how='vertical_relaxed')


def weather_table(f: pl.DataFrame) -> pl.DataFrame:
    return f.select(pl.col('fips_code').alias('county_fips'), utc(pl.col('model_run_utc')).alias('model_run_time'),
                    utc(pl.col('model_run_utc')).alias('availability_time'), utc(pl.col('valid_time_utc')).alias('target_time'),
                    *[pl.lit(None, dtype=pl.Float64).alias(c) for c in BASIC],
                    *[pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in WF])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--start', required=True, help='first issue day (UTC), e.g. 2025-06-01')
    p.add_argument('--end', required=True, help='day after the last issue day, e.g. 2025-09-01')
    p.add_argument('--counties', default=','.join(COUNTIES), help='scored counties (subset of the five official counties)')
    p.add_argument('--weather-00z', required=True, type=Path, nargs='+', help='annual 00Z statistics file(s)')
    p.add_argument('--weather-18z-dir', type=Path)
    p.add_argument('--snow-axis', required=True, type=Path)
    p.add_argument('--outages', required=True, type=Path, nargs='+', help='annual EAGLE-I count file(s)')
    p.add_argument('--radar', type=Path)
    p.add_argument('--outages-from', help='first day (UTC) of outage history to keep, e.g. 2024-12-01 (default: all given)')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(argv)
    start, end = epoch(a.start), epoch(a.end)
    counties = a.counties.split(',')
    out = a.out
    tables = {'requests/forecast_requests': requests(start, end, counties),
              'mcc/MCC': pl.DataFrame({'county_fips': counties, 'total_customers': [COUNTIES[c] for c in counties]})}
    w = weather_00z(a.weather_00z, start - 86400, end - 86400, pl.read_parquet(a.snow_axis))
    if a.weather_18z_dir:
        w = pl.concat([w, weather_18z(a.weather_18z_dir, start - 86400, end - 2 * 86400).select(w.columns)], how='vertical_relaxed')
    tables['weather/weather_forecasts'] = weather_table(w)
    o = pl.concat([pl.read_parquet(p) for p in a.outages]).unique(['county', 'timestamp'], keep='last', maintain_order=True).filter(pl.col('county').is_in(counties) & (pl.col('timestamp') < end)).sort('county', 'timestamp')
    if a.outages_from:
        o = o.filter(pl.col('timestamp') >= epoch(a.outages_from))
    tables['eaglei/eaglei_outages'] = o.select(pl.col('county').alias('county_fips'), utc(pl.col('timestamp')).alias('timestamp'),
                                               pl.col('count').alias('customers_out'), utc(pl.col('timestamp')).alias('source_record_time'))
    if a.radar:
        r = pl.read_parquet(a.radar).filter(pl.col('county').is_in(counties)).sort('county', 'grid_time')
        tables['radar/mrms_county_series'] = r.select(
            pl.col('county').alias('county_fips'), utc(pl.col('grid_time')).alias('grid_time'),
            utc(pl.col('grid_time') + RADAR_LATENCY_S).alias('availability_time'), pl.col('avail').cast(pl.Boolean).cast(pl.Float64),
            *[pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).cast(pl.Float64).alias(n)
              for c, n in (('f35', 'f35'), ('f50', 'f50'), ('max', 'max_dbz'), ('ring35', 'ring35'))])
    for name, f in tables.items():
        path = out / f'{name}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        f.write_parquet(path, compression='zstd', compression_level=19)
        print(f'{name}: {f.height} rows -> {path}')


if __name__ == '__main__':
    main()
