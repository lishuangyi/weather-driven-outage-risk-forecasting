"""Checks of the full-year input directories built by build_years_local.sh (not used by infer.py).

    python data_pipeline/validate_years.py [--inputs data/inputs] [--years 2022 2023 2024 2025]

Per year (data/inputs/input_<year>_full):
  requests   one row per official grid point: days x (48 + 4 x 72) x 5 counties x 4 architectures
  weather    a 00Z run for every issue day and a previous-day 18Z run for every issue day, for all 40 counties, with
             HRRR statistics; missing run days are listed
  outages    time range, records per scored county, largest gap
  radar      time range, availability share, days without any frame
A missing or incomplete run passes only when the stored source records show that the public archive does not have it
(00Z: no VALID row for the run in the annual statistics file; 18Z: every object of the day answered NoSuchKey).
For 2025 the rows shared with the submitted data/inputs/input_2025 (issues 2025-06-01 .. 08-31) must be identical.
Exit code 1 when a check fails.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

UTC = timezone.utc
FAILED = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f': {detail}' if detail else ''), flush=True)
    if not ok:
        FAILED.append(name)


def days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


# Optional raw-data archive used only to explain known source gaps (set OUTAGE_RAW_ROOT; skipped when absent).
RAW = Path(os.environ.get('OUTAGE_RAW_ROOT', Path(__file__).resolve().parents[1] / 'data' / 'raw'))
W00 = RAW / 'weather/current_40c'
D18 = RAW / 'weather/hrrr_18z_current40/daily'


def archive_gaps_00z(year: int) -> set[date]:
    """00Z runs whose statistics are not all VALID in the annual file (source object missing or undecodable)."""
    out = set()
    for y in (year - 1, year):
        path = W00 / f'weather_{y}_full_polygon_robust.parquet'
        if path.exists():
            bad = (pl.scan_parquet(path).filter(pl.col('quality_status') != 'VALID')
                   .select(pl.col('model_run_utc').str.slice(0, 10).unique()).collect().to_series().to_list())
            out |= {date.fromisoformat(d) for d in bad}
    return out


def archive_gap_18z(day: date) -> bool:
    """True when the day's receipt shows that no object of the 18Z run exists in the archive (NoSuchKey)."""
    import json
    path = D18 / f'{day:%Y%m%d}_18z.json'
    if not path.exists():
        return False
    objects = json.loads(path.read_text())['objects']
    return bool(objects) and all(o['status'] != 'OK' and 'NoSuchKey' in o.get('error', '') for o in objects)


def year_checks(directory: Path, year: int) -> None:
    print(f'== {directory.name}', flush=True)
    first, last = date(year, 1, 1), date(year, 12, 31)
    n_days = (last - first).days + 1
    r = pl.scan_parquet(directory / 'requests/forecast_requests.parquet')
    rows = r.select(pl.len()).collect().item()
    check('requests', rows == n_days * (48 + 4 * 72) * 20, f'{rows} rows for {n_days} issue days')
    issues = r.select(pl.col('issue_time').min().alias('a'), pl.col('issue_time').max().alias('b')).collect().row(0)
    check('request window', issues == (f'{year}-01-01T00:00:00Z', f'{year}-12-31T18:00:00Z'), f'{issues[0]} .. {issues[1]}')

    w = pl.scan_parquet(directory / 'weather/weather_forecasts.parquet')
    runs = (w.group_by('model_run_time').agg(pl.col('county_fips').n_unique().alias('counties'), pl.len().alias('rows'),
                                             pl.col('TMP__mean').is_not_null().mean().alias('tmp'),
                                             pl.col('GUST__mean').is_not_null().mean().alias('gust')).collect())
    for hour, wanted, label in ((0, days(first - timedelta(days=1), last), '00Z runs (day before the first issue .. last issue day)'),
                                (18, days(first - timedelta(days=1), last - timedelta(days=1)), 'previous-day 18Z runs')):
        have = runs.filter(pl.col('model_run_time').dt.hour() == hour)
        got = set(have['model_run_time'].dt.date().to_list())
        missing = [d for d in wanted if d not in got]
        thin = have.filter((pl.col('counties') < 40) | (pl.col('tmp') < 0.99) | (pl.col('gust') < 0.99))['model_run_time'].dt.date().to_list()
        if hour == 0:   # the run of 2021-12-31 is before the first annual file
            gaps = archive_gaps_00z(year) | {date(2021, 12, 31)}
        else:
            gaps = {d for d in missing if archive_gap_18z(d)}
        real, real_thin = [d for d in missing if d not in gaps], [d for d in thin if d not in gaps]
        known = sorted(d for d in set(missing) | set(thin) if d in gaps)
        check(label, not real and not real_thin, f'{len(got)} of {len(wanted)} days'
              + (f'; UNEXPLAINED missing {[str(d) for d in real][:12]}{" ..." if len(real) > 12 else ""}' if real else '')
              + (f'; UNEXPLAINED incomplete {[str(d) for d in real_thin][:8]}' if real_thin else '')
              + (f'; {len(known)} not (fully) in the public archive: {[str(d) for d in known]}' if known else ''))
    lead = w.select(((pl.col('target_time') - pl.col('model_run_time')).dt.total_hours()).max()).collect().item()
    check('weather leads', lead >= 48, f'up to {lead} h; {w.select(pl.len()).collect().item()} rows')

    o = pl.read_parquet(directory / 'eaglei/eaglei_outages.parquet')
    per = o.group_by('county_fips').agg(pl.len().alias('n'), pl.col('timestamp').min().alias('a'), pl.col('timestamp').max().alias('b'),
                                        pl.col('timestamp').sort().diff().max().alias('gap')).sort('county_fips')
    start_ok = per['a'].max() <= datetime(year, 1, 1, tzinfo=UTC) - timedelta(days=7)
    end_ok = per['b'].min() >= datetime(year, 12, 31, tzinfo=UTC)
    check('outages', bool(per.height == 5 and start_ok and end_ok),
          '; '.join(f"{c} {n} records, largest gap {g.total_seconds() / 3600:.0f} h" for c, n, _, _, g in per.iter_rows())
          + f"; {o['timestamp'].min():%Y-%m-%d} .. {o['timestamp'].max():%Y-%m-%d %H:%M}")

    rd = pl.read_parquet(directory / 'radar/mrms_county_series.parquet')
    got = set(rd.filter(pl.col('avail') > 0)['grid_time'].dt.date().unique().to_list())
    missing = [d for d in days(first - timedelta(days=1), last) if d not in got]
    check('radar', bool(rd['county_fips'].n_unique() == 5 and len(missing) <= 3),
          f"{rd.height} rows, availability {rd['avail'].mean():.3f}, {rd['grid_time'].min():%Y-%m-%d} .. {rd['grid_time'].max():%Y-%m-%d}"
          + (f'; days without a frame: {[str(d) for d in missing][:10]}' if missing else ''))


def same_rows(name: str, new: pl.DataFrame, old: pl.DataFrame, keys: list[str]) -> None:
    """Every row of old must be in new with identical values (nulls equal)."""
    joined = old.join(new, on=keys, how='left', suffix='__new', coalesce=False)
    absent = joined.filter(pl.col(keys[-1] + '__new').is_null()).height
    values = [c for c in old.columns if c not in keys]
    differing = joined.filter(pl.any_horizontal(*[~pl.col(c).eq_missing(pl.col(c + '__new')) for c in values])).height if values else 0
    check(name, absent == 0 and differing == 0, f'{old.height} submitted rows: {absent} absent from the full year, {differing} with a different value')


def compare_2025(full: Path, submitted: Path) -> None:
    print(f'== {full.name} against the submitted {submitted.name}', flush=True)
    for table, keys in (('weather/weather_forecasts', ['county_fips', 'model_run_time', 'target_time']),
                        ('eaglei/eaglei_outages', ['county_fips', 'timestamp']),
                        ('radar/mrms_county_series', ['county_fips', 'grid_time']),
                        ('requests/forecast_requests', ['task_id', 'architecture_id', 'county_fips', 'issue_time', 'target_time']),
                        ('mcc/MCC', ['county_fips'])):
        old = pl.read_parquet(submitted / f'{table}.parquet')
        new = pl.read_parquet(full / f'{table}.parquet')
        check(f'{table} columns', new.columns == old.columns and new.schema == old.schema)
        same_rows(table, new, old, keys)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'inputs')
    p.add_argument('--years', type=int, nargs='+', default=[2022, 2023, 2024, 2025])
    a = p.parse_args()
    for year in a.years:
        directory = a.inputs / f'input_{year}_full'
        if not directory.is_dir():
            check(f'input_{year}_full', False, 'not built')
            continue
        year_checks(directory, year)
        if year == 2025 and (a.inputs / 'input_2025').is_dir():
            compare_2025(directory, a.inputs / 'input_2025')
    print('\nALL CHECKS PASSED' if not FAILED else f'\nFAILED: {FAILED}')
    sys.exit(1 if FAILED else 0)
