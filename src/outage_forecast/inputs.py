"""Organizer inputs: read once with Polars (CSV or Parquet), validate, convert times to UTC epoch seconds.

Layout of the input directory (challenge input schema; radar is an optional addition):

  requests/forecast_requests.{csv,parquet}   task_id, architecture_id, county_fips, issue_time, target_time
  mcc/MCC.{csv,parquet}                      county_fips, total_customers
  eaglei/eaglei_outages.{csv,parquet}        county_fips, timestamp, customers_out, source_record_time
  weather/weather_forecasts.{csv,parquet}    county_fips, model_run_time, availability_time, target_time, five basic
                                             fields, optional HRRR county statistics <VARIABLE>__<mean|std|max|p95>
  radar/mrms_county_series.{csv,parquet}     county_fips, grid_time, availability_time, avail, f35, f50, max_dbz, ring35

Missing files: requests and MCC are required. A missing outage or weather file counts as a table with a header and no
rows (the affected rows use the fallback models; a warning names the file); an empty (zero-byte), unreadable or
incomplete file is an error. When both formats of a table exist the CSV file is used. Times are ISO 8601 strings with a UTC offset (or Parquet
datetimes); request keys are written back exactly as given.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from .config import Config

REQUEST_KEYS = ('task_id', 'architecture_id', 'county_fips', 'issue_time', 'target_time')
BASIC_WEATHER = ('temperature_c', 'relative_humidity_pct', 'wind_speed_mps', 'precipitation_mm', 'pressure_hpa')
VARIABLES = ('TMP', 'DPT', 'UGRD', 'VGRD', 'GUST', 'CAPE', 'RH', 'PRES', 'MSLMA', 'TCDC', 'VIS', 'PRATE',
             'APCP_1hr_acc_fcst', 'ASNOW_acc_fcst')
STATS = ('mean', 'std', 'max', 'p95')
WEATHER_FIELDS = [f'{v}__{s}' for v in VARIABLES for s in STATS]
RADAR_VALUES = ('avail', 'f35', 'f50', 'max_dbz', 'ring35')
TABLES = {'requests': 'requests/forecast_requests', 'mcc': 'mcc/MCC', 'outages': 'eaglei/eaglei_outages',
          'weather': 'weather/weather_forecasts', 'radar': 'radar/mrms_county_series'}
REQUIRED = ('requests', 'mcc')
OPTIONAL_EMPTY = ('outages', 'weather')   # a missing file = header without rows
_FIPS = re.compile(r'\d{1,5}')
log = logging.getLogger('outage_forecast')


class InputError(ValueError):
    """The organizer input is missing or invalid; the run stops with exit code 2."""


@dataclass
class InputTables:
    requests: pl.DataFrame   # raw key strings + task, arch, county, issue, target (s), lead_minutes, row
    mcc: dict[str, int]
    outages: pl.DataFrame    # county, timestamp (s), source_time (s), count (nullable), row
    weather: pl.DataFrame    # county, run (s), availability (s), target (s), 56 HRRR fields (Float64, null = missing)
    radar: pl.DataFrame | None  # county, grid (s), availability (s), avail, f35, f50, max_dbz, ring35, row


def locate(input_dir: Path, table: str) -> Path | None:
    for suffix in ('.csv', '.parquet'):
        path = input_dir / (TABLES[table] + suffix)
        if path.is_file():
            return path
    return None


def check_layout(input_dir: Path) -> None:
    if not input_dir.is_dir():
        raise InputError(f'input directory not found: {input_dir}')
    missing = [TABLES[t] + '.csv (or .parquet)' for t in REQUIRED if locate(input_dir, t) is None]
    if missing:
        raise InputError(f'required input files missing: {missing}')


def _read_or_empty(input_dir: Path, table: str, required: set[str]) -> pl.DataFrame:
    """The table, or for a missing optional file the same columns as strings with no rows."""
    path = locate(input_dir, table)
    if path is not None:
        return _read(path, required)
    log.warning('WARNING: %s.csv (or .parquet) is missing: read as a table without rows; the affected requests use the '
                'fallback models', TABLES[table])
    return pl.DataFrame({c: pl.Series(c, [], pl.String) for c in sorted(required)}).with_row_index('row')


def _read(path: Path, required: set[str]) -> pl.DataFrame:
    """All columns as raw strings (CSV) or as stored (Parquet); 'row' = source row order."""
    if path.stat().st_size == 0:
        raise InputError(f'{path} is empty (not even a header)')
    try:
        if path.suffix == '.csv':
            frame = pl.read_csv(path, infer_schema=False, encoding='utf8-lossy')
            frame = frame.rename({c: c.lstrip('\ufeff') for c in frame.columns})
        else:
            frame = pl.read_parquet(path)
    except pl.exceptions.PolarsError as error:
        raise InputError(f'{path} cannot be read: {error}') from error
    missing = required - set(frame.columns)
    if missing:
        raise InputError(f'{path} is missing columns: {sorted(missing)}')
    return frame.with_row_index('row')


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError as error:
        raise InputError(f'invalid UTC timestamp: {value!r}') from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise InputError(f'timestamp must include a UTC offset: {value!r}')
    return parsed.astimezone(timezone.utc)


def _seconds(frame: pl.DataFrame, column: str) -> pl.Series:
    """UTC epoch seconds (Float64) of a time column given as ISO strings or Parquet datetimes."""
    s = frame[column]
    if s.dtype == pl.Datetime:
        if s.dtype.time_zone is None:
            raise InputError(f'{column} must be time-zone aware (UTC)')
        if s.null_count():
            raise InputError(f'{column} has empty timestamps')
        return (s.dt.epoch('us') / 1e6).alias(column)
    s = s.cast(pl.String).str.strip_chars()
    unique = s.unique()
    lookup = {v: _parse_utc(v).timestamp() for v in unique.to_list() if v is not None}
    if None in unique.to_list() or '' in lookup:
        raise InputError(f'{column} has empty timestamps')
    if not lookup:  # no rows
        return pl.Series(column, [], pl.Float64)
    return s.replace_strict(lookup, return_dtype=pl.Float64).alias(column)


def _fips(frame: pl.DataFrame, column: str = 'county_fips') -> pl.Series:
    s = frame[column].cast(pl.String).str.strip_chars()
    bad = [v for v in s.unique().to_list() if v is None or not _FIPS.fullmatch(v)]
    if bad:
        raise InputError(f'invalid county FIPS: {bad[:3]!r}')
    return s.str.zfill(5).alias('county')


def _numbers(frame: pl.DataFrame, column: str, what: str) -> pl.Series:
    """Nullable Float64: blank -> null; anything else must parse as a number."""
    s = frame[column]
    if s.dtype == pl.String:
        s = s.str.strip_chars()
        s = pl.select(pl.when(s == '').then(None).otherwise(s).alias(column)).to_series()  # blank or empty -> null
        parsed = s.cast(pl.Float64, strict=False)
        if (parsed.is_null() & s.is_not_null()).any():
            raise InputError(f'invalid {what} value in {column}')
        return parsed
    return s.cast(pl.Float64)


def read_requests(input_dir: Path, config: Config) -> pl.DataFrame:
    path = locate(input_dir, 'requests')
    frame = _read(path, set(REQUEST_KEYS))
    if frame.height == 0:
        raise InputError('forecast_requests holds no requests')
    def key(k):  # the request key as it will be written back; Parquet datetimes in the ISO 'Z' form
        if frame[k].dtype == pl.Datetime:
            return pl.col(k).dt.convert_time_zone('UTC').dt.strftime('%Y-%m-%dT%H:%M:%SZ').alias('key_' + k)
        return pl.col(k).cast(pl.String).alias('key_' + k)
    raw = frame.select(pl.col('row'), *[key(k) for k in REQUEST_KEYS])
    out = raw.with_columns(
        pl.col('key_task_id').str.strip_chars().alias('task'),
        pl.col('key_architecture_id').str.strip_chars().alias('arch'),
        _fips(frame),
        _seconds(frame, 'issue_time').alias('issue_s'),
        _seconds(frame, 'target_time').alias('target_s'))
    bad = out.filter(~pl.col('task').is_in(['A', 'B']).fill_null(False) | ~pl.col('county').is_in(list(config.counties)))
    if bad.height:
        raise InputError(f'unsupported task or county in request row {bad["row"][0] + 2}')
    bad = out.filter(~pl.col('arch').is_in(list(config['risk']['x0'])).fill_null(False))
    if bad.height:
        raise InputError(f'unsupported architecture in request row {bad["row"][0] + 2}')
    out = out.with_columns(((pl.col('target_s') - pl.col('issue_s')) / 60).alias('lead_f'),
                           pl.col('issue_s').cast(pl.Int64).alias('issue'), pl.col('target_s').cast(pl.Int64).alias('target'))
    for task, grid in config['requests'].items():
        part = out.filter(pl.col('task') == task)
        lead = part['lead_f']
        if ((lead < grid['lead_min_minutes']) | (lead > grid['lead_max_minutes']) | (lead % grid['lead_step_minutes'] != 0)).any():
            raise InputError(f'request outside the Task {task} lead grid')
        issue = part['issue_s']
        if ((issue % 3600 != 0) | ~((issue // 3600) % 24).is_in(grid['issue_hours_utc'])).any():
            raise InputError(f'request has an invalid Task {task} issue slot')
    if out.select('task', 'arch', 'county', 'issue_s', 'target_s').is_duplicated().any():
        raise InputError('duplicate forecast request')
    return out.with_columns(pl.col('lead_f').cast(pl.Int64).alias('lead_minutes')).drop('lead_f', 'issue_s', 'target_s')


def read_mcc(input_dir: Path, config: Config, requested: set[str]) -> dict[str, int]:
    frame = _read(locate(input_dir, 'mcc'), {'county_fips', 'total_customers'})
    county = _fips(frame).to_list()
    totals = frame['total_customers'].cast(pl.String).to_list()
    result: dict[str, int] = {}
    for code, raw in zip(county, totals):
        try:
            total = int(raw)
        except (TypeError, ValueError) as error:
            raise InputError(f'invalid MCC denominator for {code}') from error
        if total <= 0 or code in result:
            raise InputError(f'invalid or duplicate MCC denominator for {code}')
        result[code] = total
    for code, fixed in config.counties.items():
        if code in result and result[code] != int(fixed):
            raise InputError(f'MCC denominator of {code} differs from the fixed official value {fixed}')
    missing = requested - set(result)
    if missing:
        raise InputError(f'MCC.csv has no denominator for requested counties {sorted(missing)}')
    return {c: result[c] for c in config.counties if c in result}


def read_outages(input_dir: Path) -> pl.DataFrame:
    frame = _read_or_empty(input_dir, 'outages', {'county_fips', 'timestamp', 'customers_out', 'source_record_time'})
    count = _numbers(frame, 'customers_out', 'customers_out')
    if (count.is_nan() | count.is_infinite() | (count < 0)).any():
        raise InputError('invalid customers_out (negative or non-finite)')
    return pl.DataFrame([frame['row'], _fips(frame), _seconds(frame, 'timestamp').alias('timestamp_s'),
                         _seconds(frame, 'source_record_time').alias('source_s'), count.alias('count')])


def read_weather(input_dir: Path) -> pl.DataFrame:
    """One row per forecast record with at least one usable value; the five basic fields fill TMP/RH/PRES means only
    where the HRRR statistic is missing (temperature to K, pressure to Pa). Wind speed and precipitation without an
    interval are not mapped."""
    frame = _read_or_empty(input_dir, 'weather', {'county_fips', 'model_run_time', 'availability_time', 'target_time', *BASIC_WEATHER})
    basic = {c: _numbers(frame, c, 'weather') for c in BASIC_WEATHER}
    for c, s in basic.items():
        if (s.is_nan() | s.is_infinite()).any():
            raise InputError(f'invalid weather {c}')
    fields = {c: _numbers(frame, c, 'weather') if c in frame.columns else pl.Series(c, [None] * frame.height, pl.Float64)
              for c in WEATHER_FIELDS}
    for source, destination, scale, offset in (('temperature_c', 'TMP__mean', 1.0, 273.15),
                                               ('relative_humidity_pct', 'RH__mean', 1.0, 0.0),
                                               ('pressure_hpa', 'PRES__mean', 100.0, 0.0)):
        fields[destination] = pl.select(pl.when(fields[destination].is_not_null()).then(fields[destination])
                                        .otherwise(basic[source] * scale + offset)).to_series().alias(destination)
    out = pl.DataFrame([frame['row'], _fips(frame), _seconds(frame, 'model_run_time').alias('run'),
                        _seconds(frame, 'availability_time').alias('availability'),
                        _seconds(frame, 'target_time').cast(pl.Int64).alias('target'), *fields.values()])
    return out.filter(pl.any_horizontal(pl.col(WEATHER_FIELDS).is_not_null()))


def read_radar(input_dir: Path) -> pl.DataFrame | None:
    path = locate(input_dir, 'radar')
    if path is None or path.stat().st_size == 0:  # optional table; an empty file means no radar
        return None
    frame = _read(path, {'county_fips', 'grid_time', 'availability_time', *RADAR_VALUES})
    return pl.DataFrame([frame['row'], _fips(frame), _seconds(frame, 'grid_time').cast(pl.Int64).alias('grid'),
                         _seconds(frame, 'availability_time').alias('availability'),
                         *[_numbers(frame, c, 'radar').fill_null(math.nan) for c in RADAR_VALUES]])


def read_inputs(input_dir: Path, config: Config) -> InputTables:
    check_layout(input_dir)
    try:
        requests = read_requests(input_dir, config)
        mcc = read_mcc(input_dir, config, set(requests['county'].unique().to_list()))
        return InputTables(requests, mcc, read_outages(input_dir), read_weather(input_dir), read_radar(input_dir))
    except pl.exceptions.PolarsError as error:
        raise InputError(f'invalid input values: {error}') from error
