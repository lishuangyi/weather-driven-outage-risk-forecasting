"""Which requests a demo run predicts (demo tool; never used by infer.py).

Without any selection option the input directory is used as it is. With --counties, --start/--end, --tasks or
--generate-requests, a new input directory is made inside the run directory:
  requests/forecast_requests   the organizer request table filtered to the selection (rows kept exactly as given), or,
                               with --generate-requests or when the input has no request table, requests generated on
                               the official grid: Task A issues at 00 UTC with leads 60..2880 min every 60 min, Task B
                               issues at 00/06/12/18 UTC with leads 5..360 min every 5 min (configs/inference.json),
                               for every architecture ARC01..ARC04 and every selected county
  the other tables             symbolic links to the original files (a copy where links are not possible)
The original input directory is never written to.

Input ablation (--drop radar, --drop weather; may be repeated): the new input directory leaves out that table (CSV and
Parquet), so the frozen infer.py runs its fallback for missing input. It is a stress test of the frozen system for the
demonstration, not a model change, and never the official output.

Time window: issue times T with start <= T < end (UTC). A date means 00:00 UTC of that day; --start alone selects one
day (end = start + 1 day).

Part of the official grid (--a-leads, --b-leads as MIN:MAX[:STEP] in minutes, --b-issue-hours as UTC hours): fewer leads
or fewer Task B issue hours, for generated and for filtered requests alike. Only a part of the official grid can be
chosen: infer.py rejects any other request (the frozen models exist for these issue hours and leads only).
"""
from __future__ import annotations

import re
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

TABLES = {'requests': 'requests/forecast_requests', 'mcc': 'mcc/MCC', 'outages': 'eaglei/eaglei_outages',
          'weather': 'weather/weather_forecasts', 'radar': 'radar/mrms_county_series'}
KEYS = ['task_id', 'architecture_id', 'county_fips', 'issue_time', 'target_time']
ALIASES = {'los angeles': '06037', 'la': '06037', 'miami-dade': '12086', 'miami dade': '12086', 'miami': '12086',
           'wayne': '26163', 'rockingham': '33015', 'harris': '48201'}


class SelectionError(ValueError):
    """The selection cannot be made (unknown county, bad time, nothing left); the demo stops with exit code 2."""


def locate(data: Path, table: str) -> Path | None:
    for suffix in ('.csv', '.parquet'):
        if (data / (TABLES[table] + suffix)).is_file():
            return data / (TABLES[table] + suffix)
    return None


def parse_counties(text: str | None, official) -> list[str] | None:
    """'48201,06037' or names ('Harris, Los Angeles') -> sorted official FIPS codes."""
    if text is None:
        return None
    out = set()
    for item in (x.strip() for x in text.split(',')):
        if not item:
            continue
        code = ALIASES.get(item.lower(), item.zfill(5) if item.isdigit() else item)
        if code not in official:
            raise SelectionError(f'unknown county {item!r}: use one of {", ".join(sorted(official))} '
                                 f'(or Los Angeles, Miami-Dade, Wayne, Rockingham, Harris)')
        out.add(code)
    if not out:
        raise SelectionError('--counties is empty')
    return sorted(out)


def parse_tasks(text: str | None) -> list[str]:
    if text is None:
        return ['A', 'B']
    tasks = sorted({t.strip().upper() for t in text.split(',') if t.strip()})
    if not tasks or any(t not in ('A', 'B') for t in tasks):
        raise SelectionError(f'--tasks must be A, B or A,B (got {text!r})')
    return tasks


def parse_time(text: str, name: str) -> datetime:
    """UTC time from 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM' or a full ISO time ('Z' or offset; no offset = UTC)."""
    try:
        text = re.sub(r'^(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)', lambda m: f'{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}', str(text).strip())
        t = datetime.fromisoformat(text.replace('Z', '+00:00'))      # 2025-9-8 is accepted as 2025-09-08
    except ValueError as error:
        raise SelectionError(f'{name} {text!r} is not a date or time (example: 2025-07-08 or 2025-07-08T06:00)') from error
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def window(start: str | None, end: str | None) -> tuple[datetime, datetime] | None:
    if start is None and end is None:
        return None
    if start is None:
        raise SelectionError('--end needs --start')
    s = parse_time(start, '--start')
    e = parse_time(end, '--end') if end is not None else s + timedelta(days=1)
    if e <= s:
        raise SelectionError(f'--end ({e:%Y-%m-%d %H:%M}) must be after --start ({s:%Y-%m-%d %H:%M})')
    return s, e


def parse_grid(args, config) -> tuple[dict, bool]:
    """(grid per task, whether it is narrower than the official one). Options: a_leads / b_leads 'MIN:MAX[:STEP]' in
    minutes, b_issue_hours '0,12'."""
    grid = {task: dict(config['requests'][task], issue_hours_utc=list(config['requests'][task]['issue_hours_utc']))
            for task in ('A', 'B')}
    narrowed = False
    for task in ('A', 'B'):
        text, g, name = getattr(args, f'{task.lower()}_leads', None), grid[task], f'--{task.lower()}-leads'
        official = f"{g['lead_min_minutes']}:{g['lead_max_minutes']}:{g['lead_step_minutes']}"
        if text is None:
            continue
        try:
            parts = [int(x) for x in str(text).split(':')]
            lo, hi, step = (parts + [g['lead_step_minutes']])[:3] if len(parts) in (2, 3) else (None, None, None)
        except ValueError:
            lo = None
        if lo is None:
            raise SelectionError(f'{name} {text!r} must be MIN:MAX or MIN:MAX:STEP in minutes (official grid {official})')
        if (lo < g['lead_min_minutes'] or hi > g['lead_max_minutes'] or lo > hi or step <= 0
                or lo % g['lead_step_minutes'] or step % g['lead_step_minutes']):
            raise SelectionError(f'{name} {text!r} is not part of the official Task {task} lead grid {official} (minutes): '
                                 'the frozen models predict these leads only')
        g.update(lead_min_minutes=lo, lead_max_minutes=hi, lead_step_minutes=step)
        narrowed = True
    hours = getattr(args, 'b_issue_hours', None)
    if hours is not None:
        try:
            chosen = sorted({int(x) for x in str(hours).split(',') if x.strip()})
        except ValueError:
            chosen = []
        if not chosen or not set(chosen) <= set(grid['B']['issue_hours_utc']):
            raise SelectionError(f"--b-issue-hours {hours!r} must be UTC hours out of {grid['B']['issue_hours_utc']} "
                                 '(the official Task B issue hours)')
        grid['B']['issue_hours_utc'] = chosen
        narrowed = True
    return grid, narrowed


def generate(counties: list[str], tasks: list[str], span: tuple[datetime, datetime], config, grids: dict | None = None) -> pl.DataFrame:
    """Requests on the official grid (or the chosen part of it) for the issue times in [start, end)."""
    start, end = span
    first = start.replace(minute=0, second=0, microsecond=0)
    if first < start:
        first += timedelta(hours=1)
    rows = []
    archs = sorted(config['risk']['x0'])
    t = first
    while t < end:
        for task in tasks:
            grid = (grids or config['requests'])[task]
            if t.hour not in grid['issue_hours_utc']:
                continue
            issue = t.strftime('%Y-%m-%dT%H:%M:%SZ')
            targets = [(t + timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M:%SZ')
                       for m in range(grid['lead_min_minutes'], grid['lead_max_minutes'] + 1, grid['lead_step_minutes'])]
            rows += [(task, arch, county, issue, target) for arch in archs for county in counties for target in targets]
        t += timedelta(hours=1)
    if not rows:
        raise SelectionError(f'no issue time of the selected tasks between {start:%Y-%m-%d %H:%M} and {end:%Y-%m-%d %H:%M} UTC '
                             '(Task A issues at 00 UTC, Task B at 00/06/12/18 UTC)')
    return pl.DataFrame(rows, schema=KEYS, orient='row')


def filter_requests(frame: pl.DataFrame, counties, tasks, span, grids: dict | None = None) -> pl.DataFrame:
    """The rows of the organizer request table inside the selection, unchanged and in their original order. Issue times
    are read exactly as infer.py reads them (outage_forecast.inputs); no column is added to the table."""
    s = lambda c: pl.col(c).cast(pl.String).str.strip_chars()  # noqa: E731
    keep = frame.select(s('task_id').str.to_uppercase().is_in(tasks).fill_null(False)).to_series()
    if counties is not None:
        keep &= frame.select(s('county_fips').str.zfill(5).is_in(counties).fill_null(False)).to_series()
    if span is not None:
        from outage_forecast.inputs import InputError, _seconds
        try:
            issue = _seconds(frame, 'issue_time')
        except (InputError, pl.exceptions.PolarsError) as error:
            raise SelectionError(f'issue_time of the request table cannot be read, so the time window cannot be applied ({error})') from error
        keep &= (issue >= span[0].timestamp()) & (issue < span[1].timestamp())
    if grids is not None:   # only the chosen part of the official grid
        from outage_forecast.inputs import InputError, _seconds
        try:
            issue, target = _seconds(frame, 'issue_time'), _seconds(frame, 'target_time')
        except (InputError, pl.exceptions.PolarsError) as error:
            raise SelectionError(f'the request times cannot be read, so the lead selection cannot be applied ({error})') from error
        lead, hour = (target - issue) / 60, (issue // 3600) % 24
        task = frame.select(s('task_id').str.to_uppercase()).to_series()
        inside = pl.Series([False] * frame.height)
        for name, g in grids.items():
            inside |= ((task == name) & (lead >= g['lead_min_minutes']) & (lead <= g['lead_max_minutes'])
                       & ((lead - g['lead_min_minutes']) % g['lead_step_minutes'] == 0)
                       & hour.is_in(g['issue_hours_utc'])).fill_null(False)
        keep &= inside
    return frame.filter(keep)


def data_defaults(data: Path, official: set[str]) -> tuple[list[str] | None, tuple[datetime, datetime] | None]:
    """What the input directory itself says when it has no request table: the official counties that have a customer
    total in the MCC (and outage records, when there are any), and the issue times the outage records cover (from one
    day after the first record, so that the first issues have a history, up to the last record). None where the
    files do not tell (missing or unreadable); infer.py remains the judge of the input."""
    def read(table):
        path = locate(data, table)
        if path is None or path.stat().st_size == 0:
            return None
        try:
            frame = pl.read_csv(path, infer_schema=False) if path.suffix == '.csv' else pl.read_parquet(path)
            return frame.with_columns(pl.col('county_fips').cast(pl.String).str.strip_chars().str.zfill(5))
        except Exception:  # noqa: BLE001
            return None

    mcc, outages = read('mcc'), read('outages')
    counties = None if mcc is None else set(mcc['county_fips'].to_list()) & official
    span = None
    if outages is not None and outages.height and 'timestamp' in outages.columns:
        outages = outages.filter(pl.col('county_fips').is_in(sorted(counties if counties else official)))
        ts = outages['timestamp']
        if ts.dtype == pl.String:
            ts = ts.str.strip_chars().str.replace('Z$', '+00:00').str.to_datetime(time_zone='UTC', strict=False)
        ts = ts.drop_nulls()
        if len(ts):
            if counties:
                counties &= set(outages['county_fips'].unique().to_list())
            first, last = ts.min().astimezone(timezone.utc), ts.max().astimezone(timezone.utc)
            start = first + timedelta(days=1) if last - first > timedelta(days=2) else first
            span = (start, last + timedelta(seconds=1))
    return (sorted(counties) if counties else None), span


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare(data: Path, run_dir: Path, args, config) -> tuple[Path, dict]:
    """(input directory to use, description of the selection)."""
    official = set(config.counties)
    counties = parse_counties(args.counties, official)
    tasks = parse_tasks(args.tasks)
    span = window(args.start, args.end)
    grids, narrowed = parse_grid(args, config)
    requests = locate(data, 'requests')
    generate_mode = args.generate_requests or requests is None
    drop = sorted(set(getattr(args, 'drop', None) or []))
    selecting = generate_mode or counties is not None or span is not None or args.tasks is not None or narrowed
    if not selecting and not drop:
        return data, {'mode': 'request table of the input directory, as given'}
    assumed = []
    if generate_mode and (span is None or counties is None):   # take what is not given from the data themselves
        found_counties, found_span = data_defaults(data, official)
        if counties is None and found_counties:
            counties = found_counties
            assumed.append('counties from the MCC and the outage records')
        if span is None and found_span:
            span = found_span
            assumed.append('all issue times covered by the outage records')
    if generate_mode and span is None:
        raise SelectionError('generating requests needs a time window: --start (and optionally --end)'
                             + ('' if requests is not None else ' — the input directory has no request table and no '
                                                                'outage records to take the period from'))
    target = run_dir / 'input'
    for table in ('mcc', 'outages', 'weather', 'radar'):
        found = locate(data, table)
        if found is not None and table not in drop:
            _link(found, target / found.relative_to(data))
    info = {'counties': counties or 'all in the request table', 'tasks': tasks, 'dropped': drop,
            'window_utc': None if span is None else [f'{span[0]:%Y-%m-%d %H:%M}', f'{span[1]:%Y-%m-%d %H:%M}'],
            'grid': {t: grids[t] for t in tasks} if narrowed else 'official'}
    if not selecting:   # ablation only: the request table is used as given
        _link(requests, target / requests.relative_to(data))
        info |= {'mode': 'request table of the input directory, as given', 'requests': None}
        return target, info
    out_path = target / (TABLES['requests'] + '.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if generate_mode:
        chosen = counties or sorted(official)
        frame = generate(chosen, tasks, span, config, grids)
        frame.write_csv(out_path)
        info |= {'mode': 'requests generated on ' + ('a part of the official grid' if narrowed else 'the official grid'),
                 'counties': chosen}
        if assumed:
            info['mode'] += ' (' + '; '.join(assumed) + ')'
    else:
        try:   # an unreadable request table is an input error (exit code 2), as in infer.py
            if requests.stat().st_size == 0:
                raise SelectionError(f'{requests.name} is empty (not even a header)')
            frame = pl.read_csv(requests, infer_schema=False) if requests.suffix == '.csv' else pl.read_parquet(requests)
        except (OSError, pl.exceptions.PolarsError) as error:
            raise SelectionError(f'{requests.name} cannot be read: {type(error).__name__}: {str(error).splitlines()[0]}') from error
        if not {*KEYS} <= set(frame.columns):
            raise SelectionError(f'{requests.name} lacks request columns {sorted(set(KEYS) - set(frame.columns))}')
        frame = filter_requests(frame, counties, tasks, span, grids if narrowed else None)
        if requests.suffix == '.csv':
            frame.write_csv(out_path)
        else:
            out_path = out_path.with_suffix('.parquet')
            frame.write_parquet(out_path)
        info |= {'mode': 'request table of the input directory, filtered'}
    if frame.height == 0:
        raise SelectionError('no request left after the selection: check --counties, --start/--end and --tasks against the '
                             'request table')
    info['requests'] = frame.height
    return target, info
