"""Run diagnostics: which model path every request actually used, and how complete the inputs were.

Observation only: everything here is read from values the models already computed (route masks, the outage model
view, weather columns, radar coverage); nothing feeds back into a prediction, and no method raises: the first error
switches the diagnostics off with a warning, and the predictions are written as usual. The summary goes to stderr; the
demo tool asks for a JSON copy with the hidden --diagnostics-json option.

Final paths per request (mutually exclusive; the counts of a task add up to its requests):
  Task A  main (0.4 GRU + 0.4 joint tree + 0.2 Q95) | main_no_radar (joint tree -> radar-free event tree)
          | no_weather (weather-free tree) | no_history (history-free tree)
  Task B  main (0.5 joint tree + 0.5 GRU) | main_no_radar (joint tree -> five-member tree system)
          | no_history (monthly climatology) | anchor (+5 / +10 min: latest observed proportion at the issue)
Weather coverage of the model rows (not exclusive): any weather value; HRRR statistics (any field other than the TMP,
RH and PRES means that the five basic fields can fill); GUST or CAPE mean; basic fields only.
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from .inputs import WEATHER_FIELDS

log = logging.getLogger('outage_forecast')
DAY, WEEK = 86400, 7 * 86400
BASIC_MAPPED = ('TMP__mean', 'RH__mean', 'PRES__mean')
HRRR_ONLY = [c for c in WEATHER_FIELDS if c not in BASIC_MAPPED]


def guarded(method):
    """Run a collection method; on any error switch the diagnostics off (warning) instead of raising."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.error is not None:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as error:  # noqa: BLE001
            self.error = f'{type(error).__name__}: {error}'
            log.warning('WARNING: run diagnostics switched off (%s); the predictions are not affected', self.error)
            return None
    return wrapper


def _weather_coverage(frame: pl.DataFrame) -> dict:
    values = np.isfinite(frame.select(WEATHER_FIELDS).to_numpy())
    basic = np.isfinite(frame.select(BASIC_MAPPED).to_numpy()).any(axis=1)
    hrrr = np.isfinite(frame.select(HRRR_ONLY).to_numpy()).any(axis=1)
    core = np.isfinite(frame.select('GUST__mean', 'CAPE__mean').to_numpy()).any(axis=1)
    return {'model_rows': int(frame.height), 'with_weather': int(values.any(axis=1).sum()), 'with_hrrr_statistics': int(hrrr.sum()),
            'with_gust_or_cape': int(core.sum()), 'basic_fields_only': int((basic & ~hrrr).sum())}


class Diagnostics:
    def __init__(self):
        self.error: str | None = None
        self.routes: dict[str, pl.DataFrame] = {}
        self.weather: dict[str, dict] = {}
        self.groups: list[dict] = []
        self.outages: dict[str, dict] = {}

    # ----------------------------------------------------------------- collection (called by the pipeline and tasks)
    @guarded
    def outage_view(self, raw: pl.DataFrame, view: pl.DataFrame) -> None:
        """Records per county in the input and in the 15-minute model view (all versions, blanks included)."""
        n_raw = dict(raw.group_by('county').len().iter_rows())
        n_view = dict(view.group_by('county').len().iter_rows())
        for county in sorted(n_raw):
            self.outages[county] = {'records': n_raw[county], 'off_grid_ignored': n_raw[county] - n_view.get(county, 0)}

    @guarded
    def history(self, task: str, history, groups) -> None:
        """Usable 15-minute observations at each issue: count in the last 24 h (of 96 grid times) and 7 days (of 672),
        age of the latest one."""
        for county, issue in groups.select('county', 'issue').sort('county', 'issue').iter_rows():
            ts, _ = history.at(county, issue)
            self.groups.append({'task': task, 'county': county, 'issue': int(issue),
                                'grid_24h': int(np.sum(ts > issue - DAY)), 'grid_7d': int(np.sum(ts > issue - WEEK)),
                                'latest_age_min': float((issue - ts[-1]) / 60) if len(ts) else None,
                                'usable_records': int(len(ts))})

    @guarded
    def task_a(self, frame, latest, no_history, no_weather, no_radar) -> None:
        labels = np.where(no_history, 'no_history', np.where(no_weather, 'no_weather', np.where(no_radar, 'main_no_radar', 'main')))
        self.routes['A'] = frame.select('county', 'issue', 'lead_minutes').with_columns(pl.Series('route', labels))
        self.weather['A'] = _weather_coverage(latest)

    @guarded
    def task_b(self, frame, no_history, no_radar) -> None:
        labels = np.where(no_history, 'no_history', np.where(no_radar, 'main_no_radar', 'main'))
        self.routes['B'] = frame.select('county', 'issue', 'lead_minutes').with_columns(pl.Series('route', labels))
        self.weather['B'] = _weather_coverage(frame)

    # ----------------------------------------------------------------- summary
    def summarize(self, requests: pl.DataFrame, anchors: pl.DataFrame, extra: dict) -> dict:
        paths = []
        a = requests.filter(pl.col('task') == 'A')
        if a.height:
            paths.append(a.join(self.routes['A'], on=['county', 'issue', 'lead_minutes'], how='left').select('task', 'county', 'route'))
        b = requests.filter(pl.col('task') == 'B')
        if b.height:
            native = self.routes['B']
            b = (b.with_columns((pl.col('lead_minutes') // 15 * 15).alias('native_lead'))
                  .join(native.rename({'lead_minutes': 'native_lead'}), on=['county', 'issue', 'native_lead'], how='left')
                  .join(native.filter(pl.col('lead_minutes') == 15).select('county', 'issue', pl.col('route').alias('route15')),
                        on=['county', 'issue'], how='left')
                  .join(anchors, on=['county', 'issue'], how='left'))
            anchored = (pl.col('native_lead') == 0) & pl.col('anchor').is_finite().fill_null(False)
            b = b.with_columns(pl.when(pl.col('native_lead') > 0).then(pl.col('route'))
                                 .when(anchored).then(pl.lit('anchor')).otherwise(pl.col('route15')).alias('route'))
            paths.append(b.select('task', 'county', 'route'))
        counts = pl.concat(paths).group_by('task', 'county', 'route').len()
        if counts['route'].is_null().any():
            raise RuntimeError('a request has no recorded model path')
        tasks = {}
        groups = pl.DataFrame(self.groups)
        for task in ('A', 'B'):
            part = counts.filter(pl.col('task') == task)
            if not part.height:
                continue
            g = groups.filter(pl.col('task') == task)
            tasks[task] = {
                'requests': int(part['len'].sum()),
                'paths': {r: int(n) for r, n in part.group_by('route').agg(pl.col('len').sum()).sort('route').iter_rows()},
                'paths_by_county': {c: {r: int(n) for _, r, n in part.filter(pl.col('county') == c).sort('route').select('county', 'route', 'len').iter_rows()}
                                    for c in sorted(part['county'].unique().to_list())},
                'issues': int(g.height),
                'issues_without_history': int((g['usable_records'] == 0).sum()),
                'issues_without_recent_24h': int((g['grid_24h'] == 0).sum()),
                'min_grid_24h_of_96': int(g['grid_24h'].min()), 'min_grid_7d_of_672': int(g['grid_7d'].min()),
                'max_latest_age_min': g['latest_age_min'].max(),
                'weather': self.weather.get(task, {})}
            if tasks[task]['requests'] != requests.filter(pl.col('task') == task).height:
                raise RuntimeError(f'Task {task} path counts do not add up to its requests')
        summary = {'tasks': tasks, 'outages': self.outages, **extra}
        unusable = [c for c, v in self.outages.items() if v['records'] and v['records'] == v['off_grid_ignored']]
        if unusable:
            summary['no_usable_grid_history'] = unusable
        return summary


def report(diag: Diagnostics, requests: pl.DataFrame, anchors: pl.DataFrame, extra: dict, json_path: Path | None) -> None:
    """Summary lines on stderr (and the full summary as JSON when asked). Never raises."""
    try:
        if diag.error is not None:
            raise RuntimeError(diag.error)
        summary = diag.summarize(requests, anchors, extra)
        for task, s in summary['tasks'].items():
            paths = ', '.join(f'{r} {n}' for r, n in s['paths'].items())
            log.info('Task %s: %d requests, %d issues; paths: %s', task, s['requests'], s['issues'], paths)
            w = s['weather']
            if w:
                log.info('Task %s weather: %d of %d model rows with weather values (%d with HRRR statistics, %d with GUST or '
                         'CAPE, %d with basic fields only)', task, w['with_weather'], w['model_rows'], w['with_hrrr_statistics'],
                         w['with_gust_or_cape'], w['basic_fields_only'])
            if s['issues_without_history'] or s['issues_without_recent_24h']:
                log.warning('WARNING: Task %s: %d issues without any usable outage record, %d without one in the last 24 h',
                            task, s['issues_without_history'], s['issues_without_recent_24h'])
        off = {c: v['off_grid_ignored'] for c, v in summary['outages'].items() if v['off_grid_ignored']}
        if off:
            log.info('outage records between 15-minute grid times (not used by the models): %s', off)
        if summary.get('no_usable_grid_history'):
            log.warning('WARNING: counties with outage records but none on the 15-minute grid (history fallback used): %s',
                        summary['no_usable_grid_history'])
        log.info('radar: %s', 'present' if summary.get('radar_rows') else 'none (radar-free paths)')
    except Exception as error:  # noqa: BLE001
        log.warning('WARNING: run diagnostics unavailable (%s: %s); the predictions are not affected', type(error).__name__, error)
        summary = {'error': f'{type(error).__name__}: {error}'}
    if json_path is not None:
        partial = None
        try:
            partial = json_path.with_name(json_path.name + '.partial')
            partial.write_text(json.dumps(summary, indent=1, default=str) + '\n', encoding='utf-8')
            partial.replace(json_path)   # complete file or none
        except Exception as error:  # noqa: BLE001
            log.warning('WARNING: diagnostics file not written (%s: %s)', type(error).__name__, error)
            try:
                if partial is not None:
                    partial.unlink(missing_ok=True)
            except OSError:
                pass
