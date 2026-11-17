"""Outage-history features, built only from EAGLE-I records available at the issue time.

Model view: only observations on the 15-minute grid (timestamp a whole multiple of 900 s) feed the models, as in
training (EAGLE-I 2022-2024 is a 15-minute series). Records between grid times, e.g. of a native 5-minute series, are
ignored, never rounded or interpolated (model_outage_view).

Visibility rule: a record is visible at issue T when timestamp <= T and source_record_time <= T. For each county and
observation time the visible record with the latest (source_record_time, source row) wins; if that record is blank the
observation is unavailable (an older value is not reused). Counts are divided by the county MCC.

  history_features   30 base columns: latest value and age, lags 15 min .. 24 h, rolling mean/max/min over
                     60 min .. 24 h (both ends closed), slopes, missing-input count
  event_features     9 ev_* columns describing the outage under way (look-back 168 h), Task A trees
  sequence_inputs    GRU inputs: 7 days of 15-minute values (240 steps x 6 channels) and 11 event-state values
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..networks import STATE_COLUMNS

HISTORY_NAMES = (['hist_latest_x', 'hist_latest_observation_age_minutes', 'hist_available', 'hist_missing_flags']
                 + [f'hist_lag_{n}m_x' for n in (15, 30, 60, 120, 180, 360, 720, 1440)]
                 + [f'roll_{s}_{n}' for n in (60, 180, 360, 720, 1440) for s in ('mean', 'max', 'min')]
                 + [f'slope_{n}' for n in (60, 180, 360)])
LAGS = (15, 30, 60, 120, 180, 360, 720, 1440)
WINDOWS = (30, 60, 180, 360, 720, 1440)
MISSING_INPUTS = {  # inputs counted by hist_missing_flags (as in training)
    'A': {'lag': (60, 180, 360, 720, 1440), 'rolling': (180, 360, 720, 1440), 'slope': (60, 180, 360)},
    'B': {'lag': LAGS, 'rolling': (60, 180, 360, 1440), 'slope': (30, 60, 180)}}
EVENT_NAMES = ['ev_peak_72h', 'ev_peak_168h', 'ev_auc_72h', 'ev_lag_2880m', 'ev_lag_4320m', 'ev_dur_q95_h', 'ev_dur_2e4_h',
               'ev_since_peak168_h', 'ev_recover168']
EVENT_GAP_S = 1800
NATIVE, HOURLY = 96, 144              # last 24 h at 15 minutes, preceded by 6 days of hourly means
RAW_STEPS = NATIVE + 4 * HOURLY       # 672 fifteen-minute steps = 7 days


def model_outage_view(outages: pl.DataFrame) -> pl.DataFrame:
    """The validated outage table restricted to observations on the 15-minute grid. Only the observation time is tested
    (with any fraction of a second, so 00:15:00.5 is off the grid); source_record_time, versions, blanks and row order
    are kept. For input that is already on the grid this is the identity."""
    return outages.filter((pl.col('timestamp_s') % 900) == 0)


class OutageHistory:
    """Per-county record arrays sorted by (timestamp, source_record_time, source row)."""

    def __init__(self, outages: pl.DataFrame):
        self.series = {}
        for (county,), g in outages.sort('county', 'timestamp_s', 'source_s', 'row').group_by('county', maintain_order=True):
            count = g['count'].to_numpy().astype(np.float64)
            self.series[county] = (g['timestamp_s'].to_numpy(), g['source_s'].to_numpy(), count, g['count'].is_null().to_numpy())

    def at(self, county: str, issue: int) -> tuple[np.ndarray, np.ndarray]:
        """(timestamps in seconds, float64 with any fraction kept; counts float64) of the county's usable observations at
        the issue, sorted by time. Event features and GRU inputs use whole seconds (see whole_seconds)."""
        if county not in self.series:
            return np.zeros(0), np.zeros(0)
        ts, src, count, blank = self.series[county]
        idx = np.flatnonzero((ts <= issue) & (src <= issue))
        if len(idx):
            t = ts[idx]
            idx = idx[np.r_[t[1:] != t[:-1], True]]   # latest visible version of each observation time
            idx = idx[~blank[idx]]
        return ts[idx], count[idx]


def whole_seconds(ts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The series on whole seconds, the time base of the event features and GRU inputs: times truncated to whole seconds
    and the records ordered by (second, count), as in training; where two records fall in the same second the GRU
    history keeps the last of them."""
    t = ts.astype(np.int64)
    order = np.lexsort((counts, t))
    return t[order], counts[order]


def history_features(ts: np.ndarray, counts: np.ndarray, issue: int, total: int, task: str) -> dict[str, float]:
    """The 30 base history columns (and hist_latest_x in float64 for the persistence anchor). ts in seconds (float)."""
    nan = math.nan
    n = len(ts)
    latest = counts[-1] if n else nan
    v = {'hist_latest_x': latest / total if n else nan,
         'hist_latest_observation_age_minutes': (issue - ts[-1]) / 60 if n else nan,
         'hist_available': float(n > 0)}
    lag_count, missing = {}, {}
    for lag in LAGS:
        p = int(np.searchsorted(ts, issue - lag * 60, 'right')) - 1
        lag_count[lag] = float(counts[p]) if p >= 0 else nan
        missing[('lag', lag)] = float(p < 0)
        v[f'hist_lag_{lag}m_x'] = lag_count[lag] / total
    rolling, slope = {}, {}
    for window in WINDOWS:
        lo = int(np.searchsorted(ts, issue - window * 60, 'left'))
        hi = int(np.searchsorted(ts, issue, 'right'))
        points = counts[lo:hi]
        for stat, fn in (('mean', np.mean), ('max', max), ('min', min)):
            rolling[stat, window] = float(fn(points)) if len(points) else nan
            missing[('rolling', window, stat)] = float(not len(points))
        change = (float(latest) - lag_count[window]) if window in (60, 180, 360) else nan
        if task == 'A':
            s = change / window
        elif len(points) >= 2:
            x = (ts[lo:hi] - issue) / 3600
            y = np.array(points)
            denominator = float(np.sum((x - x.mean()) ** 2))
            s = float(np.sum((x - x.mean()) * (y - y.mean())) / denominator) if denominator else 0.0
        else:
            s = nan
        slope[window] = s
        missing[('slope', window)] = float(not math.isfinite(s))
    flags = MISSING_INPUTS[task]
    v['hist_missing_flags'] = (sum(missing[('lag', w)] for w in flags['lag'])
                               + sum(missing[('rolling', w, s)] for w in flags['rolling'] for s in ('mean', 'max'))
                               + sum(missing[('slope', w)] for w in flags['slope']))
    for window in (60, 180, 360, 720, 1440):
        for stat in ('mean', 'max', 'min'):
            v[f'roll_{stat}_{window}'] = rolling[stat, window] / total
    for window in (60, 180, 360):
        v[f'slope_{window}'] = slope[window] / total
    return v


# ----------------------------------------------------------------------------- events (Task A)
def _run_start(ts, x, cut):
    above = x >= cut
    brk = np.ones(len(ts), bool)
    brk[1:] = ~above[:-1] | (np.diff(ts) > EVENT_GAP_S)
    start_idx = np.maximum.accumulate(np.where(above & brk, np.arange(len(ts)), -1))
    return np.where(above & (start_idx >= 0), ts[np.maximum(start_idx, 0)], -1)


def event_features(ts: np.ndarray, x: np.ndarray, issue: int, q95: float) -> dict[str, float]:
    """ev_* at one issue from the county's usable series (x = count / MCC). A gap over 30 minutes ends a run; the
    latest value counts as current only within 30 minutes of the issue; durations are capped at 336 h."""
    if not len(ts):
        return {k: (0. if k in ('ev_peak_72h', 'ev_peak_168h', 'ev_auc_72h', 'ev_dur_q95_h', 'ev_dur_2e4_h') else np.nan)
                for k in EVENT_NAMES}
    out = {k: np.nan for k in EVENT_NAMES}
    starts = {cut: _run_start(ts, x, cut) for cut in (q95, 2e-4)}
    t = issue
    pos = np.searchsorted(ts, t, 'right')
    live = pos > 0 and t - ts[pos - 1] <= EVENT_GAP_S
    latest = x[pos - 1] if live else 0.
    for h, name in ((72, 'ev_peak_72h'), (168, 'ev_peak_168h')):
        lo = np.searchsorted(ts, t - h * 3600, 'right')
        out[name] = x[lo:pos].max() if pos > lo else 0.
    lo72 = np.searchsorted(ts, t - 72 * 3600, 'right')
    out['ev_auc_72h'] = x[lo72:pos].sum() * .25
    for lag in (2880, 4320):
        p = np.searchsorted(ts, t - lag * 60, 'right')
        if p > 0 and t - lag * 60 - ts[p - 1] <= 3600:
            out[f'ev_lag_{lag}m'] = x[p - 1]
    for cut, name in ((q95, 'ev_dur_q95_h'), (2e-4, 'ev_dur_2e4_h')):
        s = starts[cut][pos - 1] if live else -1
        out[name] = min((t - s) / 3600, 336.) if s >= 0 else 0.
    lo168 = np.searchsorted(ts, t - 168 * 3600, 'right')
    if pos > lo168:
        k = lo168 + int(np.argmax(x[lo168:pos]))
        out['ev_since_peak168_h'] = (t - ts[k]) / 3600
        out['ev_recover168'] = latest / x[k] if x[k] > 0 else np.nan
    return {k: float(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- GRU sequence inputs
def raw_sequence(ts: np.ndarray, x: np.ndarray, issue: int) -> np.ndarray:
    """672 fifteen-minute values ending at the issue itself (float32, NaN where no record). Only records on the
    15-minute grid with x in [0, 1] are used; there is no filling."""
    grid = issue - 900 * np.arange(RAW_STEPS - 1, -1, -1)
    out = np.full(RAW_STEPS, np.nan, np.float64)
    if len(ts):
        ok = (ts % 900 == 0) & (x >= 0) & (x <= 1) & (ts <= issue)
        t, v = ts[ok], x[ok]
        pos = np.searchsorted(t, grid, 'right') - 1   # the last record of a grid time
        hit = pos >= 0
        hit[hit] = t[pos[hit]] == grid[hit]
        out[hit] = v[pos[hit]]
    return out.astype(np.float32)


def _last_finite(a):
    ok = np.isfinite(a)
    idx = np.where(ok, np.arange(a.shape[1])[None, :], -1).max(1)
    val = np.where(idx >= 0, a[np.arange(len(a)), np.maximum(idx, 0)], np.nan)
    return val, np.where(idx >= 0, a.shape[1] - 1 - idx, 10 ** 6)


def sequence_inputs(raw: np.ndarray, issue: np.ndarray, q95: float):
    """raw [n, 672] -> history [n, 240, 6] float16, state [n, 11] float32 (unstandardised, STATE_COLUMNS order), x0 [n]."""
    dt = .25
    n = len(raw)
    older = raw[:, :4 * HOURLY].reshape(n, HOURLY, 4)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            hourly = np.nanmean(older, 2)
    seq = np.concatenate([hourly, raw[:, -NATIVE:]], 1)
    ok = np.isfinite(seq)
    z = np.log1p(np.where(ok, seq, 0.) / 1e-4)
    hist = np.zeros((n, HOURLY + NATIVE, 6), np.float16)
    hist[..., 0] = z
    hist[..., 1] = ok
    hist[:, 1:, 2] = np.diff(z, axis=1)
    hist[:, HOURLY:, 3] = 1
    end = issue[:, None].astype(np.float64)
    t = np.concatenate([end - NATIVE * 900 - (HOURLY - np.arange(HOURLY))[None, :] * 3600 + 1800,
                        end - (NATIVE - 1 - np.arange(NATIVE))[None, :] * 900], 1)
    ang = 2 * np.pi * (t % 86400) / 86400
    hist[..., 4] = np.sin(ang)
    hist[..., 5] = np.cos(ang)

    x0, _ = _last_finite(raw[:, -8:])
    known = np.isfinite(x0)
    x0 = np.where(known, x0, 0.)
    filled = np.where(np.isfinite(raw), raw, -1.)
    peak7 = np.maximum(filled.max(1), 0.)
    arg7 = filled.argmax(1)
    peak24 = np.maximum(filled[:, -NATIVE:].max(1), 0.)
    v2, _ = _last_finite(raw[:, -16:-8])
    v6, _ = _last_finite(raw[:, -32:-24])
    lz = lambda v: np.log1p(np.nan_to_num(v, nan=0.) / 1e-4)  # noqa: E731
    slope2 = np.where(known & np.isfinite(v2), lz(x0) - lz(v2), 0.)
    slope6 = np.where(known & np.isfinite(v6), lz(x0) - lz(v6), 0.)
    above = np.isfinite(raw) & (raw >= q95)
    run = np.argmin(above[:, ::-1], 1)
    run = np.where(above.all(1), RAW_STEPS, run)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            area24 = np.nansum(raw[:, -NATIVE:], 1) * dt
            obs24 = np.isfinite(raw[:, -NATIVE:]).mean(1)
    columns = {'x0': lz(x0), 'x0_known': known, 'peak7': lz(peak7), 'peak24': lz(peak24),
               'since_peak7_h': np.where(peak7 > 0, (RAW_STEPS - 1 - arg7) * dt, 168.) / 24,
               'ratio7': np.where(peak7 > 0, x0 / np.maximum(peak7, 1e-12), 0.), 'slope2h': slope2, 'slope6h': slope6,
               'dur_q95_h': run * dt / 24, 'area24': np.log1p(area24 / 1e-4), 'obs24': obs24}
    state = np.stack([columns[c] for c in STATE_COLUMNS], 1).astype(np.float32)
    return hist, state, x0.astype(np.float32)
