"""Native prediction frame: one row per (task, county, issue, native lead) with the 150 base columns.

Native leads are the full model paths (Task A 60 .. 2880 min hourly, Task B 15 .. 360 min every 15 minutes) for every
requested (task, county, issue), whatever subset of targets is requested. Base columns: lead, issue / target time of
day and issue day of year as sine and cosine, log MCC, 40-county one-hot, 30 history columns (float32; weather columns
are filled by features.weather). 'persistence' keeps the latest observed proportion in float64.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import polars as pl

from .history import HISTORY_NAMES, OutageHistory, history_features


def calendar(issue: int, lead: int) -> tuple[float, ...]:
    d = datetime.fromtimestamp(issue, timezone.utc)
    t = datetime.fromtimestamp(issue + lead * 60, timezone.utc)
    yday = d.timetuple().tm_yday
    return (lead / 60, math.sin(d.hour * math.pi / 12), math.cos(d.hour * math.pi / 12),
            math.sin(yday * 2 * math.pi / 365.25), math.cos(yday * 2 * math.pi / 365.25),
            math.sin((t.hour + t.minute / 60) * math.pi / 12), math.cos((t.hour + t.minute / 60) * math.pi / 12))


def native_frame(task: str, groups: list[tuple[str, int]], history: OutageHistory, mcc: dict[str, int], config) -> pl.DataFrame:
    """Rows for the (county, issue) groups of one task, sorted by (county, issue, lead_minutes)."""
    first, last, step = config['native_leads_minutes'][task]
    leads = list(range(first, last + 1, step))
    features = config.features('base')
    column = {name: i for i, name in enumerate(features)}
    denominators = {**config.denominators, **mcc}
    counties = config.county_order
    rows = np.full((len(groups) * len(leads), len(features)), np.nan, np.float64)
    meta = {'county': [], 'issue': [], 'lead_minutes': [], 'persistence': []}
    calendar_cols = [column[n] for n in ('lead_hours', 'issue_sin', 'issue_cos', 'day_sin', 'day_cos', 'target_sin', 'target_cos')]
    history_cols = [column[n] for n in HISTORY_NAMES]
    r = 0
    for county, issue in sorted(groups):
        ts, counts = history.at(county, issue)
        h = history_features(ts, counts, issue, mcc[county], task)
        hist_values = [h[n] for n in HISTORY_NAMES]
        one_hot = np.array([float(code == county) for code in counties])
        for lead in leads:
            rows[r, calendar_cols] = calendar(issue, lead)
            rows[r, column['log_mcc']] = math.log(denominators[county])
            rows[r, column['county_' + counties[0]]:column['county_' + counties[0]] + len(counties)] = one_hot
            rows[r, history_cols] = hist_values
            meta['county'].append(county)
            meta['issue'].append(issue)
            meta['lead_minutes'].append(lead)
            meta['persistence'].append(h['hist_latest_x'])
            r += 1
    frame = pl.DataFrame({'county': pl.Series(meta['county'], dtype=pl.String), 'issue': pl.Series(meta['issue'], dtype=pl.Int64),
                          'lead_minutes': pl.Series(meta['lead_minutes'], dtype=pl.Int64),
                          'persistence': pl.Series(meta['persistence'], dtype=pl.Float64)})
    frame = frame.hstack(pl.DataFrame(rows.astype(np.float32), schema=features, orient='row'))
    return frame.with_columns((pl.col('issue') + pl.col('lead_minutes') * 60).alias('target'))
