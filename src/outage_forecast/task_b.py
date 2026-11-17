"""Task B (15-minute native leads to 6 h, issues at 00/06/12/18 UTC): 0.5 joint Tweedie tree + 0.5 GRU history network.

Fallbacks:
  radar missing (rx_cov24 = 0 or no county series): the joint tree is replaced by the five-member tree system
      (shared / regional ladder and level families, a long-lead ladder after 180 min; radar columns left empty)
  history missing (no usable outage record): monthly climatology median of the county
  weather missing: same main models (they were trained with missing weather)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import polars as pl

from . import sequence
from .features import history as hist
from .features.radar import RX
from .features.weather import add_process_features, task_b_weather
from .model_store import feature_matrix
from .tree_heads import combine


def tree_system(frame: pl.DataFrame, store, config) -> np.ndarray:
    """Five-member tree system for rows without radar (frame sorted by county, issue, lead)."""
    system = config['task_b']['radar_missing']
    f = frame.with_columns(*[pl.lit(np.nan, dtype=pl.Float64).alias(k) for k in RX])
    processed = add_process_features(f)
    assert processed.select('county', 'issue', 'lead_minutes').equals(f.select('county', 'issue', 'lead_minutes'))
    persistence = np.nan_to_num(f['persistence'].to_numpy(), nan=0.)
    county = f['county'].to_numpy()
    p = {}
    for name, family in system['families'].items():
        source = processed if family['process'] else f
        out = np.zeros(f.height)
        for region, heads in family['regions'].items():
            rows = np.ones(f.height, bool) if region == 'shared40' else county == region
            if not rows.any():
                continue
            x = feature_matrix(source.filter(pl.Series(rows)), family['features'])
            raw = {head: store.tree(spec['path'], x) for head, spec in heads.items()}
            out[rows] = combine(family['design'], raw, persistence[rows], family['constants'][region],
                                config['constants']['up_ladder'], config['constants']['level_ladder'])
        p[name] = out
    lead = f['lead_minutes'].to_numpy() // 15 * 15
    result = 0.
    for part, (shared, regional), share in (('ladder', ('shared_ladder', 'regional_ladder'), 1 - system['level_weight']),
                                            ('level', ('shared_level', 'regional_level'), system['level_weight'])):
        state = system[part]
        w = np.array([state['route'][c] for c in county])
        mixed = (1 - w) * p[shared] + w * p[regional]
        lw = np.interp(lead, [180, 360], [0, state['long']])
        result = result + share * np.clip((1 - lw) * mixed + lw * p['long_ladder'], 0, 1)
    return result


def climatology(frame: pl.DataFrame, config) -> np.ndarray:
    spec = config['task_b']['history_missing']
    values = json.loads(config.path(spec['path']).read_text())['values']
    return np.array([values[c][str(datetime.fromtimestamp(t, timezone.utc).month)][spec['statistic']]
                     for c, t in zip(frame['county'], frame['target'])])


def predict(frame: pl.DataFrame, tables, history: hist.OutageHistory, radar, store, config, diag=None) -> tuple[np.ndarray, pl.DataFrame]:
    """frame: Task B native rows sorted by (county, issue, lead_minutes). Returns the predicted proportions (float64) and
    the frame with its weather columns filled (same row order)."""
    spec = config['task_b']
    q95 = config['constants']['q95']
    frame = task_b_weather(frame, tables.weather, config.county_order, config.region_of)

    groups = frame.select('county', 'issue').unique(maintain_order=True)
    rx, raw = {}, {}
    for county, issue in groups.iter_rows():
        ts, counts = history.at(county, issue)
        rx[county, issue] = radar.exposure(county, issue)
        ts, counts = hist.whole_seconds(ts, counts)
        raw[county, issue] = hist.raw_sequence(ts, counts / config.denominators[county], issue)
    row_keys = list(zip(frame['county'].to_list(), frame['issue'].to_list()))
    no_radar = np.array([not rx[g]['rx_cov24'] > 0 for g in row_keys])

    members = spec['members']
    model, norm = store.network('task_b')
    gru = sequence.predict(model, norm, frame, raw, 15, 24, config, store.device, q95)
    w_tree, w_gru = members['joint_tree']['weight'], members['gru']['weight']
    main = np.empty(frame.height)
    if (~no_radar).any():  # float32 throughout, as the reference mixture
        f = add_process_features(frame.filter(pl.Series(~no_radar)))
        f = f.with_columns(*[pl.Series(k, np.array([rx[g][k] for g in zip(f['county'], f['issue'])], np.float64)) for k in RX])
        joint = np.clip(store.tree(members['joint_tree']['path'], feature_matrix(f, config.features('task_b_joint_tree'))) / 1000, 0, 1)
        main[~no_radar] = 0 + w_tree * joint + w_gru * gru[~no_radar]
    if no_radar.any():  # tree system output is float64
        system = tree_system(frame.filter(pl.Series(no_radar)), store, config)
        main[no_radar] = 0 + w_tree * system + w_gru * gru[no_radar]

    no_history = np.nan_to_num(frame['hist_available'].to_numpy(), nan=0.) <= 0
    out = np.empty(frame.height)
    out[~no_history] = main[~no_history]
    if no_history.any():
        out[no_history] = climatology(frame, config)[no_history]
    if diag is not None:  # observation only; Diagnostics never raises
        diag.task_b(frame, no_history, no_radar)
    return np.clip(out, 0, 1), frame
