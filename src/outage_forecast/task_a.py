"""Task A (48 hourly leads, 00 UTC issues): 0.4 GRU history network + 0.4 joint feature tree + 0.2 Q95 tail member.

Fallbacks, decided per row as in training:
  radar missing (rx_cov24 = 0 or no county series): the joint tree is replaced by the radar-free event tree
  weather missing (no usable run, or none of the TMP/GUST/CAPE means available): 150-column tree anchored on persistence
  history missing (no usable outage record): 185-column tree trained with masked inputs
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import sequence
from .features import history as hist
from .features.radar import RX
from .features.weather import add_path_features, select_weather, with_lagavg, with_weather
from .model_store import feature_matrix


def tree_output(store, spec: dict, frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Booster output on the proportion scale for the Task A tree kinds."""
    p = store.tree(spec['path'], feature_matrix(frame, features), spec.get('rounds'))
    kind = spec['output']
    if kind == 'log_ratio':
        b = np.nan_to_num(frame['persistence'].to_numpy(), nan=0.)
        p = (b + 1e-4) * np.exp(np.clip(p, -12, 12)) - 1e-4
    else:  # Tweedie / winsorised targets are fitted on 1000 x proportion
        p = p / 1000
    return np.clip(p, 0, 1)


def predict(frame: pl.DataFrame, tables, history: hist.OutageHistory, radar, store, config, diag=None) -> np.ndarray:
    """frame: Task A native rows sorted by (county, issue, lead_minutes). Returns predicted proportions (float64)."""
    spec = config['task_a']
    q95 = config['constants']['q95']
    keys = frame.select('county', 'issue', 'target')
    counties, region_of = config.county_order, config.region_of
    latest = with_weather(frame, select_weather(tables.weather, keys, 48, 'latest', counties, region_of))
    previous18 = with_weather(frame, select_weather(tables.weather, keys, 42, 'previous18', counties, region_of))
    combined = add_path_features(latest)
    lagged = add_path_features(with_lagavg(latest, previous18))
    for f in (latest, previous18, combined, lagged):
        assert f.select('county', 'issue', 'lead_minutes').equals(frame.select('county', 'issue', 'lead_minutes'))

    # per (county, issue): event columns, radar exposure, raw GRU history
    groups = frame.select('county', 'issue').unique(maintain_order=True)
    ev, rx, raw = {}, {}, {}
    for county, issue in groups.iter_rows():
        ts, counts = history.at(county, issue)
        ts, counts = hist.whole_seconds(ts, counts)
        x = counts / config.denominators[county]
        ev[county, issue] = hist.event_features(ts, x, issue, q95)
        rx[county, issue] = radar.exposure(county, issue)
        raw[county, issue] = hist.raw_sequence(ts, x, issue)
    row_keys = list(zip(frame['county'].to_list(), frame['issue'].to_list()))
    ev_cols = [pl.Series(k, np.array([ev[g][k] for g in row_keys], np.float32)) for k in hist.EVENT_NAMES]
    rx_cols = [pl.Series(k, np.array([rx[g][k] for g in row_keys], np.float64)) for k in RX]

    members = spec['members']
    model, norm = store.network('task_a')
    gru = sequence.predict(model, norm, combined, raw, 60, members['gru']['leads'], config, store.device, q95)

    joint = tree_output(store, members['joint_tree'], lagged.with_columns(*ev_cols, *rx_cols), config.features('task_a_joint_tree'))
    no_radar = np.array([not rx[g]['rx_cov24'] > 0 for g in row_keys])
    if no_radar.any():
        alt = tree_output(store, spec['radar_missing'], combined.with_columns(*ev_cols), config.features('task_a_event_tree'))
        joint = np.where(no_radar, alt, joint)

    q = members['q95']
    p95 = store.tree(q['path'], feature_matrix(lagged, config.features(q['features'])))
    tail = (1 - p95) * q['mu0'] + p95 * q['mu1']

    main = 0
    for name, member in (('gru', gru), ('joint_tree', joint), ('q95', tail)):
        main = main + members[name]['weight'] * member

    no_history = np.nan_to_num(frame['hist_available'].to_numpy(), nan=0.) <= 0
    no_weather = ((np.nan_to_num(latest['weather_available'].to_numpy(), nan=0.) <= 0)
                  | ~np.isfinite(latest.select('TMP__mean', 'GUST__mean', 'CAPE__mean').to_numpy()).any(axis=1))
    out = np.empty(frame.height)
    out[~no_history & ~no_weather] = main[~no_history & ~no_weather]
    if (~no_history & no_weather).any():
        p = tree_output(store, spec['weather_missing'], latest, config.features('base'))
        out[~no_history & no_weather] = p[~no_history & no_weather]
    if no_history.any():
        p = tree_output(store, spec['history_missing'], combined, config.features('task_a_combined'))
        out[no_history] = p[no_history]
    if diag is not None:  # observation only; Diagnostics never raises
        diag.task_a(frame, latest, no_history, no_weather, no_radar)
    return np.clip(out, 0, 1)
