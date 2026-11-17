"""End-to-end inference: inputs -> native frames -> Task A and Task B models -> seven-column predictions."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from . import diagnostics, output, task_a, task_b
from .config import Config
from .features.base import native_frame
from .features.history import OutageHistory, model_outage_view
from .features.radar import RadarSeries
from .inputs import read_inputs
from .model_store import ModelStore

log = logging.getLogger('outage_forecast')


def resolve_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return device


def run(input_dir: Path, output_path: Path, device: str = 'auto', config: Config | None = None,
        diagnostics_json: Path | None = None) -> pl.DataFrame:
    started = time.perf_counter()
    config = config or Config.load()
    device = resolve_device(device)
    torch.set_num_threads(4)
    tables = read_inputs(input_dir, config)
    log.info('inputs read: %d requests (%.1f s)', tables.requests.height, time.perf_counter() - started)
    model_outages = model_outage_view(tables.outages)
    history = OutageHistory(model_outages)
    diag = diagnostics.Diagnostics()
    diag.outage_view(tables.outages, model_outages)
    radar = RadarSeries(tables.radar)
    store = ModelStore(config, device)

    groups = tables.requests.select('task', 'county', 'issue').unique()
    native, anchors = {}, None
    for task in ('A', 'B'):
        g = groups.filter(pl.col('task') == task)
        if not g.height:
            native[task] = pl.DataFrame(schema={'county': pl.String, 'issue': pl.Int64, 'lead_minutes': pl.Int64, 'p': pl.Float64})
            continue
        frame = native_frame(task, list(g.select('county', 'issue').iter_rows()), history, tables.mcc, config)
        diag.history(task, history, g)
        if task == 'A':
            p = task_a.predict(frame, tables, history, radar, store, config, diag)
        else:
            p, frame = task_b.predict(frame, tables, history, radar, store, config, diag)
            anchors = frame.group_by('county', 'issue').agg(pl.col('persistence').first().alias('anchor'))
        native[task] = frame.select('county', 'issue', 'lead_minutes').with_columns(pl.Series('p', np.asarray(p, np.float64)))
        log.info('Task %s: %d native rows (%.1f s)', task, frame.height, time.perf_counter() - started)
    if anchors is None:
        anchors = pl.DataFrame(schema={'county': pl.String, 'issue': pl.Int64, 'anchor': pl.Float64})
    result = output.assemble(tables.requests, native, anchors, config)
    output.write(result, output_path)
    log.info('wrote %d rows to %s (%.1f s)', result.height, output_path, time.perf_counter() - started)
    extra = {'device': device, 'radar_rows': 0 if tables.radar is None else tables.radar.height,
             'weather_rows': tables.weather.height, 'outage_rows': tables.outages.height}
    diagnostics.report(diag, tables.requests, anchors, extra, diagnostics_json)   # observation only; never raises
    return result
