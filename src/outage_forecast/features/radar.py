"""MRMS radar exposure (20 rx_* columns) from the county frame series, frames with availability_time <= issue.

The 24 h before the issue are 144 ten-minute slots ending at issue - 4 min. For a slot with several available rows,
the last one in file order is used. A county without any series gets NaN for every column; a county whose series has
no usable frame gets zero sums and rx_cov24 = 0 (the models' missing-radar fallbacks key on rx_cov24 = 0).
"""
from __future__ import annotations

import numpy as np
import polars as pl

RX = ['rx_e35_1h', 'rx_e50_1h', 'rx_e35_3h', 'rx_e50_3h', 'rx_e35_6h', 'rx_e50_6h', 'rx_ring35_6h', 'rx_e35_12h', 'rx_e50_12h',
      'rx_e35_24h', 'rx_e50_24h', 'rx_ring35_24h', 'rx_max_1h', 'rx_max_6h', 'rx_max_24h', 'rx_now35', 'rx_cov24', 'rx_since35_h',
      'rx_peak35_24h', 'rx_since_peak_h']
FRAME_S, LATENCY_S, SLOTS = 600, 240, 144


class RadarSeries:
    def __init__(self, radar: pl.DataFrame | None):
        self.series = {}
        if radar is None:
            return
        for (county,), g in radar.sort('county', 'grid', 'row').group_by('county', maintain_order=True):
            self.series[county] = (g['grid'].to_numpy(), g['availability'].to_numpy(),
                                   g.select('avail', 'f35', 'f50', 'max_dbz', 'ring35').to_numpy().astype(np.float64))

    def exposure(self, county: str, issue: int) -> dict[str, float]:
        if county not in self.series:
            return {k: np.nan for k in RX}
        grid, availability, values = self.series[county]
        slots = issue - LATENCY_S - FRAME_S * np.arange(SLOTS - 1, -1, -1)  # oldest first
        frame = np.full((SLOTS, 5), np.nan)
        found = np.zeros(SLOTS, bool)
        lo = np.searchsorted(grid, slots, 'left')
        hi = np.searchsorted(grid, slots, 'right')
        for i in np.flatnonzero(hi > lo):
            usable = np.flatnonzero(availability[lo[i]:hi[i]] <= issue)
            if len(usable):  # rows of one grid time are in file order: the last usable one wins
                frame[i] = values[lo[i] + usable[-1]]
                found[i] = True
        av = np.where(found, frame[:, 0], 0.)
        return _exposure(av, frame[:, 1], frame[:, 2], frame[:, 3], frame[:, 4])


def _exposure(av, f35, f50, mx, ring35):
    out = {}
    for w in (1, 3, 6, 12, 24):
        sl = slice(SLOTS - w * 6, SLOTS)
        out[f'rx_e35_{w}h'] = np.nansum(f35[sl]) / 6
        out[f'rx_e50_{w}h'] = np.nansum(f50[sl]) / 6
        if w in (6, 24):
            out[f'rx_ring35_{w}h'] = np.nansum(ring35[sl]) / 6
    for w in (1, 6, 24):
        part = mx[-w * 6:]
        out[f'rx_max_{w}h'] = float(np.nanmax(part)) if np.isfinite(part).any() else np.nan
    out['rx_now35'] = f35[-1]
    out['rx_cov24'] = av.sum() / SLOTS
    hit = np.nan_to_num(f35, nan=0.) >= .1
    out['rx_since35_h'] = np.argmax(hit[::-1]) / 6 if hit.any() else np.nan
    pk = np.nan_to_num(f35, nan=0.)
    out['rx_peak35_24h'] = pk.max()
    out['rx_since_peak_h'] = (SLOTS - 1 - np.argmax(pk)) / 6 if pk.max() > 0 else np.nan
    return {k: float(out[k]) for k in RX}
