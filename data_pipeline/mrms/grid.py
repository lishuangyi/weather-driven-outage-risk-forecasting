"""MRMS CONUS grid, the five regional crops, and the county cells and rings used for the county frame statistics."""
import math
import warnings
from pathlib import Path

import numpy as np
import polars as pl

# (lat_min, lat_max, lon_min, lon_max): the regional county extents plus about 3 degrees (~300 km)
REGIONS = {'CA': (29.8, 38.6, -124.3, -110.8), 'FL': (21.8, 30.1, -85.6, -76.7), 'MI': (39.0, 46.1, -88.1, -78.7),
           'NE': (39.5, 46.7, -75.8, -66.6), 'TX': (26.1, 33.4, -99.4, -91.3)}
LAT0, LON0, STEP, NI, NJ = 54.995, -129.995, .01, 7000, 3500  # MRMS CONUS grid (checked against the GRIB keys per file)
POOL, KM = 3, 111.2                                            # 3 x 3 max pooling (~3 km); km per degree
STATE_REGION = {'06': 'CA', '12': 'FL', '26': 'MI', '23': 'NE', '25': 'NE', '33': 'NE', '48': 'TX'}
SPATIAL_PLAN = Path(__file__).resolve().parents[1] / 'reference' / 'hrrr_grid' / 'spatial_plan.parquet'


def window(region):
    """Row and column range of a regional crop on the CONUS grid."""
    a, b, c, d = REGIONS[region]
    r0, r1 = int(round((LAT0 - b) / STEP)), int(round((LAT0 - a) / STEP)) + 1
    c0, c1 = int(round((c - LON0) / STEP)), int(round((d - LON0) / STEP)) + 1
    return max(r0, 0), min(r1, NJ), max(c0, 0), min(c1, NI)


def counties(plan: Path = SPATIAL_PLAN):
    """Per county: region, area-weighted centroid, equal-area radius R (km) and its HRRR cells on the pooled crop."""
    import pyproj
    sp = pl.read_parquet(plan)
    p = pyproj.Proj(proj='lcc', lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=-97.5, a=6371229, b=6371229)
    x0, y0 = p(-122.719528, 21.138123)
    lon, lat = p(x0 + 3000 * sp['global_x'].to_numpy(), y0 + 3000 * sp['global_y'].to_numpy(), inverse=True)
    sp = sp.with_columns(pl.Series('lat', lat), pl.Series('lon', lon))
    out = {}
    for (fips,), g in sp.group_by('fips_code'):
        w = g['county_weight'].to_numpy()
        region = STATE_REGION[fips[:2]]
        r0, _, c0, _ = window(region)
        rows = np.round((LAT0 - g['lat'].to_numpy()) / STEP).astype(int) - r0
        cols = np.round((g['lon'].to_numpy() - LON0) / STEP).astype(int) - c0
        out[fips] = {'region': region, 'lat': float((g['lat'] * w).sum() / w.sum()), 'lon': float((g['lon'] * w).sum() / w.sum()),
                     'R': math.sqrt(float(g['county_area_m2'][0]) / math.pi) / 1000, 'cells': (rows // POOL, cols // POOL)}
    return out


def geometry(plan: Path = SPATIAL_PLAN):
    """Per county: native-pixel slice of its crop, county mask and ring mask (R, R + 50] km on the pooled crop."""
    out = {}
    for fips, ci in counties(plan).items():
        a, b, c, d = window(ci['region'])
        h, w = (b - a) // POOL, (d - c) // POOL
        cy, cx = ci['cells']
        crow = (LAT0 - ci['lat']) / STEP - a
        ccol = (ci['lon'] - LON0) / STEP - c
        pr = math.ceil((ci['R'] + 55) / (POOL * STEP * KM))
        pc = math.ceil((ci['R'] + 55) / (POOL * STEP * KM * math.cos(math.radians(ci['lat']))))
        r0 = max(min(int(cy.min()), int(crow // POOL) - pr), 0)
        r1 = min(max(int(cy.max()) + 1, int(crow // POOL) + pr + 1), h)
        c0 = max(min(int(cx.min()), int(ccol // POOL) - pc), 0)
        c1 = min(max(int(cx.max()) + 1, int(ccol // POOL) + pc + 1), w)
        mask = np.zeros((r1 - r0, c1 - c0), bool)
        inb = (cy >= r0) & (cy < r1) & (cx >= c0) & (cx < c1)
        mask[cy[inb] - r0, cx[inb] - c0] = True
        lat = LAT0 - (a + POOL * np.arange(r0, r1) + 1) * STEP
        lon = LON0 + (c + POOL * np.arange(c0, c1) + 1) * STEP
        dist = np.hypot(((lat[:, None] - ci['lat']) * KM), (lon[None, :] - ci['lon']) * KM * math.cos(math.radians(ci['lat'])))
        out[fips] = {'region': ci['region'], 'rows': slice(r0 * POOL, r1 * POOL), 'cols': slice(c0 * POOL, c1 * POOL),
                     'mask': mask, 'ring': (dist > ci['R']) & (dist <= ci['R'] + 50), 'cells_in_crop': int(inb.sum()), 'cells': int(len(cy))}
    return out


def pooled(q):
    """(T, H, W) quantised uint8 -> (T, H/3, W/3) dBZ float32 max-pooled; NaN = no coverage."""
    t, h, w = q.shape
    q = q.reshape(t, h // POOL, POOL, w // POOL, POOL)
    d = np.where(q == 0, np.nan, np.where(q == 1, -10., q / 2 - 10)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmax(d, axis=(2, 4))
