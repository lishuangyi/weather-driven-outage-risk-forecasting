"""County frame statistics from the downloaded MRMS frames (one row per county and 10-minute grid time).

    python data_pipeline/mrms/county_series.py --frames-dir <frames> --start 2025-05-31 --end 2025-08-31 --out <series.parquet>

For each county and frame, on a crop of the county plus 50 km, 3 x 3 max-pooled:
  avail   frame present and the county has radar coverage
  f35/f50 share of county cells >= 35 / 50 dBZ;  max  county maximum dBZ;  ring35  share >= 35 dBZ in the ring (R, R+50] km
The input layer's availability_time is grid_time + 4 min (assemble_inputs.py).
"""
import argparse
import json
import multiprocessing
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrms.grid import SPATIAL_PLAN, geometry, pooled  # noqa: E402


def day_series(path: Path, geo: dict) -> pl.DataFrame:
    meta = json.loads(path.with_suffix('.json').read_text())
    present = np.array([m['key'] is not None for m in meta['frames']])
    grid = np.array([int(datetime.fromisoformat(m['grid_time']).timestamp()) for m in meta['frames']])
    replay = np.array([m.get('availability') == 'replay' for m in meta['frames']])
    z = np.load(path)
    parts = []
    for region in sorted({g['region'] for g in geo.values()}):
        fr = z[region]
        for fips, g in geo.items():
            if g['region'] != region:
                continue
            d = pooled(fr[:, g['rows'], g['cols']])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                cn = d[:, g['mask']]
                ok = np.isfinite(cn)
                n = ok.sum(1)
                rg = d[:, g['ring']]
                rn = np.isfinite(rg).sum(1)
                avail = present & (n > 0)
                parts.append(pl.DataFrame({
                    'county': [fips] * len(grid), 'grid_time': grid, 'avail': avail, 'replay': replay,
                    'f35': np.where(avail, (np.nan_to_num(cn, nan=-99) >= 35).sum(1) / np.maximum(n, 1), np.nan),
                    'f50': np.where(avail, (np.nan_to_num(cn, nan=-99) >= 50).sum(1) / np.maximum(n, 1), np.nan),
                    'max': np.where(avail, np.nanmax(np.where(ok, cn, -np.inf), 1), np.nan),
                    'ring35': np.where(present & (rn > 0), (np.nan_to_num(rg, nan=-99) >= 35).sum(1) / np.maximum(rn, 1), np.nan)}))
    return pl.concat(parts)


def _job(args):
    return day_series(*args)


def series(frames_dir: Path, first: datetime, last: datetime, workers: int = 2, plan: Path = SPATIAL_PLAN) -> pl.DataFrame:
    geo = geometry(plan)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    files = [frames_dir / f'{d:%Y%m%d}.npz' for d in days]
    files = [f for f in files if f.exists() and f.with_suffix('.json').exists()]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as ex:  # fork + polars deadlocks
        parts = list(ex.map(_job, [(f, geo) for f in files]))
    return pl.concat(parts).sort('county', 'grid_time')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--frames-dir', type=Path, required=True, help='continuous frames written by download.py')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True, help='last UTC day, inclusive')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    first, last = (datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=__import__('datetime').timezone.utc) for s in (a.start, a.end))
    s = series(a.frames_dir, first, last, a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    s.write_parquet(a.out)
    print(f'{s.height} rows, avail share {float(s["avail"].mean()):.3f} -> {a.out}')
