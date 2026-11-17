"""Download MRMS composite reflectivity (MergedReflectivityQCComposite_00.50, 2-minute, 0.01 deg) from the public AWS
archive s3://noaa-mrms-pds (anonymous) and keep the five regional crops, quantised to uint8.

    python data_pipeline/mrms/download.py --frames-dir <frames> --start 2025-05-31 --end 2025-08-31 [--workers 4]

One file per UTC day (<YYYYMMDD>.npz + .json) with 144 frames on the grid HH:M6 (00:06 ... 23:56) = issue - 4 min -
10k min for every 00/06/12/18Z issue. For each grid time the newest object with valid time in [grid - 6 min, grid] that
was available at grid + 4 min (S3 LastModified) is taken; objects back-filled into the archive more than a day after
their valid time carry no real publication time and follow the replay assumption (available at valid time + 4 min).
Quantisation: 0 = no coverage or missing frame, 1 = no echo or < -10 dBZ, else clip(round((dBZ + 10) * 2), 2, 255).
Finished days are skipped; raw objects are not kept. Requires boto3 and eccodes (requirements/data.txt).
"""
import argparse
import gzip
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import eccodes
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrms.grid import LAT0, LON0, NI, NJ, REGIONS, STEP, window  # noqa: E402

BUCKET, PRODUCT = 'noaa-mrms-pds', 'MergedReflectivityQCComposite_00.50'
SPACING, OFFSET, TOLERANCE = 10, 4, 6  # minutes
BACKFILL = timedelta(days=1)
DECODE = threading.Lock()  # eccodes is not guaranteed thread-safe; threads overlap only the network part
STAMP = re.compile(r'_(\d{8})-(\d{6})\.grib2\.gz$')
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, retries={'max_attempts': 8, 'mode': 'adaptive'}), region_name='us-east-1')


def listing(day):
    """{valid datetime: (key, LastModified)} for one UTC day."""
    out, token = {}, None
    while True:
        kw = {'Bucket': BUCKET, 'Prefix': f'CONUS/{PRODUCT}/{day:%Y%m%d}/'}
        if token:
            kw['ContinuationToken'] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get('Contents', []):
            m = STAMP.search(o['Key'])
            if m:
                out[datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)] = (o['Key'], o['LastModified'])
        if not r.get('IsTruncated'):
            return out
        token = r['NextContinuationToken']


def decode(key):
    """Download and decode one object into the regional crops; one retry, then the error propagates."""
    for attempt in (0, 1):
        try:
            raw = gzip.decompress(s3.get_object(Bucket=BUCKET, Key=key)['Body'].read())
            with DECODE:
                return _decode(raw, key)
        except (eccodes.CodesInternalError, OSError, EOFError) as e:
            if attempt:
                raise
            print(json.dumps({'retry': key, 'error': repr(e)}), flush=True)


def _decode(raw, key):
    h = eccodes.codes_new_from_message(raw)
    try:
        grid = (eccodes.codes_get(h, 'Ni'), eccodes.codes_get(h, 'Nj'), round(eccodes.codes_get(h, 'latitudeOfFirstGridPointInDegrees'), 3),
                round(eccodes.codes_get(h, 'longitudeOfFirstGridPointInDegrees') - 360, 3), eccodes.codes_get(h, 'jScansPositively'))
        if grid != (NI, NJ, LAT0, LON0, 0):
            raise ValueError(f'unexpected grid {grid} in {key}')
        v = eccodes.codes_get_values(h).reshape(NJ, NI)
    finally:
        eccodes.codes_release(h)
    q = np.clip(np.round((v + 10) * 2), 2, 255).astype(np.uint8)
    q[v < -10] = 1
    q[v <= -998] = 0
    return {r: q[a:b, c:d].copy() for r, (a, b, c, d) in ((r, window(r)) for r in REGIONS)}, len(raw)


def day_frames(day, frames_dir: Path):
    dst = frames_dir / f'{day:%Y%m%d}.npz'
    if dst.exists() and dst.with_suffix('.json').exists():
        return {'day': f'{day:%Y%m%d}', 'skip': True}
    objects = listing(day)
    n = 24 * 60 // SPACING
    frames, meta, nbytes = {r: np.zeros((n, b - a, d - c), np.uint8) for r, (a, b, c, d) in ((r, window(r)) for r in REGIONS)}, [], 0
    for i in range(n):
        grid = day + timedelta(minutes=SPACING - OFFSET + SPACING * i)
        ok = [t for t, (_, lm) in objects.items()
              if grid - timedelta(minutes=TOLERANCE) <= t <= grid and (lm <= grid + timedelta(minutes=OFFSET) or lm - t > BACKFILL)]
        if not ok:
            meta.append({'grid_time': grid.isoformat(), 'valid_time': None, 'key': None, 'last_modified': None})
            continue
        bad, crops = [], None
        for t in sorted(ok, reverse=True):  # newest first; a corrupt object falls back to the next one in the window
            key, lm = objects[t]
            try:
                crops, b = decode(key)
                nbytes += b
                break
            except (eccodes.CodesInternalError, OSError, EOFError) as e:
                bad.append({'key': key, 'error': repr(e)})
        if crops is None:
            meta.append({'grid_time': grid.isoformat(), 'valid_time': None, 'key': None, 'last_modified': None, 'bad_keys': bad})
            continue
        for r in REGIONS:
            frames[r][i] = crops[r]
        meta.append({'grid_time': grid.isoformat(), 'valid_time': t.isoformat(), 'key': key, 'last_modified': lm.isoformat(),
                     'availability': 'archive' if lm <= grid + timedelta(minutes=OFFSET) else 'replay', **({'bad_keys': bad} if bad else {})})
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + '.tmp.npz')
    np.savez_compressed(tmp, **frames)
    tmp.rename(dst)
    dst.with_suffix('.json').write_text(json.dumps({
        'day': f'{day:%Y-%m-%d}', 'product': PRODUCT, 'frames': meta, 'missing_frames': sum(m['key'] is None for m in meta),
        'regions': {r: {'bounds_lat_lon': REGIONS[r], 'rows_cols': window(r)} for r in REGIONS},
        'grid': {'lat0': LAT0, 'lon0': LON0, 'step': STEP}, 'quantisation': 'dBZ = q / 2 - 10; 0 = no coverage or missing frame; 1 = no echo / < -10 dBZ',
        'replay_rule': f'grid HH:M6, valid in [grid - {TOLERANCE} min, grid], and (S3 LastModified <= grid + {OFFSET} min, or '
                       f'LastModified > valid + 1 day = back-filled -> replay: available at valid + {OFFSET} min)',
        'replay_frames': sum(m.get('availability') == 'replay' for m in meta), 'decompressed_bytes': nbytes}, indent=1) + '\n')
    return {'day': f'{day:%Y%m%d}', 'missing_frames': sum(m['key'] is None for m in meta), 'mb_in': round(nbytes / 1e6, 1)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--frames-dir', type=Path, required=True)
    p.add_argument('--start', required=True, help='first UTC day YYYY-MM-DD (the day before the first issue)')
    p.add_argument('--end', required=True, help='last UTC day YYYY-MM-DD, inclusive')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    first, last = (datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=timezone.utc) for s in (a.start, a.end))
    jobs = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    started = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for done, r in enumerate(ex.map(lambda d: day_frames(d, a.frames_dir), jobs), 1):
            print(json.dumps({**r, 'progress': f'{done}/{len(jobs)}', 'elapsed_min': round((time.time() - started) / 60, 1)}), flush=True)
