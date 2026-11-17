"""ASNOW lead axis of the 00Z HRRR runs: the accumulated-snow array of a run is stored with one step per hour, but its
verified time coordinate skips some leads. This table maps each stored time to the verified target time.

    python data_pipeline/hrrr/snow_axis.py <coordinates_dir> <out.parquet>

<coordinates_dir> holds the per-run coordinate audits written by the HRRR download stage (<YYYYMMDD>_00z.json).
Runs without a verified ASNOW coordinate are absent (their ASNOW statistics are not used).
"""
import json
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def build(directory: Path) -> pl.DataFrame:
    rows = []
    for path in sorted(directory.glob('*_00z.json')):
        run = int(datetime.strptime(path.stem[:8], '%Y%m%d').replace(tzinfo=timezone.utc).timestamp())
        audit = json.loads(path.read_text())['audit']
        a = next((a for a in audit if a['variable'] == 'ASNOW_acc_fcst' and a['status'] == 'COORDINATE_VERIFIED'), None)
        if a:
            rows += [(run, run + i * 3600, run + int(lead) * 3600) for i, lead in enumerate(a['actual_leads'], 1)]
    return pl.DataFrame(rows, schema={'model_run_utc': pl.Int64, 'valid_time_utc': pl.Int64, 'corrected_time_utc': pl.Int64}, orient='row')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('coordinates_dir', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    f = build(a.coordinates_dir)
    f.write_parquet(a.out)
    print(f'{f["model_run_utc"].n_unique()} runs, {f.height} rows -> {a.out}')
