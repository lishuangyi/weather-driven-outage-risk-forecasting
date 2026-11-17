"""Object plan of the 00Z HRRR runs of a date window (input of download.py).

    python data_pipeline/hrrr/plan_window.py --start 2025-05-30 --end 2025-08-31 --plan-out plan.parquet \
        --variables-out variables.parquet

One row per run, variable and chunk: the s3://hrrrzarr key of the forecast chunk covering the 40 counties
(reference/hrrr_grid/configuration.json: variable levels and the nine chunk ids), with the Zarr dtype, fill value and
forecast-hour range of the variable (reference/hrrr_plans/hrrr00z_2025_variables.parquet, the audited metadata of the
same archive layout). For 2025-05-30 .. 2025-08-31 the result equals the shipped hrrr00z_2025_plan.parquet.
"""
import argparse
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

REFERENCE = Path(__file__).resolve().parents[1] / 'reference'
VARIABLES = REFERENCE / 'hrrr_plans' / 'hrrr00z_2025_variables.parquet'


def plan(first: datetime, last: datetime) -> pl.DataFrame:
    grid = json.loads((REFERENCE / 'hrrr_grid' / 'configuration.json').read_text())
    specs = {r['variable']: r for r in pl.read_parquet(VARIABLES).iter_rows(named=True)}
    rows = []
    day = first
    while day <= last:
        stamp = day.strftime('%Y%m%d')
        for variable in sorted(grid['levels']):
            level, spec = grid['levels'][variable], specs[variable]
            for chunk in grid['chunks']:
                rows.append({'year': day.year, 'model_run_utc': day.strftime('%Y-%m-%dT00:00:00Z'), 'variable': variable,
                             'level': level, 'chunk_id': chunk,
                             'remote_object_key': f'sfc/{stamp}/{stamp}_00z_fcst.zarr/{level}/{variable}/{level}/{variable}/{chunk}',
                             'expected_bytes': None, 'dtype': spec['dtype'], 'fill_value': spec['fill_value'],
                             'forecast_hour_min': 1, 'forecast_hour_max': json.loads(spec['shape'])[0]})
        day += timedelta(days=1)
    return pl.DataFrame(rows, schema={'year': pl.Int64, 'model_run_utc': pl.String, 'variable': pl.String, 'level': pl.String,
                                      'chunk_id': pl.String, 'remote_object_key': pl.String, 'expected_bytes': pl.Null,
                                      'dtype': pl.String, 'fill_value': pl.Float64, 'forecast_hour_min': pl.Int64,
                                      'forecast_hour_max': pl.Int64})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--start', required=True, help='first run day YYYY-MM-DD')
    p.add_argument('--end', required=True, help='last run day YYYY-MM-DD, inclusive')
    p.add_argument('--plan-out', type=Path, required=True)
    p.add_argument('--variables-out', type=Path, required=True)
    a = p.parse_args()
    first, last = (datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=timezone.utc) for s in (a.start, a.end))
    f = plan(first, last)
    a.plan_out.parent.mkdir(parents=True, exist_ok=True)
    f.write_parquet(a.plan_out)
    shutil.copyfile(VARIABLES, a.variables_out)
    print(f'{f["model_run_utc"].n_unique()} runs, {f.height} objects -> {a.plan_out}')
