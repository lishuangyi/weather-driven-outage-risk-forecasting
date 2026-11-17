"""Convert an input directory in the organizer CSV layout to the same layout in Parquet (zstd).

    python data_pipeline/to_parquet.py <csv_input_dir> <parquet_output_dir>

Request keys stay strings (they are written back unchanged); times become UTC datetimes; values Float64 with blank
as null; FIPS codes strings. infer.py reads either format and produces identical predictions.
"""
import argparse
import sys
from pathlib import Path

import polars as pl

TABLES = {
    'requests/forecast_requests': ([], []),
    'mcc/MCC': ([], ['total_customers']),
    'eaglei/eaglei_outages': (['timestamp', 'source_record_time'], ['customers_out']),
    'weather/weather_forecasts': (['model_run_time', 'availability_time', 'target_time'], None),
    'radar/mrms_county_series': (['grid_time', 'availability_time'], ['avail', 'f35', 'f50', 'max_dbz', 'ring35']),
}


def utc(col):
    return pl.col(col).str.strip_chars().str.replace('Z$', '+00:00').str.to_datetime(time_unit='us', time_zone='UTC').alias(col)


def number(col):
    s = pl.col(col).str.strip_chars()
    return pl.when(s == '').then(None).otherwise(s).cast(pl.Float64).alias(col)


def convert(src: Path, dst: Path) -> None:
    for name, (times, values) in TABLES.items():
        path = src / f'{name}.csv'
        if not path.exists():
            continue
        f = pl.read_csv(path, infer_schema=False)
        if values is None:  # weather: every column except keys and times is numeric
            values = [c for c in f.columns if c not in ('county_fips', *times)]
        exprs = [utc(c) for c in times] + [number(c) for c in values]
        if name == 'mcc/MCC':
            exprs = [pl.col('total_customers').str.strip_chars().cast(pl.Int64)]
        f = f.with_columns(exprs)
        out = dst / f'{name}.parquet'
        out.parent.mkdir(parents=True, exist_ok=True)
        f.write_parquet(out, compression='zstd', compression_level=19)
        print(f'{name}: {f.height} rows -> {out} ({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('csv_input_dir', type=Path)
    p.add_argument('parquet_output_dir', type=Path)
    a = p.parse_args()
    convert(a.csv_input_dir, a.parquet_output_dir)
