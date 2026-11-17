"""EAGLE-I annual file -> customers out per county and 15-minute time (Parquet: county, timestamp, count).

    python data_pipeline/sources/eaglei_counts.py <eaglei_outages_YYYY.csv> <out.parquet> [--counties <file>]

Keeps the listed counties (default: data_pipeline/reference/counties_40.txt); run_start_time is the UTC observation time;
a blank count stays null; for a repeated (county, time) the last row of the file wins.
"""
import argparse
from pathlib import Path

import polars as pl

REFERENCE = Path(__file__).resolve().parents[1] / 'reference' / 'counties_40.txt'


def counts(source: Path, counties: list[str]) -> pl.DataFrame:
    head = pl.read_csv(source, n_rows=0).columns
    count_column = 'customers_out' if 'customers_out' in head else 'sum'
    f = (pl.scan_csv(source, schema_overrides={'fips_code': pl.String, count_column: pl.String, 'run_start_time': pl.String},
                     row_index_name='_order')
         .filter(pl.col('fips_code').str.zfill(5).is_in(counties))
         .select(pl.col('fips_code').str.zfill(5).alias('county'),
                 pl.col('run_start_time').str.replace(' ', 'T').add('Z').str.to_datetime(time_zone='UTC').dt.epoch('s').alias('timestamp'),
                 pl.col(count_column).cast(pl.Float64, strict=False).alias('count'), '_order')
         .collect(engine='streaming'))
    return f.sort('_order').unique(['county', 'timestamp'], keep='last').sort('county', 'timestamp')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('source', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--counties', type=Path, default=REFERENCE)
    a = p.parse_args()
    f = counts(a.source, a.counties.read_text().replace(',', ' ').split())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    f.write_parquet(a.out)
    print(f'{f.height} rows, {f["count"].null_count()} blank -> {a.out}')
