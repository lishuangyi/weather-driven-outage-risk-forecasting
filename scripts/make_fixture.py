"""Cut a small, self-contained input directory out of a full input directory (used for tests and CI).

    python scripts/make_fixture.py --src data/inputs/input_2025 --start 2025-07-08 --end 2025-07-10 \
        --out tests/fixtures/mini_input

Kept rows:
  requests   issues in [start, end)
  eaglei     timestamps in [start - 8 days, end + 2 days)   (7-day history for the networks + truth after the issues)
  weather    model runs in [start - 1 day, end)              (00Z runs and the previous-day 18Z runs)
  radar      frame times in [start - 1 day, end)
  mcc        unchanged
All values are copied unchanged, so predictions on the fixture equal the predictions of the same issues on the source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl


def utc(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    start, end = utc(a.start), utc(a.end)

    def write(table: str, frame: pl.DataFrame) -> None:
        path = a.out / f'{table}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path, compression='zstd', compression_level=19, statistics=False)
        print(f'{table:30s} {frame.height:>8,d} rows  {path.stat().st_size / 1024:8.1f} KiB')

    req = pl.read_parquet(a.src / 'requests/forecast_requests.parquet')
    issue = pl.col('issue_time').str.to_datetime(time_zone='UTC')
    write('requests/forecast_requests', req.filter((issue >= start) & (issue < end)))
    eag = pl.read_parquet(a.src / 'eaglei/eaglei_outages.parquet')
    write('eaglei/eaglei_outages', eag.filter(pl.col('timestamp').is_between(start - timedelta(days=8), end + timedelta(days=2),
                                                                          closed='left')))
    wx = pl.read_parquet(a.src / 'weather/weather_forecasts.parquet')
    write('weather/weather_forecasts', wx.filter(pl.col('model_run_time').is_between(start - timedelta(days=1), end, closed='left')))
    rad = pl.read_parquet(a.src / 'radar/mrms_county_series.parquet')
    write('radar/mrms_county_series', rad.filter(pl.col('grid_time').is_between(start - timedelta(days=1), end, closed='left')))
    write('mcc/MCC', pl.read_parquet(a.src / 'mcc/MCC.parquet'))


if __name__ == '__main__':
    main()
