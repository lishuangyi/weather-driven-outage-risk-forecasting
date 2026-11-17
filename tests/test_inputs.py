"""Input validation: invalid inputs stop the run with exit code 2 and write nothing."""
import polars as pl

from conftest import requires_models, run_inference

pytestmark = requires_models


def test_missing_request_table_is_rejected(fixture_copy, tmp_path):
    (fixture_copy / 'requests' / 'forecast_requests.parquet').unlink()
    out = tmp_path / 'p.csv'
    assert run_inference(fixture_copy, out) == 2
    assert not out.exists()


def test_wrong_customer_denominator_is_rejected(fixture_copy, tmp_path):
    path = fixture_copy / 'mcc' / 'MCC.parquet'
    pl.read_parquet(path).with_columns(pl.col('total_customers') + 1).write_parquet(path)
    out = tmp_path / 'p.csv'
    assert run_inference(fixture_copy, out) == 2
    assert not out.exists()


def test_off_grid_lead_is_rejected(fixture_copy, tmp_path):
    path = fixture_copy / 'requests' / 'forecast_requests.parquet'
    r = pl.read_parquet(path)
    r = r.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(pl.lit('2025-07-08T00:07:00+00:00'))
                       .otherwise(pl.col('target_time')).alias('target_time'))
    r.write_parquet(path)
    out = tmp_path / 'p.csv'
    assert run_inference(fixture_copy, out) == 2


def test_missing_radar_uses_fallback(fixture_copy, tmp_path):
    (fixture_copy / 'radar' / 'mrms_county_series.parquet').unlink()
    out = tmp_path / 'p.csv'
    assert run_inference(fixture_copy, out) == 0
    assert out.exists()
