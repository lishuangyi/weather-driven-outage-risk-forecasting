"""Temporal-causality test: data that became available after an issue time must not change that issue's forecast.

Every record published after the cut-off (outage records, weather runs, radar frames) is replaced by an extreme value.
Forecasts issued at or before the cut-off must stay bit-identical; forecasts issued later are expected to change (which
shows the perturbation actually reaches the models).
"""
from datetime import datetime, timezone

import polars as pl

from conftest import KEYS, read_predictions, requires_models, run_inference

pytestmark = requires_models
CUTOFF = datetime(2025, 7, 9, 0, 0, tzinfo=timezone.utc)


def perturb_future(input_dir):
    path = input_dir / 'eaglei' / 'eaglei_outages.parquet'
    t = pl.read_parquet(path)
    late = pl.col('source_record_time') > CUTOFF
    t.with_columns(pl.when(late).then(pl.col('customers_out') * 50 + 5000).otherwise(pl.col('customers_out'))
                   .alias('customers_out')).write_parquet(path)

    path = input_dir / 'weather' / 'weather_forecasts.parquet'
    w = pl.read_parquet(path)
    late = pl.col('availability_time') > CUTOFF
    numeric = [c for c, d in w.schema.items() if d == pl.Float64]
    w.with_columns([pl.when(late).then(pl.col(c) * 3 + 40).otherwise(pl.col(c)).alias(c) for c in numeric]).write_parquet(path)

    path = input_dir / 'radar' / 'mrms_county_series.parquet'
    r = pl.read_parquet(path)
    late = pl.col('availability_time') > CUTOFF
    r.with_columns([pl.when(late).then(pl.lit(1.0)).otherwise(pl.col(c)).alias(c) for c in ('avail', 'f35', 'f50', 'ring35')]
                   + [pl.when(late).then(pl.lit(70.0)).otherwise(pl.col('max_dbz')).alias('max_dbz')]).write_parquet(path)


def test_future_data_does_not_change_past_forecasts(fixture_copy, fixture_predictions, tmp_path):
    perturb_future(fixture_copy)
    out = tmp_path / 'perturbed.csv'
    assert run_inference(fixture_copy, out) == 0
    perturbed = read_predictions(out)

    issue = pl.col('issue_time').str.to_datetime(time_zone='UTC')
    joined = fixture_predictions.join(perturbed, on=KEYS, suffix='_p')
    past = joined.filter(issue <= CUTOFF)
    future = joined.filter(issue > CUTOFF)
    assert past.height > 0 and future.height > 0
    assert (past['predicted_x'] == past['predicted_x_p']).all(), 'a forecast used data published after its issue time'
    assert (future['predicted_x'] != future['predicted_x_p']).any(), 'perturbation did not reach the models'
