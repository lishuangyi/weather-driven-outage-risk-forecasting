"""Reference trainer: training set from the warehouse, purging, prediction output format."""
from datetime import datetime, timezone

import numpy as np
import polars as pl

from conftest import FIXTURE, KEYS, read_predictions
from outage_forecast.reference import LABEL, examples, feature_names, fit, labelled, predict, write_predictions
from outage_forecast.warehouse import build


def test_examples_cover_every_request_once():
    frame = examples(FIXTURE)
    requests = pl.read_parquet(FIXTURE / 'requests' / 'forecast_requests.parquet')
    assert frame.height == requests.select('task_id', 'county_fips', 'issue_time', 'target_time').unique().height
    assert frame.select('task', 'county', 'issue_s', 'target_s').is_duplicated().sum() == 0
    features = feature_names(frame)
    assert LABEL not in features and 'issue_s' not in features
    assert {'hist_latest_x', 'wx_gust_max', 'rx_f35_max_24h', 'lead_minutes'} <= set(features)


def test_purging_drops_targets_in_the_evaluation_period():
    frame = examples(FIXTURE)
    boundary = datetime(2025, 7, 9, tzinfo=timezone.utc).timestamp()
    kept = labelled(frame, before=boundary)
    assert kept.height > 0 and (kept['target_s'] < boundary).all()


def test_predictions_have_the_standard_format(tmp_path):
    frame = examples(FIXTURE)
    features = feature_names(frame)
    models = {t: (fit(labelled(frame.filter(pl.col('task') == t)), features, 'cpu', 5, v), v) for t, v in (('A', 'plain'), ('B', 'anchored'))}
    out = tmp_path / 'reference.csv'
    con = build(FIXTURE)
    write_predictions(con, predict(models, frame, features), out)
    con.close()
    p = read_predictions(out)
    requests = pl.read_parquet(FIXTURE / 'requests' / 'forecast_requests.parquet')
    assert p.columns == KEYS + ['predicted_x', 'predicted_risk']
    assert p.height == requests.height
    assert p['predicted_x'].is_between(0, 1).all() and p['predicted_risk'].is_between(0, 1).all()
    x0 = p['architecture_id'].replace_strict({'ARC01': 0.001, 'ARC02': 0.00117, 'ARC03': 0.00143, 'ARC04': 0.0016}).cast(pl.Float64)
    np.testing.assert_allclose(p['predicted_risk'], 1 / (1 + np.exp(-4000 * (p['predicted_x'] - x0))), atol=1e-12)
