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


# ---------------------------------------------------------------- NULL handling and purging (review fixes, 2026-11-18)
def test_fetch_keeps_sql_null_apart_from_zero():
    import duckdb

    from outage_forecast.reference import fetch

    f = fetch(duckdb.connect(), 'SELECT * FROM (VALUES (1.5, 1), (NULL, NULL), (0.0, 0)) t(x, n)')
    assert f['x'].to_list() == [1.5, None, 0.0]
    assert f['n'].to_list() == [1, None, 0]


def test_missing_labels_stay_missing():
    con = build(FIXTURE)
    sql_missing = con.execute('SELECT count(*) FROM gold.examples WHERE truth_x IS NULL').fetchall()[0][0]
    con.close()
    frame = examples(FIXTURE)
    assert sql_missing > 0, 'the fixture should contain examples without truth'
    assert frame[LABEL].null_count() == sql_missing
    assert labelled(frame)[LABEL].null_count() == 0
    assert 'truth_published_s' not in feature_names(frame)


def _add_revision(input_dir, county, observed, published, customers_out):
    """Append a later version of one outage observation (a correction, or a blank record when customers_out is None)."""
    path = input_dir / 'eaglei' / 'eaglei_outages.parquet'
    t = pl.read_parquet(path)
    row = pl.DataFrame({'county_fips': [county], 'timestamp': [observed], 'customers_out': [customers_out],
                        'source_record_time': [published]}, schema=t.schema)
    pl.concat([t, row]).write_parquet(path)


OBSERVED = datetime(2025, 7, 9, 12, 0, tzinfo=timezone.utc)
CUTOFF = datetime(2025, 7, 10, tzinfo=timezone.utc).timestamp()


def _rows_using(frame, observed):
    """Task B examples whose label is the observation at `observed` (targets in its 15-minute interval)."""
    t = observed.timestamp()
    return frame.filter((pl.col('task') == 'B') & (pl.col('county') == '48201')
                        & (pl.col('target_s') >= t) & (pl.col('target_s') < t + 900))


def test_late_correction_is_purged_from_training(fixture_copy):
    before = _rows_using(labelled(examples(FIXTURE), before=CUTOFF), OBSERVED)
    assert before.height > 0
    _add_revision(fixture_copy, '48201', OBSERVED, datetime(2025, 7, 12, tzinfo=timezone.utc), 99_999.0)
    frame = examples(fixture_copy)
    corrected = _rows_using(frame, OBSERVED)
    assert corrected[LABEL].round(6).unique().to_list() == [round(99_999.0 / 1_827_686, 6)]   # final truth = correction
    assert _rows_using(labelled(frame, before=CUTOFF), OBSERVED).height == 0                    # but not before C
    assert _rows_using(labelled(frame), OBSERVED).height == corrected.height                   # retrospective use


def test_blank_latest_revision_makes_the_label_missing(fixture_copy):
    _add_revision(fixture_copy, '48201', OBSERVED, datetime(2025, 7, 9, 13, 0, tzinfo=timezone.utc), None)
    frame = examples(fixture_copy)
    rows = _rows_using(frame, OBSERVED)
    assert rows.height > 0 and rows[LABEL].null_count() == rows.height
    assert _rows_using(labelled(frame), OBSERVED).height == 0


def test_targets_after_the_cutoff_are_purged():
    frame = labelled(examples(FIXTURE), before=CUTOFF)
    assert frame.height > 0
    assert (frame['target_s'] < CUTOFF).all() and (frame['truth_published_s'] < CUTOFF).all()
