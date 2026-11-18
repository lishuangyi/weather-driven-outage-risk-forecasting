"""Scoring cohort: the predictions file must cover the scope exactly once; model and persistence share one cohort."""
import polars as pl
import pytest

from conftest import EXPECTED, FIXTURE
from outage_forecast.warehouse import PredictionError, build

DAY1 = ('2025-07-08', '2025-07-09')


def _write(frame: pl.DataFrame, tmp_path, name='p.parquet'):
    path = tmp_path / name
    frame.write_parquet(path)
    return path


@pytest.fixture
def expected():
    return pl.read_parquet(EXPECTED)


def test_complete_file_scores_model_and_persistence_on_one_cohort(expected, tmp_path):
    con = build(FIXTURE, predictions=_write(expected, tmp_path))
    keys = 'task, arch, county, issue_s, target_s'
    model = con.execute(f"SELECT {keys} FROM gold.scored_cases WHERE model = 'model' ORDER BY ALL").fetchall()
    base = con.execute(f"SELECT {keys} FROM gold.scored_cases WHERE model = 'persistence' ORDER BY ALL").fetchall()
    cohort = con.execute(f'SELECT {keys} FROM gold.score_cohort ORDER BY ALL').fetchall()
    assert model == base == cohort and len(cohort) > 0
    assert con.execute("SELECT count(*) FROM gold.prediction_checks WHERE status = 'FAIL'").fetchall() == [(0,)]


@pytest.mark.parametrize('broken, message', [
    (lambda p: pl.concat([p, p.head(2)]), 'duplicate prediction keys'),
    (lambda p: p.slice(5), 'requests without a prediction'),
    (lambda p: pl.concat([p, p.head(1).with_columns(pl.lit('2025-07-08T00:30:00Z').alias('target_time'))]),
     'predictions without a request'),
    (lambda p: p.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(float('nan'))
                              .otherwise(pl.col('predicted_risk')).alias('predicted_risk')), 'out-of-range'),
    (lambda p: p.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(1.5)
                              .otherwise(pl.col('predicted_x')).alias('predicted_x')), 'out-of-range'),
])
def test_incomplete_or_invalid_files_are_rejected(expected, tmp_path, broken, message):
    with pytest.raises(PredictionError, match=message):
        build(FIXTURE, predictions=_write(broken(expected), tmp_path))


def test_subset_needs_an_explicit_scope(expected, tmp_path):
    issue = pl.col('issue_time').str.to_datetime(time_zone='UTC')
    subset = _write(expected.filter(issue < pl.datetime(2025, 7, 9, time_zone='UTC')), tmp_path)
    with pytest.raises(PredictionError, match='requests without a prediction'):
        build(FIXTURE, predictions=subset)
    con = build(FIXTURE, predictions=subset, scope=DAY1)
    assert con.execute("SELECT count(*) FROM gold.prediction_checks WHERE status = 'FAIL'").fetchall() == [(0,)]


def test_rows_outside_the_scope_are_ignored_and_counted(expected, tmp_path):
    con = build(FIXTURE, predictions=_write(expected, tmp_path), scope=DAY1)
    ignored = con.execute("SELECT violations FROM gold.prediction_checks "
                          "WHERE check_name LIKE 'prediction rows outside%'").fetchall()[0][0]
    in_scope = con.execute('SELECT count(*) FROM gold.score_scope').fetchall()[0][0]
    assert ignored > 0 and ignored + in_scope == expected.height
    assert con.execute("SELECT max(issue_s) < epoch(TIMESTAMPTZ '2025-07-09 00:00:00+00') FROM gold.score_cohort").fetchall() == [(True,)]


def test_coverage_accounts_for_every_requested_case(expected, tmp_path):
    con = build(FIXTURE, predictions=_write(expected, tmp_path))
    req, scored, no_truth, invalid = con.execute(
        'SELECT sum(requested), sum(scored), sum(excluded_no_truth), sum(excluded_invalid_truth) FROM gold.coverage').fetchall()[0]
    assert req == expected.height == scored + no_truth + invalid
    assert scored == con.execute('SELECT count(*) FROM gold.score_cohort').fetchall()[0][0]
    assert no_truth > 0                     # the fixture has blank records, so some targets have no truth
