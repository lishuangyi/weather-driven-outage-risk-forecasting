"""The DuckDB warehouse (sql/) against the Python implementation: same features, labels and scores.

Two independent implementations of the same definitions; any difference is a bug in one of them.
"""
import math
from pathlib import Path

import polars as pl
import pytest

from conftest import EXPECTED, FIXTURE, ROOT, to_polars
from outage_forecast.config import Config
from outage_forecast.features.history import HISTORY_NAMES, OutageHistory, history_features, model_outage_view
from outage_forecast.inputs import read_mcc, read_outages
from outage_forecast.warehouse import build

FULL_2025 = ROOT / 'data' / 'inputs' / 'input_2025'


@pytest.fixture(scope='module')
def predictions_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp('pred') / 'predictions.csv'
    pl.read_parquet(EXPECTED).write_csv(path)
    return path


@pytest.fixture(scope='module')
def warehouse(predictions_csv):
    con = build(FIXTURE, predictions=predictions_csv)
    yield con
    con.close()


def python_history(input_dir: Path, issues: pl.DataFrame) -> dict:
    """{(task, county, issue_s): {feature: value}} from the Python feature code."""
    config = Config.load()
    outages = model_outage_view(read_outages(input_dir))
    history = OutageHistory(outages)
    mcc = read_mcc(input_dir, config, set(issues['county'].unique().to_list()))
    out = {}
    for task, county, issue in issues.select('task', 'county', 'issue_s').iter_rows():
        ts, counts = history.at(county, int(issue))
        out[task, county, int(issue)] = history_features(ts, counts, int(issue), mcc[county], task)
    return out


def assert_same(sql_value, py_value, label):
    py_missing = py_value is None or (isinstance(py_value, float) and math.isnan(py_value))
    sql_missing = sql_value is None or (isinstance(sql_value, float) and math.isnan(sql_value))
    assert sql_missing == py_missing, f'{label}: SQL {sql_value!r} vs Python {py_value!r}'
    if not py_missing:
        assert sql_value == pytest.approx(py_value, rel=1e-12, abs=1e-15), label


def test_history_features_match_python(warehouse):
    sql = to_polars(warehouse.sql('SELECT * EXCLUDE (issue_time) FROM gold.history_features'))
    assert sql.height == warehouse.sql('SELECT count(*) FROM silver.issues').fetchall()[0][0]
    expected = python_history(FIXTURE, sql)
    for row in sql.iter_rows(named=True):
        py = expected[row['task'], row['county'], int(row['issue_s'])]
        for name in HISTORY_NAMES:
            assert_same(row[name], py[name], f"{row['task']} {row['county']} {row['issue_s']} {name}")


def test_labels_match_scorer(warehouse, predictions_csv):
    import scoring

    joined = scoring.frames(predictions_csv, FIXTURE, Config.load())
    sql = to_polars(warehouse.sql('SELECT task, arch, county, issue_s, target_s, truth_x FROM gold.labels'))
    for task in ('A', 'B'):
        py = joined[task].select('task', 'arch', 'county', pl.col('issue').alias('issue_s'), pl.col('target').alias('target_s'), 'truth')
        both = py.join(sql.filter(pl.col('task') == task), on=['task', 'arch', 'county', 'issue_s', 'target_s'], how='full', coalesce=True)
        assert both.height == py.height
        assert (both['truth'].is_null() == both['truth_x'].is_null()).all()
        assert (both['truth'].is_nan().fill_null(False) == both['truth_x'].is_nan().fill_null(False)).all()
        ok = both.filter(pl.col('truth').is_not_null() & pl.col('truth').is_finite())
        assert (ok['truth'] == ok['truth_x']).all(), f'Task {task} labels differ'


def compare_scores(con, predictions: Path, input_dir: Path):
    import scoring

    py = scoring.score(predictions, input_dir, Config.load())
    metrics = {(t, d, g): s for t, d, g, s in con.execute(
        "SELECT task, dimension, grp, score FROM gold.metrics WHERE model = 'model'").fetchall()}
    for task, entry in py['tasks'].items():
        assert metrics[task, 'pooled', 'all'] == pytest.approx(entry['pooled']['score'], abs=1e-9)
        for arch, s in entry['per_architecture'].items():
            assert metrics[task, 'architecture', arch] == pytest.approx(s['score'], abs=1e-9)
        for county, s in entry['per_county_diagnostic'].items():
            assert metrics[task, 'county', county] == pytest.approx(s['score'], abs=1e-9)
    a, b, g, total = con.execute("SELECT task_a, task_b, generalization, technical_score FROM gold.scoreboard "
                                 "WHERE model = 'model'").fetchall()[0]
    assert g == pytest.approx(py['generalization'], abs=1e-9)
    assert total == pytest.approx(py['technical_score'], abs=1e-9)
    return a, b


def test_scores_match_scorer(warehouse, predictions_csv):
    compare_scores(warehouse, predictions_csv, FIXTURE)


def test_quality_checks_pass_on_clean_input(warehouse):
    failed = warehouse.sql("SELECT check_name FROM silver.quality_checks WHERE status = 'FAIL'").fetchall()
    assert failed == []


def test_quality_checks_catch_broken_input(fixture_copy):
    path = fixture_copy / 'requests' / 'forecast_requests.parquet'
    r = pl.read_parquet(path)
    pl.concat([r, r.head(3)]).write_parquet(path)                          # duplicated requests
    path = fixture_copy / 'eaglei' / 'eaglei_outages.parquet'
    o = pl.read_parquet(path)
    o.with_columns(pl.when(pl.int_range(pl.len()) < 2).then(pl.col('timestamp') - pl.duration(hours=1))
                   .otherwise(pl.col('source_record_time')).alias('source_record_time')).write_parquet(path)  # published early
    con = build(fixture_copy)
    failed = dict(con.execute("SELECT check_name, violations FROM silver.quality_checks WHERE status = 'FAIL'").fetchall())
    assert failed == {'requests: duplicate request key': 3,
                      'outages: published before observed (source_record_time < timestamp)': 2}


def test_sql_features_ignore_future_data(fixture_copy):
    """The no-lookahead property for the SQL features (same perturbation as tests/test_no_lookahead.py)."""
    from test_no_lookahead import CUTOFF, perturb_future

    before = to_polars(build(FIXTURE).sql('SELECT * EXCLUDE (issue_time) FROM gold.history_features'))
    perturb_future(fixture_copy)
    after = to_polars(build(fixture_copy).sql('SELECT * EXCLUDE (issue_time) FROM gold.history_features'))
    past = lambda f: f.filter(pl.col('issue_s') <= CUTOFF.timestamp()).sort('task', 'county', 'issue_s')  # noqa: E731
    future = lambda f: f.filter(pl.col('issue_s') > CUTOFF.timestamp()).sort('task', 'county', 'issue_s')  # noqa: E731
    assert past(before).height > 0
    assert past(before).equals(past(after))
    assert not future(before).equals(future(after))


@pytest.mark.skipif(not (FULL_2025 / 'requests').is_dir(), reason='2025 inputs not downloaded')
def test_full_window_matches(tmp_path):
    """Full June-August 2025 window (618,240 requests): every history feature and the scores of the frozen release."""
    predictions = ROOT / 'out' / 'predictions.csv'
    if not predictions.is_file():
        pytest.skip('run `make predict` first')
    con = build(FULL_2025, predictions=predictions)
    sql = to_polars(con.sql('SELECT * EXCLUDE (issue_time) FROM gold.history_features'))
    expected = python_history(FULL_2025, sql)
    for row in sql.iter_rows(named=True):
        py = expected[row['task'], row['county'], int(row['issue_s'])]
        for name in HISTORY_NAMES:
            assert_same(row[name], py[name], f"{row['task']} {row['county']} {row['issue_s']} {name}")
    a, b = compare_scores(con, predictions, FULL_2025)
    assert (round(a, 4), round(b, 4)) == (54.5219, 75.1347)


@pytest.mark.parametrize('query', sorted((ROOT / 'sql' / 'analysis').glob('*.sql')), ids=lambda p: p.stem)
def test_analysis_queries_run(warehouse, query):
    rows = warehouse.sql(query.read_text(encoding='utf-8')).fetchall()
    assert rows, f'{query.name} returned no rows'
