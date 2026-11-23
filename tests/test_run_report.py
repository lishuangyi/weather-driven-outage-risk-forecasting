"""Run report: provenance and scores on the fixture; the documentation tables match the committed report."""
import json

import pytest

from conftest import EXPECTED, FIXTURE
from outage_forecast.config import PACKAGE_ROOT
from outage_forecast.run_report import REPORT, render, run, sha256
from outage_forecast.warehouse import build

DAY_1, DAY_2, DAY_3 = '2025-07-08', '2025-07-09', '2025-07-10'


def _spec(tmp_path, predictions):
    spec = {'predictions': {'p': {'path': str(predictions), 'model': 'frozen ensemble', 'command': 'fixture'}},
            'inputs': {'mini': str(FIXTURE)},
            'evaluations': [
                {'id': 'all', 'label': 'all', 'role': 'test', 'predictions': 'p', 'input': 'mini', 'window': None},
                {'id': 'first', 'label': 'day 1', 'role': 'test', 'predictions': 'p', 'input': 'mini',
                 'window': [[DAY_1, DAY_2]]},
                {'id': 'second', 'label': 'day 2', 'role': 'test', 'predictions': 'p', 'input': 'mini',
                 'window': [[DAY_2, DAY_3]]},
                {'id': 'both', 'label': 'two intervals', 'role': 'test', 'predictions': 'p', 'input': 'mini',
                 'window': [[DAY_1, DAY_2], [DAY_2, DAY_3]]}],
            'uncertainty': []}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    return path


@pytest.fixture(scope='module')
def fixture_report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('run_report')
    report = run(_spec(tmp, EXPECTED), tmp / 'report.json')
    assert json.loads((tmp / 'report.json').read_text())['evaluations'].keys() == report['evaluations'].keys()
    return report


def test_report_records_code_inputs_predictions_and_environment(fixture_report):
    r = fixture_report
    assert r['code']['commit'] and isinstance(r['code']['uncommitted_changes'], list)
    assert r['predictions']['p']['sha256'] == sha256(EXPECTED)
    mini = r['inputs']['mini']
    assert mini['files']['requests/forecast_requests.parquet'] == sha256(FIXTURE / 'requests' / 'forecast_requests.parquet')
    assert sum(t['rows'] for t in mini['requests'].values()) == 13_440
    assert r['environment']['python'] and r['environment']['packages']['duckdb']
    assert r['artifacts']['frozen_ensemble']['model_files'] > 0
    assert r['recipes']['reference']['seed'] == 42


def test_report_scores_match_the_warehouse(fixture_report):
    con = build(FIXTURE, predictions=EXPECTED)
    board = {m: (a, b) for m, a, b in con.execute('SELECT model, task_a, task_b FROM gold.scoreboard').fetchall()}
    con.close()
    tasks = fixture_report['evaluations']['all']['tasks']
    for model in ('model', 'persistence'):
        assert (tasks['A'][model]['score'], tasks['B'][model]['score']) == pytest.approx(board[model], abs=1e-9)
        assert tasks['A'][model]['cases'] == tasks['A']['coverage']['scored']


def test_a_window_of_two_intervals_is_their_union(fixture_report):
    e = fixture_report['evaluations']
    assert e['both']['window'] == [[DAY_1, DAY_2], [DAY_2, DAY_3]]
    for task in ('A', 'B'):
        parts = [e[k]['tasks'][task]['coverage'] for k in ('first', 'second')]
        assert all(p['scored'] > 0 for p in parts)
        both = e['both']['tasks'][task]
        assert both['coverage']['scored'] == sum(p['scored'] for p in parts) == e['all']['tasks'][task]['coverage']['scored']
        assert both['model']['score'] == pytest.approx(e['all']['tasks'][task]['model']['score'], abs=1e-9)


def test_documentation_tables_match_the_committed_report():
    assert REPORT.is_file(), 'artifacts/runs/benchmark.json is missing'
    assert render(check=True) == [], 'run `python -m outage_forecast.run_report render` (or re-run the report)'
    report = json.loads(REPORT.read_text())
    assert report['code']['uncommitted_changes'] == [], 'the committed report must come from a clean checkout'
    for doc in ('README.md', 'docs/RESULTS.md'):
        assert '<!-- run-report:' in (PACKAGE_ROOT / doc).read_text()
