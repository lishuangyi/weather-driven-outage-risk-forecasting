"""Alert-budget replay: onset definition (SQL), budget cutoffs, one-to-one matching, no lookahead."""
from datetime import datetime, timezone

import duckdb
import numpy as np
import polars as pl
import pytest

from conftest import EXPECTED, FIXTURE
from outage_forecast.alerts import PROTOCOL, _sql, event_tables, fit_cutoff, ratio_difference, replay, summary, window_issue
from outage_forecast.warehouse import build

X0 = PROTOCOL['threshold_x']
LOW, HIGH = 0.0002, 0.003
T0 = 1_750_000_000 // 86400 * 86400          # a 00 UTC issue time


def onsets_of(hourly: list, county: str = '1') -> list[int]:
    """Onset hours (index of the hour) of an hourly series; None = a missing hour, 'partial' = 3 of 4 quarters."""
    con = duckdb.connect()
    con.execute('CREATE SCHEMA gold')
    con.execute('CREATE TABLE gold.final_outages (county VARCHAR, ts_s BIGINT, src_s BIGINT, x DOUBLE)')
    rows = []
    for h, v in enumerate(hourly):
        end = T0 + h * 3600
        if v is None:
            continue
        quarters = (2700, 1800, 900) if v == 'partial' else (2700, 1800, 900, 0)
        rows += [(county, end - q, end - q, LOW if v == 'partial' else v) for q in quarters]
    con.executemany('INSERT INTO gold.final_outages VALUES (?, ?, ?, ?)', rows)
    con.execute(_sql('01_hourly_truth.sql', PROTOCOL, X0))
    return [(h - T0) // 3600 for _, h in con.execute(_sql('02_onsets.sql', PROTOCOL, X0)).fetchall()]


def test_onset_needs_six_complete_quiet_hours():
    assert onsets_of([LOW] * 6 + [HIGH]) == [6]
    assert onsets_of([LOW] * 5 + [HIGH]) == []
    assert onsets_of([LOW] * 6 + [X0]) == [6]                          # reaching the threshold counts
    assert onsets_of([LOW] * 6 + [X0 * 0.999]) == []


def test_a_gap_or_a_partial_hour_breaks_the_quiet_run():
    assert onsets_of([LOW] * 3 + [None] + [LOW] * 2 + [HIGH]) == []
    assert onsets_of([LOW] * 3 + ['partial'] + [LOW] * 2 + [HIGH]) == []
    assert onsets_of([LOW] * 3 + [None] + [LOW] * 6 + [HIGH]) == [10]   # quiet again after the gap
    assert onsets_of([LOW] * 6 + [None, HIGH]) == []                    # first high after a gap: unknown


def test_one_episode_gives_one_onset_and_rearms_after_six_quiet_hours():
    series = [LOW] * 6 + [HIGH] * 40 + [LOW] * 5 + [HIGH] + [LOW] * 6 + [HIGH]
    assert onsets_of(series) == [6, 58]


def test_daily_windows_tile_the_time_axis():
    assert window_issue(T0 + 3600) == T0
    assert window_issue(T0 + 86400) == T0                               # (T, T + 24 h] is closed on the right
    assert window_issue(T0) == T0 - 86400


def test_cutoff_respects_the_budget_and_excludes_ties():
    days = 30
    assert fit_cutoff([1, 2, 3, 4, 5, 6], 4, days) == 2                 # 4 scores > 2
    assert sum(s > fit_cutoff([1, 2, 3, 3, 3, 6], 2, days) for s in [1, 2, 3, 3, 3, 6]) == 1   # ties at the boundary stay out
    assert fit_cutoff([None, 1, None], 4, days) == -np.inf             # fewer eligible issues than the allowance
    assert fit_cutoff([5, 6, 7], 1, 15) == 7                            # floor(1 * 15 / 30) = 0 alerts


def _issues(rows):
    keys = ('county', 'issue_s', 'gate_ok', 'model_score', 'verified')
    out = {k: [r[i] for r in rows] for i, k in enumerate(keys)}
    out['latest_x'], out['weather_score'] = out['model_score'], out['model_score']
    return out


def test_matching_is_one_to_one_and_counts_only_verified_windows():
    protocol = PROTOCOL | {'fit': ['2020-01-01', '2020-01-31'], 'evaluate': ['2020-02-01', '2020-03-01']}
    fit_t = [int(datetime(2020, 1, d, tzinfo=timezone.utc).timestamp()) for d in range(1, 31)]
    ev = [int(datetime(2020, 2, d, tzinfo=timezone.utc).timestamp()) for d in (1, 2, 3, 4)]
    rows = [('1', t, True, 0.1, True) for t in fit_t[:-1]] + [('1', fit_t[-1], True, 0.9, True)]   # cutoff 0.1
    rows += [('1', ev[0], True, 0.5, True),      # alert, two onsets in the window
             ('1', ev[1], True, 0.5, False),     # alert, unverified window
             ('1', ev[2], False, 0.5, True),     # abstains (not quiet), one onset: a miss
             ('1', ev[3], True, 0.5, True)]      # alert, no onset: false alarm
    onsets = [('1', ev[0] + 5 * 3600), ('1', ev[0] + 9 * 3600), ('1', ev[1] + 3600), ('1', ev[2] + 7200)]
    r = replay(_issues(rows), onsets, 'model', 1, protocol)
    assert r['cutoffs'] == {'1': 0.1}
    s = summary(r['issues'], 4)
    assert (s['onsets_in_verified_windows'], s['matched_onsets'], s['recall']) == (3, 1, pytest.approx(1 / 3))
    assert (s['alerts'], s['verified_alerts'], s['unverified_alerts'], s['precision']) == (3, 2, 1, 0.5)
    assert s['median_lead_hours'] == 5 and s['abstained_issues'] == 1


def test_empty_denominators_are_undefined():
    s = summary([{'county': '1', 'issue_s': 0, 'eligible': True, 'alert': False, 'verified': True, 'onsets': 0,
                  'hit': False, 'lead_hours': None}], 1)
    assert s['recall'] is None and s['precision'] is None and s['median_lead_hours'] is None


def test_identical_methods_have_a_zero_difference():
    rng = np.random.default_rng(0)
    c = {'onsets': rng.integers(0, 3, 60).astype(float), 'verified_alerts': np.ones(60)}
    c['matched'] = np.minimum(c['onsets'], 1)
    r = ratio_difference(c, c, 'matched', 'onsets', 7, 100, 0)
    assert r['estimate'] == 0 and r['ci95'] == [0, 0]


# ------------------------------------------------------------------------------------------------ on the fixture
def _issue_view(input_dir):
    con = build(input_dir, predictions=EXPECTED)
    t = event_tables(con)
    con.close()
    return t['issues']


def test_issue_view_on_the_fixture():
    v = _issue_view(FIXTURE)
    requests = pl.read_parquet(FIXTURE / 'requests' / 'forecast_requests.parquet').filter(pl.col('task_id') == 'A')
    assert len(v['county']) == requests.select('county_fips', 'issue_time').unique().height
    assert all(s is not None for s in v['model_score'])
    assert any(v['gate_ok']) and all(isinstance(g, bool) for g in v['gate_ok'])


def test_later_truth_cannot_change_an_issued_alert(fixture_copy):
    before = _issue_view(FIXTURE)
    first = min(before['issue_s'])
    path = fixture_copy / 'eaglei' / 'eaglei_outages.parquet'
    t = pl.read_parquet(path)
    later = pl.col('timestamp') > datetime.fromtimestamp(first, timezone.utc)
    pl.concat([t.filter(~later), t.filter(later).with_columns(pl.col('customers_out') * 50)]).write_parquet(path)
    after = _issue_view(fixture_copy)
    rows = lambda v: [tuple(v[k][i] for k in ('county', 'gate_ok', 'latest_x', 'model_score', 'weather_score'))  # noqa: E731
                      for i in range(len(v['county'])) if v['issue_s'][i] == first]
    assert rows(after) == rows(before)                                   # issued at T: unchanged
    second = lambda v: [v['latest_x'][i] for i in range(len(v['county'])) if v['issue_s'][i] > first]  # noqa: E731
    assert second(after) != second(before)                               # positive control: the change is visible later
