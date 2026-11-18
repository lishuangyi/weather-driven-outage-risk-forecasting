"""Paired block bootstrap: resampling scheme and known answers."""
import numpy as np
import pytest

from conftest import EXPECTED, FIXTURE
from outage_forecast.evaluation import block_days, load_cohort, mae_difference, score_difference
from outage_forecast.metrics import task_score


def test_blocks_are_runs_of_consecutive_days():
    rng = np.random.default_rng(1)
    d = block_days(30, 7, rng)
    assert len(d) == 30 and d.min() >= 0 and d.max() < 30
    for k in range(0, 28, 7):                                    # each full block is a run of consecutive days
        assert np.all(np.diff(d[k:k + 7]) == 1)
    assert np.array_equal(block_days(30, 7, np.random.default_rng(1)), d)   # seeded


def test_block_longer_than_the_series_takes_the_whole_series():
    assert np.array_equal(block_days(5, 7, np.random.default_rng(0)), np.arange(5))


def test_constant_improvement_has_a_zero_width_interval():
    n = np.full(40, 10.0)
    base = np.random.default_rng(0).uniform(1, 3, 40)
    r = mae_difference(base - 0.5, base, n, 7, 200, 0)             # every day: 10 cases, 0.05 better each
    assert r['estimate'] == pytest.approx(-0.05)
    assert r['ci95'] == pytest.approx([-0.05, -0.05])
    assert r['excludes_zero']


def test_noise_without_a_real_difference_covers_zero():
    rng = np.random.default_rng(3)
    n = np.full(120, 20.0)
    a, b = rng.normal(5, 1, 120), rng.normal(5, 1, 120)
    r = mae_difference(a, b, n, 7, 500, 0)
    assert r['ci95'][0] < 0 < r['ci95'][1] and not r['excludes_zero']


def test_score_difference_point_estimate_matches_the_full_sample():
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 1, 600)
    model, base = np.clip(true + rng.normal(0, 0.1, 600), 0, 1), rng.uniform(0, 1, 600)
    days = [np.arange(i, i + 20) for i in range(0, 600, 20)]
    r = score_difference(days, true, model, base, 3, 50, 0)
    assert r['estimate'] == pytest.approx(task_score(true, model)['score'] - task_score(true, base)['score'])
    assert r['excludes_zero'] and r['ci95'][0] > 0                # a clearly better model


def test_cohort_is_paired_and_grouped_by_issue_day(tmp_path):
    import polars as pl

    path = tmp_path / 'p.parquet'
    pl.read_parquet(EXPECTED).write_parquet(path)
    c = load_cohort(FIXTURE, path, 'A')
    assert c['n_days'] == 2 and len(c['true']) == len(c['model']) == len(c['base']) == sum(len(i) for i in c['day_cases'])
