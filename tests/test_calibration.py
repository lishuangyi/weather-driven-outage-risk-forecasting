"""Event probability: the fitted map, the fit cut-off, reliability bins, and a run on the fixture."""
import numpy as np
import pytest

from conftest import EXPECTED, FIXTURE
from outage_forecast.alerts import epoch
from outage_forecast.calibration import PROTOCOL, analyse, brier, fit, fit_mask, log_loss, margin, predict, reliability


def test_fit_recovers_a_known_logistic_map():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 200_000)
    y = (rng.uniform(size=z.size) < 1 / (1 + np.exp(-(-1.0 + 1.5 * z)))).astype(float)
    p = fit(z, y)
    assert p['a'] == pytest.approx(-1.0, abs=0.03) and p['b'] == pytest.approx(1.5, abs=0.03)


def test_slope_is_never_negative():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 20_000)
    y = (rng.uniform(size=z.size) < 1 / (1 + np.exp(2 * z))).astype(float)      # events where z is LOW
    p = fit(z, y)
    assert p['b'] == 0
    assert predict(p, z) == pytest.approx(np.full(z.size, y.mean()), abs=1e-6)  # the base rate


def test_map_keeps_the_order_of_shares_and_architectures():
    p = {'a': -2.0, 'b': 1.3}
    x_hat = np.linspace(0, 0.01, 50)
    for x0 in (0.001, 0.00117, 0.00143, 0.0016):
        assert np.all(np.diff(predict(p, margin(x_hat, x0, 4000))) >= 0)
    at = [predict(p, margin(np.array([0.0013]), x0, 4000))[0] for x0 in (0.001, 0.00117, 0.00143, 0.0016)]
    assert at == sorted(at, reverse=True)                                       # lower threshold, higher probability


def test_fit_cases_end_before_the_cutoff():
    cut = epoch(PROTOCOL['cutoff'])
    issue = np.array([cut - 86400] * 3 + [cut], dtype=float)
    cases = {'issue_s': issue, 'target_s': np.array([cut - 3600, cut + 3600, cut - 3600, cut + 3600], dtype=float),
             'published_s': np.array([cut - 3600, cut - 3600, cut + 60, cut + 3600], dtype=float)}
    assert fit_mask(cases).tolist() == [True, False, False, False]              # late target / late correction / later issue


def test_scores_and_reliability_bins():
    y = np.array([0, 0, 1, 1.0])
    p = np.array([0.0, 0.2, 0.8, 1.0])
    assert brier(p, y) == pytest.approx(0.02)
    assert log_loss(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(-np.log(1e-6))
    bins = reliability(p, y)
    assert sum(b['cases'] for b in bins) == 4 and bins[-1]['bin'] == [0.9, 1.0]   # 1.0 falls in the top bin


def test_analysis_runs_on_the_fixture():
    protocol = PROTOCOL | {'fit': ['2025-07-08', '2025-07-09'], 'evaluate': ['2025-07-09', '2025-07-10'],
                           'cutoff': '2025-07-09', 'primary': PROTOCOL['primary'] | {'reps': 50}}
    r = analyse(FIXTURE, EXPECTED, 'fixture', protocol)
    for task in ('A', 'B'):
        t = r['tasks'][task]
        assert t['fit_cases'] > 0 and t['evaluate_cases'] > 0 and t['evaluate_issue_days'] == 1
        assert 0 <= t['brier']['p_event'] <= 1 and t['params']['b'] >= 0
        assert sum(b['cases'] for b in t['reliability']['p_event']) == t['evaluate_cases']
