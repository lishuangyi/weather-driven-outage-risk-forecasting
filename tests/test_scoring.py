"""The evaluation metric: average precision with tied scores, and the pooled task score."""
import numpy as np
import pytest

from scoring import average_precision, task_score


def test_average_precision_matches_scikit_learn():
    sklearn = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.random(500) < 0.1
        s = np.round(rng.random(500), 2)        # many ties
        assert average_precision(y, s) == pytest.approx(sklearn.average_precision_score(y, s), abs=1e-12)


def test_perfect_forecast_scores_100():
    risk = np.array([0.01, 0.2, 0.7, 0.99])
    assert task_score(risk, risk)['score'] == pytest.approx(100.0)


def test_constant_forecast_gets_no_ranking_credit():
    risk = np.array([0.01, 0.2, 0.7, 0.99])
    result = task_score(risk, np.full(4, 0.5))
    assert result['auprc_norm'] == pytest.approx(0.0)
