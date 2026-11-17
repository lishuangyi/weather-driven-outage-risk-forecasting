"""End-to-end inference on the mini input: output contract and regression against the recorded predictions."""
import numpy as np
import polars as pl

from conftest import EXPECTED, FIXTURE, KEYS, requires_models

pytestmark = requires_models


def test_one_row_per_request_in_request_order(fixture_predictions):
    requests = pl.read_parquet(FIXTURE / 'requests' / 'forecast_requests.parquet')
    assert fixture_predictions.columns == KEYS + ['predicted_x', 'predicted_risk']
    assert fixture_predictions.height == requests.height
    assert fixture_predictions.select(KEYS).sort(KEYS).equals(requests.select(KEYS).sort(KEYS))


def test_values_are_probabilities_and_follow_the_risk_formula(fixture_predictions):
    from outage_forecast.config import Config

    risk = Config.load()['risk']
    x = fixture_predictions['predicted_x'].to_numpy()
    p = fixture_predictions['predicted_risk'].to_numpy()
    x0 = fixture_predictions['architecture_id'].replace_strict(risk['x0']).cast(pl.Float64).to_numpy()
    assert np.isfinite(x).all() and ((0 <= x) & (x <= 1)).all()
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-risk['k'] * (x - x0))), rtol=0, atol=1e-12)


def test_matches_recorded_predictions(fixture_predictions):
    """Regression guard: same predictions as the frozen release (tolerance covers CPU instruction-set rounding)."""
    expected = pl.read_parquet(EXPECTED)
    joined = expected.join(fixture_predictions, on=KEYS, how='left', suffix='_new')
    assert joined['predicted_x_new'].null_count() == 0
    np.testing.assert_allclose(joined['predicted_x_new'], joined['predicted_x'], rtol=0, atol=1e-6)
    np.testing.assert_allclose(joined['predicted_risk_new'], joined['predicted_risk'], rtol=0, atol=1e-4)
