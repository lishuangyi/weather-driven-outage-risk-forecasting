"""Loading and evaluation of the submitted models (XGBoost boosters and the two GRU history networks)."""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import xgboost as xgb

from .config import Config
from .networks import HistoryGRUForecaster


def feature_matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Model input as float32, non-finite values as NaN (the boosters' missing value)."""
    x = np.array(frame.select(features).to_numpy(), dtype=np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    return x


class ModelStore:
    """Caches boosters and networks; device is 'cuda' or 'cpu'."""

    def __init__(self, config: Config, device: str):
        self.config, self.device, self.cache = config, device, {}

    def booster(self, path: str, rounds: int | None = None) -> xgb.Booster:
        key = (path, rounds)
        if key not in self.cache:
            m = xgb.Booster()
            m.load_model(self.config.path(path))
            m.set_param({'device': self.device, 'nthread': 4})
            if rounds and m.num_boosted_rounds() > rounds:
                m = m[:rounds]
            self.cache[key] = m
        return self.cache[key]

    def tree(self, path: str, x: np.ndarray, rounds: int | None = None) -> np.ndarray:
        """Raw booster output (float32) for the rows of x."""
        m = self.booster(path, rounds)
        if not len(x):
            return np.zeros(0, np.float32)
        if self.device == 'cuda':
            return m.predict(xgb.DMatrix(x, nthread=4))
        return m.inplace_predict(x)

    def network(self, name: str) -> tuple[HistoryGRUForecaster, dict]:
        """Task A ('task_a') or Task B ('task_b') GRU network and its fitted normalisers."""
        if name not in self.cache:
            if name == 'task_a':
                spec = self.config['task_a']['members']['gru']
                features = self.config.features(spec['features'])
                model = HistoryGRUForecaster(features, leads=spec['leads'])
                model.load_state_dict(torch.load(self.config.path(spec['weights']), map_location=self.device, weights_only=True))
                n = np.load(self.config.path(spec['normalizer']))
                norm = {k: n[k] for k in ('feature_mean', 'feature_std', 'state_mean', 'state_std')}
            else:
                spec = self.config['task_b']['members']['gru']
                ck = torch.load(self.config.path(spec['checkpoint']), map_location=self.device, weights_only=True)
                features = list(ck['features'])
                if features != self.config.features(spec['features']):
                    raise ValueError('Task B network feature schema differs from the configuration')
                model = HistoryGRUForecaster(features, leads=24)
                model.load_state_dict(ck['state_dict'])
                norm = {k: ck[k].cpu().numpy() for k in ('feature_mean', 'feature_std', 'state_mean', 'state_std')}
            model.to(self.device).eval()
            self.cache[name] = (model, {'features': features, **norm})
        return self.cache[name]
