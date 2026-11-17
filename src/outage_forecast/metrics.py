"""The challenge metric in numpy (the SQL version is sql/scoring/02_metrics.sql; tests check that they agree).

    score = 100 * (0.75 * AUPRC_norm + 0.25 * (1 - MAE)) on the architecture risk scale, label = true risk >= 0.5.
"""
from __future__ import annotations

import numpy as np


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision with tied scores as one threshold (as scikit-learn's average_precision_score)."""
    order = np.argsort(-score, kind='mergesort')
    s, t = score[order], y[order].astype(np.float64)
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]   # last index of each tie group
    tps = np.cumsum(t)[last]
    fps = (last + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def task_score(risk: np.ndarray, predicted: np.ndarray) -> dict:
    y = risk >= 0.5
    n, positives = len(y), int(y.sum())
    pi = positives / n
    ok = 0 < positives < n
    ap = average_precision(y, predicted) if ok else 0.0
    norm = float(np.clip((ap - pi) / (1 - pi), 0, 1)) if ok else 0.0
    mae = float(np.abs(predicted - risk).mean())
    return {'score': 100 * (0.75 * norm + 0.25 * (1 - mae)), 'auprc': ap, 'auprc_norm': norm, 'mae_risk': mae,
            'positive_rate': pi, 'cases': n, 'degenerate_labels': not ok}


def _risk(x: np.ndarray, x0: np.ndarray, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(k * (x - x0), -700, 700)))
