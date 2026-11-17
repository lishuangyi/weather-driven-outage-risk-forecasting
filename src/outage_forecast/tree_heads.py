"""Combination of the tree heads of the Task B radar-free tree system (five families of XGBoost heads).

ladder design  up heads: P(rise > level k) for the up-ladder levels; down heads: P(fall) and log1p(1000 x fall size)
               prediction = persistence + E[rise] (trapezoid over the monotone exceedance curve, plus the tail mean
               beyond the top level) - P(fall) x expm1(severity) / 1000
level design   heads: P(x > level k) for the level ladder; prediction = sum_k P_k x (level_k - level_{k-1}) on the
               monotone exceedance curve
"""
import numpy as np


def ladder_rise(p: dict, tail_mean: float, up_ladder: list[float]) -> np.ndarray:
    probs = np.minimum.accumulate(np.stack([p[f'up_{k}'] for k in range(len(up_ladder))], 1), axis=1)
    return (probs[:, :-1] + probs[:, 1:]) / 2 @ np.diff(up_ladder) + probs[:, -1] * tail_mean


def combine(design: str, p: dict, persistence: np.ndarray, constants: dict, up_ladder: list[float],
            level_ladder: list[float]) -> np.ndarray:
    """Member prediction (outage proportion) from its raw head outputs p (head name -> array)."""
    if design == 'level':
        probs = np.minimum.accumulate(np.stack([p[f'level_{k}'] for k in range(len(level_ladder))], 1), axis=1)
        return probs @ np.diff([0.] + list(level_ladder))
    down = p['down_gate'] * np.expm1(np.clip(p['down_severity'], 0, 12)) / 1000
    return persistence + ladder_rise(p, constants['tail_mean'], up_ladder) - down
