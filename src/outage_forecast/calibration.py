"""Event probability: a calibrated P(outage share >= threshold) next to the challenge risk.

    python -m outage_forecast.calibration --input-dir data/inputs/input_2025_full \
        --predictions out/frozen_2025_full.csv --out artifacts/evaluation/calibration_2025_frozen.json

The challenge risk sigmoid(k (x_hat - x0)), k = 4000, is a steep soft threshold of a point forecast, not a
probability: in the top bin it promises 0.98 where 0.64 of the cases are events (RESULTS section 6). This module
fits a separate event probability p_event and leaves the challenge output and score unchanged.

Pre-specified protocol (PROTOCOL below; committed before the results were computed, docs/RESULTS.md section 11):

- Map, one per task, shared by leads and architectures: p_event = sigmoid(a + b * asinh(k (x_hat - x0))), b >= 0,
  fitted by maximum likelihood with a small ridge (1e-3 on a and b). asinh compresses the margin of large
  forecasts; the shared, non-negative slope keeps the order of the four architectures at a target.
- Fit: the scorable cases issued 2025-06-01 ... 08-31 whose target AND truth publication are before 2025-09-01.
  Evaluate: the cases issued 2025-09-01 ... 12-31 (chronological replay; June-August also selected the ensemble).
- Event: true risk >= 0.5 (truth_x >= x0), the event label of the challenge AUPRC.
- Primary statistic: Brier score of p_event minus Brier score of the challenge risk, per task; paired moving blocks
  of 7 UTC issue days, 2,000 resamples, seed 0, 95 % percentile interval; 14-day blocks as sensitivity. Secondary:
  log loss (both clipped to [1e-6, 1 - 1e-6]); reliability in ten fixed bins with counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .alerts import epoch
from .evaluation import mae_difference
from .warehouse import build

PROTOCOL = {'fit': ['2025-06-01', '2025-09-01'], 'evaluate': ['2025-09-01', '2026-01-01'], 'cutoff': '2025-09-01',
            'ridge': 1e-3, 'clip': 1e-6, 'bins': 10,
            'primary': {'statistic': 'Brier difference, p_event - challenge risk', 'block_days': 7, 'reps': 2000,
                        'seed': 0}}


def margin(x_hat: np.ndarray, x0: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.arcsinh(k * (x_hat - x0))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(0.5 * z))


def fit(z: np.ndarray, y: np.ndarray, ridge: float = PROTOCOL['ridge']) -> dict:
    """Logistic regression y ~ a + b z by Newton's method, with b >= 0 (b = 0 gives the base rate)."""
    def newton(design):
        w = np.zeros(design.shape[1])
        for _ in range(100):
            p = _sigmoid(design @ w)
            grad = design.T @ (p - y) + ridge * w
            hess = (design * (p * (1 - p))[:, None]).T @ design + ridge * np.eye(len(w))
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-10:
                break
        return w
    a, b = newton(np.column_stack([np.ones_like(z), z]))
    if b < 0:
        (a,), b = newton(np.ones((len(z), 1))), 0.0
    return {'a': float(a), 'b': float(b)}


def predict(params: dict, z: np.ndarray) -> np.ndarray:
    return _sigmoid(params['a'] + params['b'] * z)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, clip: float = PROTOCOL['clip']) -> float:
    p = np.clip(p, clip, 1 - clip)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = PROTOCOL['bins']) -> list[dict]:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for i in range(bins):
        m = idx == i
        if m.any():
            out.append({'bin': [i / bins, (i + 1) / bins], 'cases': int(m.sum()),
                        'mean_predicted': float(p[m].mean()), 'event_rate': float(y[m].mean())})
    return out


def load_cases(input_dir: Path, predictions: Path) -> dict:
    """Scorable cases (one cohort) with the predicted share and risk, the truth and its publication time."""
    con = build(input_dir, predictions=predictions)
    rows = con.execute("""
        SELECT c.task, c.issue_s, c.target_s, c.truth_published_s, c.x0, c.k, c.true_risk,
               p.predicted_x, p.predicted_risk
        FROM gold.score_cohort c JOIN silver.predictions p USING (task, arch, county, issue_s, target_s)
        ORDER BY c.task, c.issue_s, c.county, c.arch, c.target_s""").fetchall()
    con.close()
    names = ('task', 'issue_s', 'target_s', 'published_s', 'x0', 'k', 'true_risk', 'predicted_x', 'predicted_risk')
    cols = list(zip(*rows)) if rows else [[] for _ in names]
    out = {n: np.asarray(c, dtype=object if n == 'task' else float) for n, c in zip(names, cols)}
    out['event'] = (out['true_risk'] >= 0.5).astype(float)
    return out


def fit_mask(cases: dict, protocol: dict = PROTOCOL) -> np.ndarray:
    """Fit cases: issued in the fit window, with target and truth publication before the cut-off."""
    f0, f1 = (epoch(d) for d in protocol['fit'])
    cut = epoch(protocol['cutoff'])
    return ((cases['issue_s'] >= f0) & (cases['issue_s'] < f1) & (cases['target_s'] < cut)
            & (cases['published_s'] < cut))


def eval_mask(cases: dict, protocol: dict = PROTOCOL) -> np.ndarray:
    e0, e1 = (epoch(d) for d in protocol['evaluate'])
    return (cases['issue_s'] >= e0) & (cases['issue_s'] < e1)


def _day_sums(issue_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    days, inverse = np.unique((issue_s // 86400).astype(np.int64), return_inverse=True)
    return np.bincount(inverse, weights=values), np.bincount(inverse).astype(float)


def analyse(input_dir: Path, predictions: Path, name: str = 'model', protocol: dict = PROTOCOL) -> dict:
    cases = load_cases(input_dir, predictions)
    p = protocol['primary']
    out = {'model': name, 'predictions': Path(predictions).name, 'protocol': protocol, 'tasks': {}}
    for task in ('A', 'B'):
        t = cases['task'] == task
        f, e = t & fit_mask(cases, protocol), t & eval_mask(cases, protocol)
        z = margin(cases['predicted_x'], cases['x0'], cases['k'])
        params = fit(z[f], cases['event'][f], protocol['ridge'])
        y, raw, cal = cases['event'][e], cases['predicted_risk'][e], predict(params, z[e])
        sq_cal, n = _day_sums(cases['issue_s'][e], (cal - y) ** 2)
        sq_raw, _ = _day_sums(cases['issue_s'][e], (raw - y) ** 2)
        clip = protocol['clip']
        ll = lambda q: -(y * np.log(np.clip(q, clip, 1 - clip)) + (1 - y) * np.log(1 - np.clip(q, clip, 1 - clip)))  # noqa: E731
        ll_cal, _ = _day_sums(cases['issue_s'][e], ll(cal))
        ll_raw, _ = _day_sums(cases['issue_s'][e], ll(raw))
        out['tasks'][task] = {
            'params': params, 'fit_cases': int(f.sum()), 'fit_event_rate': float(cases['event'][f].mean()),
            'evaluate_cases': int(e.sum()), 'evaluate_issue_days': len(n), 'evaluate_event_rate': float(y.mean()),
            'brier': {'challenge_risk': brier(raw, y), 'p_event': brier(cal, y)},
            'log_loss': {'challenge_risk': log_loss(raw, y, clip), 'p_event': log_loss(cal, y, clip)},
            'brier_difference': {f'{b}-day blocks': mae_difference(sq_cal, sq_raw, n, b, p['reps'], p['seed'])
                                 for b in (7, 14)},
            'log_loss_difference': {'7-day blocks': mae_difference(ll_cal, ll_raw, n, 7, p['reps'], p['seed'])},
            'reliability': {'challenge_risk': reliability(raw, y, protocol['bins']),
                            'p_event': reliability(cal, y, protocol['bins'])}}
    return out


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--input-dir', type=Path, required=True)
    a.add_argument('--predictions', type=Path, required=True)
    a.add_argument('--name', default='frozen ensemble')
    a.add_argument('--out', type=Path)
    args = a.parse_args(argv)
    text = json.dumps(analyse(args.input_dir, args.predictions, args.name), indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
