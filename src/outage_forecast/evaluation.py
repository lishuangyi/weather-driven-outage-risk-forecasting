"""Paired moving-block bootstrap of the difference between a model and persistence on one scoring cohort.

    python -m outage_forecast.evaluation --input-dir data/inputs/input_2025_full \
        --predictions out/frozen_2025_full.csv --name "frozen ensemble" --out artifacts/evaluation/frozen_2025.json

Pre-specified analysis (written down before the results were seen, docs/RESULTS.md):
  primary    Task A, risk MAE of the model minus risk MAE of persistence, on the cohort of sql/scoring
             (negative = the model is better); moving blocks of 7 consecutive UTC issue days, 2,000 resamples,
             seed 0, 95 % percentile interval
  sensitivity  the same with 3- and 14-day blocks
  secondary  difference of the full challenge score (model minus persistence) per task, recomputed from scratch in
             every resample (AUPRC does not decompose into per-case losses); 7-day blocks, 1,000 resamples

Resampling unit: an issue day with all its counties, architectures and leads, so that cases that share weather and
outage events stay together; blocks of consecutive days keep the dependence between neighbouring days. Model and
persistence are paired: every resample uses the same cases for both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import task_score
from .warehouse import build

PRIMARY = {'task': 'A', 'statistic': 'risk MAE difference', 'block_days': 7, 'reps': 2000, 'seed': 0}


def block_days(n_days: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Day indices of one moving-block resample: blocks of `block` consecutive days with uniform random starts,
    concatenated and cut to n_days (the series is not wrapped around)."""
    block = min(block, n_days)
    starts = rng.integers(0, n_days - block + 1, size=-(-n_days // block))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_days]


def mae_difference(day_sum_model: np.ndarray, day_sum_base: np.ndarray, day_n: np.ndarray, block: int, reps: int,
                   seed: int) -> dict:
    """Bootstrap of (sum |e_model| - sum |e_base|) / n over resampled days. Inputs are per-day sums and counts."""
    rng = np.random.default_rng(seed)
    point = (day_sum_model.sum() - day_sum_base.sum()) / day_n.sum()
    draws = np.empty(reps)
    for r in range(reps):
        d = block_days(len(day_n), block, rng)
        draws[r] = (day_sum_model[d].sum() - day_sum_base[d].sum()) / day_n[d].sum()
    return summarise(point, draws)


def score_difference(day_cases: list[np.ndarray], true_risk: np.ndarray, pred_model: np.ndarray, pred_base: np.ndarray,
                     block: int, reps: int, seed: int) -> dict:
    """Bootstrap of the challenge-score difference, both scores recomputed on every resample."""
    rng = np.random.default_rng(seed)
    point = task_score(true_risk, pred_model)['score'] - task_score(true_risk, pred_base)['score']
    draws = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([day_cases[d] for d in block_days(len(day_cases), block, rng)])
        draws[r] = task_score(true_risk[idx], pred_model[idx])['score'] - task_score(true_risk[idx], pred_base[idx])['score']
    return summarise(point, draws)


def summarise(point: float, draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'estimate': float(point), 'ci95': [float(lo), float(hi)], 'bootstrap_sd': float(draws.std(ddof=1)),
            'excludes_zero': bool(lo > 0 or hi < 0), 'reps': len(draws)}


def load_cohort(input_dir: Path, predictions: Path, task: str, scope: tuple[str, str] | None = None) -> dict:
    """Paired cases of one task, sorted by issue time: true risk and both predicted risks, grouped by issue day."""
    con = build(input_dir, predictions=predictions, scope=scope)
    rows = con.execute(f"""
        SELECT m.issue_s, m.true_risk, m.predicted_risk AS model, p.predicted_risk AS base
        FROM gold.scored_cases m
        JOIN gold.scored_cases p USING (task, arch, county, issue_s, target_s)
        WHERE m.model = 'model' AND p.model = 'persistence' AND m.task = '{task}'
        ORDER BY m.issue_s, m.county, m.arch, m.target_s""").fetchnumpy()
    con.close()
    day = (np.asarray(rows['issue_s']) // 86400).astype(np.int64)
    days, start = np.unique(day, return_index=True)
    bounds = np.r_[start, len(day)]
    return {'true': np.asarray(rows['true_risk']), 'model': np.asarray(rows['model']), 'base': np.asarray(rows['base']),
            'day_cases': [np.arange(bounds[i], bounds[i + 1]) for i in range(len(days))], 'n_days': len(days)}


def analyse(input_dir: Path, predictions: Path, name: str, scope: tuple[str, str] | None = None,
            score_reps: int = 1000) -> dict:
    out = {'model': name, 'predictions': Path(predictions).name, 'scope': scope, 'primary': PRIMARY, 'tasks': {}}
    for task in ('A', 'B'):
        c = load_cohort(input_dir, predictions, task, scope)
        err_m, err_b = np.abs(c['model'] - c['true']), np.abs(c['base'] - c['true'])
        sum_m = np.array([err_m[i].sum() for i in c['day_cases']])
        sum_b = np.array([err_b[i].sum() for i in c['day_cases']])
        n = np.array([len(i) for i in c['day_cases']], dtype=float)
        entry = {'issue_days': c['n_days'], 'cases': int(n.sum()),
                 'mae_model': float(err_m.mean()), 'mae_persistence': float(err_b.mean()),
                 'mae_difference': {f'{b}-day blocks': mae_difference(sum_m, sum_b, n, b, PRIMARY['reps'], PRIMARY['seed'])
                                    for b in (7, 3, 14)},
                 'score_difference': {'7-day blocks': score_difference(c['day_cases'], c['true'], c['model'], c['base'],
                                                                       7, score_reps, PRIMARY['seed'])}}
        out['tasks'][task] = entry
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--name', default='model')
    p.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--score-reps', type=int, default=1000)
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    scope = (a.start or '1900-01-01', a.end or '2200-01-01') if (a.start or a.end) else None
    result = analyse(a.input_dir, a.predictions, a.name, scope, a.score_reps)
    text = json.dumps(result, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
