"""Local reference score of a finished predictions.csv against outage observations (demo tool; never used by infer.py).

Official formula (kickoff slide): per task, pooled over architectures, counties, issues and targets,
    score = 100 * (0.75 * AUPRC_norm + 0.25 * (1 - MAE)),
on architecture risk P = 1 / (1 + exp(-k (x - x0_a))): y = 1[P >= 0.5], pi = mean(y),
AUPRC_norm = clip((AP - pi) / (1 - pi), 0, 1) with AP the average precision of predicted_risk (tied predictions form one
threshold), MAE = mean |predicted_risk - P|. Label sets without positives or without negatives score AUPRC_norm = 0
(local convention).

Truth (challenge specification sections 6 and 8): for every (county, observation time) the record with the latest (source_record_time,
source row) counts; a blank value is unavailable, an absent time is unavailable (never zero), x = customers_out / MCC.
  Task A target T: mean of x at T-45, T-30, T-15 and T min; unavailable if any of the four is unavailable, invalid if
                   any of the four lies outside [0, 1].
  Task B target T: b_truth='auto' (default): the observation at T when the series has a record at T (native 5-minute
                   data; a blank record is unavailable), otherwise the 15-minute observation at floor(T / 15 min)
                   (15-minute data: last observation carried forward within its own interval, challenge specification section 6.3).
                   Decided per target, so mixed inputs (15-minute history, 5-minute recent data) are handled.
                   'native' always uses the observation at T, '15min' always floor(T / 15 min).
Cases without truth are excluded and counted. The technical score (0.4 A + 0.4 B + 0.1 G) is given only when both tasks
cover all four architectures. The predictions file is only read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

KEYS = ['task_id', 'architecture_id', 'county_fips', 'issue_time', 'target_time']


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


def truth_table(input_dir: Path, config, counties: set[str]) -> tuple[pl.DataFrame, dict, dict]:
    """(county, t, x) of the resolved observation times (x null for a blank record), MCC, and per county the number of
    observation times on the 15-minute grid and between grid times (information only)."""
    from outage_forecast.inputs import read_mcc, read_outages
    mcc = read_mcc(input_dir, config, counties)
    o = read_outages(input_dir)
    resolved = o.sort('county', 'timestamp_s', 'source_s', 'row').group_by('county', 'timestamp_s', maintain_order=True).last()
    sampling = {c: {'grid_15min_times': int(((g['timestamp_s'] % 900) == 0).sum()),
                    'times_between_grid': int(((g['timestamp_s'] % 900) != 0).sum())}
                for (c,), g in resolved.group_by('county')}
    x = resolved.select('county', pl.col('timestamp_s').alias('t'),
                        (pl.col('count') / pl.col('county').replace_strict(mcc, return_dtype=pl.Float64, default=None)).alias('x'))
    return x, mcc, sampling


def frames(predictions: Path, input_dir: Path, config, b_truth: str = 'auto') -> dict:
    """The predictions joined with their truth: {'A': frame, 'B': frame, 'sampling': ...}. Each frame keeps the request
    rows of that task with task, arch, county, issue and target (s), pred (risk), px (predicted_x) and truth (null =
    unavailable, NaN = invalid); Task B also has truth_at_t. Used by score() and by the summary charts."""
    p = pl.read_csv(predictions, infer_schema=False)
    if list(p.columns) != KEYS + ['predicted_x', 'predicted_risk']:
        raise ValueError(f'unexpected columns in {predictions}: {p.columns}')
    ts = lambda c: pl.col(c).str.to_datetime(time_zone='UTC', time_unit='us').dt.epoch('us') / 1e6  # noqa: E731
    # computation on normalized keys (as infer.py reads them); the file keeps the request keys as given
    p = p.with_columns(pl.col('task_id').str.strip_chars().alias('task'), pl.col('architecture_id').str.strip_chars().alias('arch'),
                       pl.col('county_fips').str.strip_chars().str.zfill(5).alias('county'),
                       pl.col('target_time').str.strip_chars().str.replace('Z$', '+00:00').alias('_target'),
                       pl.col('issue_time').str.strip_chars().str.replace('Z$', '+00:00').alias('_issue'),
                       pl.col('predicted_risk').cast(pl.Float64).alias('pred'),
                       pl.col('predicted_x').cast(pl.Float64).alias('px'))
    p = p.with_columns(ts('_target').alias('target'), ts('_issue').alias('issue'))
    truth, mcc, sampling = truth_table(input_dir, config, set(p['county'].unique().to_list()))
    truth = truth.with_columns(pl.lit(True).alias('present'))
    lookup = lambda frame, col, name: frame.join(truth.rename({'t': col, 'x': name, 'present': name + '_present'}),  # noqa: E731
                                                 on=['county', col], how='left')

    a = p.filter(pl.col('task') == 'A')
    for m in (45, 30, 15, 0):
        a = lookup(a.with_columns((pl.col('target') - 60 * m).alias(f't{m}')), f't{m}', f'x{m}')
    parts = [pl.col(f'x{m}') for m in (45, 30, 15, 0)]
    a = a.with_columns(pl.when(pl.any_horizontal(*[x.is_null() for x in parts])).then(None)                       # unavailable
                         .when(pl.all_horizontal(*[x.is_between(0, 1) for x in parts])).then(sum(parts) / 4)     # valid
                         .otherwise(float('nan')).alias('truth'))                                                # invalid
    b = p.filter(pl.col('task') == 'B').with_columns(((pl.col('target') // 900) * 900).alias('t15'))
    b = lookup(lookup(b, 'target', 'xt'), 't15', 'x15')
    at_t = pl.col('xt_present').fill_null(False) if b_truth == 'auto' else pl.lit(b_truth == 'native')
    b = b.with_columns(pl.when(at_t).then(pl.col('xt')).otherwise(pl.col('x15')).alias('truth'),
                       at_t.alias('truth_at_t'))
    return {'A': a, 'B': b, 'sampling': sampling}


def score(predictions: Path, input_dir: Path, config, b_truth: str = 'auto', joined: dict | None = None) -> dict:
    joined = joined or frames(predictions, input_dir, config, b_truth)
    a, b, sampling = joined['A'], joined['B'], joined['sampling']
    k, x0 = float(config['risk']['k']), {arc: float(v) for arc, v in config['risk']['x0'].items()}
    b_rule = {'auto': 'observation at T when recorded, else the 15-minute value at floor(T)', 'native': 'observation at T',
              '15min': '15-minute value at floor(T)'}[b_truth]
    result = {'truth_rule': {'task_A': 'mean of T-45/-30/-15/T', 'task_B': b_rule,
                             # targets between 15-minute grid times (on grid times both rules agree)
                             'task_B_targets_at_T': int((b['truth_at_t'] & (b['target'] % 900 != 0)).sum()) if b.height else 0,
                             'task_B_targets_floor15': int((~b['truth_at_t'] & (b['target'] % 900 != 0)).sum()) if b.height else 0,
                             'sampling': sampling}, 'tasks': {}}
    for task, frame in (('A', a), ('B', b)):
        if not frame.height:
            continue
        valid = frame.filter(pl.col('truth').is_not_null() & pl.col('truth').is_finite() & pl.col('truth').is_between(0, 1))
        entry = {'requests': frame.height, 'scored': valid.height, 'excluded_no_truth': int(frame['truth'].is_null().sum()),
                 'excluded_invalid_truth': frame.height - valid.height - int(frame['truth'].is_null().sum())}
        if valid.height:
            risk = _risk(valid['truth'].to_numpy(), valid['arch'].replace_strict(x0, return_dtype=pl.Float64).to_numpy(), k)
            pred = valid['pred'].to_numpy()
            entry['pooled'] = task_score(risk, pred)
            arcs = valid['arch'].to_numpy()
            entry['per_architecture'] = {arc: task_score(risk[arcs == arc], pred[arcs == arc]) for arc in sorted(set(arcs))}
            county = valid['county'].to_numpy()
            entry['per_county_diagnostic'] = {c: task_score(risk[county == c], pred[county == c]) for c in sorted(set(county))}
        result['tasks'][task] = entry
    t = result['tasks']
    arcs = sorted(x0)
    if all(task in t and 'pooled' in t[task] and sorted(t[task]['per_architecture']) == arcs for task in ('A', 'B')):
        s = np.array([(t['A']['per_architecture'][c]['score'] + t['B']['per_architecture'][c]['score']) / 200 for c in arcs])
        g = float(100 * np.clip(s.mean() - s.std(), 0, 1))
        total = 0.4 * t['A']['pooled']['score'] + 0.4 * t['B']['pooled']['score'] + 0.1 * g
        if np.isfinite(g) and np.isfinite(total):
            result['generalization'], result['technical_score'] = g, total
    if 'technical_score' not in result:
        result['technical_score_note'] = 'not computed: both tasks must be scored on all four architectures'
    return result
