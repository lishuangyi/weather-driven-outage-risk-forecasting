"""Reference model: one XGBoost (Tweedie) model per task, trained from scratch on the SQL warehouse examples.

    python -m outage_forecast.reference train            # 3 stages below, writes models/reference/ and the report
    python -m outage_forecast.reference predict --input-dir data/inputs/input_2025 --output out/reference.csv

This is the clean, reproducible counterpart of the frozen ensemble (whose weights come from the original experiment
code): one documented recipe, one feature table (gold.examples, built in SQL), fixed hyper-parameters and a
chronological protocol in which every number is computed on data the model has not been fitted or tuned on.

  1. selection   fit 2022, score 2023 with the challenge metric -> per task: variant and number of boosting rounds
                 variants: 'plain' Tweedie regression of the outage share, or 'anchored' = the same with the
                 persistence forecast as base margin (log link), so the trees learn a multiplicative change to it
  2. test        fit 2022-2023 with those rounds, score 2024 -> test score (2024 untouched until here)
  3. final       fit 2022-2024 with those rounds, score 2025 -> the saved model; 2025 was never used by this recipe
Purging: training examples whose target lies in or after the evaluation period are dropped.
Scores use the warehouse metric (sql/scoring), next to the persistence baseline and, when its predictions file is
given, the frozen ensemble.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from .config import PACKAGE_ROOT, Config
from .metrics import task_score
from .warehouse import build, run_layer, source

KEYS = ['task', 'county', 'issue_s', 'target_s']
LABEL = 'truth_x'
PARAMS = {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.3, 'eval_metric': 'tweedie-nloglik@1.3',
          'eta': 0.05, 'max_depth': 7, 'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'lambda': 1.0, 'tree_method': 'hist', 'seed': 42}
MAX_ROUNDS = 1200
ROUND_GRID = (25, 50, 100, 200, 400, 800, 1200)
VARIANTS = ('plain', 'anchored')
INPUTS = PACKAGE_ROOT / 'data' / 'inputs'
YEARS = {2022: INPUTS / 'input_2022_full', 2023: INPUTS / 'input_2023_full', 2024: INPUTS / 'input_2024_full',
         2025: INPUTS / 'input_2025_full'}
MODEL_DIR = PACKAGE_ROOT / 'models' / 'reference'


def year_start(year: int) -> float:
    return datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()


def fetch(con, query: str) -> pl.DataFrame:
    """Query result as polars (numpy transfer, no pyarrow)."""
    return pl.DataFrame(con.execute(query).fetchnumpy())


def examples(input_dir: Path) -> pl.DataFrame:
    con = build(input_dir)
    frame = fetch(con, 'SELECT * FROM gold.examples')
    con.close()
    return frame


def feature_names(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in KEYS + [LABEL]]


def matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    x = frame.select(pl.col(features).cast(pl.Float32)).to_numpy()
    x[~np.isfinite(x)] = np.nan
    return x


def labelled(frame: pl.DataFrame, before: float | None = None) -> pl.DataFrame:
    """Rows with a valid label; with `before`, only targets before that time (purging at the period boundary)."""
    out = frame.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite() & pl.col(LABEL).is_between(0, 1))
    return out.filter(pl.col('target_s') < before) if before is not None else out


def anchor(frame: pl.DataFrame) -> np.ndarray:
    """Base margin of the 'anchored' variant: log of the persistence forecast (floored at 1e-6)."""
    return np.log(np.maximum(np.nan_to_num(frame['hist_latest_x'].to_numpy(), nan=0.0), 1e-6))


def dmatrix(frame: pl.DataFrame, features: list[str], variant: str, label: bool = True) -> xgb.DMatrix:
    d = xgb.DMatrix(matrix(frame, features), label=frame[LABEL].to_numpy() if label else None, feature_names=features)
    if variant == 'anchored':
        d.set_base_margin(anchor(frame))
    return d


def fit(train: pl.DataFrame, features: list[str], device: str, rounds: int, variant: str = 'plain') -> xgb.Booster:
    return xgb.train(PARAMS | {'device': device}, dmatrix(train, features, variant), num_boost_round=rounds)


def official_score(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Challenge score of example-level forecasts, expanded to the four architectures."""
    risk = Config.load()['risk']
    x0 = np.array([float(v) for v in risk['x0'].values()])
    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(risk['k'] * (x[:, None] - x0), -700, 700)))  # noqa: E731
    return task_score(sigmoid(x_true).ravel(), sigmoid(x_pred).ravel())['score']


def predict(models: dict[str, tuple[xgb.Booster, str]], frame: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Example-level predicted_x (clipped to [0, 1]); models: task -> (booster, variant)."""
    parts = []
    for task, (model, variant) in models.items():
        part = frame.filter(pl.col('task') == task)
        if part.height:
            x = np.clip(model.predict(dmatrix(part, features, variant, label=False)), 0, 1)
            parts.append(part.select(KEYS).with_columns(pl.Series('predicted_x', x.astype(np.float64))))
    return pl.concat(parts)


def write_predictions(con, example_predictions: pl.DataFrame, output: Path) -> None:
    """Expand example-level predictions to every request (all architectures) in the standard 7-column format."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix('.examples.csv')
    example_predictions.write_csv(staged)
    con.execute(f"""
        COPY (
            SELECT r.task AS task_id, r.arch AS architecture_id, r.county AS county_fips,
                   strftime(r.issue_time, '%Y-%m-%dT%H:%M:%SZ') AS issue_time,
                   strftime(r.target_time, '%Y-%m-%dT%H:%M:%SZ') AS target_time,
                   p.predicted_x,
                   1 / (1 + exp(-greatest(least(a.k * (p.predicted_x - a.x0), 700), -700))) AS predicted_risk
            FROM silver.requests r
            JOIN read_csv({_q(staged)}, types = {{'county': 'VARCHAR'}}) p
              ON p.task = r.task AND p.county = r.county AND p.issue_s = r.issue_s AND p.target_s = r.target_s
            JOIN ref.architectures a ON a.arch = r.arch
            ORDER BY r.row_id
        ) TO {_q(output)} (HEADER, DELIMITER ',')""")
    staged.unlink()


def _q(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def score(input_dir: Path, predictions: dict[str, Path], window: tuple[str, str] | None = None) -> dict:
    """Pooled scores of each named predictions file (and of persistence) with the warehouse metric."""
    out = {}
    for name, path in predictions.items():
        con = build(input_dir)
        expr = source(Path(path).resolve())
        if window:
            expr = (f"(SELECT * FROM {expr} WHERE CAST(issue_time AS TIMESTAMPTZ) >= '{window[0]}' "
                    f"AND CAST(issue_time AS TIMESTAMPTZ) < '{window[1]}')")
        run_layer(con, 'scoring', {'predictions': expr})
        rows = con.execute('SELECT model, task_a, task_b FROM gold.scoreboard').fetchall()
        con.close()
        for model, a, b in rows:
            out[name if model == 'model' else 'persistence'] = {'task_a': round(a, 2), 'task_b': round(b, 2)}
    return out


def train(device: str, frozen_2025: Path | None, workdir: Path) -> dict:
    t0 = time.time()
    data = {year: examples(path) for year, path in YEARS.items()}
    features = feature_names(data[2022])
    report = {'features': features, 'params': PARAMS, 'round_grid': ROUND_GRID, 'variants': VARIANTS,
              'rows': {str(y): {t: int(labelled(f.filter(pl.col('task') == t)).height) for t in ('A', 'B')}
                       for y, f in data.items()}}
    choice, models, selection = {}, {}, {}

    # 1. selection: fit 2022, challenge score on 2023 for each variant and number of rounds
    for task in ('A', 'B'):
        train_rows = labelled(data[2022].filter(pl.col('task') == task), before=year_start(2023))
        valid_rows = labelled(data[2023].filter(pl.col('task') == task))
        truth = valid_rows[LABEL].to_numpy()
        table = {'persistence': round(official_score(truth, np.nan_to_num(valid_rows['hist_latest_x'].to_numpy())), 2)}
        for variant in VARIANTS:
            model = fit(train_rows, features, device, MAX_ROUNDS, variant)
            dvalid = dmatrix(valid_rows, features, variant)
            table[variant] = {n: round(official_score(truth, np.clip(model.predict(dvalid, iteration_range=(0, n)), 0, 1)), 2)
                              for n in ROUND_GRID}
        best = max(((v, n) for v in VARIANTS for n in ROUND_GRID), key=lambda vn: table[vn[0]][vn[1]])
        choice[task], selection[task] = {'variant': best[0], 'rounds': best[1]}, table
    report['selection_2023'] = selection
    report['choice'] = choice

    # 2. test: fit 2022-2023, score 2024
    for task in ('A', 'B'):
        train_rows = pl.concat([labelled(data[y].filter(pl.col('task') == task), before=year_start(2024)) for y in (2022, 2023)])
        models[task] = (fit(train_rows, features, device, choice[task]['rounds'], choice[task]['variant']), choice[task]['variant'])
    path = workdir / 'reference_2024.csv'
    con = build(YEARS[2024])
    write_predictions(con, predict(models, data[2024], features), path)
    con.close()
    report['test_2024'] = score(YEARS[2024], {'reference': path})

    # 3. final: fit 2022-2024, score 2025 (full year and the June-August window)
    for task in ('A', 'B'):
        train_rows = pl.concat([labelled(data[y].filter(pl.col('task') == task), before=year_start(2025)) for y in (2022, 2023, 2024)])
        models[task] = (fit(train_rows, features, device, choice[task]['rounds'], choice[task]['variant']), choice[task]['variant'])
    path = workdir / 'reference_2025_full.csv'
    con = build(YEARS[2025])
    write_predictions(con, predict(models, data[2025], features), path)
    con.close()
    files = {'reference': path} | ({'frozen ensemble': frozen_2025} if frozen_2025 and frozen_2025.is_file() else {})
    report['eval_2025_full_year'] = score(YEARS[2025], files)
    report['eval_2025_jun_aug'] = score(YEARS[2025], files, ('2025-06-01', '2025-09-01'))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for task, (model, _variant) in models.items():
        model.save_model(MODEL_DIR / f'task_{task.lower()}.ubj')
        gain = model.get_score(importance_type='total_gain')
        report.setdefault('top_features', {})[task] = [k for k, _ in sorted(gain.items(), key=lambda kv: -kv[1])[:12]]
    report['seconds'] = round(time.time() - t0)
    (MODEL_DIR / 'reference.json').write_text(json.dumps(report, indent=2))
    return report


def load_models() -> tuple[dict[str, tuple[xgb.Booster, str]], list[str]]:
    meta = json.loads((MODEL_DIR / 'reference.json').read_text())
    models = {}
    for task in ('A', 'B'):
        m = xgb.Booster()
        m.load_model(MODEL_DIR / f'task_{task.lower()}.ubj')
        models[task] = (m, meta['choice'][task]['variant'])
    return models, meta['features']


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train', help='run the three stages and save models/reference/')
    t.add_argument('--device', default='cuda' if xgb.build_info().get('USE_CUDA') else 'cpu')
    t.add_argument('--frozen-2025', type=Path, default=PACKAGE_ROOT / 'out' / 'frozen_2025_full.csv',
                   help='predictions of the frozen ensemble on input_2025_full (compared when present)')
    t.add_argument('--workdir', type=Path, default=PACKAGE_ROOT / 'out' / 'reference')
    q = sub.add_parser('predict', help='predict an input directory with the saved reference model')
    q.add_argument('--input-dir', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == 'train':
        report = train(a.device, a.frozen_2025, a.workdir)
        print(json.dumps({k: report[k] for k in ('rows', 'selection_2023', 'choice', 'test_2024', 'eval_2025_full_year',
                                                  'eval_2025_jun_aug', 'top_features', 'seconds')}, indent=2))
        return 0
    models, features = load_models()
    con = build(a.input_dir)
    frame = fetch(con, 'SELECT * FROM gold.examples')
    write_predictions(con, predict(models, frame, features), a.output)
    con.close()
    print(f'Predictions written: {a.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
