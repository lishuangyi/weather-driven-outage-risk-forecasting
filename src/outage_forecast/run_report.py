"""Versioned run report: which code, artifacts, inputs and environment produced the published scores.

    python -m outage_forecast.run_report run                # score configs/benchmark.json -> artifacts/runs/benchmark.json
    python -m outage_forecast.run_report render             # write the generated tables in README.md and docs/
    python -m outage_forecast.run_report render --check     # exit 1 when a table differs from the report (CI)

`run` needs the inputs and prediction files named in the spec. It builds the DuckDB warehouse once per input, scores
every evaluation (predictions file x window of UTC issue times) with sql/scoring, and records for each one the case
counts, coverage and pooled metrics of the model and of persistence on the same cohort. It also records the git
commit (and any uncommitted changes), the SHA-256 of every input file, prediction file and model file, the recipes
and seeds, and the machine. `render` needs only the committed report: the benchmark tables in the documentation are
generated from it, between `<!-- run-report:NAME -->` and `<!-- /run-report:NAME -->` markers.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .config import PACKAGE_ROOT, Config
from .warehouse import build, intervals, run_scoring

SPEC = PACKAGE_ROOT / 'configs' / 'benchmark.json'
REPORT = PACKAGE_ROOT / 'artifacts' / 'runs' / 'benchmark.json'
DOCS = ('README.md', 'docs/RESULTS.md', 'docs/TRAINING.md', 'docs/MODEL_CARD.md')
PACKAGES = ('duckdb', 'polars', 'numpy', 'xgboost', 'torch', 'scikit-learn')
COUNTIES = {'26163': 'Wayne, MI', '48201': 'Harris, TX', '33015': 'Rockingham, NH', '06037': 'Los Angeles, CA',
            '12086': 'Miami-Dade, FL'}
MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December')


# ------------------------------------------------------------------------------------------------ provenance
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str:
    done = subprocess.run(['git', '-C', str(PACKAGE_ROOT), *args], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ''


def code_info() -> dict:
    """The commit the run used, and the tracked files that differed from it (empty for a clean checkout)."""
    changed = [line[3:] for line in _git('status', '--porcelain', '--untracked-files=no').splitlines()]
    return {'commit': _git('rev-parse', 'HEAD') or None, 'describe': _git('describe', '--always', '--dirty') or None,
            'uncommitted_changes': changed}


def environment() -> dict:
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor()
    memory = None
    if Path('/proc/meminfo').exists():
        kb = int(Path('/proc/meminfo').read_text().split('\n')[0].split()[1])
        memory = round(kb / 1024 ** 2, 1)
    gpu = None
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        gpu = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'cpu': cpu,
            'logical_cpus': os.cpu_count(), 'memory_gb': memory, 'gpu': gpu, 'packages': versions,
            'note': 'scoring runs on the CPU; GPU only matters for `make train`'}


def model_artifacts() -> dict:
    config_file = PACKAGE_ROOT / 'configs' / 'inference.json'
    frozen = {'config': 'configs/inference.json', 'config_sha256': sha256(config_file),
              'model_files': len(Config.load(verify_models=False).artifacts())}
    try:
        Config.load(verify_models=True)
        frozen['model_files_verified'] = True
    except RuntimeError as e:          # models not downloaded: the scores are still valid, the check is recorded
        frozen['model_files_verified'] = str(e)
    reference_dir = PACKAGE_ROOT / 'models' / 'reference'
    reference = {'files': {p.name: sha256(p) for p in sorted(reference_dir.glob('*')) if p.is_file()}}
    committed = PACKAGE_ROOT / 'artifacts' / 'reference' / 'reference_v2.json'
    if (reference_dir / 'reference.json').is_file():
        reference['training_report_matches'] = f'{committed.relative_to(PACKAGE_ROOT)}: ' + \
            ('yes' if sha256(reference_dir / 'reference.json') == sha256(committed) else 'NO')
    manifest = PACKAGE_ROOT / 'artifacts' / 'release_v1.0.0.json'
    return {'release_manifest': {'path': 'artifacts/release_v1.0.0.json', 'sha256': sha256(manifest)},
            'frozen_ensemble': frozen, 'reference': reference}


def recipes() -> dict:
    out = {'frozen_ensemble': {'inference': 'deterministic (no sampling); configs/inference.json fixes members, '
                                            'weights and fallbacks',
                               'training': 'experiment code not included; weights frozen in release v1.0.0'}}
    committed = PACKAGE_ROOT / 'artifacts' / 'reference' / 'reference_v2.json'
    if committed.is_file():
        r = json.loads(committed.read_text())
        out['reference'] = {'training_report': 'artifacts/reference/reference_v2.json', 'params': r['params'],
                            'seed': r['params'].get('seed'), 'variants': r['variants'], 'round_grid': r['round_grid'],
                            'choice': r['choice'], 'labelled_rows': r['rows'],
                            'protocol': {'selection': 'fit 2022, choose on 2023 (cut-off 2024-01-01)',
                                         'test': 'fit 2022-2023, score 2024', 'final': 'fit 2022-2024, score 2025'}}
    return out


def input_info(path: Path, con) -> dict:
    files = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*'))
             if p.is_file() and p.name != 'SHA256SUMS'}
    info = {'path': str(path.relative_to(PACKAGE_ROOT)) if path.is_relative_to(PACKAGE_ROOT) else path.name,
            'files': files}
    sums = path / 'SHA256SUMS'
    if sums.is_file():
        recorded = dict(reversed(line.split(maxsplit=1)) for line in sums.read_text().splitlines() if line.strip())
        info['matches_SHA256SUMS'] = all(recorded.get(k) == v for k, v in files.items()) and set(recorded) <= set(files)
    info['requests'] = {t: {'rows': n, 'first_issue': a, 'last_issue': b} for t, n, a, b in con.execute(
        "SELECT task, count(*), strftime(min(issue_time) AT TIME ZONE 'UTC', '%Y-%m-%d %H:%M UTC'), "
        "strftime(max(issue_time) AT TIME ZONE 'UTC', '%Y-%m-%d %H:%M UTC') FROM silver.requests "
        'GROUP BY task ORDER BY task').fetchall()}
    return info


# ------------------------------------------------------------------------------------------------ scoring
def _rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def evaluate(con, predictions: Path, window) -> dict:
    """Score one predictions file on one window, on the warehouse `con` (gold layer built)."""
    run_scoring(con, predictions, window)
    checks = {r['check_name']: {'status': r['status'], 'violations': r['violations']}
              for r in _rows(con, 'SELECT check_name, status, violations FROM gold.prediction_checks')}
    metrics = _rows(con, "SELECT model, task, cases, positives, positive_rate, auprc_norm, mae_risk, score "
                         "FROM gold.metrics WHERE dimension = 'pooled' ORDER BY model, task")
    tasks = {}
    for m in metrics:
        tasks.setdefault(m['task'], {})[m['model']] = {k: m[k] for k in ('cases', 'positives', 'positive_rate',
                                                                         'auprc_norm', 'mae_risk', 'score')}
    for task, models in tasks.items():
        assert models['model']['cases'] == models['persistence']['cases'], 'model and persistence on different cases'
    coverage = _rows(con, 'SELECT task, county, month, requested, scored, excluded_no_truth, excluded_invalid_truth '
                          'FROM gold.coverage ORDER BY task, county, month')
    for task in tasks:
        rows = [r for r in coverage if r['task'] == task]
        counties = {}
        for county in sorted({r['county'] for r in rows}):
            c = [r for r in rows if r['county'] == county]
            low = min(c, key=lambda r: (r['scored'] / r['requested'], r['month']))
            counties[county] = {'requested': sum(r['requested'] for r in c), 'scored': sum(r['scored'] for r in c),
                                'lowest_month': low['month'], 'lowest_month_share': low['scored'] / low['requested']}
        tasks[task]['coverage'] = {'requested': sum(r['requested'] for r in rows),
                                   'scored': sum(r['scored'] for r in rows),
                                   'excluded_no_truth': sum(r['excluded_no_truth'] for r in rows),
                                   'excluded_invalid_truth': sum(r['excluded_invalid_truth'] for r in rows),
                                   'counties': counties}
    issue_days = con.execute("SELECT count(DISTINCT floor(issue_s / 86400)) FROM gold.score_cohort").fetchall()[0][0]
    return {'prediction_checks': checks, 'issue_days_scored': issue_days, 'tasks': tasks}


def run(spec_path: Path = SPEC, out: Path = REPORT) -> dict:
    t0 = time.time()
    spec = json.loads(Path(spec_path).read_text())
    resolve = lambda p: (PACKAGE_ROOT / p).resolve()  # noqa: E731  (absolute paths stay absolute)
    report = {'schema': 1, 'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'spec': {'path': str(Path(spec_path).resolve().relative_to(PACKAGE_ROOT))
                       if Path(spec_path).resolve().is_relative_to(PACKAGE_ROOT) else Path(spec_path).name,
                       'sha256': sha256(spec_path)},
              'code': code_info(), 'environment': environment(), 'artifacts': model_artifacts(), 'recipes': recipes(),
              'seeds': {'reference_training': 42, 'bootstrap': 0, 'inference': 'not random'},
              'inputs': {}, 'predictions': {}, 'evaluations': {}}
    for key, p in spec['predictions'].items():
        path = resolve(p['path'])
        report['predictions'][key] = {'path': p['path'], 'model': p['model'], 'command': p['command'],
                                      'sha256': sha256(path), 'bytes': path.stat().st_size}
    for input_key in dict.fromkeys(e['input'] for e in spec['evaluations']):
        path = resolve(spec['inputs'][input_key])
        print(f'building the warehouse for {input_key}', flush=True)
        con = build(path)
        report['inputs'][input_key] = input_info(path, con)
        for e in (e for e in spec['evaluations'] if e['input'] == input_key):
            print(f"  scoring {e['id']}", flush=True)
            result = evaluate(con, resolve(spec['predictions'][e['predictions']]['path']), e['window'])
            report['evaluations'][e['id']] = {'label': e['label'], 'role': e['role'], 'input': input_key,
                                              'predictions': e['predictions'],
                                              'model': spec['predictions'][e['predictions']]['model'],
                                              'window': [list(i) for i in intervals(e['window'])] or None, **result}
        con.close()
    report['uncertainty'] = {p: {'sha256': sha256(resolve(p)),
                                 'predictions': json.loads(resolve(p).read_text())['predictions']}
                             for p in spec.get('uncertainty', [])}
    report['analyses'] = {name: {'path': p, 'sha256': sha256(resolve(p)),
                                 'predictions': json.loads(resolve(p).read_text())['predictions']}
                          for name, p in spec.get('analyses', {}).items()}
    report['seconds'] = round(time.time() - t0)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + '\n')
    return report


# ------------------------------------------------------------------------------------------------ tables
def _signed(v: float, digits: int) -> str:
    text = f'{v:+.{digits}f}'
    return text.replace('-', '−')


def _ci(r: dict, digits: int) -> str:
    lo, hi = r['ci95']
    return f"{_signed(r['estimate'], digits)} [{_signed(lo, digits)}, {_signed(hi, digits)}]"


def _pair(e: dict) -> str:
    """'model / persistence' for both tasks."""
    return ' | '.join(f"{e['tasks'][t]['model']['score']:.2f} / {e['tasks'][t]['persistence']['score']:.2f}"
                      for t in ('A', 'B'))


def _score(e: dict, task: str, model: str = 'model') -> str:
    return f"{e['tasks'][task][model]['score']:.2f}"


def _month(m: str) -> str:
    return MONTHS[int(m[5:7]) - 1]


def table_benchmarks(r: dict, ctx: dict) -> str:
    e = r['evaluations']
    rows = ['| Model | Data window | Role of the window in development | Task A model / persistence | Task B model / persistence |',
            '|---|---|---|---:|---:|',
            '| Frozen ensemble | 2022–2024 | training | — | — |']
    for key, model, mark in (('frozen_2025_jun_aug', 'Frozen ensemble', ''), ('frozen_2025_rest', 'Frozen ensemble', '¹'),
                             ('reference_2024', 'Reference trainer²', ''), ('reference_2025_year', 'Reference trainer²', '')):
        rows.append(f"| {model} | {e[key]['label']} | {e[key]['role'].replace('-', '–')}{mark} | {_pair(e[key])} |")
    return '\n'.join(rows)


def table_headline(r: dict, ctx: dict) -> str:
    e = r['evaluations']
    rows = ['| Window (UTC issue time) | Role of the window in development | Task A model / persistence | Task B model / persistence |',
            '|---|---|---:|---:|']
    for key in ('frozen_2025_jun_aug', 'frozen_2025_rest', 'frozen_2025_year', 'frozen_package'):
        a, b = (f"**{_score(e[key], t)}** / {_score(e[key], t, 'persistence')}" for t in ('A', 'B'))
        rows.append(f"| {e[key]['label']} | {e[key]['role'].replace('-', '–')} | {a} | {b} |")
    return '\n'.join(rows)


def table_reference(r: dict, ctx: dict) -> str:
    e = r['evaluations']
    both = lambda key, model: ' / '.join(_score(e[key], t, model) for t in ('A', 'B'))  # noqa: E731
    rows = ['| Evaluation (Task A / B) | Reference | Persistence | Frozen ensemble |', '|---|---:|---:|---:|',
            f"| 2024 chronological test | **{both('reference_2024', 'model')}** | {both('reference_2024', 'persistence')} "
            '| in its training data |']
    for ref, frozen, label in (('reference_2025_year', 'frozen_2025_year', '2025 full year (retrospective)'),
                               ('reference_2025_jun_aug', 'frozen_2025_jun_aug', '2025 June–August')):
        for t in ('A', 'B'):      # one cohort per window: persistence is the same for both models (up to summation order)
            assert e[ref]['tasks'][t]['persistence']['cases'] == e[frozen]['tasks'][t]['persistence']['cases']
            assert abs(e[ref]['tasks'][t]['persistence']['score'] - e[frozen]['tasks'][t]['persistence']['score']) < 1e-9
        rows.append(f"| {label} | {both(ref, 'model')} | {both(ref, 'persistence')} | **{both(frozen, 'model')}** |")
    return '\n'.join(rows)


def table_model_card(r: dict, ctx: dict) -> str:
    e = r['evaluations']
    rows = ['| | Task A | Task B |', '|---|---:|---:|']
    for key, label in (('frozen_2025_jun_aug', 'June–August 2025 (selection window)'),
                       ('frozen_2025_rest', 'other 9 months of 2025 (retrospective)')):
        a, b = (f"{_score(e[key], t)} (persistence {_score(e[key], t, 'persistence')})" for t in ('A', 'B'))
        rows.append(f'| {label} | {a} | {b} |')
    return '\n'.join(rows)


def table_coverage(r: dict, ctx: dict) -> str:
    year, summer = r['evaluations']['frozen_2025_year'], r['evaluations']['frozen_2025_jun_aug']
    rows = ['| 2025, full year | Task A scored / requested | Share | Lowest month | Task B scored / requested | Share | Lowest month |',
            '|---|---:|---:|---|---:|---:|---|']
    def cells(c: dict, lowest: bool = True) -> str:
        text = f"{c['scored']:,} / {c['requested']:,} | {c['scored'] / c['requested']:.1%}".replace('%', ' %')
        return text + (f" | {c['lowest_month_share']:.1%} ({_month(c['lowest_month'])})".replace('%', ' %') if lowest else ' | ')
    rows.append('| all counties | ' + ' | '.join(cells(year['tasks'][t]['coverage'], False) for t in ('A', 'B')) + ' |')
    for fips, name in COUNTIES.items():
        rows.append(f'| {name} | ' + ' | '.join(cells(year['tasks'][t]['coverage']['counties'][fips]) for t in ('A', 'B')) + ' |')
    invalid = sum(year['tasks'][t]['coverage']['excluded_invalid_truth'] for t in ('A', 'B'))
    share = {t: summer['tasks'][t]['coverage']['scored'] / summer['tasks'][t]['coverage']['requested'] for t in ('A', 'B')}
    rows += ['', f"{'No case is' if invalid == 0 else f'{invalid:,} cases are'} excluded for invalid truth. For June–August 2025, "
                 f"{share['A']:.1%} (A) and {share['B']:.1%} (B) of the requests are scored.".replace('%', ' %')]
    return '\n'.join(rows)


def _bootstrap_rows(ctx: dict) -> list[tuple[str, str, dict]]:
    out = []
    for path, data in ctx['bootstrap'].items():
        name = 'Frozen ensemble' if data['model'] == 'frozen ensemble' else 'Reference model'
        for task in ('A', 'B'):
            out.append((name, task, data['tasks'][task]))
    return out


def table_bootstrap(r: dict, ctx: dict) -> str:
    rows_in = _bootstrap_rows(ctx)
    days = {t['issue_days'] for _, _, t in rows_in}
    rows = [f"| 2025, {'/'.join(map(str, sorted(days)))} issue days | Cases | Risk MAE, model / persistence "
            '| MAE difference, 95 % CI | Score difference, 95 % CI |', '|---|---:|---|---|---|']
    same = True
    for name, task, t in rows_in:
        primary = name == 'Frozen ensemble' and task == 'A'
        label = f'**{name}, Task {task}** (primary)' if primary else f'{name}, Task {task}'
        mae = _ci(t['mae_difference']['7-day blocks'], 3)
        rows.append(f"| {label} | {t['cases']:,} | {t['mae_model']:.3f} / {t['mae_persistence']:.3f} | "
                    f"{f'**{mae}**' if primary else mae} | {_ci(t['score_difference']['7-day blocks'], 1)} |")
        verdict = lambda d: (d['excludes_zero'], d['estimate'] > 0)  # noqa: E731
        same &= all(verdict(t['mae_difference'][f'{b}-day blocks']) == verdict(t['mae_difference']['7-day blocks'])
                    for b in (3, 14))
    rows += ['', 'With 3- and 14-day blocks, every MAE interval reaches the same conclusion as with 7-day blocks.'
             if same else 'With 3- or 14-day blocks, at least one MAE interval reaches a different conclusion than '
                          'with 7-day blocks; see the JSON files.']
    return '\n'.join(rows)


def table_bootstrap_summary(r: dict, ctx: dict) -> str:
    rows = ['| | Score difference vs persistence, 95 % CI |', '|---|---|']
    rows += [f"| {name}, Task {task} | {_ci(t['score_difference']['7-day blocks'], 1)} |"
             for name, task, t in _bootstrap_rows(ctx)]
    return '\n'.join(rows)


def table_selection(r: dict, ctx: dict) -> str:
    s = ctx['reference']['selection_2023']
    grid = ctx['reference']['round_grid']
    choice = ctx['reference']['choice']
    rows = ['| Rounds | ' + ' | '.join(map(str, grid)) + ' |', '|---|' + '---:|' * len(grid)]
    for task in ('A', 'B'):
        for variant in ctx['reference']['variants']:
            cells = [f"**{s[task][variant][str(n)]:.2f}**" if choice[task] == {'variant': variant, 'rounds': n}
                     else f"{s[task][variant][str(n)]:.2f}" for n in grid]
            rows.append(f'| {task} {variant} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def _pct(v: float | None, digits: int = 1) -> str:
    return '–' if v is None else f'{100 * v:.{digits}f} %'


def table_calibration(r: dict, ctx: dict) -> str:
    c = ctx['calibration']
    rows = ['| Sep–Dec 2025 | Cases | Event rate, fit → evaluation | Brier, challenge risk / p_event '
            '| Brier difference, 95 % CI | Log loss, challenge risk / p_event | Log-loss difference, 95 % CI |',
            '|---|---:|---|---|---|---|---|']
    for task in ('A', 'B'):
        t = c['tasks'][task]
        rows.append(f"| Task {task} | {t['evaluate_cases']:,} | {_pct(t['fit_event_rate'])} → {_pct(t['evaluate_event_rate'])} "
                    f"| {t['brier']['challenge_risk']:.4f} / {t['brier']['p_event']:.4f} "
                    f"| {_ci(t['brier_difference']['7-day blocks'], 4)} "
                    f"| {t['log_loss']['challenge_risk']:.3f} / {t['log_loss']['p_event']:.3f} "
                    f"| {_ci(t['log_loss_difference']['7-day blocks'], 3)} |")
    same = all(t['brier_difference']['14-day blocks']['excludes_zero'] == t['brier_difference']['7-day blocks']['excludes_zero']
               for t in c['tasks'].values())
    rows += ['', 'Fitted maps: ' + '; '.join(
        f"Task {k}: p_event = sigmoid({t['params']['a']:.3f} + {t['params']['b']:.3f} · asinh(4000 (x̂ − x₀)))".replace('-', '−')
        for k, t in c['tasks'].items()) + '. With 14-day blocks the Brier intervals reach '
        + ('the same conclusions.' if same else 'a different conclusion for at least one task; see the JSON file.')]
    return '\n'.join(rows)


def table_reliability(r: dict, ctx: dict) -> str:
    c = ctx['calibration']
    rows = ['| Predicted | Task A challenge risk | Task A p_event | Task B challenge risk | Task B p_event |',
            '|---|---|---|---|---|']
    cell = lambda b: f"{b['mean_predicted']:.2f} → {b['event_rate']:.2f} ({b['cases']:,})" if b else '–'  # noqa: E731
    bins = {(task, kind): {round(b['bin'][0], 1): b for b in c['tasks'][task]['reliability'][kind]}
            for task in ('A', 'B') for kind in ('challenge_risk', 'p_event')}
    for i in range(10):
        lo = round(i / 10, 1)
        rows.append(f'| {lo:.1f}–{lo + 0.1:.1f} | ' + ' | '.join(
            cell(bins[(task, kind)].get(lo)) for task in ('A', 'B') for kind in ('challenge_risk', 'p_event')) + ' |')
    return '\n'.join(rows)


METHOD = {'model': 'model (max predicted share)', 'persistence': 'near-threshold persistence', 'weather': 'max HRRR gust'}


def table_alerts(r: dict, ctx: dict) -> str:
    a = ctx['alerts']
    rows = ['| Budget (per county, 30 days) | Score | Alerts issued (per county, 30 days) | Hits / verified alerts '
            '| Precision | Matched / onsets | Recall | Median lead |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for budget, e in a['budgets'].items():
        for m in a['protocol']['methods']:
            s = e['pooled'][m]
            lead = '–' if s['median_lead_hours'] is None else f"{s['median_lead_hours']:.0f} h"
            rows.append(f"| {budget} | {METHOD[m]} | {s['alerts']} ({s['alerts_per_county_per_30_days']:.1f}) "
                        f"| {s['hits']} / {s['verified_alerts']} | {_pct(s['precision'], 0)} "
                        f"| {s['matched_onsets']} / {s['onsets_in_verified_windows']} | {_pct(s['recall'], 0)} | {lead} |")
    rows += ['', '| Budget | Statistic | Model − persistence, 95 % CI | Model − gust, 95 % CI |', '|---|---|---|---|']
    for budget, e in a['budgets'].items():
        d = e['differences']
        for stat in ('recall', 'precision'):
            primary = budget == str(a['protocol']['primary']['budget']) and stat == 'recall'
            cells = [d[f'model - {base}'].get(f'{stat}, 7-day blocks') for base in ('persistence', 'weather')]
            text = ['–' if c is None else _ci({'estimate': 100 * c['estimate'], 'ci95': [100 * x for x in c['ci95']]}, 1)
                    + ' pt' for c in cells]
            label = f'**{stat}** (primary)' if primary else stat
            rows.append(f'| {budget} | {label} | ' + ' | '.join(f'**{t}**' if primary and i == 0 else t
                                                               for i, t in enumerate(text)) + ' |')
    large = a['budgets'][str(a['protocol']['primary']['budget'])]['large_outages']
    pre = large['preceded_by_an_alert']
    rows += ['', f"Large outages ({100 * large['threshold_x']:.1f} %): {large['onsets']} onsets in September–December; "
                 f"an alert at {a['protocol']['primary']['budget']} per 30 days came before "
                 f"{pre['model']} (model), {pre['persistence']} (persistence) and {pre['weather']} (gust) of them."]
    return '\n'.join(rows)


def table_alerts_by_county(r: dict, ctx: dict) -> str:
    a = ctx['alerts']
    budget = str(a['protocol']['primary']['budget'])
    e = a['budgets'][budget]
    rows = [f'| County (budget {budget}) | Onsets in verified windows | Matched: model / persistence / gust '
            '| Alerts: model / persistence / gust | Quiet issues (eligible) |', '|---|---:|---|---|---:|']
    for fips, name in COUNTIES.items():
        d = e['by_county'][fips]
        rows.append(f"| {name} | {d['model']['onsets_in_verified_windows']} | "
                    + ' / '.join(str(d[m]['matched_onsets']) for m in ('model', 'persistence', 'weather')) + ' | '
                    + ' / '.join(str(d[m]['alerts']) for m in ('model', 'persistence', 'weather'))
                    + f" | {d['model']['issues'] - d['model']['abstained_issues']} of {d['model']['issues']} |")
    return '\n'.join(rows)


TABLES = {'benchmarks': table_benchmarks, 'headline': table_headline, 'reference': table_reference,
          'model-card': table_model_card, 'coverage': table_coverage, 'bootstrap': table_bootstrap,
          'bootstrap-summary': table_bootstrap_summary, 'selection': table_selection,
          'calibration': table_calibration, 'reliability': table_reliability, 'alerts': table_alerts,
          'alerts-by-county': table_alerts_by_county}
BLOCK = re.compile(r'(<!-- run-report:(?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- /run-report:(?P=name) -->)', re.S)


def context(report: dict) -> dict:
    """Files the tables read besides the report; each must be the version the report recorded."""
    ctx = {'bootstrap': {}}
    for path, meta in report['uncertainty'].items():
        if sha256(PACKAGE_ROOT / path) != meta['sha256']:
            raise RuntimeError(f'{path} changed after the run report was made; run `run_report run` again')
        ctx['bootstrap'][path] = json.loads((PACKAGE_ROOT / path).read_text())
    for name, meta in report.get('analyses', {}).items():
        if sha256(PACKAGE_ROOT / meta['path']) != meta['sha256']:
            raise RuntimeError(f"{meta['path']} changed after the run report was made; run `run_report run` again")
        ctx[name] = json.loads((PACKAGE_ROOT / meta['path']).read_text())
    ctx['reference'] = json.loads((PACKAGE_ROOT / report['recipes']['reference']['training_report']).read_text())
    return ctx


def render(report_path: Path = REPORT, check: bool = False, docs=DOCS) -> list[str]:
    """Fill every generated table; return the files that changed (with check, nothing is written)."""
    report = json.loads(Path(report_path).read_text())
    ctx = context(report)
    changed = []
    for doc in docs:
        path = PACKAGE_ROOT / doc
        text = path.read_text(encoding='utf-8')
        def fill(m: re.Match) -> str:
            if m['name'] not in TABLES:
                raise KeyError(f"{doc}: unknown generated table '{m['name']}'")
            return m.group(1) + TABLES[m['name']](report, ctx) + '\n' + m.group(4)
        new = BLOCK.sub(fill, text)
        if new != text:
            changed.append(doc)
            if check:
                sys.stdout.writelines(difflib.unified_diff(text.splitlines(True), new.splitlines(True), doc, doc + ' (report)'))
            else:
                path.write_text(new, encoding='utf-8')
    return changed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='score the evaluations of the spec and write the report')
    r.add_argument('--spec', type=Path, default=SPEC)
    r.add_argument('--out', type=Path, default=REPORT)
    d = sub.add_parser('render', help='write the generated tables of the documentation from the report')
    d.add_argument('--report', type=Path, default=REPORT)
    d.add_argument('--check', action='store_true', help='only compare; exit 1 when a table is out of date')
    a = p.parse_args(argv)
    if a.command == 'run':
        report = run(a.spec, a.out)
        print(f"wrote {a.out}: {len(report['evaluations'])} evaluations, commit {report['code']['describe']}, "
              f"{report['seconds']} s")
        return 0
    changed = render(a.report, a.check)
    if a.check:
        print('generated tables are up to date' if not changed else f"out of date: {', '.join(changed)}")
        return 1 if changed else 0
    print('updated: ' + (', '.join(changed) or 'nothing'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
