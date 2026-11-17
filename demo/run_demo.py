#!/usr/bin/env python3
"""One-command demo run: environment check -> input overview -> official inference -> output check -> reference score
-> summary page.

    python demo/run_demo.py --data <input directory> [--out <new run directory>] [--device cpu|cuda|auto] [--no-score]
                            [--b-truth auto|native|15min]
                            [--counties 48201,06037] [--start 2025-07-08 [--end 2025-07-10]] [--tasks A|B|A,B]
                            [--generate-requests] [--a-leads 60:1440] [--b-leads 5:120] [--b-issue-hours 0,12]
                            [--drop radar] [--drop weather] [--config run.json]
    python demo/run_demo.py --check-only          (environment and model files only)

Selection (demo/select_requests.py): --counties, --start/--end (issue times in [start, end), UTC) and --tasks keep the
matching rows of the request table; --generate-requests (automatic when the input has no request table) builds the
requests on the official grid instead. The selected requests go to <run directory>/input/ with links to the other
tables; the input directory itself is never written to.

The predictions come only from the official entry point (infer.py, run as a separate process with the same arguments
the committee uses, plus a JSON copy of its run diagnostics). This tool never changes a prediction or an input file;
everything it writes goes to the run directory, which must lie outside the package and the input directory (default:
~/aidc_demo_runs/run_<UTC time>):
  predictions.csv   the official output          infer.log      the inference log
  diagnostics.json  paths and input coverage     summary.json   everything below, machine-readable
  summary.html      one-page summary with forecast-against-observation charts (opens offline)
Exit code: 0 success, 2 the input is missing or invalid (message from infer.py), 1 any other failure.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE / 'src'), str(PACKAGE), str(Path(__file__).resolve().parent)]

TABLES = {'requests': 'requests/forecast_requests', 'mcc': 'mcc/MCC', 'outages': 'eaglei/eaglei_outages',
          'weather': 'weather/weather_forecasts', 'radar': 'radar/mrms_county_series'}
COUNTY_NAMES = {'06037': 'Los Angeles, CA', '12086': 'Miami-Dade, FL', '26163': 'Wayne, MI', '33015': 'Rockingham, NH',
                '48201': 'Harris, TX'}
try:
    from charts import STYLE as CHART_STYLE
except Exception:  # noqa: BLE001  (charts are optional; the page is complete without them)
    CHART_STYLE = ''
PATH_NAMES = {'main': 'full model', 'main_no_radar': 'full model, radar-free tree', 'no_weather': 'weather-free tree',
              'no_history': 'history fallback', 'anchor': 'latest observation (+5/+10 min)'}


class Run:
    def __init__(self, out: Path):
        self.out = out
        self.status = 'running'
        self.checks: list[dict] = []
        self.warnings: list[str] = []
        self.info: dict = {}
        self.joined: dict | None = None     # predictions joined with the observations (demo/scoring.py)
        self.joined_error = 'not computed'
        self.charts: dict = {}              # chart series (demo/charts.py)

    def check(self, name: str, ok: bool | None, detail: str = '') -> bool:
        """ok True = pass, False = fail, None = warning."""
        mark = {True: 'OK  ', False: 'FAIL', None: 'WARN'}[ok]
        self.checks.append({'check': name, 'result': mark.strip(), 'detail': detail})
        print(f'  [{mark}] {name}' + (f': {detail}' if detail else ''), flush=True)
        if ok is None:
            self.warnings.append(f'{name}: {detail}')
        return ok is not False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(data: Path, table: str) -> Path | None:
    for suffix in ('.csv', '.parquet'):
        if (data / (TABLES[table] + suffix)).is_file():
            return data / (TABLES[table] + suffix)
    return None


# ----------------------------------------------------------------------------- 1. environment
def check_environment(run: Run, device: str) -> bool:
    print('1. Environment', flush=True)
    ok = run.check('Python', sys.version_info[:2] == (3, 12) or None, platform.python_version()
                   + ('' if sys.version_info[:2] == (3, 12) else ' (tested with 3.12)'))
    pins = {}
    for name in ('requirements/cpu.txt', 'requirements/cuda.txt'):
        for line in (PACKAGE / name).read_text().splitlines():
            if '==' in line and not line.startswith('#'):
                pkg, version = line.split('==')
                pins.setdefault(pkg.strip().lower(), set()).add(version.strip())
    versions = {}
    for module, pkg in (('numpy', 'numpy'), ('polars', 'polars'), ('torch', 'torch'), ('xgboost', 'xgboost')):
        try:
            versions[pkg] = __import__(module).__version__
        except Exception as error:  # noqa: BLE001
            return run.check(f'import {module}', False, f'{type(error).__name__}: {error} (install requirements/cpu.txt)')
        pinned = pins.get(pkg, set())
        base = versions[pkg].split('+')[0]
        ok &= run.check(f'{pkg} {versions[pkg]}', True if any(base == v.split('+')[0] for v in pinned) else None,
                        '' if any(base == v.split('+')[0] for v in pinned) else f'locked version {sorted(pinned)}')
    import torch
    cuda = torch.cuda.is_available()
    run.info['device'] = 'cuda' if device == 'cuda' or (device == 'auto' and cuda) else 'cpu'
    if device == 'cuda' and not cuda:
        return run.check('CUDA', False, 'requested with --device cuda but not available')
    run.check('device', True, run.info['device'] + (f' ({torch.cuda.get_device_name(0)})' if run.info['device'] == 'cuda' else ''))
    try:
        from outage_forecast.config import Config
        Config.load()
        ok &= run.check('model files', True, 'all files match the SHA-256 in configs/inference.json')
    except Exception as error:  # noqa: BLE001
        return run.check('model files', False, str(error))
    free = shutil.disk_usage(run.out).free / 1e9
    ok &= run.check('free disk space', True if free > 1 else None, f'{free:.1f} GB in the run directory')
    run.info['versions'] = versions
    return ok


# ----------------------------------------------------------------------------- 2. input overview
def overview(run: Run, data: Path) -> bool:
    import polars as pl
    found = {t: locate(data, t) for t in TABLES}
    if found['requests'] is None or found['mcc'] is None:
        missing = [TABLES[t] + '.csv' for t in ('requests', 'mcc') if found[t] is None]
        return run.check('required tables', False, f'not found in {data}: {missing}. Expected layout: '
                         + ', '.join(v + '.csv' for v in TABLES.values()) + ' (radar optional)')

    def read(t):
        path = found[t]
        return pl.read_csv(path, infer_schema=False) if path.suffix == '.csv' else pl.read_parquet(path)

    for t in ('outages', 'weather', 'radar'):   # same semantics as infer.py; infer.py remains the judge of validity
        if found[t] is None:
            continue
        if found[t].stat().st_size == 0:
            if t == 'radar':
                run.check(TABLES[t], True, 'empty file: no radar (radar-free paths)')
            else:
                run.check(TABLES[t], None, 'empty (zero-byte) file: infer.py will stop with exit code 2')
            found[t] = None
            continue
        try:
            read(t).head(1)
        except Exception as error:  # noqa: BLE001
            run.check(TABLES[t], None, f'cannot be read for the overview ({type(error).__name__}); infer.py decides')
            found[t] = None

    def utc(frame, column):
        s = frame[column]
        if s.dtype == pl.String:
            s = s.str.strip_chars().str.replace('Z$', '+00:00').str.to_datetime(time_zone='UTC', strict=False)
        return s

    info = {}
    req = read('requests')
    issues = utc(req, 'issue_time')
    info['requests'] = {'rows': req.height, 'tasks': dict(sorted(req.group_by('task_id').len().iter_rows())),
                        'counties': sorted(req['county_fips'].cast(pl.String).unique().to_list()),
                        'issues': int(issues.n_unique()), 'first_issue': str(issues.min()), 'last_issue': str(issues.max())}
    run.check('requests', True, f"{req.height} rows; tasks {info['requests']['tasks']}; counties {info['requests']['counties']}; "
              f"{info['requests']['issues']} issues from {issues.min()} to {issues.max()}")
    for t in ('outages', 'weather', 'radar'):
        if found[t] is None:
            if any(c['check'] == TABLES[t] for c in run.checks):
                continue
            run.check(TABLES[t], None if t != 'radar' else True,
                      'file not present: ' + ('the fallback models are used' if t != 'radar' else 'radar-free paths (optional table)'))
            continue
        f = read(t)
        entry = {'file': found[t].name, 'rows': f.height}
        if t == 'outages' and f.height:
            ts = utc(f, 'timestamp')
            seconds = ts.dt.epoch('us') / 1e6
            entry |= {'first': str(ts.min()), 'last': str(ts.max()), 'off_15min_grid': int(((seconds % 900) != 0).sum()),
                      'history_days_before_first_issue': round((issues.min() - ts.min()).total_seconds() / 86400, 1)}
            step = seconds.unique().sort().diff().drop_nulls()
            entry['typical_step_min'] = float(step.median() / 60) if len(step) else None
            detail = (f"{f.height} records, {entry['first']} to {entry['last']}, typical step {entry['typical_step_min']} min, "
                      f"{entry['history_days_before_first_issue']} days of history before the first issue")
            ok = None if entry['history_days_before_first_issue'] < 7 else True
            run.check('outages', ok, detail + ('' if ok else ' (the models look back up to 7 days; earlier issues have less history)'))
            if entry['off_15min_grid']:
                run.check('5-minute records', True, f"{entry['off_15min_grid']} records between 15-minute grid times: "
                          'kept for scoring; the models use the 15-minute grid as in training')
        elif t == 'weather' and f.height:
            hrrr = [c for c in f.columns if c.endswith('__mean')]
            entry |= {'hrrr_statistics': len(hrrr) > 0, 'runs': int(f['model_run_time'].n_unique())}
            run.check('weather', True, f"{f.height} rows, {entry['runs']} model runs, "
                      + ('HRRR county statistics present' if hrrr else 'basic fields only (temperature, humidity, pressure used)'))
        else:
            run.check(TABLES[t], True if t == 'radar' else None, f'{f.height} rows' + (' (header only)' if not f.height else ''))
        info[t] = entry
    run.info['input'] = info
    return True


# ----------------------------------------------------------------------------- 3. inference
def infer(run: Run, data: Path, device: str) -> int:
    print('3. Inference (official entry point infer.py)', flush=True)
    cmd = [sys.executable, str(PACKAGE / 'infer.py'), '--input-dir', str(data), '--output', str(run.out / 'predictions.csv'),
           '--device', device, '--diagnostics-json', str(run.out / 'diagnostics.json')]
    run.info['command'] = ' '.join(cmd[1:])
    started = time.perf_counter()
    with open(run.out / 'infer.log', 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        for line in proc.stdout:
            log.write(line)
            print('     ' + line.rstrip(), flush=True)
        code = proc.wait()
    run.info['inference_seconds'] = round(time.perf_counter() - started, 1)
    if code != 0:
        text = (run.out / 'infer.log').read_text(encoding='utf-8')
        message = next((l.split('Inference cannot run:', 1)[1].strip() for l in text.splitlines() if 'Inference cannot run:' in l),
                       text.strip().splitlines()[-1] if text.strip() else 'no message')
        run.check('inference', False, f'exit code {code}: {message}')
        return 2 if code == 2 else 1
    run.check('inference', True, f"finished in {run.info['inference_seconds']} s")
    return 0


# ----------------------------------------------------------------------------- 4. output contract
def check_output(run: Run, data: Path) -> bool:
    import numpy as np
    import polars as pl
    from outage_forecast.config import Config
    print('4. Output check', flush=True)
    config = Config.load()
    keys = ['task_id', 'architecture_id', 'county_fips', 'issue_time', 'target_time']
    p = pl.read_csv(run.out / 'predictions.csv', infer_schema=False)
    ok = run.check('columns', list(p.columns) == keys + ['predicted_x', 'predicted_risk'], ', '.join(p.columns))
    path = locate(data, 'requests')
    r = pl.read_csv(path, infer_schema=False) if path.suffix == '.csv' else pl.read_parquet(path)
    if path.suffix != '.csv':
        r = r.with_columns([pl.col(c).dt.convert_time_zone('UTC').dt.strftime('%Y-%m-%dT%H:%M:%SZ') if r[c].dtype == pl.Datetime
                            else pl.col(c).cast(pl.String) for c in keys])
    same = p.height == r.height and p.select(keys).sort(keys).equals(r.select(keys).sort(keys))
    ok &= run.check('one row per request', same, f'{p.height} predictions for {r.height} requests'
                    + ('' if same else ' (keys differ)'))
    x, risk = p['predicted_x'].cast(pl.Float64).to_numpy(), p['predicted_risk'].cast(pl.Float64).to_numpy()
    ok &= run.check('values', bool(np.isfinite(x).all() and np.isfinite(risk).all() and (x >= 0).all() and (x <= 1).all()
                                   and (risk >= 0).all() and (risk <= 1).all()), 'predicted_x and predicted_risk finite, in [0, 1]')
    x0 = p['architecture_id'].str.strip_chars().replace_strict({a: float(v) for a, v in config['risk']['x0'].items()},
                                                                return_dtype=pl.Float64).to_numpy()
    expected = 1 / (1 + np.exp(-np.clip(float(config['risk']['k']) * (x - x0), -700, 700)))
    gap = float(np.abs(expected - risk).max()) if len(risk) else 0.0
    ok &= run.check('risk formula', gap < 1e-12, f'max |risk - sigmoid(k (x - x0))| = {gap:.1e}')
    run.info['output'] = {'rows': p.height, 'sha256': sha256(run.out / 'predictions.csv')}
    return ok


# ----------------------------------------------------------------------------- 5. reference score
def join_truth(run: Run, data: Path, b_truth: str) -> None:
    """Predictions joined with the observations (for the score and the charts); kept in run.joined, None on failure."""
    import scoring
    from outage_forecast.config import Config
    before = sha256(run.out / 'predictions.csv')
    try:
        run.joined = scoring.frames(run.out / 'predictions.csv', data, Config.load(), b_truth)
    except Exception as error:  # noqa: BLE001
        run.joined, run.joined_error = None, f'{type(error).__name__}: {error}'
    assert sha256(run.out / 'predictions.csv') == before


def build_charts(run: Run) -> None:
    """Chart series for the summary page; without observations the forecasts alone are drawn."""
    import charts
    import polars as pl
    try:
        joined = run.joined
        if joined is None:
            p = pl.read_csv(run.out / 'predictions.csv', infer_schema=False)
            sec = lambda c: (pl.col(c).str.strip_chars().str.replace('Z$', '+00:00')  # noqa: E731
                             .str.to_datetime(time_zone='UTC', time_unit='us').dt.epoch('us') / 1e6)
            p = p.select(pl.col('task_id').str.strip_chars().alias('task'), pl.col('architecture_id').str.strip_chars().alias('arch'),
                         pl.col('county_fips').str.strip_chars().str.zfill(5).alias('county'), sec('issue_time').alias('issue'),
                         sec('target_time').alias('target'), pl.col('predicted_x').cast(pl.Float64).alias('px'))
            joined = {task: p.filter(pl.col('task') == task) for task in ('A', 'B')}
        run.charts = charts.series(joined)
    except Exception as error:  # noqa: BLE001  (the charts are an illustration; the run stays valid without them)
        run.check('charts', None, f'not drawn ({type(error).__name__}: {error})')


def reference_score(run: Run, data: Path, b_truth: str) -> None:
    import scoring
    from outage_forecast.config import Config
    print('5. Reference score (local, official formula)', flush=True)
    if run.joined is None:
        run.check('reference score', None, f'not available ({run.joined_error})')
        return
    try:
        result = scoring.score(run.out / 'predictions.csv', data, Config.load(), b_truth, joined=run.joined)
    except Exception as error:  # noqa: BLE001
        run.check('reference score', None, f'not available ({type(error).__name__}: {error})')
        return
    for task, t in result['tasks'].items():
        if 'pooled' in t:
            s = t['pooled']
            note = ' (no positive or no negative case: AUPRC term 0)' if s['degenerate_labels'] else ''
            run.check(f'Task {task}', None if s['degenerate_labels'] else True,
                      f"score {s['score']:.2f} (AUPRC_norm {s['auprc_norm']:.3f}, risk MAE {s['mae_risk']:.3f}) on {t['scored']} of "
                      f"{t['requests']} requests; {t['excluded_no_truth']} without truth, {t['excluded_invalid_truth']} invalid; "
                      f"positive rate {100 * s['positive_rate']:.1f}%{note}")
        else:
            run.check(f'Task {task}', None, f"not scorable: no valid truth for the {t['requests']} requests "
                      f"({t['excluded_no_truth']} without truth, {t['excluded_invalid_truth']} invalid)")
    rule = result['truth_rule']
    run.check('Task B truth', True, f"{rule['task_B']}; of the targets between 15-minute grid times "
              f"{rule['task_B_targets_at_T']} use the observation at T, {rule['task_B_targets_floor15']} the 15-minute value")
    run.info['score'] = result


# ----------------------------------------------------------------------------- summary page
def write_summary(run: Run) -> None:
    diag = {}
    if (run.out / 'diagnostics.json').is_file():
        try:
            diag = json.loads((run.out / 'diagnostics.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            run.check('diagnostics', None, f'unreadable ({type(error).__name__}); paths not shown')
    run.info['diagnostics'] = diag
    summary = {'status': run.status, 'checks': run.checks, 'warnings': run.warnings, **run.info}
    chart_html = ''
    if run.charts:
        try:
            import charts
            from outage_forecast.config import Config
            chart_html = charts.section(run.charts, COUNTY_NAMES, {a: float(v) for a, v in Config.load()['risk']['x0'].items()})
        except Exception as error:  # noqa: BLE001
            run.check('charts', None, f'not drawn ({type(error).__name__}: {error})')
            summary['checks'], summary['warnings'] = run.checks, run.warnings
    (run.out / 'summary.html').write_text(page(summary, chart_html), encoding='utf-8')
    (run.out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n', encoding='utf-8')


def _when(text: str) -> str:
    """'2025-07-08 00:00:00+00:00' -> '8 Jul 2025 00:00'."""
    try:
        return datetime.fromisoformat(str(text)).strftime('%d %b %Y %H:%M').lstrip('0')
    except ValueError:
        return str(text)


PAGE_STYLE = """
:root{--bg:#edf0f3;--surface:#fff;--ink:#14202c;--muted:#566575;--line:#d3dae1;--band:rgba(86,101,117,.2);
--ok:#1f7a4d;--warn:#a66300;--fail:#b3261e;--series-1:#2a78d6;--series-2:#eb6834;
--display:"Bahnschrift","DIN Alternate","Avenir Next Condensed","Roboto Condensed","Segoe UI",system-ui,sans-serif;
--text:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Noto Sans","PingFang SC","Microsoft YaHei",sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#10161c;--surface:#18212a;--ink:#e6eaee;--muted:#96a3b0;--line:#2c3945;
--band:rgba(150,163,176,.24);--ok:#5cc28f;--warn:#e0a54a;--fail:#ff7a6e;--series-1:#3987e5;--series-2:#d95926}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 var(--text)}
main{max-width:1180px;margin:0 auto;padding:0 28px 64px}
h1,h2,h3,.num{font-family:var(--display)}
header{border-top:6px solid var(--ink);padding:26px 0 4px}
.team{margin:0;color:var(--muted);font-size:14px}
h1{font-size:40px;line-height:1.1;font-weight:600;margin:6px 0 8px;letter-spacing:.005em}
.period{font-size:19px;margin:0 0 16px}
.state{display:inline-block;font:600 14px var(--display);letter-spacing:.03em;padding:4px 12px;border-radius:3px;color:#fff;background:var(--ok)}
.state.failed{background:var(--fail)}
.facts{display:flex;flex-wrap:wrap;gap:6px 36px;margin:18px 0 0;padding:0}
.facts div{min-width:120px}.facts dt{font-size:12.5px;color:var(--muted)}.facts dd{margin:0;font-size:15px;font-variant-numeric:tabular-nums}
h2{font-size:24px;font-weight:600;margin:44px 0 12px;letter-spacing:.01em}
h3{font-size:18px;font-weight:600;margin:20px 0 8px}
h3 small{font:400 14px var(--text);color:var(--muted);margin-left:12px}
.scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));background:var(--surface);border:1px solid var(--line);border-radius:4px}
.scores>div{padding:16px 22px 18px;border-left:1px solid var(--line)}.scores>div:first-child{border-left:0}
.scores .what{font-size:14px;color:var(--muted)}
.num{font-size:52px;line-height:1.05;font-weight:600;font-variant-numeric:tabular-nums;margin:4px 0 6px}
.num small{font:400 15px var(--text);color:var(--muted);margin-left:6px}
.scores p{margin:0;font-size:13.5px;color:var(--muted);line-height:1.45}
.scores .total{background:var(--ink);color:var(--bg)}.scores .total .what,.scores .total p,.scores .total .num small{color:inherit;opacity:.78}
.note{color:var(--muted);font-size:13.5px;max-width:78ch;margin:10px 0 0}
.paths{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:8px 44px}
table{border-collapse:collapse;width:100%;font-size:14.5px}
td,th{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12.5px}
.n{text-align:right;font-variant-numeric:tabular-nums;padding-right:14px}
.bar{display:inline-block;height:9px;background:var(--series-1);border-radius:1px;vertical-align:middle}
.mark{font:600 13px var(--display);letter-spacing:.03em;width:4em;white-space:nowrap}
tr.ok .mark{color:var(--ok)}tr.warn .mark{color:var(--warn)}tr.fail .mark{color:var(--fail)}
.checks td:nth-child(2){width:15em;font-weight:600}
details.all{margin-top:10px}details.all summary{cursor:pointer;color:var(--muted);font-size:14.5px;padding:6px 0}
code{font-size:13px;word-break:break-all}
footer{margin-top:44px;border-top:1px solid var(--line);padding-top:12px;color:var(--muted);font-size:13px}
footer p{margin:4px 0}
:focus-visible{outline:2px solid var(--series-1);outline-offset:2px}
@media (max-width:640px){main{padding:0 16px 48px}h1{font-size:30px}.num{font-size:40px}.scores>div{border-left:0;border-top:1px solid var(--line)}.scores>div:first-child{border-top:0}}
"""


def page(s: dict, chart_html: str = '') -> str:
    e = lambda v: html.escape(str(v))  # noqa: E731
    ok = s['status'] == 'success'
    req = s.get('input', {}).get('requests', {})
    counties = req.get('counties', [])
    names = ', '.join(COUNTY_NAMES.get(c, c).split(',')[0] for c in counties)
    title = (f"Outage-risk forecast for {len(counties)} {'county' if len(counties) == 1 else 'counties'}" if counties
             else 'Outage-risk forecast')
    period = (f"Forecasts issued from {_when(req['first_issue'])} to {_when(req['last_issue'])} UTC" if req.get('first_issue')
              else 'The run stopped before the requests were read')
    selection = s.get('selection', {})
    dropped = f"; left out for this run: {', '.join(selection['dropped'])}" if selection.get('dropped') else ''
    facts = [('Counties', names or 'none'), ('Requests', f"{req['rows']:,}" if req.get('rows') else 'none'),
             ('Inference', f"{s['inference_seconds']} s on {s.get('device', '')}" if s.get('inference_seconds') else 'not finished'),
             ('Requests from', f"{selection.get('mode', 'not selected')}{dropped}"), ('Run started', f"{s.get('started', '')} UTC")]
    facts_html = ''.join(f'<div><dt>{e(k)}</dt><dd>{e(v)}</dd></div>' for k, v in facts)

    score = ''
    sc = s.get('score', {}).get('tasks', {})
    if sc:
        cells = ''
        for k, v in sc.items():
            if 'pooled' in v:
                p = v['pooled']
                cells += (f"<div><div class='what'>Task {e(k)} score</div><div class='num'>{p['score']:.2f}</div>"
                          f"<p>Event ranking (AUPRC, normalised) {p['auprc_norm']:.3f}<br>Mean risk error {p['mae_risk']:.3f}<br>"
                          f"{v['scored']:,} of {v['requests']:,} requests scored; {100 * p['positive_rate']:.1f}% were events"
                          + ('<br>No event or only events in this period, so the ranking part is 0' if p['degenerate_labels'] else '')
                          + '</p></div>')
            else:
                cells += (f"<div><div class='what'>Task {e(k)} score</div><div class='num'>none</div><p>No observation to score against: "
                          f"{v['excluded_no_truth']:,} requests without one, {v['excluded_invalid_truth']:,} invalid</p></div>")
        score = (f"<h2>Reference score</h2><div class='scores'>{cells}</div><p class='note'>Computed here with the official formula, "
                 "after the predictions were written, against the observations in the input directory. A short period with few "
                 "events gives an unstable score.</p>")

    paths = ''
    for task, t in s.get('diagnostics', {}).get('tasks', {}).items():
        total = t['requests']
        rows = ''.join(f"<tr><td>{e(PATH_NAMES.get(r, r))}</td><td class='n'>{n:,}</td><td class='n'>{100 * n / total:.1f}%</td>"
                       f"<td><span class='bar' style='width:{max(2, round(150 * n / total))}px'></span></td></tr>"
                       for r, n in t['paths'].items())
        w = t.get('weather') or {}
        weather = (f"{w.get('with_weather', 0):,} of {w.get('model_rows', 0):,} model rows had weather values "
                   f"({w.get('with_hrrr_statistics', 0):,} with HRRR statistics, {w.get('basic_fields_only', 0):,} with basic fields only). "
                   if w else '')
        history = (f"Outage history before each issue: at least {t['min_grid_24h_of_96']} of 96 quarter-hours in the last day and "
                   f"{t['min_grid_7d_of_672']} of 672 in the last week; {t['issues_without_history']} issues had no record at all.")
        paths += (f"<div><h3>Task {e(task)}<small>{total:,} requests, {t['issues']:,} issues</small></h3><table>"
                  f"<tr><th>Model path</th><th class='n'>Requests</th><th class='n'>Share</th><th></th></tr>{rows}</table>"
                  f"<p class='note'>{e(weather + history)}</p></div>")
    paths_html = (f"<h2>Which model produced the forecasts</h2><div class='paths'>{paths}</div>" if paths else
                  "<h2>Which model produced the forecasts</h2><p class='note'>Not available: the inference did not finish.</p>")

    line = lambda c: (f"<tr class='{c['result'].lower()}'><td class='mark'>{e(c['result'])}</td><td>{e(c['check'])}</td>"  # noqa: E731
                      f"<td>{e(c['detail'])}</td></tr>")
    problems = [c for c in s['checks'] if c['result'] != 'OK']
    passed = [c for c in s['checks'] if c['result'] == 'OK']
    heading = (f"{len(passed)} checks passed" + (f", {len(problems)} need attention" if problems else ''))
    checks = f"<h2>Checks</h2><p class='note'>{heading}.</p>"
    if problems:
        checks += f"<table class='checks'>{''.join(line(c) for c in problems)}</table>"
    if passed:
        checks += (f"<details class='all'{'' if ok and not problems else ''}><summary>Show the {len(passed)} checks that passed</summary>"
                   f"<table class='checks'>{''.join(line(c) for c in passed)}</table></details>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}: demo run summary</title><style>{PAGE_STYLE}{CHART_STYLE}</style></head><body><main>
<header><p class="team">Weather-Driven Power-Outage Risk Forecasting · demo run summary</p>
<h1>{e(title)}</h1><p class="period">{e(period)}</p>
<span class="state{'' if ok else ' failed'}">{'Completed' if ok else 'Failed: no predictions are shown'}</span>
<dl class="facts">{facts_html}</dl></header>
{score}
{chart_html}
{paths_html}
{checks}
<footer><p>Predictions file SHA-256: <code>{e(s.get('output', {}).get('sha256', 'not written'))}</code></p>
<p>Command: <code>python {e(s.get('command', ''))}</code></p></footer>
</main></body></html>
"""


def select(run: Run, data: Path, args) -> Path | None:
    """Apply --counties / --start / --end / --tasks / --generate-requests; None when the selection is impossible."""
    import select_requests
    from outage_forecast.config import Config
    try:
        used, info = select_requests.prepare(data, run.out, args, Config.load())
    except select_requests.SelectionError as error:
        run.check('request selection', False, str(error))
        return None
    run.info['selection'] = info
    run.info['input_used'] = str(used)
    if used == data:
        run.check('request selection', True, 'all requests of the input directory')
    elif info.get('requests') is None:
        run.check('request selection', True, f'all requests of the input directory; written to {used}')
    else:
        window = info['window_utc']
        run.check('request selection', True, f"{info['mode']}: {info['requests']} requests; tasks {','.join(info['tasks'])}; "
                  f"counties {info['counties'] if isinstance(info['counties'], str) else ','.join(info['counties'])}"
                  + (f'; issues from {window[0]} to before {window[1]} UTC' if window else '')
                  + ('' if info.get('grid', 'official') == 'official' else '; ' + '; '.join(
                      f"Task {t} leads {g['lead_min_minutes']}..{g['lead_max_minutes']} min every {g['lead_step_minutes']} min, "
                      f"issues at {','.join(f'{h:02d}' for h in g['issue_hours_utc'])} UTC" for t, g in info['grid'].items()))
                  + f'; written to {used}')
    if info.get('dropped'):
        run.check('input ablation', None, f"left out: {', '.join(info['dropped'])} (stress test of the frozen system with that "
                  'input missing; not the official output)')
    return used


def history_checks(run: Run) -> None:
    """Read-only warnings from the inference diagnostics: stale or missing outage history, missing weather."""
    try:
        diag = json.loads((run.out / 'diagnostics.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    for task, t in diag.get('tasks', {}).items():
        age = t.get('max_latest_age_min')
        warned = False
        if t.get('issues_without_history'):
            warned = run.check(f'Task {task} outage history', None, f"{t['issues_without_history']} of {t['issues']} issues have no "
                               'usable outage record before them: history fallback used') is not None
        stale = t.get('issues_without_recent_24h') or 0
        if stale and stale >= 0.05 * t['issues']:   # the outage data do not reach these issues (e.g. 2026 requests on 2025 data)
            warned = run.check(f'Task {task} outage history', None, f"stale: {stale} of {t['issues']} issues have no usable record "
                               f'in the 24 h before them (latest record up to {age / 60:.1f} h before an issue); check that the '
                               'outage data cover the requested period') is not None
        elif stale:                                  # a few issues inside long gaps of the source records
            warned = run.check(f'Task {task} outage history', True, f"{stale} of {t['issues']} issues fall in gaps of the outage "
                               f'records longer than 24 h (longest {age / 60:.1f} h): their forecasts rest on older history')
        if not warned and age is not None:
            run.check(f'Task {task} outage history', True, 'every issue has records in the 24 h before it; latest usable record '
                      + (f'at most {age:.0f} min before each issue' if age <= 60 else
                         f'up to {age / 60:.1f} h before an issue (gaps of the source records)'))
        w = t.get('weather') or {}
        if w.get('model_rows') and not w.get('with_weather'):
            run.check(f'Task {task} weather', None, 'no usable weather for any model row')
        elif w.get('model_rows'):
            kind = 'basic fields only' if not w.get('with_hrrr_statistics') else 'HRRR statistics'
            run.check(f'Task {task} weather', True, f"{w['with_weather']} of {w['model_rows']} model rows with weather ({kind})")


def execute(run: Run, data: Path, args) -> int:
    if not check_environment(run, args.device):
        return 1
    print('2. Input', flush=True)
    data = select(run, data, args)
    if data is None:
        return 2
    try:
        if not overview(run, data):
            return 2
    except Exception as error:  # noqa: BLE001  (the overview is informative only; infer.py judges the input)
        run.check('input overview', None, f'incomplete ({type(error).__name__}: {error}); infer.py checks the input')
    code = infer(run, data, run.info['device'])
    if code:
        return code
    history_checks(run)
    if not check_output(run, data):
        return 1
    join_truth(run, data, args.b_truth)
    if not args.no_score:
        reference_score(run, data, args.b_truth)
    build_charts(run)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data', type=Path, help='input directory (organizer layout)')
    parser.add_argument('--out', type=Path, help='new run directory outside the package (default ~/aidc_demo_runs/run_<UTC time>)')
    parser.add_argument('--config', type=Path, help='JSON file with any of these options (e.g. {"data": "...", "start": '
                        '"2025-07-08", "counties": ["Harris"]}); options given on the command line win')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default=None, help='default cpu')
    parser.add_argument('--no-score', action='store_true', help='skip the reference score')
    parser.add_argument('--b-truth', choices=('auto', 'native', '15min'), default=None,
                        help='Task B truth: the observation at T when recorded, else the 15-minute value (auto, default); '
                             'or always one rule')
    parser.add_argument('--counties', help='only these counties: FIPS codes or names, comma-separated (e.g. 48201,06037 or Harris)')
    parser.add_argument('--start', help='first issue time to predict, UTC (e.g. 2025-07-08 or 2025-07-08T06:00)')
    parser.add_argument('--end', help='issue times before this UTC time (default: --start + 1 day)')
    parser.add_argument('--tasks', help='A, B or A,B (default both)')
    parser.add_argument('--generate-requests', action='store_true',
                        help='build the requests on the official grid for --start/--end instead of filtering the request table')
    parser.add_argument('--a-leads', help='part of the Task A leads, MIN:MAX[:STEP] in minutes (official 60:2880:60), e.g. 60:1440')
    parser.add_argument('--b-leads', help='part of the Task B leads, MIN:MAX[:STEP] in minutes (official 5:360:5), e.g. 5:120')
    parser.add_argument('--b-issue-hours', help='part of the Task B issue hours, UTC (official 0,6,12,18), e.g. 0,12')
    parser.add_argument('--drop', action='append', choices=('radar', 'weather'),
                        help='leave out this input (repeatable) to show the fallback of the frozen system; not the official output')
    parser.add_argument('--check-only', action='store_true', help='check the environment and the model files, then stop')
    args = parser.parse_args(argv)
    if args.config is not None:
        try:
            options = json.loads(args.config.read_text(encoding='utf-8'))
            if not isinstance(options, dict):
                raise ValueError('the file must hold one JSON object')
        except (OSError, ValueError) as error:
            parser.error(f'--config {args.config}: {error}')
        known = {k for k in vars(args) if k not in ('config', 'check_only')}
        for key, value in options.items():
            name = key.replace('-', '_')
            if name not in known:
                parser.error(f"--config {args.config}: unknown option {key!r} (known: {', '.join(sorted(known))})")
            if getattr(args, name) not in (None, False):
                continue                                    # the command line wins
            if name in ('data', 'out'):
                value = Path(value).expanduser()
            elif name == 'drop':
                value = [value] if isinstance(value, str) else list(value)
                if any(v not in ('radar', 'weather') for v in value):
                    parser.error(f'--config {args.config}: drop must be radar and/or weather')
            elif name in ('generate_requests', 'no_score'):
                value = bool(value)
            elif isinstance(value, list):
                value = ','.join(str(v) for v in value)     # counties, tasks, b_issue_hours
            elif value is not None:
                value = str(value)
            setattr(args, name, value)
        if args.device not in (None, 'cpu', 'cuda', 'auto') or args.b_truth not in (None, 'auto', 'native', '15min'):
            parser.error(f'--config {args.config}: device must be cpu, cuda or auto; b_truth auto, native or 15min')
    args.device, args.b_truth = args.device or 'cpu', args.b_truth or 'auto'
    started = datetime.now(timezone.utc)
    if args.check_only:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ok = check_environment(Run(Path(tmp)), args.device)
        print('\nENVIRONMENT READY' if ok else '\nENVIRONMENT NOT READY', flush=True)
        return 0 if ok else 1
    if args.data is None:
        parser.error('--data is required (or use --check-only)')
    data = args.data.resolve()
    out = (args.out or Path.home() / 'aidc_demo_runs' / f"run_{started:%Y%m%d_%H%M%S}").resolve()
    if not data.is_dir():
        print(f'Input directory not found: {data}', file=sys.stderr)
        return 2
    if out.is_relative_to(data) or out.is_relative_to(PACKAGE):
        print(f'The run directory {out} must be outside the input directory and the package: choose another --out', file=sys.stderr)
        return 1
    if out.exists() and any(out.iterdir()):
        print(f'Run directory {out} already exists and is not empty: choose a new --out', file=sys.stderr)
        return 1
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f'Cannot create the run directory {out}: {error}. Choose a writable --out', file=sys.stderr)
        return 1
    run = Run(out)
    run.info |= {'started': f'{started:%Y-%m-%d %H:%M:%S}', 'data': str(data), 'package': str(PACKAGE),
                 'config_sha256': sha256(PACKAGE / 'configs' / 'inference.json')}
    print(f'Demo run -> {out}', flush=True)
    try:
        code = execute(run, data, args)
    except Exception as error:  # noqa: BLE001  (reported as a failed run, never as success)
        run.check('demo tool', False, f'{type(error).__name__}: {error}')
        code = 1
    run.status = 'success' if code == 0 else 'failed'
    try:
        try:
            write_summary(run)
        except Exception as error:  # noqa: BLE001
            code, run.status = 1, 'failed'
            run.check('summary', False, f'{type(error).__name__}: {error}')
            try:
                (out / 'summary.html').unlink(missing_ok=True)
                (out / 'summary.json').write_text(json.dumps({'status': 'failed', 'checks': run.checks}, indent=1, default=str)
                                                  + '\n', encoding='utf-8')
            except Exception as second:  # noqa: BLE001
                print(f'summary not written: {type(second).__name__}: {second}', file=sys.stderr)
    finally:
        if code:   # a failed run never shows predictions (decided after the final exit code)
            try:
                if (out / 'predictions.csv').exists():
                    (out / 'predictions.csv').rename(out / 'predictions.rejected.csv')
            except OSError as error:
                print(f'could not set aside {out / "predictions.csv"}: {error}; do not use it', file=sys.stderr)
    print(f"\n{'COMPLETED' if code == 0 else 'FAILED'}: summary {out / ('summary.html' if code == 0 or (out / 'summary.html').exists() else 'summary.json')}",
          flush=True)
    if run.warnings:
        print('Warnings:\n  ' + '\n  '.join(run.warnings), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
