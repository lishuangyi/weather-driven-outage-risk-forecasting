"""Notebook front end of the demo tool (used by notebooks/demo.ipynb; never used by infer.py).

    import notebook_api as demo
    demo.check_environment()
    run = demo.run(data='data/inputs/input_2025', start='2025-07-08', end='2025-07-15', counties=['Harris'])
    run.scores()            table: one row per task (by='architecture' or 'county' for the breakdown)
    run.plot('A')           figure: forecast against observation, one panel per county
    run.paths()             table: which model path produced the forecasts
    run.checks()            table: every check of the run
    run.predictions()       table: predictions.csv
    demo.compare(full=run, no_radar=demo.run(..., drop='radar'))

run() takes the options of demo/run_demo.py as keyword arguments and calls run_demo.main() with them, so a notebook
run is the same run as `python demo/run_demo.py ...`: the predictions come from infer.py in a separate process, the
run directory gets the same files, and a failed run shows no predictions. The results are returned as ordinary
notebook objects: polars tables and matplotlib figures, read from the files of the run directory. Nothing here changes
a prediction or an input file.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [p for p in (str(Path(__file__).resolve().parent),) if p not in sys.path]

import polars as pl  # noqa: E402

import charts  # noqa: E402
import run_demo  # noqa: E402

pl.Config.set_fmt_str_lengths(400)        # the check details are sentences: show them whole
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_hide_dataframe_shape(True)

FORECAST, OBSERVED, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#14202c', '#566575', '#d3dae1'


def _argv(options: dict) -> list[str]:
    argv = []
    for name, value in options.items():
        flag = '--' + name.replace('_', '-')
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif name == 'drop':
            for v in ([value] if isinstance(value, str) else value):
                argv += [flag, str(v)]
        elif isinstance(value, (list, tuple, set)):
            argv += [flag, ','.join(str(v) for v in value)]
        else:
            argv += [flag, str(value)]
    return argv


def _path(value) -> Path:
    """Relative paths are taken from the package folder, wherever the notebook was started."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else PACKAGE / p


def check_environment(device: str = 'cpu') -> bool:
    """Packages, device and model files; True when the environment is ready."""
    return run_demo.main(['--check-only', '--device', device]) == 0


def run(data='data/inputs/input_2025', *, start=None, end=None, counties=None, tasks=None, generate_requests=False,
        a_leads=None, b_leads=None, b_issue_hours=None, drop=None, device='cpu', b_truth='auto', no_score=False,
        out=None) -> 'Result':
    """One demo run; the arguments are the options of demo/run_demo.py (see docs/DEMO.md).

    counties: FIPS codes or names; start/end: issue times in [start, end), UTC; tasks: 'A', 'B' or both;
    a_leads/b_leads: 'MIN:MAX[:STEP]' in minutes; b_issue_hours: UTC hours; drop: 'radar' and/or 'weather'.
    """
    if out is None:
        base = Path.home() / 'aidc_demo_runs' / f"nb_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        out, n = base, 1
        while out.exists():
            out, n = base.with_name(f'{base.name}_{n}'), n + 1
    out = _path(out)
    options = dict(data=_path(data), out=out, start=start, end=end, counties=counties, tasks=tasks,
                   generate_requests=generate_requests, a_leads=a_leads, b_leads=b_leads, b_issue_hours=b_issue_hours,
                   drop=drop, device=device, b_truth=b_truth, no_score=no_score)
    try:
        code = run_demo.main(_argv(options))
    except SystemExit as stop:          # an option the command line would have rejected
        code = stop.code if isinstance(stop.code, int) else 1
    return Result(out, code, b_truth)


class Result:
    """A finished run: its directory, its summary (summary.json) and tables and figures made from them."""

    def __init__(self, out: Path, code: int = 0, b_truth: str = 'auto'):
        self.out, self.code, self.ok, self.b_truth = Path(out), code, code == 0, b_truth
        try:
            self.summary = json.loads((self.out / 'summary.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            self.summary = {}
        self._series = None

    def __repr__(self) -> str:
        return f"<demo run {'completed' if self.ok else f'FAILED (exit code {self.code})'}: {self.out}>"

    def _need(self) -> None:
        if not self.ok:
            why = '; '.join(f"{c['check']}: {c['detail']}" for c in self.summary.get('checks', []) if c['result'] == 'FAIL')
            raise RuntimeError(f"the run failed, so no predictions are shown. {why or 'See the output of the run cell'} ({self.out})")

    # ------------------------------------------------------------------ tables
    def score(self, task: str):
        """Reference score of Task A or B as a number (None when the run has none)."""
        return self.summary.get('score', {}).get('tasks', {}).get(task, {}).get('pooled', {}).get('score')

    def scores(self, by: str = 'task') -> pl.DataFrame:
        """Reference scores (official formula, out of 100). by='task' (the pooled score of each task, the one that
        counts), 'architecture' or 'county' (breakdowns, for diagnosis only)."""
        self._need()
        group = {'task': None, 'architecture': 'per_architecture', 'county': 'per_county_diagnostic'}[by]
        rows = []
        for task, t in self.summary.get('score', {}).get('tasks', {}).items():
            parts = {None: t.get('pooled')} if group is None else t.get(group, {})
            for name, s in parts.items():
                row = {'task': task}
                if group:
                    row[by] = run_demo.COUNTY_NAMES.get(name, name) if by == 'county' else name
                if not s:
                    rows.append(row | {'requests': t['requests'], 'scored': 0})
                    continue
                rows.append(row | {'score': round(s['score'], 2), 'AUPRC (normalised)': round(s['auprc_norm'], 3),
                                   'risk MAE': round(s['mae_risk'], 3), 'event rate %': round(100 * s['positive_rate'], 1),
                                   'scored': s.get('cases', t['scored'])} | ({} if group else {'requests': t['requests']}))
        if not rows:
            raise RuntimeError('this run has no reference score (no_score=True, or the input has no observations)')
        return pl.DataFrame(rows)

    def paths(self) -> pl.DataFrame:
        """Model path actually used, per task: requests and share."""
        self._need()
        rows = [{'task': task, 'model path': run_demo.PATH_NAMES.get(p, p), 'requests': n,
                 'share %': round(100 * n / t['requests'], 1)}
                for task, t in self.summary.get('diagnostics', {}).get('tasks', {}).items() for p, n in t['paths'].items()]
        return pl.DataFrame(rows)

    def checks(self, problems_only: bool = False) -> pl.DataFrame:
        """Every check of the run (OK / WARN / FAIL); problems_only=True leaves the passed ones out."""
        rows = [c for c in self.summary.get('checks', []) if not problems_only or c['result'] != 'OK']
        return pl.DataFrame(rows, schema={'result': pl.String, 'check': pl.String, 'detail': pl.String})

    def predictions(self) -> pl.DataFrame:
        """predictions.csv of the run as a table (the official output, unchanged)."""
        self._need()
        return pl.read_csv(self.out / 'predictions.csv', schema_overrides={'county_fips': pl.String})

    # ------------------------------------------------------------------ figure
    def series(self) -> dict:
        """Forecast and observation per task and county (the series of the summary page; see demo/charts.py)."""
        self._need()
        if self._series is None:
            r = run_demo.Run(self.out)
            data = Path(self.summary.get('input_used') or self.summary['data'])
            run_demo.join_truth(r, data, self.b_truth)
            run_demo.build_charts(r)
            self._series = r.charts
        return self._series

    def plot(self, task: str = 'A', counties=None):
        """Forecast against observation for one task, one panel per county on a shared time axis.

        Blue: forecast share of customers without power (predicted_x; where a time was forecast more than once, the
        most recent forecast). Orange: observed share (the truth used for the score). Grey band: the risk thresholds
        of the four architectures. The vertical axis is linear near zero and logarithmic above about 0.05 %.
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from outage_forecast.config import Config
        entry = self.series().get(task)
        if not entry:
            raise ValueError(f'this run has no Task {task} forecast')
        import select_requests
        wanted = None if counties is None else set(select_requests.parse_counties(
            counties if isinstance(counties, str) else ','.join(counties), list(run_demo.COUNTY_NAMES)))
        data = {c: d for c, d in entry['counties'].items() if wanted is None or c in wanted}
        if not data:
            raise ValueError(f'none of {counties} is in this run ({sorted(entry["counties"])})')
        x0 = [float(v) for v in Config.load()['risk']['x0'].values()]
        lo, hi = min(x0), max(x0)
        values = [v for d in data.values() for v in d['pred'] + d['obs'] if v is not None]
        top = next((t for t in charts.TOPS if t >= max(max(values), hi * 1.5)), 1.0)
        step = entry.get('step_seconds', 0)

        def line(d, key):
            """x, y with a break (NaN) where a value is missing or no target time was requested."""
            xs, ys = [], []
            for seg in charts._segments(d['t'], d[key], step):
                xs += [datetime.fromtimestamp(t, timezone.utc).replace(tzinfo=None) for t, _ in seg]
                ys += [v for _, v in seg]
                xs.append(xs[-1])
                ys.append(math.nan)
            return xs, ys

        fig, axes = plt.subplots(len(data), 1, sharex=True, sharey=True, figsize=(11, 1.0 + max(1.55 * len(data), 2.5)), squeeze=False,
                                 layout='constrained')
        ticks = [t for t in charts.Y_TICKS if t <= top]
        has_obs = False
        for ax, (county, d) in zip(axes[:, 0], data.items()):
            ax.axhspan(lo, hi, color=MUTED, alpha=0.2, lw=0)
            ox, oy = line(d, 'obs')
            if ox:
                has_obs = True
                ax.fill_between(ox, oy, 0, color=OBSERVED, alpha=0.13, lw=0)
                ax.plot(ox, oy, color=OBSERVED, lw=1.2, marker='o', ms=1.5)
            px, py = line(d, 'pred')
            ax.plot(px, py, color=FORECAST, lw=1.8)
            ax.set_yscale('asinh', linear_width=charts.KNEE)
            ax.set_ylim(0, top)
            ax.set_yticks(ticks, [charts._pct(t) for t in ticks])
            ax.minorticks_off()
            ax.grid(axis='y', color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
            obs = [v for v in d['obs'] if v is not None]
            fact = (f"observed above {100 * lo:.2f}% in {100 * sum(v > lo for v in obs) / len(obs):.0f}% of the times, "
                    f"peak {100 * max(obs):.2f}%" if obs else 'no observation in the input')
            ax.set_title(run_demo.COUNTY_NAMES.get(county, county), loc='left', fontsize=10.5, fontweight='bold', color=INK, pad=3)
            ax.set_title(fact, loc='right', fontsize=8.5, color=MUTED, pad=3)
        last = axes[-1, 0]
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        last.xaxis.set_major_locator(locator)
        last.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        last.tick_params(axis='x', length=3, color=MUTED)
        last.set_xlabel('target time (UTC)', fontsize=9, color=MUTED)
        fig.supylabel('customers without power', fontsize=9, color=MUTED)
        title, sub = charts.TASK_TITLES.get(task, (f'Task {task}', ''))
        note = '' if entry['buckets_merged'] == 1 else f"; each point is the highest value within {entry['bucket_minutes']:g} min"
        fig.suptitle(f'{title}: forecast and observation\n', x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK)
        fig.text(0.01, 1 - 0.36 / fig.get_figheight(), sub + note, ha='left', va='top', fontsize=9, color=MUTED)
        keys = [Line2D([], [], color=FORECAST, lw=1.8, label='forecast (predicted_x)')]
        keys += [Line2D([], [], color=OBSERVED, lw=1.2, label='observed')] if has_obs else []
        keys += [Patch(color=MUTED, alpha=0.2, lw=0, label=f'risk thresholds of the four architectures, {100 * lo:.2f}% to {100 * hi:.2f}%')]
        fig.legend(handles=keys, loc='outside lower center', ncols=3, frameon=False, fontsize=9, labelcolor=INK)
        plt.close(fig)          # returned to the notebook, which shows it once
        return fig

    def summary_page(self, height: int = 900):
        """The run's summary.html (the page the command-line tool writes) inside the notebook."""
        import html
        from IPython.display import HTML
        page = (self.out / 'summary.html').read_text(encoding='utf-8')
        return HTML(f'<div><iframe srcdoc="{html.escape(page, quote=True)}" style="width:100%;height:{height}px;border:0"></iframe></div>')


def compare(**runs: Result) -> pl.DataFrame:
    """Table of the reference scores of several runs, e.g. compare(full=run, no_radar=other)."""
    rows = []
    for name, r in runs.items():
        selection = r.summary.get('selection', {})
        a, b = r.score('A'), r.score('B')
        rows.append({'run': name, 'status': 'completed' if r.ok else 'failed',
                     'left out': ', '.join(selection.get('dropped') or []) or 'nothing',
                     'Task A score': None if a is None else round(a, 2), 'Task B score': None if b is None else round(b, 2),
                     'requests': r.summary.get('input', {}).get('requests', {}).get('rows'),
                     'inference s': r.summary.get('inference_seconds')})
    return pl.DataFrame(rows, schema={'run': pl.String, 'status': pl.String, 'left out': pl.String, 'Task A score': pl.Float64,
                                      'Task B score': pl.Float64, 'requests': pl.Int64, 'inference s': pl.Float64})
