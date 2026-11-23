"""Alert-budget replay: how often does a budgeted daily alert come before a new outage?

    python -m outage_forecast.alerts --input-dir data/inputs/input_2025_full \
        --predictions out/frozen_2025_full.csv --out artifacts/evaluation/alerts_2025_frozen.json

Pre-specified protocol (PROTOCOL below; committed before the results were computed, docs/RESULTS.md section 11):

- Event: an onset is the first hourly share (Task A truth: mean of four quarter-hours) at or above 0.100 % (ARC01)
  after 6 complete hours below it (sql/events/02_onsets.sql). A gap breaks the quiet run; one episode gives one onset.
- Alerting: once a day at the Task A issue time (00 UTC), per county. A county is eligible only when it has been
  quiet for 6 complete hours in the records visible at T; otherwise the rule abstains. An eligible county is alerted
  when its score exceeds a cutoff. Scores: the model's maximum predicted share over T+1 h ... T+24 h;
  near-threshold persistence (the latest visible hourly share); the maximum HRRR gust over the same 24 h.
- Budget: B alerts per county per 30 days, B = 4 and 8. The cutoff per county and method is fitted on the issues of
  June-August 2025 from the scores alone (no truth): at most floor(B * days / 30) eligible issues exceed it, ties
  excluded. It is then frozen and applied to September-December 2025.
- Scoring: an alert at T is a hit when an onset falls in (T, T+24 h]; it is credited with one onset (the first).
  Only verified windows count (all hourly truths from T-5 h to T+24 h exist), so every onset in them is known.
  Recall = matched onsets / onsets in verified windows (abstained days count as misses); precision = hits /
  verified alerts; lead = first onset - T.
- Primary statistic: recall of the model minus recall of near-threshold persistence at B = 4; paired moving blocks of
  7 issue days, 2,000 resamples, seed 0, 95 % percentile interval. Secondary: the same against the weather score,
  precision differences, B = 8, 14-day blocks. Large outages (0.5 %): counts only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .evaluation import block_days, summarise
from .warehouse import SQL_DIR, build, render

PROTOCOL = {
    'threshold_x': 0.001, 'large_outage_x': 0.005, 'quiet_hours': 6, 'horizon_hours': 24,
    'fit': ['2025-06-01', '2025-09-01'], 'evaluate': ['2025-09-01', '2026-01-01'],
    'budgets_per_30_days': [4, 8], 'methods': ['model', 'persistence', 'weather'],
    'primary': {'statistic': 'recall difference, model - persistence', 'budget': 4, 'block_days': 7, 'reps': 2000,
                'seed': 0},
}
SCORE = {'model': 'model_score', 'persistence': 'latest_x', 'weather': 'weather_score'}
HOUR, DAY = 3600, 86400


def epoch(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


# ------------------------------------------------------------------------------------------------ queries
def _sql(name: str, protocol: dict, x0: float) -> str:
    text = (SQL_DIR / 'events' / name).read_text(encoding='utf-8')
    return render(text, {'x0': repr(float(x0)), 'quiet_hours': str(int(protocol['quiet_hours'])),
                         'horizon_hours': str(int(protocol['horizon_hours']))})


def event_tables(con, protocol: dict = PROTOCOL) -> dict:
    """Issue view and onsets (both thresholds) from a warehouse with gold layer and scored predictions."""
    con.execute(_sql('01_hourly_truth.sql', protocol, protocol['threshold_x']))
    cur = con.execute(_sql('03_issue_view.sql', protocol, protocol['threshold_x']))
    names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    issues = {n: [r[i] for r in rows] for i, n in enumerate(names)}
    onsets = {key: con.execute(_sql('02_onsets.sql', protocol, protocol[key])).fetchall()
              for key in ('threshold_x', 'large_outage_x')}
    return {'issues': issues, 'onsets': onsets}


# ------------------------------------------------------------------------------------------------ the rule
def window_issue(hour_s: int, horizon_hours: int = 24) -> int:
    """The daily issue whose window (T, T + horizon] contains the hour ending at hour_s (daily 00 UTC issues)."""
    assert horizon_hours == 24, 'daily issues with a 24 h window tile the time axis'
    return (int(hour_s) - 1) // DAY * DAY


def fit_cutoff(scores: list, budget: float, days: int) -> float:
    """Cutoff c such that at most floor(budget * days / 30) of the eligible scores are > c (ties excluded).
    Scores of abstaining issues are None. With fewer eligible issues than the allowance every one is alerted."""
    allowed = int(np.floor(budget * days / 30 + 1e-9))
    eligible = np.sort(np.array([s for s in scores if s is not None], dtype=float))[::-1]
    if len(eligible) <= allowed:
        return -np.inf
    return float(eligible[allowed])          # the (allowed+1)-th largest: only strictly larger scores alert


def replay(issues: dict, onsets: list, method: str, budget: float, protocol: dict = PROTOCOL) -> dict:
    """Alerts of one method and budget: fitted cutoffs per county, and per evaluated issue the outcome."""
    fit0, fit1 = (epoch(d) for d in protocol['fit'])
    ev0, ev1 = (epoch(d) for d in protocol['evaluate'])
    score = [s if g else None for s, g in zip(issues[SCORE[method]], issues['gate_ok'])]
    by_window = {}
    for county, hour_s in onsets:
        by_window.setdefault((county, window_issue(hour_s, protocol['horizon_hours'])), []).append(hour_s)
    counties = sorted(set(issues['county']))
    fit_days = len({t for t in issues['issue_s'] if fit0 <= t < fit1})
    cutoffs = {}
    for c in counties:
        fit_scores = [s for s, cc, t in zip(score, issues['county'], issues['issue_s']) if cc == c and fit0 <= t < fit1]
        cutoffs[c] = fit_cutoff(fit_scores, budget, fit_days)
    out = []
    for c, t, s, verified in zip(issues['county'], issues['issue_s'], score, issues['verified']):
        if not ev0 <= t < ev1:
            continue
        alert = s is not None and s > cutoffs[c]
        window = sorted(by_window.get((c, t), []))
        out.append({'county': c, 'issue_s': t, 'eligible': s is not None, 'alert': alert, 'verified': bool(verified),
                    'onsets': len(window) if verified else 0, 'hit': bool(alert and verified and window),
                    'lead_hours': (window[0] - t) / HOUR if alert and verified and window else None})
    return {'cutoffs': cutoffs, 'fit_days': fit_days, 'issues': out}


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def summary(rows: list[dict], days: int) -> dict:
    onsets = sum(r['onsets'] for r in rows)
    matched = sum(r['hit'] for r in rows)            # one onset credited per hit
    alerts = sum(r['alert'] for r in rows)
    verified_alerts = sum(r['alert'] and r['verified'] for r in rows)
    counties = len({r['county'] for r in rows}) or 1
    leads = [r['lead_hours'] for r in rows if r['lead_hours'] is not None]
    return {'onsets_in_verified_windows': onsets, 'matched_onsets': matched, 'recall': _ratio(matched, onsets),
            'alerts': alerts, 'verified_alerts': verified_alerts, 'unverified_alerts': alerts - verified_alerts,
            'hits': matched, 'precision': _ratio(matched, verified_alerts),
            'alerts_per_county_per_30_days': alerts / counties / days * 30 if days else None,
            'median_lead_hours': float(np.median(leads)) if leads else None,
            'abstained_issues': sum(not r['eligible'] for r in rows), 'issues': len(rows)}


def day_counts(rows: list[dict], days: list[int]) -> dict:
    """Per issue day (all counties): onsets, matched onsets, verified alerts."""
    index = {d: i for i, d in enumerate(days)}
    out = {k: np.zeros(len(days)) for k in ('onsets', 'matched', 'verified_alerts')}
    for r in rows:
        i = index[r['issue_s']]
        out['onsets'][i] += r['onsets']
        out['matched'][i] += r['hit']
        out['verified_alerts'][i] += r['alert'] and r['verified']
    return out


def ratio_difference(a: dict, b: dict, num: str, den: str, block: int, reps: int, seed: int) -> dict | None:
    """Paired block bootstrap of num/den (method a) minus num/den (method b) over issue days."""
    def stat(idx):
        da, db = a[den][idx].sum(), b[den][idx].sum()
        return a[num][idx].sum() / da - b[num][idx].sum() / db if da and db else np.nan
    n = len(a[num])
    point = stat(np.arange(n))
    if np.isnan(point):
        return None
    rng = np.random.default_rng(seed)
    draws = np.array([stat(block_days(n, block, rng)) for _ in range(reps)])
    result = summarise(point, draws[~np.isnan(draws)])
    result['undefined_resamples'] = int(np.isnan(draws).sum())
    return result


def large_outages(issues: dict, onsets: list, replays: dict, protocol: dict = PROTOCOL) -> dict:
    """Onsets at the large-outage threshold in the evaluation period, and how many fell in an alerted window."""
    ev0, ev1 = (epoch(d) for d in protocol['evaluate'])
    alerted = {m: {(r['county'], r['issue_s']) for r in rep['issues'] if r['alert']} for m, rep in replays.items()}
    events = [(c, h) for c, h in onsets if ev0 <= window_issue(h, protocol['horizon_hours']) < ev1]
    return {'threshold_x': protocol['large_outage_x'], 'onsets': len(events),
            'by_county': {c: sum(cc == c for cc, _ in events) for c in sorted({c for c, _ in events})},
            'preceded_by_an_alert': {m: sum((c, window_issue(h, protocol['horizon_hours'])) in a for c, h in events)
                                     for m, a in alerted.items()}}


# ------------------------------------------------------------------------------------------------ analysis
def analyse(input_dir: Path, predictions: Path, name: str = 'model', protocol: dict = PROTOCOL) -> dict:
    con = build(input_dir, predictions=predictions)
    tables = event_tables(con, protocol)
    con.close()
    issues, onsets = tables['issues'], tables['onsets']
    ev0, ev1 = (epoch(d) for d in protocol['evaluate'])
    days = sorted({t for t in issues['issue_s'] if ev0 <= t < ev1})
    out = {'model': name, 'predictions': Path(predictions).name, 'protocol': protocol, 'evaluation_days': len(days),
           'onsets': {k: {'fit': sum(epoch(protocol['fit'][0]) <= window_issue(h) < epoch(protocol['fit'][1])
                                     for _, h in v),
                          'evaluate': sum(ev0 <= window_issue(h) < ev1 for _, h in v)}
                      for k, v in onsets.items()},
           'budgets': {}}
    p = protocol['primary']
    for budget in protocol['budgets_per_30_days']:
        replays = {m: replay(issues, onsets['threshold_x'], m, budget, protocol) for m in protocol['methods']}
        counties = sorted(set(issues['county']))
        entry = {'cutoffs': {m: r['cutoffs'] for m, r in replays.items()},
                 'pooled': {m: summary(r['issues'], len(days)) for m, r in replays.items()},
                 'by_county': {c: {m: summary([i for i in r['issues'] if i['county'] == c], len(days))
                                   for m, r in replays.items()} for c in counties},
                 'large_outages': large_outages(issues, onsets['large_outage_x'], replays, protocol)}
        counts = {m: day_counts(r['issues'], days) for m, r in replays.items()}
        entry['differences'] = {}
        for base in ('persistence', 'weather'):
            blocks = (7, 14) if base == 'persistence' else (7,)
            entry['differences'][f'model - {base}'] = {
                **{f'recall, {b}-day blocks': ratio_difference(counts['model'], counts[base], 'matched', 'onsets',
                                                               b, p['reps'], p['seed']) for b in blocks},
                'precision, 7-day blocks': ratio_difference(counts['model'], counts[base], 'matched',
                                                            'verified_alerts', 7, p['reps'], p['seed'])}
        out['budgets'][str(budget)] = entry
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
