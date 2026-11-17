#!/usr/bin/env python3
"""Run the demo tool on every calendar month of the full-year inputs and tabulate the results (test tool).

    python demo/check_months.py [--years 2022 2023 2024 2025] [--device cuda] [--out ~/aidc_demo_runs/monthly]

For each month: python demo/run_demo.py --data data/inputs/input_<year>_full --start <first day> --end <first day of the
next month>, in <out>/<year>_<month>. A month whose run already finished successfully is not run again. The table goes
to <out>/monthly_results.csv and is printed. Exit code 1 when any month failed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
FIELDS = ['year', 'month', 'status', 'requests', 'inference_s', 'A_score', 'A_scored', 'A_positive_pct', 'B_score', 'B_scored',
          'B_positive_pct', 'technical_of_90', 'A_paths', 'B_paths', 'warnings']


def row(year: int, month: int, run: Path) -> dict:
    out = {'year': year, 'month': month, 'status': 'no summary'}
    try:
        s = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return out
    out['status'] = s.get('status')
    out['requests'] = s.get('input', {}).get('requests', {}).get('rows')
    out['inference_s'] = s.get('inference_seconds')
    for task in ('A', 'B'):
        t = s.get('score', {}).get('tasks', {}).get(task, {})
        if 'pooled' in t:
            out[f'{task}_score'] = round(t['pooled']['score'], 2)
            out[f'{task}_positive_pct'] = round(100 * t['pooled']['positive_rate'], 1)
        out[f'{task}_scored'] = t.get('scored')
        paths = s.get('diagnostics', {}).get('tasks', {}).get(task, {}).get('paths', {})
        out[f'{task}_paths'] = ' '.join(f'{k}={v}' for k, v in paths.items())
    if 'technical_score' in s.get('score', {}):
        out['technical_of_90'] = round(s['score']['technical_score'], 2)
    out['warnings'] = ' | '.join(w[:110] for w in s.get('warnings', []))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--years', type=int, nargs='+', default=[2022, 2023, 2024, 2025])
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, default=Path.home() / 'aidc_demo_runs' / 'monthly')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in a.years:
        for month in range(1, 13):
            run = a.out / f'{year}_{month:02d}'
            done = row(year, month, run)
            if done['status'] != 'success':
                if run.exists():
                    subprocess.run(['rm', '-rf', str(run)], check=True)
                end = f'{year + (month == 12)}-{month % 12 + 1:02d}-01'
                cmd = [sys.executable, str(PACKAGE / 'demo' / 'run_demo.py'), '--data', str(PACKAGE / 'data' / 'inputs' / f'input_{year}_full'),
                       '--start', f'{year}-{month:02d}-01', '--end', end, '--device', a.device, '--out', str(run)]
                code = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
                done = row(year, month, run)
                if code != 0 and done['status'] == 'success':
                    done['status'] = f'exit {code}'
            rows.append(done)
            print(f"{year}-{month:02d} {done['status']}: A {done.get('A_score')} B {done.get('B_score')} "
                  f"({done.get('requests')} requests, {done.get('inference_s')} s)" + (f" WARN {done['warnings']}" if done.get('warnings') else ''),
                  flush=True)
            with open(a.out / 'monthly_results.csv', 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, FIELDS)
                w.writeheader()
                w.writerows(rows)
    failed = [f"{r['year']}-{r['month']:02d}" for r in rows if r['status'] != 'success']
    print(f"\n{len(rows) - len(failed)} of {len(rows)} months succeeded" + (f'; FAILED: {failed}' if failed else '')
          + f"; table: {a.out / 'monthly_results.csv'}")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
