#!/usr/bin/env python3
"""Render the summary page for periods of different length and check that the charts are drawn correctly (test tool).

    python demo/check_summary.py [--device cuda] [--out ~/aidc_demo_runs/summary_test] [--year] [--shots]

Cases (from data/inputs/input_2025_full when it exists, otherwise data/inputs/input_2025): one day, one week, one month,
three months, a narrowed request grid for two counties, Task B only, and with --year the whole year. For each page:
  the run finished and no check warns that the charts were not drawn
  one strip per county and task of the run
  every strip has a forecast line; no 'nan' or 'inf' anywhere in a chart
  every coordinate of every line, area, dot and band lies inside the drawing area of its strip
  the embedded hover data has a forecast value for every time of every county
--shots also writes a PNG of each page with headless Chrome (path in the CHROME environment variable, or the default
Windows location under WSL), light and dark, for looking at. Exit code 1 when a check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
CHROME = os.environ.get('CHROME', 'google-chrome')
NUMBER = re.compile(r'-?\d+(?:\.\d+)?(?:e-?\d+)?')


def cases(data: Path, year: bool) -> list[tuple[str, list[str]]]:
    d = ['--data', str(data)]
    out = [('day', d + ['--start', '2025-07-10', '--end', '2025-07-11']),
           ('week', d + ['--start', '2025-07-08', '--end', '2025-07-15']),
           ('month', d + ['--start', '2025-07-01', '--end', '2025-08-01']),
           ('three_months', d + ['--start', '2025-06-01', '--end', '2025-09-01']),
           ('two_counties_narrow_grid', d + ['--start', '2025-07-08', '--end', '2025-07-15', '--counties', 'Harris,Los Angeles',
                                             '--generate-requests', '--b-issue-hours', '0,12', '--b-leads', '5:120']),
           ('task_b_only', d + ['--start', '2025-08-01', '--end', '2025-08-04', '--tasks', 'B'])]
    if year:
        out.append(('year', d))
    return out


def check_page(run: Path) -> list[str]:
    """Problems of one finished run directory (empty when the page is right)."""
    try:
        s = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        page = (run / 'summary.html').read_text(encoding='utf-8')
    except (OSError, ValueError) as error:
        return [f'no summary ({error})']
    bad = []
    if s.get('status') != 'success':
        bad.append(f"status {s.get('status')}")
    bad += [f"check {c['check']}: {c['detail']}" for c in s.get('checks', []) if c['check'] == 'charts' and c['result'] != 'OK']
    requests = s.get('input', {}).get('requests', {})
    wanted = len(requests.get('counties', [])) * len(requests.get('tasks', []))
    strips = re.findall(r"<svg viewBox='0 0 (\d+) (\d+)' data-key='([^']+)'.*?</svg>", page, flags=re.S)
    bodies = re.findall(r"<svg viewBox='0 0 \d+ \d+' data-key='[^']+'.*?</svg>", page, flags=re.S)
    if len(strips) != wanted or wanted == 0:
        bad.append(f'{len(strips)} strips for {wanted} county-task pairs')
    for (w, h, key), body in zip(strips, bodies):
        w, h = float(w), float(h)
        if re.search(r'nan|inf', body, flags=re.I):
            bad.append(f'{key}: nan or inf in the drawing')
        paths = dict(re.findall(r"<path class='([a-z-]+)' d='([^']*)'", body))
        if not paths.get('pred'):
            bad.append(f'{key}: no forecast line')
        for name, d in paths.items():
            values = [float(v) for v in NUMBER.findall(d)]
            xs, ys = values[0::2], values[1::2]
            if len(values) % 2 or not xs or min(xs) < 0 or max(xs) > w or min(ys) < 0 or max(ys) > h:
                bad.append(f'{key}: {name} leaves the drawing area')
        for cx, cy in re.findall(r"<circle class='obs-dot' cx='([^']+)' cy='([^']+)'", body):
            if not (0 <= float(cx) <= w and 0 <= float(cy) <= h):
                bad.append(f'{key}: a dot leaves the drawing area')
    found = re.search(r"<script type='application/json' id='chart-data'>(.*?)</script>", page, flags=re.S)
    if not found:
        bad.append('no hover data')
    else:
        hover = json.loads(found.group(1))
        if sorted(hover) != sorted(key for _, _, key in strips):
            bad.append('hover data and strips do not match')
        for key, c in hover.items():
            if len(c['t']) != len(c['pred']) or len(c['t']) != len(c['obs']) or not c['t']:
                bad.append(f'{key}: hover data lengths differ')
            elif any(v is None or not math.isfinite(v) for v in c['pred']):
                bad.append(f'{key}: a time without a forecast value')
    return bad


def shoot(run: Path) -> None:
    page = (run / 'summary.html').read_text(encoding='utf-8')
    for mode in ('light', 'dark'):
        forced = run / f'summary_{mode}.html'   # the page follows the system setting; force each mode for the picture
        forced.write_text(page.replace('<head>', f"<head><meta name='color-scheme' content='{mode}'>", 1)
                          .replace('@media (prefers-color-scheme:dark)', '@media all' if mode == 'dark' else '@media not all'),
                          encoding='utf-8')
        win = lambda p, flag: subprocess.run(['wslpath', flag, str(p)], capture_output=True, text=True).stdout.strip() or str(p)  # noqa: E731
        subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--window-size=1280,3000',
                        f'--screenshot={win(run / f"summary_{mode}.png", "-w")}', f'file://{win(forced, "-m")}'],
                       cwd=run, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, default=Path.home() / 'aidc_demo_runs' / 'summary_test')
    p.add_argument('--year', action='store_true', help='also the whole year (a few minutes)')
    p.add_argument('--shots', action='store_true', help='write PNG pictures of the pages')
    a = p.parse_args()
    data = PACKAGE / 'data' / 'inputs' / 'input_2025_full'
    if not data.is_dir():
        data = PACKAGE / 'data' / 'inputs' / 'input_2025'
    failed = 0
    for name, options in cases(data, a.year):
        run = a.out / name
        shutil.rmtree(run, ignore_errors=True)
        cmd = [sys.executable, str(PACKAGE / 'demo' / 'run_demo.py'), *options, '--device', a.device, '--out', str(run)]
        code = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
        bad = check_page(run) + ([f'exit code {code}'] if code else [])
        failed += bool(bad)
        print(f"[{'FAIL' if bad else 'OK  '}] {name}: {run / 'summary.html'}" + ''.join(f'\n       {b}' for b in bad[:8]), flush=True)
        if a.shots and (run / 'summary.html').exists():
            shoot(run)
    print(f"\n{'ALL PAGES CORRECT' if not failed else f'{failed} PAGE(S) WRONG'}")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
