"""README figure: Task A forecasts against observed outages for one storm event.

    python scripts/make_figures.py --predictions out/predictions.csv --inputs data/inputs/input_2025 \
        --county 26163 --start 2025-06-26 --end 2025-06-30 --out docs/assets/event_wayne_2025-06.png
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

NAMES = {'06037': 'Los Angeles County, CA', '12086': 'Miami-Dade County, FL', '26163': 'Wayne County, MI',
         '33015': 'Rockingham County, NH', '48201': 'Harris County, TX'}
X0 = {'ARC01': 0.00100, 'ARC04': 0.00160}
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
ISSUE_COLORS = ['#d9480f', '#0b6e99', '#2b8a3e', '#7048e8']


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--county', default='26163')
    p.add_argument('--start', default='2025-06-26')
    p.add_argument('--end', default='2025-06-30')
    p.add_argument('--issues', default='2025-06-27,2025-06-28', help='issue dates (00Z) to draw, comma-separated')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)

    mcc = pl.read_parquet(a.inputs / 'mcc/MCC.parquet').filter(pl.col('county_fips') == a.county)['total_customers'][0]
    obs = (pl.read_parquet(a.inputs / 'eaglei/eaglei_outages.parquet')
           .filter((pl.col('county_fips') == a.county) & pl.col('timestamp').is_between(start, end))
           .sort('timestamp').with_columns(x=pl.col('customers_out') / mcc))
    pred = (pl.read_csv(a.predictions, infer_schema_length=0)
            .filter((pl.col('task_id') == 'A') & (pl.col('architecture_id') == 'ARC01') & (pl.col('county_fips') == a.county))
            .with_columns(pl.col('issue_time', 'target_time').str.to_datetime(time_zone='UTC'),
                          pl.col('predicted_x').cast(pl.Float64))
            .filter(pl.col('issue_time').dt.date().cast(pl.String).is_in(a.issues.split(','))))

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.grid(True, axis='y', color=GRID, lw=0.8)
    ax.axhspan(X0['ARC01'], X0['ARC04'], color='#ffe8a3', zorder=0, label='risk thresholds of the four data-center architectures')
    ax.plot(obs['timestamp'], obs['x'].clip(1e-5), color=INK, lw=1.6, label='observed (EAGLE-I, 15 min)')
    for i, (issue, g) in enumerate(pred.group_by('issue_time', maintain_order=True)):
        g = g.sort('target_time')
        color = ISSUE_COLORS[i % len(ISSUE_COLORS)]
        ax.plot(g['target_time'], g['predicted_x'].clip(1e-5), color=color, lw=1.8, ls='--',
                label=f'forecast issued {issue[0]:%b %d} 00:00 UTC (+1..+48 h)')
        ax.axvline(issue[0], color=color, lw=0.8, ls=':', alpha=0.8)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 0.06)
    ax.set_xlim(start, end)
    ax.set_ylabel('share of customers without power', color=INK)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(colors=MUTED)
    ax.set_title(f'{NAMES.get(a.county, a.county)}: 48-hour forecasts (dashed) vs. observed outages',
                 loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, loc='upper right', fontsize=8.5)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(a.out)


if __name__ == '__main__':
    main()
