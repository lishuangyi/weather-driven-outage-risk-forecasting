"""Seven-column output: request keys exactly as given, predicted_x and the official architecture risk.

Task B native 15-minute predictions are held over their interval: a request at lead L uses the prediction at
floor(L / 15) x 15; leads of 5 and 10 minutes use the latest observed proportion at the issue (the 15-minute
prediction when none is available). Rows are written by issue time, then in request-file order; numbers with '.17g'.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from .inputs import InputError

COLUMNS = ('task_id', 'architecture_id', 'county_fips', 'issue_time', 'target_time', 'predicted_x', 'predicted_risk')


def risk(x: float, x0: float, k: float) -> float:
    if not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError('predicted_x must be finite and in [0, 1]')
    z = max(-700.0, min(700.0, k * (x - x0)))
    return 1.0 / (1.0 + math.exp(-z))


def assemble(requests: pl.DataFrame, native: dict[str, pl.DataFrame], anchors: pl.DataFrame, config) -> pl.DataFrame:
    """native[task]: county, issue, lead_minutes, p (float64); anchors: county, issue, anchor (Task B)."""
    a = requests.filter(pl.col('task') == 'A').join(native['A'], on=['county', 'issue', 'lead_minutes'], how='left')
    b = requests.filter(pl.col('task') == 'B').with_columns((pl.col('lead_minutes') // 15 * 15).alias('native_lead'))
    nb = native['B']
    b = (b.join(nb.rename({'lead_minutes': 'native_lead'}), on=['county', 'issue', 'native_lead'], how='left')
          .join(anchors, on=['county', 'issue'], how='left')
          .join(nb.filter(pl.col('lead_minutes') == 15).select('county', 'issue', pl.col('p').alias('p15')), on=['county', 'issue'], how='left'))
    b = b.with_columns(pl.when(pl.col('native_lead') > 0).then(pl.col('p')).otherwise(pl.col('anchor')).alias('p'))
    b = b.with_columns(pl.when(pl.col('p').is_finite()).then(pl.col('p')).otherwise(pl.col('p15')).alias('p'))
    out = pl.concat([a.select(requests.columns + ['p']), b.select(requests.columns + ['p'])]).sort('issue', 'row')
    if out.height != requests.height or out['p'].is_null().any():
        raise RuntimeError('a request has no native prediction')
    # model outputs are clipped to [0, 1]; only the observed proportion used at 5 and 10 minutes can fall outside
    bad = out.filter(~pl.col('p').is_between(0, 1) | ~pl.col('p').is_finite())
    if bad.height:
        r = bad.row(0, named=True)
        raise InputError(f'latest observed outage proportion {r["p"]:.4g} is outside [0, 1] for county {r["county"]} at '
                         f'issue {r["key_issue_time"]} (customers_out above the MCC denominator)')
    k, x0 = float(config['risk']['k']), {a: float(v) for a, v in config['risk']['x0'].items()}
    p = out['p'].to_list()
    return out.with_columns(pl.Series('predicted_x', [format(v, '.17g') for v in p]),
                            pl.Series('predicted_risk', [format(risk(v, x0[arch], k), '.17g') for v, arch in zip(p, out['arch'])]))


def write(out: pl.DataFrame, path: Path) -> None:
    frame = out.select(*[pl.col('key_' + c).alias(c) for c in COLUMNS[:5]], 'predicted_x', 'predicted_risk')
    frame.write_csv(path, line_terminator='\r\n', quote_style='necessary')
