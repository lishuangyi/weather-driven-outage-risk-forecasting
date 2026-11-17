"""DuckDB warehouse: point-in-time features, labels, quality checks and the challenge metric in SQL.

    python -m outage_forecast.warehouse build --input-dir data/inputs/input_2025 \
        --predictions out/predictions.csv --db out/warehouse.duckdb
    python -m outage_forecast.warehouse report --db out/warehouse.duckdb
    python -m outage_forecast.warehouse query sql/analysis/largest_events.sql

Layers (SQL files in sql/, executed in name order):
  bronze  views over the delivered Parquet/CSV files (nothing copied or changed)
  silver  typed tables with an explicit grain, every record revision kept; data-quality checks
  gold    point-in-time outage history per issue, the 30 history features, request labels, and with --predictions
          the scored cases (model and persistence baseline), metrics by dimension and the scoreboard
Reference tables (schema ref) come from configs/inference.json: architectures (x0, k) and the request grids.

The warehouse is a second, independent implementation next to the Python pipeline; tests/test_sql_parity.py checks
that both give the same features, labels and scores.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import duckdb

from .config import PACKAGE_ROOT, Config

SQL_DIR = PACKAGE_ROOT / 'sql'
TABLES = {'requests': 'requests/forecast_requests', 'mcc': 'mcc/MCC', 'outages': 'eaglei/eaglei_outages',
          'weather': 'weather/weather_forecasts', 'radar': 'radar/mrms_county_series'}
EMPTY = {  # schema of an optional table that is absent (radar) or a missing outage/weather file
    'outages': "SELECT NULL::VARCHAR AS county_fips, NULL::TIMESTAMPTZ AS \"timestamp\", NULL::DOUBLE AS customers_out, "
               "NULL::TIMESTAMPTZ AS source_record_time, NULL::BIGINT AS file_row_number WHERE false",
    'weather': "SELECT NULL::VARCHAR AS county_fips, NULL::TIMESTAMPTZ AS model_run_time, NULL::TIMESTAMPTZ AS availability_time, "
               "NULL::TIMESTAMPTZ AS target_time, NULL::BIGINT AS file_row_number WHERE false",
    'radar': "SELECT NULL::VARCHAR AS county_fips, NULL::TIMESTAMPTZ AS grid_time, NULL::TIMESTAMPTZ AS availability_time, "
             "NULL::DOUBLE AS avail, NULL::DOUBLE AS f35, NULL::DOUBLE AS f50, NULL::DOUBLE AS max_dbz, NULL::DOUBLE AS ring35, "
             "NULL::BIGINT AS file_row_number WHERE false"}


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def source(path: Path) -> str:
    """Table expression reading one file with its 0-based delivery row number."""
    if path.suffix == '.parquet':
        return f'read_parquet({_quote(path)}, file_row_number = true)'
    # CSV: every column as text (typing happens in silver); row number in file order
    return f'(SELECT *, row_number() OVER () - 1 AS file_row_number FROM read_csv({_quote(path)}, all_varchar = true))'


def locate(input_dir: Path, table: str) -> Path | None:
    """The table's file: CSV wins when both exist (as in the inference reader)."""
    for suffix in ('.csv', '.parquet'):
        path = input_dir / (TABLES[table] + suffix)
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def render(sql: str, values: dict[str, str]) -> str:
    out = re.sub(r'\{\{(\w+)\}\}', lambda m: values[m.group(1)], sql)
    return out


def run_layer(con: duckdb.DuckDBPyConnection, layer: str, values: dict[str, str]) -> list[str]:
    done = []
    for path in sorted((SQL_DIR / layer).glob('*.sql')):
        con.execute(render(path.read_text(encoding='utf-8'), values))
        done.append(f'{layer}/{path.name}')
    return done


def reference_tables(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    risk = config['risk']
    con.execute('CREATE OR REPLACE TABLE ref.architectures (arch VARCHAR PRIMARY KEY, x0 DOUBLE, k DOUBLE)')
    con.executemany('INSERT INTO ref.architectures VALUES (?, ?, ?)',
                    [(arch, float(x0), float(risk['k'])) for arch, x0 in risk['x0'].items()])
    con.execute('CREATE OR REPLACE TABLE ref.request_grid (task VARCHAR PRIMARY KEY, lead_min_minutes INTEGER, '
                'lead_max_minutes INTEGER, lead_step_minutes INTEGER, issue_hours_utc INTEGER[])')
    con.executemany('INSERT INTO ref.request_grid VALUES (?, ?, ?, ?, ?)',
                    [(task, g['lead_min_minutes'], g['lead_max_minutes'], g['lead_step_minutes'], g['issue_hours_utc'])
                     for task, g in config['requests'].items()])


def build(input_dir: Path, db: Path | str = ':memory:', predictions: Path | None = None, lookback_minutes: int = 4320,
          config: Config | None = None) -> duckdb.DuckDBPyConnection:
    """Build the warehouse for one input directory and return the open connection."""
    config = config or Config.load()
    input_dir = Path(input_dir).resolve()
    if db != ':memory:':
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        Path(db).unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    con.execute("SET TimeZone = 'UTC'")
    for schema in ('ref', 'bronze', 'silver', 'gold'):
        con.execute(f'CREATE SCHEMA IF NOT EXISTS {schema}')
    values = {'lookback_minutes': str(int(lookback_minutes))}
    for table in TABLES:
        path = locate(input_dir, table)
        if path is None and table in ('requests', 'mcc'):
            raise FileNotFoundError(f'{TABLES[table]}.csv or .parquet missing in {input_dir}')
        values[table] = source(path) if path else f'({EMPTY[table]})'
    reference_tables(con, config)
    run_layer(con, 'bronze', values)
    run_layer(con, 'silver', values)
    for path in sorted((SQL_DIR / 'gold').glob('*.sql')):
        if path.name >= '04' and predictions is None:   # scoring needs predictions
            break
        values['predictions'] = source(Path(predictions).resolve()) if predictions else ''
        con.execute(render(path.read_text(encoding='utf-8'), values))
    return con


def report(con: duckdb.DuckDBPyConnection) -> str:
    lines = ['Data-quality checks']
    for name, severity, violations, example, status in con.sql(
            'SELECT check_name, severity, violations, example, status FROM silver.quality_checks').fetchall():
        lines.append(f'  [{status:4s}] {name}: {violations:,}' + (f'  (e.g. {example})' if example else ''))
    tables = {t for (t,) in con.sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'gold'").fetchall()}
    if 'scoreboard' in tables:
        lines.append('\nScoreboard (pooled, challenge formula; local truth)')
        lines.append(f"  {'model':12s} {'Task A':>8s} {'Task B':>8s} {'general.':>9s} {'technical':>10s}")
        for model, a, b, g, t in con.sql('SELECT * FROM gold.scoreboard').fetchall():
            fmt = lambda v: f'{v:8.2f}' if v is not None else '       -'  # noqa: E731
            lines.append(f'  {model:12s} {fmt(a)} {fmt(b)} {fmt(g):>9s} {fmt(t):>10s}')
        lines.append('\nBy lead (model; diagnostic, not additive)')
        for task, grp, cases, rate, score in con.sql(
                "SELECT task, grp, cases, positive_rate, score FROM gold.metrics "
                "WHERE model = 'model' AND dimension = 'lead' ORDER BY task, grp").fetchall():
            lines.append(f'  {grp:14s} {score:6.2f}   cases {cases:>8,}   event rate {rate:6.1%}')
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='DuckDB warehouse of one input directory')
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='build (or rebuild) the warehouse file')
    b.add_argument('--input-dir', type=Path, required=True)
    b.add_argument('--db', type=Path, default=Path('out/warehouse.duckdb'))
    b.add_argument('--predictions', type=Path, help='predictions.csv to score (adds the scoring tables)')
    b.add_argument('--lookback-minutes', type=int, default=4320)
    r = sub.add_parser('report', help='print the quality checks and the scoreboard')
    r.add_argument('--db', type=Path, default=Path('out/warehouse.duckdb'))
    q = sub.add_parser('query', help='run a SQL file (e.g. sql/analysis/*.sql) against the warehouse')
    q.add_argument('sql_file', type=Path)
    q.add_argument('--db', type=Path, default=Path('out/warehouse.duckdb'))
    a = p.parse_args(argv)
    if a.command == 'query':
        with duckdb.connect(str(a.db), read_only=True, config={'TimeZone': 'UTC'}) as con:
            con.sql(a.sql_file.read_text(encoding='utf-8')).show(max_rows=60, max_width=160)
        return 0
    if a.command == 'build':
        con = build(a.input_dir, a.db, a.predictions, a.lookback_minutes)
    else:
        con = duckdb.connect(str(a.db), read_only=True, config={'TimeZone': 'UTC'})
    print(report(con))
    failed = con.execute("SELECT count(*) FROM silver.quality_checks WHERE status = 'FAIL'").fetchall()[0][0]
    con.close()
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
