"""Warehouse build: no model files needed, atomic replacement, quality gate, issue-time cut-offs of weather and radar."""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from conftest import FIXTURE, ROOT, to_polars
from outage_forecast.config import Config
from outage_forecast.warehouse import QualityError, build

CUTOFF = datetime(2025, 7, 9, 0, 0, tzinfo=timezone.utc)
KEYS = ['task', 'county', 'issue_s', 'target_s']


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_early(input_dir):
    """Break the input: two outage records published before they were observed (a FAIL check)."""
    path = input_dir / 'eaglei' / 'eaglei_outages.parquet'
    o = pl.read_parquet(path)
    o.with_columns(pl.when(pl.int_range(pl.len()) < 2).then(pl.col('timestamp') - pl.duration(hours=1))
                   .otherwise(pl.col('source_record_time')).alias('source_record_time')).write_parquet(path)


def test_configuration_metadata_does_not_need_model_files(tmp_path):
    raw = json.loads((ROOT / 'configs' / 'inference.json').read_text())
    raw['task_a']['members']['joint_tree']['path'] = 'models/does_not_exist.ubj'
    path = tmp_path / 'inference.json'
    path.write_text(json.dumps(raw))
    assert Config.load(path, verify_models=False)['risk']['k'] == 4000.0
    with pytest.raises(RuntimeError, match='model file missing'):
        Config.load(path, verify_models=True)


def test_failed_build_keeps_the_existing_warehouse(fixture_copy, tmp_path):
    db = tmp_path / 'out' / 'warehouse.duckdb'
    build(FIXTURE, db).close()
    before = _sha(db)
    _publish_early(fixture_copy)
    with pytest.raises(QualityError, match='published before observed'):
        build(fixture_copy, db)
    assert _sha(db) == before
    assert sorted(p.name for p in db.parent.iterdir()) == ['warehouse.duckdb']     # no staged leftovers


def test_quality_gate_stops_before_gold(fixture_copy):
    _publish_early(fixture_copy)
    with pytest.raises(QualityError):
        build(fixture_copy)
    con = build(fixture_copy, on_fail='continue')             # inspection mode still builds everything
    assert con.execute('SELECT count(*) FROM gold.examples').fetchall()[0][0] > 0


def _examples(input_dir):
    return to_polars(build(input_dir, on_fail='continue').sql(
        'SELECT * EXCLUDE (truth_x, truth_published_s) FROM gold.examples')).sort(KEYS)


def _perturb(input_dir, table, time_column, columns, transform):
    path = input_dir / table
    t = pl.read_parquet(path)
    late = pl.col(time_column) > CUTOFF
    t.with_columns([pl.when(late).then(transform(pl.col(c))).otherwise(pl.col(c)).alias(c) for c in columns]).write_parquet(path)


@pytest.mark.parametrize('source', ['weather', 'radar'])
def test_context_features_ignore_data_published_after_the_issue(fixture_copy, source):
    if source == 'weather':
        cols = [c for c in pl.read_parquet(FIXTURE / 'weather/weather_forecasts.parquet').columns if '__' in c]
        _perturb(fixture_copy, 'weather/weather_forecasts.parquet', 'availability_time', cols, lambda c: c * 3 + 40)
        prefix = 'wx_'
    else:
        _perturb(fixture_copy, 'radar/mrms_county_series.parquet', 'availability_time', ['f35', 'f50', 'max_dbz', 'ring35'],
                 lambda c: c * 0 + 1)
        prefix = 'rx_'
    before, after = _examples(FIXTURE), _examples(fixture_copy)
    feats = [c for c in before.columns if c.startswith(prefix)]
    past = lambda f: f.filter(pl.col('issue_s') <= CUTOFF.timestamp()).select(KEYS + feats)  # noqa: E731
    future = lambda f: f.filter(pl.col('issue_s') > CUTOFF.timestamp()).select(KEYS + feats)  # noqa: E731
    assert past(before).height > 0
    assert past(before).equals(past(after)), f'{source} published after an issue changed its features'
    assert not future(before).equals(future(after)), 'positive control: later issues must see the change'


def test_radar_frame_from_after_the_issue_is_ignored(fixture_copy):
    """A malformed frame (frame time after the issue, availability before it) must not leak into that issue."""
    path = fixture_copy / 'radar' / 'mrms_county_series.parquet'
    r = pl.read_parquet(path)
    issue = datetime(2025, 7, 8, 12, 0, tzinfo=timezone.utc)
    bad = r.head(1).with_columns(pl.lit('48201').alias('county_fips'), pl.lit(issue + timedelta(hours=2)).alias('grid_time'),
                                 pl.lit(issue - timedelta(minutes=1)).alias('availability_time'),
                                 pl.lit(1.0).alias('f35'), pl.lit(1.0).alias('f50'), pl.lit(75.0).alias('max_dbz'))
    pl.concat([r, bad.cast(r.schema)]).write_parquet(path)
    q = f"SELECT * FROM gold.radar_as_of WHERE county = '48201' AND issue_s = {issue.timestamp()}"
    assert to_polars(build(FIXTURE).sql(q)).equals(to_polars(build(fixture_copy, on_fail='continue').sql(q)))
