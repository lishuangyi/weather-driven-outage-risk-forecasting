import shutil
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests' / 'fixtures' / 'mini_input'
EXPECTED = ROOT / 'tests' / 'fixtures' / 'expected_predictions.parquet'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'demo')]

KEYS = ['task_id', 'architecture_id', 'county_fips', 'issue_time', 'target_time']


def models_present() -> bool:
    return (ROOT / 'models' / 'task_a' / 'joint_tree.ubj').is_file()


requires_models = pytest.mark.skipif(not models_present(),
                                     reason='model files not downloaded (python scripts/download_artifacts.py models)')


def run_inference(input_dir: Path, output: Path) -> int:
    from outage_forecast.cli import main
    return main(['--input-dir', str(input_dir), '--output', str(output), '--device', 'cpu'])


def read_predictions(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=0).with_columns(pl.col('predicted_x', 'predicted_risk').cast(pl.Float64))


@pytest.fixture
def fixture_copy(tmp_path: Path) -> Path:
    """A writable copy of the mini input directory."""
    target = tmp_path / 'input'
    shutil.copytree(FIXTURE, target)
    return target


@pytest.fixture(scope='session')
def fixture_predictions(tmp_path_factory) -> pl.DataFrame:
    """Predictions of the frozen models on the unchanged mini input (computed once per session)."""
    if not models_present():
        pytest.skip('model files not downloaded')
    out = tmp_path_factory.mktemp('pred') / 'predictions.csv'
    assert run_inference(FIXTURE, out) == 0
    return read_predictions(out)


def to_polars(relation) -> pl.DataFrame:
    """DuckDB result as a polars frame without pyarrow (not a dependency of this project)."""
    return pl.DataFrame(relation.fetchall(), schema=relation.columns, orient='row', infer_schema_length=None)
