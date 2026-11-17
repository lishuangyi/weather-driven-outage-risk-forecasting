#!/usr/bin/env python3
"""Materialize exact county-polygon HRRR robust spatial statistics.

The command reads local HRRR-Zarr chunk payloads, creates one exact
Census/TIGER polygon-to-cell plan, and streams independent NOAA variables to
staging long-form ZSTD Parquet. Successful runs compact that staging layout
into one file per variable plus an optional annual long-form file. It does not
merge outage labels, create temporal or cross-variable features, train models,
or modify the raw cache.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]  # data_pipeline/
sys.path.insert(0, str(ROOT))

from hrrr.polygons import (
    build_spatial_plan,
    coverage_frame,
    load_county_geometries,
    spatial_plan_frame,
)
from hrrr.stats import robust_spatial_stats
from hrrr.decode import (
    HrrrZarrError,
    decode_chunk_with_metadata,
    iso_utc,
    parse_utc,
)

UTC = dt.timezone.utc
SUCCESS_STATUSES = frozenset({"DOWNLOADED", "REUSED_EXISTING", "REPAIRED_EXISTING"})
MISSING_STATUS = "MISSING_SOURCE_OBJECT"
EXPECTED_LABELS = (
    "APCP_1hr_acc_fcst", "ASNOW_acc_fcst", "CAPE", "DPT", "GUST", "MSLMA",
    "PRATE", "PRES", "RH", "TCDC", "TMP", "UGRD", "VGRD", "VIS",
)
SPATIAL_METHOD = "EXACT_COUNTY_POLYGON_X_HRRR_CELL_AREA_WEIGHTED"
SOURCE_MODEL = "NOAA HRRR via public hrrrzarr compact Zarr archive"
CORE_VALID_AREA_THRESHOLD = 0.99
COMPACT_STORAGE_LAYOUT = "compact_14_plus_1_long_v1"
COMPACT_ROW_GROUP_SIZE = 131_072
VARIABLE_SORT_ORDER = ("fips_code", "model_run_utc", "forecast_hour")
ANNUAL_SORT_ORDER = ("variable", "fips_code", "model_run_utc", "forecast_hour")
IDENTITY_COLUMNS = ("fips_code", "variable", "model_run_utc", "forecast_hour")
OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "fips_code": pl.Utf8, "county": pl.Utf8, "state": pl.Utf8,
    "variable": pl.Utf8, "level": pl.Utf8, "units": pl.Utf8, "year": pl.Int32,
    "model_run_utc": pl.Utf8, "forecast_hour": pl.Int32, "valid_time_utc": pl.Utf8,
    "source_model": pl.Utf8, "spatial_method": pl.Utf8, "chunk_ids_used": pl.Utf8,
    "source_object_keys": pl.Utf8, "polygon_coverage_ratio": pl.Float64,
    "coverage_status": pl.Utf8, "mean": pl.Float64, "std": pl.Float64,
    "min": pl.Float64, "max": pl.Float64, "core_min": pl.Float64, "core_max": pl.Float64,
    "p05": pl.Float64, "p10": pl.Float64, "p25": pl.Float64, "p50": pl.Float64,
    "p75": pl.Float64, "p90": pl.Float64, "p95": pl.Float64, "p99": pl.Float64,
    "iqr": pl.Float64, "spatial_range": pl.Float64, "top05_mean": pl.Float64,
    "top10_mean": pl.Float64, "valid_area_ratio": pl.Float64,
    "intersecting_cell_count": pl.Int64, "valid_cell_count": pl.Int64,
    "effective_cell_count": pl.Float64, "boundary_area_ratio": pl.Float64,
    "quality_status": pl.Utf8, "quality_reason": pl.Utf8,
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_signature(cache_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(item for item in cache_root.rglob("*") if item.is_file()):
        stat = path.stat()
        files.append({"relative_path": path.relative_to(cache_root).as_posix(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        total_bytes += stat.st_size
    return {"root": str(cache_root.resolve()), "file_count": len(files), "total_bytes": total_bytes, "files": files}


def _collect_stream(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame with the current Polars streaming spelling."""

    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:  # pragma: no cover - compatibility with older Polars
        return lazy_frame.collect(streaming=True)


def _scan_paths(paths: Iterable[Path]) -> pl.LazyFrame:
    values = [str(path) for path in paths]
    if not values:
        raise ValueError("no Parquet paths were supplied")
    return pl.scan_parquet(values)


def _schema_payload(schema: Any) -> list[dict[str, str]]:
    return [{"name": str(name), "dtype": str(dtype)} for name, dtype in dict(schema).items()]


def schema_fingerprint(schema: Any) -> str:
    return hashlib.sha256(_json(_schema_payload(schema)).encode("utf-8")).hexdigest()


def _schema_for_path(path: Path) -> tuple[Any, list[dict[str, str]]]:
    schema = pl.read_parquet_schema(path)
    return schema, _schema_payload(schema)


def _finite_number(value: Any) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_columns(schema: Any) -> list[str]:
    numeric_type_names = {
        "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
        "Float32", "Float64", "Decimal",
    }
    return [str(name) for name, dtype in dict(schema).items() if str(dtype).split("(", 1)[0] in numeric_type_names]


def semantic_fingerprint(paths: Iterable[Path]) -> dict[str, Any]:
    """Build a bounded-memory semantic profile for one or more Parquet files."""

    path_list = sorted(Path(path) for path in paths)
    if not path_list:
        raise ValueError("cannot fingerprint an empty Parquet set")
    lazy_frame = _scan_paths(path_list)
    schema = lazy_frame.collect_schema()
    schema_names = [str(name) for name in dict(schema)]
    missing_identity = sorted(set(IDENTITY_COLUMNS) - set(schema_names))
    if missing_identity:
        raise ValueError(f"semantic fingerprint is missing identity columns: {missing_identity}")

    count_row = _collect_stream(
        lazy_frame.select(
            [
                pl.len().alias("rows"),
                pl.col("fips_code").n_unique().alias("unique_fips"),
                pl.col("model_run_utc").n_unique().alias("unique_runs"),
                pl.struct(list(IDENTITY_COLUMNS)).n_unique().alias("unique_identity_keys"),
            ]
        )
    ).to_dicts()[0]
    total_rows = int(count_row["rows"])
    unique_identity_keys = int(count_row["unique_identity_keys"])

    per_variable = _collect_stream(
        lazy_frame.group_by("variable").agg(pl.len().alias("rows"))
    ).sort("variable")
    per_variable_rows = {
        str(row["variable"]): int(row["rows"])
        for row in per_variable.to_dicts()
    }
    quality = _collect_stream(
        lazy_frame.group_by("quality_status").agg(pl.len().alias("rows"))
    ).sort("quality_status")
    quality_status_counts = {
        str(row["quality_status"]): int(row["rows"])
        for row in quality.to_dicts()
    }

    null_counts_frame = _collect_stream(
        lazy_frame.select([pl.col(name).null_count().alias(name) for name in schema_names])
    )
    null_counts = {
        name: int(null_counts_frame.get_column(name)[0])
        for name in schema_names
    }

    numeric_names = _numeric_columns(schema)
    numeric_stats: dict[str, dict[str, float | int | None]] = {}
    if numeric_names:
        numeric_row = _collect_stream(
            lazy_frame.select(
                [
                    expression
                    for name in numeric_names
                    for expression in (
                        pl.col(name).cast(pl.Float64).min().alias(f"{name}__min"),
                        pl.col(name).cast(pl.Float64).max().alias(f"{name}__max"),
                        pl.col(name).cast(pl.Float64).sum().alias(f"{name}__sum"),
                        pl.col(name).cast(pl.Float64).mean().alias(f"{name}__mean"),
                    )
                ]
            )
        ).to_dicts()[0]
        numeric_stats = {
            name: {
                "min": _finite_number(numeric_row[f"{name}__min"]),
                "max": _finite_number(numeric_row[f"{name}__max"]),
                "sum": _finite_number(numeric_row[f"{name}__sum"]),
                "mean": _finite_number(numeric_row[f"{name}__mean"]),
            }
            for name in numeric_names
        }

    unique_fips = _collect_stream(
        lazy_frame.select("fips_code").unique().sort("fips_code")
    ).get_column("fips_code").to_list()
    unique_runs = _collect_stream(
        lazy_frame.select("model_run_utc").unique().sort("model_run_utc")
    ).get_column("model_run_utc").to_list()
    sample_columns = [
        name for name in (*IDENTITY_COLUMNS, "quality_status", "mean", "std", "min", "max")
        if name in schema_names
    ]
    sample_rows = _collect_stream(
        lazy_frame.select(sample_columns).sort(list(IDENTITY_COLUMNS)).head(5)
    ).to_dicts()

    return {
        "paths": [path.as_posix() for path in path_list],
        "schema": _schema_payload(schema),
        "schema_fingerprint": schema_fingerprint(schema),
        "columns": len(schema_names),
        "rows": total_rows,
        "per_variable_rows": per_variable_rows,
        "unique_fips": [str(value) for value in unique_fips],
        "unique_runs": [str(value) for value in unique_runs],
        "unique_identity_keys": unique_identity_keys,
        "duplicate_identity_keys": total_rows - unique_identity_keys,
        "quality_status_counts": quality_status_counts,
        "null_counts": null_counts,
        "numeric_stats": numeric_stats,
        "sample_rows": sample_rows,
    }


def _numbers_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def _sample_rows_equal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right):
        if set(left_row) != set(right_row):
            return False
        for key in left_row:
            left_value, right_value = left_row[key], right_row[key]
            if isinstance(left_value, (float, int)) and isinstance(right_value, (float, int)):
                if not _numbers_equal(left_value, right_value):
                    return False
            elif left_value != right_value:
                return False
    return True


def compare_semantic_fingerprints(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare source and compact profiles without requiring Parquet byte equality."""

    differences: list[str] = []
    for field in (
        "schema", "schema_fingerprint", "columns", "rows", "per_variable_rows",
        "unique_fips", "unique_runs", "unique_identity_keys", "duplicate_identity_keys",
        "quality_status_counts", "null_counts",
    ):
        if left.get(field) != right.get(field):
            differences.append(field)
    left_numeric, right_numeric = left.get("numeric_stats", {}), right.get("numeric_stats", {})
    if set(left_numeric) != set(right_numeric):
        differences.append("numeric_stats.keys")
    else:
        for name in sorted(left_numeric):
            for statistic in ("min", "max", "sum", "mean"):
                if not _numbers_equal(left_numeric[name].get(statistic), right_numeric[name].get(statistic)):
                    differences.append(f"numeric_stats.{name}.{statistic}")
    if not _sample_rows_equal(left.get("sample_rows", []), right.get("sample_rows", [])):
        differences.append("sample_rows")
    return {
        "semantic_equal": not differences,
        "status": "PASS" if not differences else "FAIL",
        "differences": differences,
        "left_rows": int(left.get("rows", 0)),
        "right_rows": int(right.get("rows", 0)),
        "left_schema_fingerprint": left.get("schema_fingerprint"),
        "right_schema_fingerprint": right.get("schema_fingerprint"),
    }


def artifact_profile(
    path: Path,
    *,
    expected_variable: str | None = None,
    expected_variables: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Read lightweight structural metadata for one compact Parquet artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Parquet artifact is missing: {path}")
    schema, schema_payload = _schema_for_path(path)
    lazy_frame = pl.scan_parquet(str(path))
    rows = int(_collect_stream(lazy_frame.select(pl.len().alias("rows"))).item())
    values = _collect_stream(
        lazy_frame.select(pl.col("variable").unique()).sort("variable")
    ).get_column("variable").to_list()
    variable_values = [str(value) for value in values]
    expected_schema = _schema_payload(OUTPUT_SCHEMA)
    expected_set = set(str(value) for value in expected_variables) if expected_variables is not None else None
    valid = schema_payload == expected_schema
    if expected_variable is not None:
        valid = valid and variable_values == [expected_variable]
    if expected_set is not None:
        valid = valid and set(variable_values) == expected_set
    return {
        "path": path.as_posix(),
        "rows": rows,
        "columns": len(schema_payload),
        "schema": schema_payload,
        "schema_fingerprint": schema_fingerprint(schema),
        "variable_values": variable_values,
        "valid": valid,
    }


def artifact_record(
    path: Path,
    output_dir: Path,
    *,
    kind: str,
    variable: str | None,
    sort_order: Iterable[str],
    expected_rows: int | None = None,
    expected_variables: Iterable[str] | None = None,
    compression: str = "zstd",
) -> dict[str, Any]:
    profile = artifact_profile(path, expected_variable=variable, expected_variables=expected_variables)
    if expected_rows is not None and profile["rows"] != int(expected_rows):
        profile["valid"] = False
    if not profile["valid"]:
        raise ValueError(f"invalid compact artifact: {path}")
    digest = sha256_file(path)
    result = {
        "kind": kind,
        "variable": variable,
        "path": path.relative_to(output_dir).as_posix(),
        "rows": profile["rows"],
        "columns": profile["columns"],
        "bytes": path.stat().st_size,
        "row_group_size": COMPACT_ROW_GROUP_SIZE,
        "row_groups_expected": math.ceil(profile["rows"] / COMPACT_ROW_GROUP_SIZE),
        "sha256": digest,
        "SHA256": digest,
        "schema_fingerprint": profile["schema_fingerprint"],
        "compression": compression,
        "sort_order": list(sort_order),
        "variable_values": profile["variable_values"],
        "semantic_validation": "PASS",
    }
    return result


def _load_compaction_module() -> Any:
    """Load the package-local compaction implementation lazily."""

    return importlib.import_module("hrrr.compaction")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    _fsync_file(temporary)
    os.replace(temporary, path)


def _atomic_write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    frame.write_parquet(temporary, compression="zstd")
    _fsync_file(temporary)
    os.replace(temporary, path)


def _publish_staged_file(source: Path, target: Path, *, overwrite: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"validated staged artifact is missing: {source}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace an existing target without --overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _new_run_id(year: int) -> str:
    now = dt.datetime.now(tz=UTC)
    return f"{year}-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"


def _staging_path(output_dir: Path, run_id: str) -> Path:
    staging_root = (output_dir / ".staging").resolve()
    staging_dir = (staging_root / run_id).resolve()
    if staging_dir.parent != staging_root or output_dir.resolve() not in staging_dir.parents:
        raise ValueError(f"unsafe staging path: {staging_dir}")
    return staging_dir


def _cleanup_staging(staging_dir: Path, *, keep: bool) -> bool:
    if keep:
        return False
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_parent = staging_dir.parent
    if staging_parent.is_dir() and not any(staging_parent.iterdir()):
        staging_parent.rmdir()
    return not staging_dir.exists()


def _fragment_paths(root: Path, variables: Iterable[str] | None = None) -> list[Path]:
    selected = set(variables) if variables is not None else None
    paths = []
    for path in sorted(root.glob("variable=*/*.parquet")):
        label = path.parent.name.removeprefix("variable=")
        if selected is None or label in selected:
            paths.append(path)
    return paths


def _fragment_labels(root: Path) -> list[str]:
    return sorted({path.parent.name.removeprefix("variable=") for path in _fragment_paths(root)})


def _stage_fragment_links(source_parts: Iterable[Path], source_root: Path, staging_dir: Path) -> None:
    """Expose existing processed parts in staging without copying or rereading raw data."""

    source_root = source_root.resolve()
    for source in source_parts:
        source = source.resolve()
        try:
            relative = source.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"source fragment is outside the declared root: {source}") from exc
        staged = staging_dir / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        if staged.exists() or staged.is_symlink():
            raise FileExistsError(f"staging source link already exists: {staged}")
        staged.symlink_to(source)


def _run_compaction_bundle(
    *,
    staging_dir: Path,
    annual_stage: Path,
    variables: list[str],
    expected_rows_by_variable: dict[str, int] | None,
) -> dict[str, Any]:
    """Call the package-local all-in-one compaction API on the staging root."""

    module = _load_compaction_module()
    function = getattr(module, "compact_existing", None)
    if not callable(function):
        raise RuntimeError(
            "hrrr.compaction must expose public compact_existing(root, ...); "
            f"available={sorted(name for name in dir(module) if not name.startswith('_'))}"
        )
    result = function(
        staging_dir,
        expected_labels=variables,
        expected_row_counts=expected_rows_by_variable,
        expected_schema=OUTPUT_SCHEMA,
        annual_path=annual_stage,
        force=False,
        row_group_size=COMPACT_ROW_GROUP_SIZE,
        include_row_hash=True,
    )
    if not isinstance(result, dict):
        raise TypeError("hrrr_compaction.compact_existing must return a dictionary")
    return result


def _validated_existing_variable(
    path: Path,
    *,
    variable: str,
    expected_rows: int | None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        profile = artifact_profile(path, expected_variable=variable)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None
    if not profile["valid"]:
        return None
    if expected_rows is not None and int(profile["rows"]) != int(expected_rows):
        return None
    return profile


def _compact_and_publish(
    *,
    output_dir: Path,
    year: int,
    source_root: Path,
    staging_dir: Path,
    variables: list[str],
    overwrite: bool,
    no_annual_file: bool,
    source_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compact validated staging/source parts and atomically publish targets."""

    final_data_dir = output_dir / "data"
    work_data_dir = staging_dir / "data"
    final_data_dir.mkdir(parents=True, exist_ok=True)
    work_data_dir.mkdir(parents=True, exist_ok=True)
    variables = list(dict.fromkeys(sorted(variables)))
    if not variables:
        raise ValueError("cannot compact an empty variable list")

    expected_rows_by_variable = (source_profile or {}).get("per_variable_rows", {})
    source_paths_by_variable = {
        variable: _fragment_paths(source_root, [variable])
        for variable in variables
    }
    to_build: list[str] = []
    reused_variables: list[str] = []
    for variable in variables:
        target = final_data_dir / f"{variable}.parquet"
        expected_rows = expected_rows_by_variable.get(variable)
        existing = _validated_existing_variable(target, variable=variable, expected_rows=expected_rows)
        if existing is None:
            if target.exists() and not overwrite:
                raise ValueError(
                    f"existing compact target is invalid or incompatible; pass --overwrite to rebuild: {target}"
                )
            to_build.append(variable)
            continue
        source_paths = source_paths_by_variable[variable]
        if source_paths:
            source_variable_profile = semantic_fingerprint(source_paths)
            target_variable_profile = semantic_fingerprint([target])
            equality = compare_semantic_fingerprints(source_variable_profile, target_variable_profile)
            if not equality["semantic_equal"]:
                if not overwrite:
                    raise ValueError(
                        f"existing compact target is not semantically equal to source fragments; "
                        f"pass --overwrite to rebuild: {target}"
                    )
                to_build.append(variable)
                continue
        reused_variables.append(variable)

    # The compaction API operates on one root.  Valid final variable artifacts are
    # represented there as symlinks, so compact_existing can validate/reuse
    # them without replacing the user's existing files.  Invalid or missing
    # targets are left absent and are written into staging atomically.
    for variable in reused_variables:
        target = final_data_dir / f"{variable}.parquet"
        staged = work_data_dir / f"{variable}.parquet"
        staged.symlink_to(target.resolve())

    annual_path = output_dir / f"weather_{year}_full_polygon_robust.parquet"
    annual_stage = staging_dir / annual_path.name
    api_result = _run_compaction_bundle(
        staging_dir=staging_dir,
        annual_stage=annual_stage,
        variables=variables,
        expected_rows_by_variable=(
            {str(key): int(value) for key, value in expected_rows_by_variable.items()}
            if expected_rows_by_variable
            else None
        ),
    )

    for variable in variables:
        staged = work_data_dir / f"{variable}.parquet"
        expected_rows = expected_rows_by_variable.get(variable)
        artifact_record(
            staged,
            staging_dir,
            kind="variable",
            variable=variable,
            sort_order=VARIABLE_SORT_ORDER,
            expected_rows=expected_rows,
        )
        if variable in to_build:
            _publish_staged_file(staged, final_data_dir / f"{variable}.parquet", overwrite=overwrite)

    variable_paths = [final_data_dir / f"{variable}.parquet" for variable in variables]
    variable_records = [
        artifact_record(
            path,
            output_dir,
            kind="variable",
            variable=variable,
            sort_order=VARIABLE_SORT_ORDER,
            expected_rows=expected_rows_by_variable.get(variable),
        )
        for variable, path in zip(variables, variable_paths)
    ]
    compact_profile = semantic_fingerprint(variable_paths)
    source_equality = None
    if source_profile is not None:
        source_equality = compare_semantic_fingerprints(source_profile, compact_profile)
        if not source_equality["semantic_equal"]:
            raise ValueError(
                "fragmented source and variable-level compact outputs are not semantically equal: "
                f"{source_equality['differences']}"
            )

    annual_record: dict[str, Any] | None = None
    annual_profile: dict[str, Any] | None = None
    annual_equality: dict[str, Any] | None = None
    if not no_annual_file:
        reuse_existing_annual = False
        annual_profile = semantic_fingerprint([annual_stage])
        annual_equality = compare_semantic_fingerprints(compact_profile, annual_profile)
        if not annual_equality["semantic_equal"]:
            raise ValueError(
                "annual artifact is not semantically equal to validated variable artifacts: "
                f"{annual_equality['differences']}"
            )
        existing_annual_profile: dict[str, Any] | None = None
        if annual_path.is_file():
            try:
                existing_candidate = artifact_profile(annual_path, expected_variables=variables)
                if existing_candidate["valid"] and int(existing_candidate["rows"]) == int(compact_profile["rows"]):
                    existing_annual_profile = semantic_fingerprint([annual_path])
                    existing_equality = compare_semantic_fingerprints(compact_profile, existing_annual_profile)
                    if not existing_equality["semantic_equal"] and not overwrite:
                        raise ValueError(
                            f"existing annual artifact is not semantically equal; pass --overwrite: {annual_path}"
                        )
                    if existing_equality["semantic_equal"]:
                        annual_profile = existing_annual_profile
                        annual_equality = existing_equality
                        reuse_existing_annual = True
                elif not overwrite:
                    raise ValueError(
                        f"existing annual artifact is invalid or incompatible; pass --overwrite: {annual_path}"
                    )
            except (OSError, ValueError, pl.exceptions.PolarsError):
                if not overwrite:
                    raise
        if not reuse_existing_annual:
            _publish_staged_file(annual_stage, annual_path, overwrite=overwrite)
            annual_api_result = {"action": "written_from_validated_staging", "api": api_result}
        else:
            annual_stage.unlink(missing_ok=True)
            annual_api_result = {"action": "reused_existing_valid", "api": api_result}
        annual_record = artifact_record(
            annual_path,
            output_dir,
            kind="annual",
            variable=None,
            sort_order=ANNUAL_SORT_ORDER,
            expected_rows=int(compact_profile["rows"]),
            expected_variables=variables,
        )
    else:
        annual_profile = semantic_fingerprint([annual_stage])
        annual_equality = compare_semantic_fingerprints(compact_profile, annual_profile)
        annual_stage.unlink(missing_ok=True)
        annual_api_result = {"action": "skipped_final_annual", "api": api_result}

    return {
        "variables": variables,
        "reused_variables": reused_variables,
        "rebuilt_variables": to_build,
        "compaction_api": api_result,
        "variable_artifacts": variable_records,
        "annual_artifact": annual_record,
        "annual_api": annual_api_result,
        "source_profile": source_profile,
        "compact_profile": compact_profile,
        "annual_profile": annual_profile,
        "semantic_equality": {
            "source_to_variables": source_equality,
            "variables_to_annual": annual_equality,
            "semantic_equal": bool(
                (source_equality is None or source_equality["semantic_equal"])
                and (no_annual_file or (annual_equality is not None and annual_equality["semantic_equal"]))
            ),
        },
    }


def _profile_without_paths(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {key: value for key, value in profile.items() if key != "paths"}


def _output_file_hashes(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    staging_root = (output_dir / ".staging").resolve()
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if staging_root in path.parents:
            continue
        resolved = path.resolve()
        if resolved == (output_dir / "manifest.json").resolve() or resolved == (output_dir / "audit.json").resolve():
            continue
        if staging_root in resolved.parents:
            continue
        records.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _finalize_output(
    *,
    output_dir: Path,
    year: int,
    summary: dict[str, Any],
    started_at: dt.datetime,
    started: float,
    raw_before: dict[str, Any] | None,
    raw_after: dict[str, Any] | None,
    staging_dir: Path,
    keep_staging: bool,
    source_parts: list[Path],
    compaction: dict[str, Any],
    sidecar_frames: dict[str, pl.DataFrame] | None,
    extra_audit: dict[str, Any],
) -> dict[str, Any]:
    source_bytes = sum(path.stat().st_size for path in source_parts if path.is_file())
    if sidecar_frames:
        for name, frame in sidecar_frames.items():
            _atomic_write_frame(frame, output_dir / name)
    if raw_before is not None:
        _atomic_write_text(output_dir / "raw_manifest_before.json", _json(raw_before))
    if raw_after is not None:
        _atomic_write_text(output_dir / "raw_manifest_after.json", _json(raw_after))

    staging_deleted = _cleanup_staging(staging_dir, keep=keep_staging)
    source_profile = compaction.get("source_profile")
    compact_profile = compaction.get("compact_profile")
    annual_profile = compaction.get("annual_profile")
    source_quality = (source_profile or {}).get("quality_status_counts", {})
    source_has_quality_findings = any(
        status != "VALID" and int(count) > 0
        for status, count in source_quality.items()
    )
    raw_unchanged = None if raw_before is None or raw_after is None else raw_before == raw_after
    semantic_equal = bool(compaction["semantic_equality"]["semantic_equal"])
    decode_failure_count = int(extra_audit.get("decode_failure_count", 0))
    status = (
        "SUCCESS"
        if raw_unchanged is True and semantic_equal and decode_failure_count == 0 and not source_has_quality_findings
        else "COMPLETED_WITH_AUDIT_FINDINGS"
    )
    finished_at = dt.datetime.now(tz=UTC)
    source_rows = int((source_profile or {}).get("rows", 0))
    compact_rows = int((compact_profile or {}).get("rows", 0))
    artifacts = list(compaction.get("variable_artifacts", []))
    annual_artifact = compaction.get("annual_artifact")
    if annual_artifact is not None:
        artifacts.append(annual_artifact)
    audit = {
        "status": status,
        "schema_version": "hrrr-full-polygon-robust-v2",
        "storage_layout": COMPACT_STORAGE_LAYOUT,
        "generated_at": iso_utc(finished_at),
        "generated_at_utc": iso_utc(finished_at),
        "started_at_utc": iso_utc(started_at),
        "finished_at_utc": iso_utc(finished_at),
        "elapsed_seconds": time.monotonic() - started,
        "raw_input_modified": None if raw_unchanged is None else not raw_unchanged,
        "raw_immutability_verified": raw_unchanged is not None,
        "raw_before_file_count": None if raw_before is None else raw_before["file_count"],
        "raw_after_file_count": None if raw_after is None else raw_after["file_count"],
        "raw_before_total_bytes": None if raw_before is None else raw_before["total_bytes"],
        "raw_after_total_bytes": None if raw_after is None else raw_after["total_bytes"],
        "source_fragment_count": len(source_parts),
        "source_fragment_rows": source_rows,
        "source_fragment_bytes": source_bytes,
        "source_fragments_retained": bool(source_parts and all(path.exists() for path in source_parts)),
        "source_quality_status_counts": source_quality,
        "compact_rows": compact_rows,
        "row_counts_by_variable": (compact_profile or {}).get("per_variable_rows", {}),
        "partition_counts_by_variable": dict(
            Counter(path.parent.name.removeprefix("variable=") for path in source_parts)
        ),
        "decoded_object_count": extra_audit.get("decoded_object_count", 0),
        "decode_failure_count": decode_failure_count,
        "decode_failure_samples": extra_audit.get("decode_failure_samples", []),
        "missing_source_by_variable": extra_audit.get("missing_source_by_variable", {}),
        "reused_variables": compaction.get("reused_variables", []),
        "rebuilt_variables": compaction.get("rebuilt_variables", []),
        "staging_path": staging_dir.relative_to(output_dir).as_posix(),
        "staging_deleted": staging_deleted,
        "keep_staging": keep_staging,
        "semantic_equality": compaction["semantic_equality"],
        "artifacts": artifacts,
        "threshold_coverage": {
            "enabled": False,
            "reason": "No threshold features configured; metadata units/semantics are not uniformly explicit",
        },
        **extra_audit,
    }
    manifest = {
        "schema_version": "hrrr-full-polygon-robust-v2",
        "storage_layout": COMPACT_STORAGE_LAYOUT,
        "generated_at": iso_utc(finished_at),
        "generated_at_utc": iso_utc(finished_at),
        "created_at_utc": iso_utc(finished_at),
        "scope": "raw HRRR to exact county polygon robust aggregation",
        "input": summary,
        "output_schema": {name: str(dtype) for name, dtype in OUTPUT_SCHEMA.items()},
        "source": {
            "layout": "fragmented_variable_parts",
            "fragment_count": len(source_parts),
            "rows": source_rows,
            "bytes": source_bytes,
            "semantic_fingerprint": _profile_without_paths(source_profile),
        },
        "artifacts": artifacts,
        "variable_artifacts": list(compaction.get("variable_artifacts", [])),
        "annual_artifact": annual_artifact,
        "semantic_equality": compaction["semantic_equality"],
        "staging": {
            "path": staging_dir.relative_to(output_dir).as_posix(),
            "deleted": staging_deleted,
            "keep_staging": keep_staging,
        },
        "files": _output_file_hashes(output_dir),
        "audit_file": "audit.json",
        "raw_manifest_before": "raw_manifest_before.json" if raw_before is not None else None,
        "raw_manifest_after": "raw_manifest_after.json" if raw_after is not None else None,
    }
    _atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(output_dir / "audit.json", json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {
        "status": status,
        **summary,
        **audit,
        "output_dir": str(output_dir),
        "total_rows": compact_rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL input is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
                rows.append(value)
    return rows


def _latest_requests(cache_root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(cache_root / "download_objects.jsonl"):
        request_id = str(row.get("request_id", ""))
        if request_id:
            latest[request_id] = row
    if not latest:
        raise ValueError("download_objects.jsonl contains no request rows")
    return latest


def _metadata_inventory(cache_root: Path, year: int) -> dict[str, dict[str, Any]]:
    root = cache_root / "metadata" / str(year)
    if not root.is_dir():
        raise FileNotFoundError(f"metadata root is missing: {root}")
    result: dict[str, dict[str, Any]] = {}
    for zarray_path in sorted(root.glob("*/*/.zarray")):
        attrs_path = zarray_path.with_name(".zattrs")
        array = json.loads(zarray_path.read_text(encoding="utf-8"))
        attrs = json.loads(attrs_path.read_text(encoding="utf-8")) if attrs_path.is_file() else {}
        variable, level = zarray_path.parent.name, zarray_path.parent.parent.name
        result[variable] = {
            "variable": variable, "level": level, "units": attrs.get("units"),
            "long_name": attrs.get("long_name"), "shape": list(array.get("shape", [])),
            "chunks": list(array.get("chunks", [])), "dtype": array.get("dtype"),
            "fill_value": array.get("fill_value"),
            "metadata_path": zarray_path.relative_to(cache_root).as_posix(),
            "attrs_path": attrs_path.relative_to(cache_root).as_posix(),
        }
    if not result:
        raise ValueError(f"no Zarr metadata was found under {root}")
    return result


def _object_inventory(
    latest: dict[str, dict[str, Any]], cache_root: Path, metadata: dict[str, dict[str, Any]]
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]], list[str]]:
    objects: dict[tuple[str, str, str], dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for raw in latest.values():
        if raw.get("kind") != "data_chunk":
            continue
        symbol, run, chunk = str(raw.get("symbol", "")), str(raw.get("model_run_utc", "")), str(raw.get("chunk_id", ""))
        if not symbol or not run or not chunk:
            continue
        run = iso_utc(parse_utc(run))
        status = str(raw.get("status", ""))
        if status == MISSING_STATUS:
            missing.append(raw)
            continue
        if status not in SUCCESS_STATUSES:
            continue
        row = dict(raw)
        relative = row.get("local_relative_path")
        local_path = Path(str(relative)) if relative else Path("")
        row["local_path"] = str(local_path if local_path.is_absolute() else cache_root / local_path)
        objects[(run, symbol, chunk)] = row
    labels = sorted(set(metadata) | {key[1] for key in objects} | {str(row.get("symbol")) for row in missing})
    return objects, missing, labels


def build_inventory_frame(
    metadata: dict[str, dict[str, Any]], objects: dict[tuple[str, str, str], dict[str, Any]],
    missing: list[dict[str, Any]], runs: list[str]
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in sorted(metadata):
        spec = metadata[label]
        success_runs = sorted({run for run, variable, _ in objects if variable == label})
        success_count = sum(1 for _, variable, _ in objects if variable == label)
        missing_rows = [row for row in missing if str(row.get("symbol")) == label]
        rows.append({
            "variable": label, "level": spec.get("level"), "units": spec.get("units"),
            "long_name": spec.get("long_name"), "metadata_path": spec.get("metadata_path"),
            "shape": _json(spec.get("shape")), "chunks": _json(spec.get("chunks")),
            "dtype": spec.get("dtype"), "fill_value": spec.get("fill_value"),
            "run_count_expected": len(runs), "run_count_with_success": len(success_runs),
            "object_count_success": success_count,
            "chunk_ids_success": _json(sorted({chunk for _, variable, chunk in objects if variable == label})),
            "missing_object_count": len(missing_rows),
            "missing_runs": _json(sorted({str(row.get("model_run_utc")) for row in missing_rows})),
            "forecast_step_count": int(spec["shape"][0]) if spec.get("shape") else None,
            "native_time_axis": "forecast_period_2" if label == "ASNOW_acc_fcst" else "forecast_period",
            "source_status": "PARTIAL_MISSING_SOURCE_OBJECT" if missing_rows else "AVAILABLE",
        })
    return pl.DataFrame(rows)


def selection_rows(selection_path: Path, max_counties: int | None) -> list[dict[str, Any]]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    raw_rows = payload.get("counties") or payload.get("selected_counties") or []
    if raw_rows:
        rows = [{"fips_code": str(row.get("fips_code")).zfill(5), "county": str(row.get("county", "")), "state": str(row.get("state", ""))} for row in raw_rows]
    else:
        rows = [{"fips_code": str(value).zfill(5), "county": "", "state": ""} for value in payload.get("fips_codes", [])]
    if not rows or len({row["fips_code"] for row in rows}) != len(rows):
        raise ValueError(f"county selection is empty or duplicated: {selection_path}")
    return rows[:max_counties] if max_counties is not None else rows


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def output_guard(cache_root: Path, output_dir: Path, overwrite: bool, *, allow_nonempty: bool = False) -> None:
    cache, output = cache_root.resolve(), output_dir.resolve()
    if output == cache or cache in output.parents:
        raise ValueError(f"output_dir must be outside cache_root: output={output}, cache={cache}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output_dir is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        if allow_nonempty:
            return
        if not overwrite:
            raise FileExistsError(f"output_dir is non-empty; pass --overwrite explicitly: {output}")
        # Normal overwrite is deliberately non-destructive.  Existing valid
        # compact targets, sidecars, legacy reports, and stale staging runs are
        # preserved until each replacement has passed validation and is ready
        # for an atomic os.replace().


def _make_schema_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=OUTPUT_SCHEMA, strict=False) if rows else pl.DataFrame(schema=OUTPUT_SCHEMA)


def _base_row(*, county: dict[str, Any], variable: str, spec: dict[str, Any], year: int, run: str, forecast_hour: int, chunks: list[str], object_keys: list[str], coverage: dict[str, Any], status: str, reason: str, valid_area_ratio: float, valid_cell_count: int, effective_cell_count: float | None) -> dict[str, Any]:
    return {
        "fips_code": county["fips_code"], "county": county["county"], "state": county["state"], "variable": variable,
        "level": spec.get("level"), "units": spec.get("units"), "year": year, "model_run_utc": run,
        "forecast_hour": forecast_hour, "valid_time_utc": iso_utc(parse_utc(run) + dt.timedelta(hours=forecast_hour)),
        "source_model": SOURCE_MODEL, "spatial_method": SPATIAL_METHOD, "chunk_ids_used": _json(chunks),
        "source_object_keys": _json(object_keys), "polygon_coverage_ratio": coverage["polygon_coverage_ratio"],
        "coverage_status": coverage["coverage_status"], "mean": None, "std": None, "min": None, "max": None,
        "core_min": None, "core_max": None, "p05": None, "p10": None, "p25": None, "p50": None, "p75": None,
        "p90": None, "p95": None, "p99": None, "iqr": None, "spatial_range": None, "top05_mean": None,
        "top10_mean": None, "valid_area_ratio": valid_area_ratio, "intersecting_cell_count": coverage["intersecting_cell_count"],
        "valid_cell_count": valid_cell_count, "effective_cell_count": effective_cell_count,
        "boundary_area_ratio": coverage["boundary_area_ratio"], "quality_status": status, "quality_reason": reason,
    }


def aggregate_row(*, county: dict[str, Any], variable: str, spec: dict[str, Any], year: int, run: str, forecast_hour: int, county_cells: dict[str, list[dict[str, Any]]], arrays: dict[str, np.ndarray], fill_value: float | None, coverage: dict[str, Any], object_rows: dict[str, dict[str, Any]], missing_chunks: list[str], decode_errors: dict[str, str]) -> dict[str, Any]:
    values: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    overlaps: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    for chunk_id in sorted(county_cells):
        rows = county_cells[chunk_id]
        flat = np.asarray([row["flat_index"] for row in rows], dtype=np.int64)
        values.append(arrays[chunk_id][forecast_hour - 1].reshape(-1)[flat].astype(np.float64, copy=False) if chunk_id in arrays else np.full(flat.size, np.nan))
        areas.append(np.asarray([row["intersection_area_m2"] for row in rows], dtype=np.float64))
        centers.append(np.asarray([row["center_inside_polygon"] for row in rows], dtype=bool))
        overlaps.append(np.asarray([row["cell_overlap_fraction"] for row in rows], dtype=np.float64))
        boundaries.append(np.asarray([row["boundary_cell"] for row in rows], dtype=bool))
    joined_values = np.concatenate(values) if values else np.array([], dtype=np.float64)
    joined_areas = np.concatenate(areas) if areas else np.array([], dtype=np.float64)
    valid = np.isfinite(joined_values)
    if fill_value is not None and np.isfinite(fill_value):
        valid &= joined_values != float(fill_value)
    stats = robust_spatial_stats(
        joined_values, joined_areas, center_inside_polygon=np.concatenate(centers),
        cell_overlap_fraction=np.concatenate(overlaps), boundary_cell=np.concatenate(boundaries),
        valid_mask=valid, intersecting_cell_count=int(joined_values.size),
    )
    chunks = sorted(arrays)
    object_keys = sorted(str(object_rows[chunk].get("object_key", "")) for chunk in chunks)
    status, reason = "VALID", "all planned intersection area represented by valid source cells"
    if decode_errors:
        status, reason = "DECODE_ERROR", _json(decode_errors)
    elif missing_chunks:
        status, reason = "PARTIAL_SOURCE_OBJECTS", f"missing source chunks: {','.join(sorted(missing_chunks))}"
    if stats["valid_cell_count"] == 0:
        if decode_errors:
            status, reason = "DECODE_ERROR", _json(decode_errors)
        elif not arrays:
            status, reason = MISSING_STATUS, "no source object was available for the planned chunks"
        else:
            status, reason = "NO_VALID_CELLS", "no finite non-fill source values represented"
    elif float(stats["valid_area_ratio"]) < CORE_VALID_AREA_THRESHOLD and status == "VALID":
        status, reason = "PARTIAL_VALID_AREA", f"valid_area_ratio={stats['valid_area_ratio']:.6f} below {CORE_VALID_AREA_THRESHOLD}"
    row = _base_row(
        county=county, variable=variable, spec=spec, year=year, run=run, forecast_hour=forecast_hour,
        chunks=chunks, object_keys=object_keys, coverage=coverage, status=status, reason=reason,
        valid_area_ratio=float(stats["valid_area_ratio"]), valid_cell_count=int(stats["valid_cell_count"]),
        effective_cell_count=stats["effective_cell_count"],
    )
    row.update({
        "mean": stats["area_mean"], "std": stats["area_std"], "min": stats["raw_min"], "max": stats["raw_max"],
        "core_min": stats["core_min"], "core_max": stats["core_max"], "p05": stats["weighted_p05"], "p10": stats["weighted_p10"],
        "p25": stats["weighted_p25"], "p50": stats["weighted_p50"], "p75": stats["weighted_p75"], "p90": stats["weighted_p90"],
        "p95": stats["weighted_p95"], "p99": stats["weighted_p99"], "iqr": stats["iqr"], "spatial_range": stats["spatial_range"],
        "top05_mean": stats["top05_area_mean"], "top10_mean": stats["top10_area_mean"], "boundary_area_ratio": stats["boundary_area_ratio"],
    })
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--county-archive", type=Path, default=ROOT / "data/training/geography/tiger2025/county/tl_2025_us_county.zip")
    parser.add_argument("--counties", type=str, default=None, help="selection JSON path or comma-separated FIPS")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--variables", type=str, default=None, help="comma-separated raw NOAA labels")
    parser.add_argument("--chunk-id", type=str, default=None, help="comma-separated Zarr chunk ids")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-counties", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compact-existing",
        action="store_true",
        help="compact existing processed variable=*/part-*.parquet files without reading raw HRRR arrays",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="retain .staging/<run-id> after a successful workflow for debugging",
    )
    parser.add_argument(
        "--no-annual-file",
        action="store_true",
        help="advanced: write validated variable files but skip the annual long-form artifact",
    )
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = (args.cache_root or ROOT / f"data/training/weather/hrrr_fips20_{args.year}").resolve()
    selection_path = cache_root / "selection_manifest.json"
    if args.counties and Path(args.counties).is_file():
        selection_path = Path(args.counties).resolve()
    if args.counties and not Path(args.counties).is_file():
        counties = [{"fips_code": value.zfill(5), "county": "", "state": ""} for value in parse_csv(args.counties) or []]
    else:
        counties = selection_rows(selection_path, args.max_counties)
    if args.smoke:
        counties = counties[:2]
    if args.max_counties is not None:
        counties = counties[:args.max_counties]
    metadata = _metadata_inventory(cache_root, args.year)
    latest = _latest_requests(cache_root)
    objects, missing, labels = _object_inventory(latest, cache_root, metadata)
    # The package downloader stores multiple calendar years in one cache.  A
    # materialization request must never silently mix those model runs into an
    # annual output merely because they share a cache root.
    year_prefix = f"{args.year:04d}-"
    objects = {key: value for key, value in objects.items() if key[0].startswith(year_prefix)}
    missing = [row for row in missing if str(row.get("model_run_utc", "")).startswith(year_prefix)]
    labels = sorted(set(metadata) | {key[1] for key in objects} | {str(row.get("symbol")) for row in missing})
    variables = list(dict.fromkeys(parse_csv(args.variables) or labels))
    unknown = sorted(set(variables) - set(metadata))
    if unknown:
        raise ValueError(f"requested variables have no local metadata: {unknown}")
    selection_payload = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.is_file()
        else {}
    )
    chunks = parse_csv(args.chunk_id) or [str(value) for value in selection_payload.get("chunk_ids", [])]
    if not chunks:
        chunks = sorted({key[2] for key in objects})
    all_runs = sorted({key[0] for key in objects} | {iso_utc(parse_utc(str(row["model_run_utc"]))) for row in missing})
    if not all_runs:
        raise ValueError(f"no downloaded HRRR objects are registered for year={args.year}")
    runs = all_runs[:args.max_runs] if args.max_runs is not None else all_runs
    if args.smoke:
        runs = runs[:1]
    output_dir = (args.output_dir or ROOT / f"data/processed/weather/hrrr_full_polygon_robust/{args.year}").resolve()
    return {"cache_root": cache_root, "selection_path": selection_path, "counties": counties, "output_dir": output_dir, "metadata": metadata, "latest": latest, "objects": objects, "missing": missing, "variables": variables, "chunk_ids": chunks, "runs": runs, "all_runs": all_runs}


def _compact_existing(args: argparse.Namespace) -> dict[str, Any]:
    started_at, started = dt.datetime.now(tz=UTC), time.monotonic()
    output_dir = (args.output_dir or ROOT / f"data/processed/weather/hrrr_full_polygon_robust/{args.year}").resolve()
    cache_root = (args.cache_root or ROOT / f"data/training/weather/hrrr_fips20_{args.year}").resolve()
    output_guard(cache_root, output_dir, args.overwrite, allow_nonempty=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = parse_csv(args.variables)
    source_labels = _fragment_labels(output_dir)
    existing_labels = sorted(path.stem for path in (output_dir / "data").glob("*.parquet")) if (output_dir / "data").is_dir() else []
    variables = list(dict.fromkeys(requested or sorted(set(source_labels) | set(existing_labels))))
    if not variables:
        raise FileNotFoundError(f"no fragmented parts or compact variable files found under {output_dir}")
    source_parts = _fragment_paths(output_dir, variables)
    existing_profiles = {
        variable: _validated_existing_variable(
            output_dir / "data" / f"{variable}.parquet",
            variable=variable,
            expected_rows=None,
        )
        for variable in variables
    }
    summary = {
        "year": args.year,
        "cache_root": str(cache_root),
        "output_dir": str(output_dir),
        "variables": variables,
        "source_fragment_count": len(source_parts),
        "source_fragment_bytes": sum(path.stat().st_size for path in source_parts),
        "existing_valid_variables": sorted(variable for variable, profile in existing_profiles.items() if profile is not None),
        "existing_missing_or_invalid_variables": sorted(variable for variable, profile in existing_profiles.items() if profile is None),
        "compact_existing": True,
        "raw_arrays_reread": False,
        "dry_run": bool(args.dry_run),
        "smoke": bool(args.smoke),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return {"status": "DRY_RUN_OK", **summary, "elapsed_seconds": time.monotonic() - started}

    # A finalized output intentionally has no source fragments.  In that
    # state --compact-existing is still useful as a read-only integrity check
    # and must reuse the already validated 14+1 artifacts instead of trying to
    # rebuild from a source layout that was deliberately removed.
    if not source_parts:
        if any(profile is None for profile in existing_profiles.values()):
            missing = sorted(variable for variable, profile in existing_profiles.items() if profile is None)
            raise FileNotFoundError(
                "no fragmented source parts remain and compact artifacts are missing or invalid: "
                f"{missing}"
            )
        module = _load_compaction_module()
        variable_paths = [output_dir / "data" / f"{variable}.parquet" for variable in variables]
        compact_profile = semantic_fingerprint(variable_paths)
        annual_path = output_dir / f"weather_{args.year}_full_polygon_robust.parquet"
        annual_profile = None
        annual_equality = None
        annual_artifact = None
        if not args.no_annual_file:
            annual_report = module.validate_dataset(
                annual_path,
                expected_labels=variables,
                expected_row_counts=compact_profile["per_variable_rows"],
                expected_schema=OUTPUT_SCHEMA,
                required_sort_columns=ANNUAL_SORT_ORDER,
                include_row_hash=True,
            )
            if not annual_report["valid"]:
                raise ValueError(f"existing annual artifact is invalid: {annual_report['errors']}")
            annual_profile = semantic_fingerprint([annual_path])
            annual_equality = compare_semantic_fingerprints(compact_profile, annual_profile)
            if not annual_equality["semantic_equal"]:
                raise ValueError(
                    "existing annual artifact is not semantically equal to variable artifacts: "
                    f"{annual_equality['differences']}"
                )
            annual_artifact = artifact_record(
                annual_path,
                output_dir,
                kind="annual",
                variable=None,
                sort_order=ANNUAL_SORT_ORDER,
                expected_rows=int(compact_profile["rows"]),
                expected_variables=variables,
            )
        raw_before = raw_signature(cache_root) if cache_root.is_dir() else None
        raw_after = raw_signature(cache_root) if cache_root.is_dir() else None
        raw_unchanged = None if raw_before is None or raw_after is None else raw_before == raw_after
        variable_artifacts = [
            artifact_record(
                path,
                output_dir,
                kind="variable",
                variable=variable,
                sort_order=VARIABLE_SORT_ORDER,
                expected_rows=int(compact_profile["per_variable_rows"][variable]),
            )
            for variable, path in zip(variables, variable_paths)
        ]
        return {
            "status": "COMPLETED" if raw_unchanged is not False else "COMPLETED_WITH_AUDIT_FINDINGS",
            "output_dir": str(output_dir),
            "year": args.year,
            "variables": variables,
            "total_rows": int(compact_profile["rows"]),
            "elapsed_seconds": time.monotonic() - started,
            "raw_input_modified": None if raw_unchanged is None else not raw_unchanged,
            "raw_arrays_reread": False,
            "compact_existing": True,
            "already_compact": True,
            "reused_variables": variables,
            "rebuilt_variables": [],
            "source_fragment_count": 0,
            "staging_deleted": True,
            "semantic_equality": {
                "source_to_variables": None,
                "variables_to_annual": annual_equality,
                "semantic_equal": args.no_annual_file or bool(annual_equality and annual_equality["semantic_equal"]),
            },
            "variable_artifacts": variable_artifacts,
            "annual_artifact": annual_artifact,
        }

    raw_before = raw_signature(cache_root) if cache_root.is_dir() else None
    run_id = _new_run_id(args.year)
    staging_dir = _staging_path(output_dir, run_id)
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        _stage_fragment_links(source_parts, output_dir, staging_dir)
        source_profile = semantic_fingerprint(source_parts) if source_parts else None
        compaction = _compact_and_publish(
            output_dir=output_dir,
            year=args.year,
            source_root=staging_dir,
            staging_dir=staging_dir,
            variables=variables,
            overwrite=args.overwrite,
            no_annual_file=args.no_annual_file,
            source_profile=source_profile,
        )
        raw_after = raw_signature(cache_root) if cache_root.is_dir() else None
        extra_audit = {
            "compact_existing": True,
            "raw_arrays_reread": False,
            "source_manifest_available": raw_before is not None,
            "decoded_object_count": 0,
            "decode_failure_count": 0,
            "decode_failure_samples": [],
            "missing_source_by_variable": {},
        }
        return _finalize_output(
            output_dir=output_dir,
            year=args.year,
            summary=summary,
            started_at=started_at,
            started=started,
            raw_before=raw_before,
            raw_after=raw_after,
            staging_dir=staging_dir,
            keep_staging=args.keep_staging,
            source_parts=source_parts,
            compaction=compaction,
            sidecar_frames=None,
            extra_audit=extra_audit,
        )
    except Exception:
        # Failed runs retain their narrow run-specific staging directory for
        # diagnosis; successful runs remove it unless --keep-staging is set.
        raise


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.compact_existing:
        return _compact_existing(args)

    started_at, started = dt.datetime.now(tz=UTC), time.monotonic()
    inputs = _resolve_inputs(args)
    cache_root, output_dir = inputs["cache_root"], inputs["output_dir"]
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache root is missing: {cache_root}")
    output_guard(cache_root, output_dir, args.overwrite and not args.dry_run, allow_nonempty=args.dry_run)
    fips = [row["fips_code"] for row in inputs["counties"]]
    geometries = load_county_geometries(args.county_archive, fips)
    plan_rows, coverage_rows = build_spatial_plan(geometries, inputs["chunk_ids"])
    coverage_by_fips = {row["fips_code"]: row for row in coverage_rows}
    inventory = build_inventory_frame(inputs["metadata"], inputs["objects"], inputs["missing"], inputs["all_runs"])
    summary = {
        "year": args.year, "cache_root": str(cache_root), "output_dir": str(output_dir), "counties": len(fips), "fips_codes": fips,
        "chunk_ids": inputs["chunk_ids"], "variables": inputs["variables"], "run_count_selected": len(inputs["runs"]),
        "run_count_available": len(inputs["all_runs"]), "plan_cell_count": len(plan_rows),
        "coverage_status_counts": dict(Counter(row["coverage_status"] for row in coverage_rows)), "missing_source_object_count": len(inputs["missing"]),
        "expected_labels_not_local": sorted(set(EXPECTED_LABELS) - set(inputs["metadata"])), "actual_labels_not_in_expected": sorted(set(inputs["metadata"]) - set(EXPECTED_LABELS)),
        "dry_run": bool(args.dry_run), "smoke": bool(args.smoke), "compact_existing": False,
        "raw_arrays_reread": True, "storage_layout": COMPACT_STORAGE_LAYOUT,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return {"status": "DRY_RUN_OK", **summary, "elapsed_seconds": time.monotonic() - started}

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_before = raw_signature(cache_root)
    run_id = _new_run_id(args.year)
    staging_dir = _staging_path(output_dir, run_id)
    staging_dir.mkdir(parents=True, exist_ok=False)
    county_cells: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in plan_rows:
        county_cells[row["fips_code"]][row["chunk_id"]].append(row)
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    partition_counts, row_counts = Counter(), Counter()
    object_decode_count, decode_failures = 0, []
    missing_counter: Counter[str] = Counter(
        str(row.get("symbol"))
        for row in inputs["missing"]
        if str(row.get("symbol")) in inputs["variables"]
        and iso_utc(parse_utc(str(row.get("model_run_utc")))) in inputs["runs"]
        and str(row.get("chunk_id")) in inputs["chunk_ids"]
    )
    runs_since_flush, flush_every_runs = 0, 8
    try:
        for run in inputs["runs"]:
            runs_since_flush += 1
            for variable in inputs["variables"]:
                spec, arrays, object_rows, missing_chunks, decode_errors = inputs["metadata"][variable], {}, {}, [], {}
                expected_steps, fill_value = int(spec["shape"][0]), spec.get("fill_value")
                for chunk_id in inputs["chunk_ids"]:
                    source = inputs["objects"].get((run, variable, chunk_id))
                    if source is None:
                        missing_chunks.append(chunk_id)
                        continue
                    local_path = Path(str(source.get("local_path", "")))
                    if not local_path.is_file():
                        decode_errors[chunk_id] = f"local source missing: {local_path}"
                        continue
                    try:
                        array, _ = decode_chunk_with_metadata(local_path, variable=variable, dtype_hint=spec.get("dtype"))
                        if int(array.shape[0]) != expected_steps:
                            decode_errors[chunk_id] = f"forecast-step mismatch: metadata={expected_steps}, decoded={array.shape[0]}"
                        else:
                            arrays[chunk_id], object_rows[chunk_id] = array, source
                            object_decode_count += 1
                    except (OSError, ValueError, HrrrZarrError) as exc:
                        decode_errors[chunk_id] = f"{type(exc).__name__}: {exc}"
                if decode_errors:
                    decode_failures.append({"run": run, "variable": variable, "errors": decode_errors})
                for county in inputs["counties"]:
                    county_info = {**county, "county": geometries[county["fips_code"]].county, "state": geometries[county["fips_code"]].state}
                    coverage = coverage_by_fips[county["fips_code"]]
                    for forecast_hour in range(1, expected_steps + 1):
                        buffers[variable].append(aggregate_row(
                            county=county_info, variable=variable, spec=spec, year=args.year, run=run, forecast_hour=forecast_hour,
                            county_cells=county_cells[county["fips_code"]], arrays=arrays, fill_value=fill_value, coverage=coverage,
                            object_rows=object_rows, missing_chunks=missing_chunks, decode_errors=decode_errors,
                        ))
            if runs_since_flush >= flush_every_runs or run == inputs["runs"][-1]:
                for variable, rows in list(buffers.items()):
                    if not rows:
                        continue
                    directory = staging_dir / f"variable={variable}"
                    directory.mkdir(parents=True, exist_ok=True)
                    part = directory / f"part-{partition_counts[variable]:05d}.parquet"
                    _atomic_write_frame(_make_schema_frame(rows), part)
                    partition_counts[variable] += 1
                    row_counts[variable] += len(rows)
                    rows.clear()
                runs_since_flush = 0

        source_parts = _fragment_paths(staging_dir, inputs["variables"])
        source_profile = semantic_fingerprint(source_parts)
        compaction = _compact_and_publish(
            output_dir=output_dir,
            year=args.year,
            source_root=staging_dir,
            staging_dir=staging_dir,
            variables=inputs["variables"],
            overwrite=args.overwrite,
            no_annual_file=args.no_annual_file,
            source_profile=source_profile,
        )
        raw_after = raw_signature(cache_root)
        extra_audit = {
            "compact_existing": False,
            "raw_arrays_reread": True,
            "decoded_object_count": object_decode_count,
            "decode_failure_count": len(decode_failures),
            "decode_failure_samples": decode_failures[:20],
            "missing_source_by_variable": dict(missing_counter),
            "materialized_row_counts_by_variable": dict(row_counts),
        }
        sidecar_frames = {
            "spatial_plan.parquet": spatial_plan_frame(plan_rows),
            "county_coverage.parquet": coverage_frame(coverage_rows),
            "variable_inventory.parquet": inventory,
        }
        return _finalize_output(
            output_dir=output_dir,
            year=args.year,
            summary=summary,
            started_at=started_at,
            started=started,
            raw_before=raw_before,
            raw_after=raw_after,
            staging_dir=staging_dir,
            keep_staging=args.keep_staging,
            source_parts=source_parts,
            compaction=compaction,
            sidecar_frames=sidecar_frames,
            extra_audit=extra_audit,
        )
    except Exception:
        # Preserve failed run staging for diagnosis; successful runs are the
        # only runs eligible for automatic staging deletion.
        raise


def main() -> int:
    try:
        args = _parse_args()
        result = materialize(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run:
        print(json.dumps({key: result[key] for key in ("status", "output_dir", "variables", "total_rows", "elapsed_seconds", "raw_input_modified") if key in result}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
