"""Polars-only compaction and semantic validation for robust HRRR tables.

The materialization step deliberately writes bounded ``variable=*/part-*.parquet``
files.  This module compacts those derived files without reopening the raw
HRRR cache.  It writes one sorted file per local variable and then one sorted
annual file made only from the validated variable files.

The public functions return JSON-serializable dictionaries so a parent CLI can
place their results in an audit or manifest without depending on a particular
CLI implementation.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import polars as pl


PathLike: TypeAlias = str | Path
TableSource: TypeAlias = PathLike | Sequence[PathLike] | pl.LazyFrame | pl.DataFrame


# These are the labels present in the local 2024 HRRR robust output.  They are
# intentionally not replaced by provider aliases such as ``APCP`` or ``WIND``.
EXPECTED_LABELS: tuple[str, ...] = (
    "APCP_1hr_acc_fcst",
    "ASNOW_acc_fcst",
    "CAPE",
    "DPT",
    "GUST",
    "MSLMA",
    "PRATE",
    "PRES",
    "RH",
    "TCDC",
    "TMP",
    "UGRD",
    "VGRD",
    "VIS",
)

# The 2024, 20-county scope has 366 runs.  All local variables have 48 steps
# except the native 45-step ASNOW forecast axis.
EXPECTED_ROW_COUNTS_2024: dict[str, int] = {
    label: (329_400 if label == "ASNOW_acc_fcst" else 351_360)
    for label in EXPECTED_LABELS
}

OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "fips_code": pl.String,
    "county": pl.String,
    "state": pl.String,
    "variable": pl.String,
    "level": pl.String,
    "units": pl.String,
    "year": pl.Int32,
    "model_run_utc": pl.String,
    "forecast_hour": pl.Int32,
    "valid_time_utc": pl.String,
    "source_model": pl.String,
    "spatial_method": pl.String,
    "chunk_ids_used": pl.String,
    "source_object_keys": pl.String,
    "polygon_coverage_ratio": pl.Float64,
    "coverage_status": pl.String,
    "mean": pl.Float64,
    "std": pl.Float64,
    "min": pl.Float64,
    "max": pl.Float64,
    "core_min": pl.Float64,
    "core_max": pl.Float64,
    "p05": pl.Float64,
    "p10": pl.Float64,
    "p25": pl.Float64,
    "p50": pl.Float64,
    "p75": pl.Float64,
    "p90": pl.Float64,
    "p95": pl.Float64,
    "p99": pl.Float64,
    "iqr": pl.Float64,
    "spatial_range": pl.Float64,
    "top05_mean": pl.Float64,
    "top10_mean": pl.Float64,
    "valid_area_ratio": pl.Float64,
    "intersecting_cell_count": pl.Int64,
    "valid_cell_count": pl.Int64,
    "effective_cell_count": pl.Float64,
    "boundary_area_ratio": pl.Float64,
    "quality_status": pl.String,
    "quality_reason": pl.String,
}

IDENTITY_COLUMNS: tuple[str, ...] = (
    "fips_code",
    "variable",
    "model_run_utc",
    "forecast_hour",
)
VARIABLE_SORT_COLUMNS: tuple[str, ...] = (
    "fips_code",
    "model_run_utc",
    "forecast_hour",
)
ANNUAL_SORT_COLUMNS: tuple[str, ...] = (
    "variable",
    "fips_code",
    "model_run_utc",
    "forecast_hour",
)

_FLOAT_DTYPES = frozenset({pl.Float32, pl.Float64})


class CompactionValidationError(ValueError):
    """Raised when source or target data fail a required validation."""

    def __init__(self, message: str, report: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = dict(report) if report is not None else None


def _schema_signature(schema: Mapping[str, pl.DataType]) -> list[dict[str, str]]:
    return [{"name": name, "dtype": str(dtype)} for name, dtype in schema.items()]


def _schema_matches(
    actual: Mapping[str, pl.DataType],
    expected: Mapping[str, pl.DataType] | None,
) -> bool:
    if expected is None:
        return True
    return _schema_signature(actual) == _schema_signature(expected)


def _schema_error(
    actual: Mapping[str, pl.DataType],
    expected: Mapping[str, pl.DataType] | None,
) -> str | None:
    if expected is None or _schema_matches(actual, expected):
        return None
    return f"schema mismatch: expected={_schema_signature(expected)!r}, actual={_schema_signature(actual)!r}"


def _json_safe(value: Any) -> Any:
    """Convert Polars scalar output into stable JSON-compatible values."""

    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "+Inf"
        if value == float("-inf"):
            return "-Inf"
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    return value


def _collect(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect through the streaming engine available in the active Polars."""

    return lazy_frame.collect(engine="streaming")


def _source_to_lazy(source: TableSource) -> pl.LazyFrame:
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if isinstance(source, (str, Path)):
        path = str(source)
        return pl.scan_parquet(path, hive_partitioning=False, low_memory=True)
    paths = [Path(item) for item in source]
    if not paths:
        raise FileNotFoundError("no Parquet source files were supplied")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Parquet source files are missing: {missing[:5]}")
    return pl.scan_parquet(
        [str(path) for path in paths],
        hive_partitioning=False,
        low_memory=True,
    )


def discover_source_fragments(
    root: PathLike,
    *,
    expected_labels: Sequence[str] | None = EXPECTED_LABELS,
) -> dict[str, tuple[Path, ...]]:
    """Discover source parts grouped by their ``variable=`` directory.

    This function only inspects the derived fragment layout.  It never follows
    any raw-cache path and never includes ``data/*.parquet`` or annual files.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"compaction root is missing: {root_path}")
    result: dict[str, tuple[Path, ...]] = {}
    for directory in sorted(root_path.glob("variable=*")):
        if not directory.is_dir():
            continue
        label = directory.name.removeprefix("variable=")
        parts = tuple(sorted(path for path in directory.glob("part-*.parquet") if path.is_file()))
        if parts:
            result[label] = parts
    if expected_labels is not None:
        expected = set(expected_labels)
        observed = set(result)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing or unexpected:
            raise CompactionValidationError(
                f"source variable labels do not match: missing={missing}, unexpected={unexpected}"
            )
    if not result:
        raise FileNotFoundError(f"no variable=*/part-*.parquet files found under {root_path}")
    return result


def _row_hash_sha256(frame: pl.DataFrame) -> str:
    """Hash all rows in their already-deterministic sorted order.

    Polars supplies the row hash, while the standard-library digest makes the
    result compact and independent of Python object formatting.  The digest is
    a semantic fingerprint, not a replacement for the schema and summary
    checks returned alongside it.
    """

    row_hashes = frame.hash_rows(
        seed=0x13579BDF,
        seed_1=0x2468ACE0,
        seed_2=0x0F1E2D3C,
        seed_3=0x4B5A6978,
    )
    digest = hashlib.sha256()
    for value in row_hashes:
        digest.update(struct.pack("<Q", int(value)))
    return digest.hexdigest()


def _group_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    grouped = frame.group_by(column, maintain_order=False).len().sort(column)
    result: dict[str, int] = {}
    for row in grouped.iter_rows(named=True):
        value = row[column]
        result["<NULL>" if value is None else str(value)] = int(row["len"])
    return result


def _numeric_summary(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    numeric = [
        (name, dtype)
        for name, dtype in frame.schema.items()
        if dtype.is_numeric()
    ]
    if not numeric:
        return {}
    expressions: list[pl.Expr] = []
    for name, dtype in numeric:
        expressions.extend(
            [
                pl.col(name).count().alias(f"{name}__non_null"),
                pl.col(name).null_count().alias(f"{name}__null"),
                pl.col(name).min().alias(f"{name}__min"),
                pl.col(name).max().alias(f"{name}__max"),
                pl.col(name).sum().alias(f"{name}__sum"),
            ]
        )
        if dtype in _FLOAT_DTYPES:
            expressions.extend(
                [
                    pl.col(name).is_nan().sum().fill_null(0).alias(f"{name}__nan"),
                    pl.col(name).is_infinite().sum().fill_null(0).alias(f"{name}__infinite"),
                ]
            )
    values = frame.select(expressions).row(0, named=True)
    result: dict[str, dict[str, Any]] = {}
    for name, dtype in numeric:
        item: dict[str, Any] = {
            "dtype": str(dtype),
            "non_null": values[f"{name}__non_null"],
            "null": values[f"{name}__null"],
            "min": values[f"{name}__min"],
            "max": values[f"{name}__max"],
            "sum": values[f"{name}__sum"],
        }
        if dtype in _FLOAT_DTYPES:
            item["nan"] = values[f"{name}__nan"]
            item["infinite"] = values[f"{name}__infinite"]
        result[name] = _json_safe(item)
    return result


def _forecast_summary(frame: pl.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        frame.group_by("variable", maintain_order=False)
        .agg(
            pl.len().alias("row_count"),
            pl.col("forecast_hour").n_unique().alias("forecast_hour_count"),
            pl.col("forecast_hour").min().alias("forecast_hour_min"),
            pl.col("forecast_hour").max().alias("forecast_hour_max"),
            pl.col("forecast_hour").unique().sort().alias("forecast_hours"),
        )
        .sort("variable")
    )
    return [_json_safe(row) for row in grouped.iter_rows(named=True)]


def _physical_sort_check(
    frame: pl.DataFrame,
    required_sort_columns: Sequence[str] | None,
) -> dict[str, Any]:
    if required_sort_columns is None:
        return {"columns": None, "is_sorted": None}
    columns = tuple(required_sort_columns)
    sorted_keys = frame.select(list(columns)).sort(
        list(columns), maintain_order=True
    )
    return {
        "columns": list(columns),
        "is_sorted": frame.select(list(columns)).equals(sorted_keys),
    }


def semantic_fingerprint(
    source: TableSource,
    *,
    expected_schema: Mapping[str, pl.DataType] | None = OUTPUT_SCHEMA,
    sort_keys: Sequence[str] = IDENTITY_COLUMNS,
    required_sort_columns: Sequence[str] | None = None,
    include_row_hash: bool = True,
) -> dict[str, Any]:
    """Return a deterministic semantic fingerprint for a Parquet table.

    ``sort_keys`` controls order-independent content comparison.  The optional
    ``required_sort_columns`` checks the physical order of a target file without
    making source fragments fail merely because their part order differs.
    """

    lazy_frame = _source_to_lazy(source)
    schema = lazy_frame.collect_schema()
    schema_problem = _schema_error(schema, expected_schema)
    if schema_problem:
        raise CompactionValidationError(schema_problem)
    missing_sort = [column for column in sort_keys if column not in schema]
    if missing_sort:
        raise CompactionValidationError(f"sort key columns are missing: {missing_sort}")
    if required_sort_columns is not None:
        missing_order = [column for column in required_sort_columns if column not in schema]
        if missing_order:
            raise CompactionValidationError(f"required sort columns are missing: {missing_order}")
    frame = _collect(lazy_frame)
    sorted_frame = frame.sort(list(sort_keys), maintain_order=True)

    identity_unique = int(
        frame.select(pl.struct(list(IDENTITY_COLUMNS)).n_unique().alias("unique_identity_keys")).item()
    )
    row_count = int(frame.height)
    null_counts = {
        name: int(value)
        for name, value in frame.select(
            [pl.col(name).null_count().alias(name) for name in frame.columns]
        ).row(0, named=True).items()
    }
    run_series = frame.get_column("model_run_utc").drop_nulls().unique().sort()
    variables = frame.get_column("variable").drop_nulls().unique().sort().to_list()
    fips = frame.get_column("fips_code").drop_nulls().unique().sort().to_list()
    result: dict[str, Any] = {
        "schema": _schema_signature(schema),
        "row_count": row_count,
        "variable_labels": [str(value) for value in variables],
        "fips_codes": [str(value) for value in fips],
        "fips_count": len(fips),
        "model_run_count": int(frame.get_column("model_run_utc").n_unique()),
        "model_run_min": run_series[0] if len(run_series) else None,
        "model_run_max": run_series[-1] if len(run_series) else None,
        "row_counts_by_variable": _group_counts(frame, "variable"),
        "forecast_summary": _forecast_summary(frame),
        "quality_status_counts": _group_counts(frame, "quality_status"),
        "null_counts": null_counts,
        "numeric_summary": _numeric_summary(frame),
        "identity": {
            "columns": list(IDENTITY_COLUMNS),
            "unique_keys": identity_unique,
            "duplicate_extra_rows": row_count - identity_unique,
            "null_key_counts": {
                column: int(null_counts.get(column, 0)) for column in IDENTITY_COLUMNS
            },
        },
        "physical_sort": _physical_sort_check(frame, required_sort_columns),
    }
    if include_row_hash:
        result["row_hash_sha256"] = _row_hash_sha256(sorted_frame)
        result["row_hash_method"] = "polars.hash_rows(sorted_by_identity; fixed_seeds)"
    return _json_safe(result)


def _fingerprint_differences(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> list[str]:
    def numbers_equal(left_value: Any, right_value: Any) -> bool:
        if left_value is None or right_value is None:
            return left_value is None and right_value is None
        if isinstance(left_value, str) or isinstance(right_value, str):
            return left_value == right_value
        try:
            left_number, right_number = float(left_value), float(right_value)
        except (TypeError, ValueError):
            return left_value == right_value
        if left_number != left_number or right_number != right_number:
            return left_number != left_number and right_number != right_number
        return abs(left_number - right_number) <= max(
            1e-12,
            1e-12 * max(abs(left_number), abs(right_number)),
        )

    def numeric_summary_equal(left_summary: Any, right_summary: Any) -> bool:
        if not isinstance(left_summary, Mapping) or not isinstance(right_summary, Mapping):
            return left_summary == right_summary
        if set(left_summary) != set(right_summary):
            return False
        exact_fields = {"dtype", "non_null", "null", "nan", "infinite"}
        numeric_fields = {"min", "max", "sum"}
        for name in left_summary:
            left_item, right_item = left_summary[name], right_summary[name]
            if not isinstance(left_item, Mapping) or not isinstance(right_item, Mapping):
                if left_item != right_item:
                    return False
                continue
            if set(left_item) != set(right_item):
                return False
            for field in exact_fields:
                if field in left_item and left_item[field] != right_item[field]:
                    return False
            for field in numeric_fields:
                if field in left_item and not numbers_equal(left_item[field], right_item[field]):
                    return False
        return True

    differences: list[str] = []
    # Physical order is validated independently.  A source directory is not
    # required to be sorted, while its compact target is.
    comparable = set(left) | set(right)
    comparable.discard("physical_sort")
    for key in sorted(comparable):
        if key == "numeric_summary":
            if not numeric_summary_equal(left.get(key), right.get(key)):
                differences.append(key)
            continue
        if left.get(key) != right.get(key):
            differences.append(key)
    return differences


def semantic_equal(
    left: TableSource,
    right: TableSource,
    *,
    expected_schema: Mapping[str, pl.DataType] | None = OUTPUT_SCHEMA,
    sort_keys: Sequence[str] = IDENTITY_COLUMNS,
    include_row_hash: bool = True,
) -> dict[str, Any]:
    """Compare two tables independent of source-file and row order."""

    left_fingerprint = semantic_fingerprint(
        left,
        expected_schema=expected_schema,
        sort_keys=sort_keys,
        include_row_hash=include_row_hash,
    )
    right_fingerprint = semantic_fingerprint(
        right,
        expected_schema=expected_schema,
        sort_keys=sort_keys,
        include_row_hash=include_row_hash,
    )
    differences = _fingerprint_differences(left_fingerprint, right_fingerprint)
    return {
        "equal": not differences,
        "difference_fields": differences,
        "left": left_fingerprint,
        "right": right_fingerprint,
    }


def validate_dataset(
    source: TableSource,
    *,
    expected_labels: Sequence[str] | None = EXPECTED_LABELS,
    expected_row_counts: Mapping[str, int] | None = EXPECTED_ROW_COUNTS_2024,
    expected_schema: Mapping[str, pl.DataType] | None = OUTPUT_SCHEMA,
    required_sort_columns: Sequence[str] | None = None,
    require_unique_identity: bool = True,
    include_row_hash: bool = True,
) -> dict[str, Any]:
    """Validate labels, rows, schema, identity keys, summaries, and ordering."""

    try:
        fingerprint = semantic_fingerprint(
            source,
            expected_schema=expected_schema,
            required_sort_columns=required_sort_columns,
            include_row_hash=include_row_hash,
        )
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "fingerprint": None,
        }

    errors: list[str] = []
    observed_labels = set(fingerprint["variable_labels"])
    expected_label_set = set(expected_labels) if expected_labels is not None else None
    if expected_label_set is not None and observed_labels != expected_label_set:
        errors.append(
            f"variable labels mismatch: expected={sorted(expected_label_set)}, "
            f"observed={sorted(observed_labels)}"
        )
    if expected_row_counts is not None:
        observed_counts = fingerprint["row_counts_by_variable"]
        for label, expected_count in expected_row_counts.items():
            actual_count = int(observed_counts.get(label, 0))
            if actual_count != int(expected_count):
                errors.append(
                    f"row count mismatch for {label}: expected={int(expected_count)}, actual={actual_count}"
                )
        unexpected_counts = sorted(set(observed_counts) - set(expected_row_counts))
        if unexpected_counts:
            errors.append(f"unexpected row-count labels: {unexpected_counts}")
    identity = fingerprint["identity"]
    if require_unique_identity and int(identity["duplicate_extra_rows"]) != 0:
        errors.append(f"duplicate identity rows: {identity['duplicate_extra_rows']}")
    null_key_counts = identity["null_key_counts"]
    for column, count in null_key_counts.items():
        if int(count) != 0:
            errors.append(f"null identity keys in {column}: {count}")
    physical_sort = fingerprint["physical_sort"]
    if required_sort_columns is not None and not physical_sort["is_sorted"]:
        errors.append(f"physical row order is not sorted by {list(required_sort_columns)}")
    return {
        "valid": not errors,
        "errors": errors,
        "fingerprint": fingerprint,
        "expected_labels": sorted(expected_label_set) if expected_label_set is not None else None,
        "expected_row_counts": (
            {str(key): int(value) for key, value in expected_row_counts.items()}
            if expected_row_counts is not None
            else None
        ),
    }


def validate_source_fragments(
    root: PathLike,
    *,
    expected_labels: Sequence[str] | None = EXPECTED_LABELS,
    expected_row_counts: Mapping[str, int] | None = EXPECTED_ROW_COUNTS_2024,
    expected_schema: Mapping[str, pl.DataType] | None = OUTPUT_SCHEMA,
    include_row_hash: bool = True,
) -> dict[str, Any]:
    """Validate the existing ``variable=*/part-*.parquet`` source layout."""

    root_path = Path(root).resolve()
    fragments = discover_source_fragments(root_path, expected_labels=expected_labels)
    errors: list[str] = []
    variables: dict[str, Any] = {}
    total_parts = 0
    total_rows = 0
    for label in sorted(fragments):
        parts = fragments[label]
        total_parts += len(parts)
        for part in parts:
            try:
                part_schema = pl.scan_parquet(
                    str(part), hive_partitioning=False, low_memory=True
                ).collect_schema()
            except Exception as exc:
                errors.append(f"{part}: {type(exc).__name__}: {exc}")
                continue
            schema_problem = _schema_error(part_schema, expected_schema)
            if schema_problem:
                errors.append(f"{part}: {schema_problem}")
        expected_for_label = None
        if expected_row_counts is not None and label in expected_row_counts:
            expected_for_label = {label: int(expected_row_counts[label])}
        report = validate_dataset(
            parts,
            expected_labels=(label,),
            expected_row_counts=expected_for_label,
            expected_schema=expected_schema,
            include_row_hash=include_row_hash,
        )
        variables[label] = {
            "part_count": len(parts),
            "parts": [str(part) for part in parts],
            **report,
        }
        if not report["valid"]:
            errors.extend(f"{label}: {error}" for error in report["errors"])
        if report["fingerprint"] is not None:
            total_rows += int(report["fingerprint"]["row_count"])
    expected_set = set(expected_labels) if expected_labels is not None else set(fragments)
    if expected_row_counts is not None:
        missing_counts = sorted(set(expected_row_counts) - set(fragments))
        if missing_counts:
            errors.append(f"missing row-count labels: {missing_counts}")
    return {
        "valid": not errors,
        "root": str(root_path),
        "source_pattern": "variable=*/part-*.parquet",
        "expected_labels": sorted(expected_set) if expected_labels is not None else None,
        "observed_labels": sorted(fragments),
        "part_count": total_parts,
        "row_count": total_rows,
        "variables": variables,
        "errors": errors,
    }


def validate_compact_file(
    path: PathLike,
    variable: str,
    *,
    expected_rows: int | None = None,
    expected_schema: Mapping[str, pl.DataType] | None = OUTPUT_SCHEMA,
    include_row_hash: bool = True,
) -> dict[str, Any]:
    """Validate one variable-level compact file and its required sort order."""

    row_counts: Mapping[str, int] | None
    if expected_rows is not None:
        row_counts = {variable: int(expected_rows)}
    elif variable in EXPECTED_ROW_COUNTS_2024:
        row_counts = {variable: EXPECTED_ROW_COUNTS_2024[variable]}
    else:
        row_counts = None
    report = validate_dataset(
        path,
        expected_labels=(variable,),
        expected_row_counts=row_counts,
        expected_schema=expected_schema,
        required_sort_columns=VARIABLE_SORT_COLUMNS,
        include_row_hash=include_row_hash,
    )
    report["path"] = str(Path(path).resolve())
    report["variable"] = variable
    return report


def _atomic_sink_parquet(
    lazy_frame: pl.LazyFrame,
    target: Path,
    *,
    force: bool,
    row_group_size: int | None,
) -> None:
    """Stream a lazy frame to a temporary file and atomically publish it."""

    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise FileExistsError(f"target exists; explicit force is required: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        lazy_frame.sink_parquet(
            temporary,
            compression="zstd",
            row_group_size=row_group_size,
            maintain_order=True,
            engine="streaming",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _require_valid(report: Mapping[str, Any], context: str) -> None:
    if not bool(report.get("valid")):
        errors = report.get("errors", [])
        raise CompactionValidationError(
            f"{context} failed validation: {errors}",
            report=report,
        )


def _compare_fingerprints(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    differences = _fingerprint_differences(left, right)
    return {
        "equal": not differences,
        "difference_fields": differences,
    }


def _default_annual_path(root: Path) -> Path:
    return root / f"weather_{root.name}_full_polygon_robust.parquet"


def compact_existing(
    root: PathLike,
    *,
    expected_labels: Sequence[str] = EXPECTED_LABELS,
    expected_row_counts: Mapping[str, int] | None = EXPECTED_ROW_COUNTS_2024,
    expected_schema: Mapping[str, pl.DataType] | None = OUTPUT_SCHEMA,
    annual_path: PathLike | None = None,
    force: bool = False,
    row_group_size: int | None = 131_072,
    include_row_hash: bool = True,
) -> dict[str, Any]:
    """Compact existing derived fragments into 14 files plus one annual file.

    Parameters
    ----------
    root:
        Existing 2024-style output root containing ``variable=*/part-*.parquet``.
    expected_labels / expected_row_counts:
        The exact local labels and their expected row counts.  Pass explicit
        values for a different year or county scope.
    annual_path:
        Optional explicit annual target.  The default is
        ``root/weather_<root.name>_full_polygon_robust.parquet``.
    force:
        Required to replace an existing target.  Existing valid targets are
        reused without rewriting when this is false.

    The source fragments remain untouched.  No raw HRRR path is opened by this
    function.
    """

    root_path = Path(root).resolve()
    labels = tuple(sorted(dict.fromkeys(expected_labels)))
    if not labels:
        raise ValueError("at least one expected variable label is required")
    source_report = validate_source_fragments(
        root_path,
        expected_labels=labels,
        expected_row_counts=expected_row_counts,
        expected_schema=expected_schema,
        include_row_hash=include_row_hash,
    )
    _require_valid(source_report, "source fragments")
    fragments = discover_source_fragments(root_path, expected_labels=labels)
    data_dir = root_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    variable_results: dict[str, Any] = {}
    compact_paths: dict[str, Path] = {}
    compact_reports: dict[str, dict[str, Any]] = {}
    for label in labels:
        target = data_dir / f"{label}.parquet"
        expected_rows = (
            int(expected_row_counts[label])
            if expected_row_counts is not None and label in expected_row_counts
            else None
        )
        source_fingerprint = source_report["variables"][label]["fingerprint"]
        target_report: dict[str, Any] | None = None
        action = "written"
        if target.exists():
            target_report = validate_compact_file(
                target,
                label,
                expected_rows=expected_rows,
                expected_schema=expected_schema,
                include_row_hash=include_row_hash,
            )
            if target_report["valid"]:
                equality = _compare_fingerprints(
                    source_fingerprint, target_report["fingerprint"]
                )
                if equality["equal"] and not force:
                    action = "reused_existing_valid"
                elif not force:
                    raise CompactionValidationError(
                        f"existing compact target is not semantically equal: {target}",
                        report={"target": target_report, "equality": equality},
                    )
            elif not force:
                raise CompactionValidationError(
                    f"existing compact target is invalid: {target}",
                    report=target_report,
                )
        if action == "written":
            source_lazy = _source_to_lazy(fragments[label]).sort(
                list(VARIABLE_SORT_COLUMNS), maintain_order=True
            )
            _atomic_sink_parquet(
                source_lazy,
                target,
                force=force,
                row_group_size=row_group_size,
            )
            target_report = validate_compact_file(
                target,
                label,
                expected_rows=expected_rows,
                expected_schema=expected_schema,
                include_row_hash=include_row_hash,
            )
            _require_valid(target_report, f"compact target {label}")
            equality = _compare_fingerprints(
                source_fingerprint, target_report["fingerprint"]
            )
            if not equality["equal"]:
                raise CompactionValidationError(
                    f"source and compact target differ for {label}",
                    report={"source": source_fingerprint, "target": target_report, "equality": equality},
                )
        else:
            assert target_report is not None
            equality = _compare_fingerprints(
                source_fingerprint, target_report["fingerprint"]
            )
        assert target_report is not None
        compact_paths[label] = target
        compact_reports[label] = target_report
        variable_results[label] = {
            "path": str(target),
            "action": action,
            "row_count": int(target_report["fingerprint"]["row_count"]),
            "source_semantic_equal": bool(equality["equal"]),
            "validation": target_report,
        }

    # Annual construction is intentionally based on the validated compact
    # files, not on the original fragments.  Lexical label order plus each
    # variable file's validated internal order gives the required annual order.
    annual_target = Path(annual_path).resolve() if annual_path is not None else _default_annual_path(root_path)
    annual_sources = [compact_paths[label] for label in labels]
    annual_lazy = pl.concat(
        [_source_to_lazy(path) for path in annual_sources],
        how="vertical",
        rechunk=False,
    )
    total_expected_rows = (
        sum(int(expected_row_counts[label]) for label in labels)
        if expected_row_counts is not None and all(label in expected_row_counts for label in labels)
        else sum(int(report["fingerprint"]["row_count"]) for report in compact_reports.values())
    )
    annual_row_counts = (
        {str(label): int(expected_row_counts[label]) for label in labels}
        if expected_row_counts is not None and all(label in expected_row_counts for label in labels)
        else {str(label): int(compact_reports[label]["fingerprint"]["row_count"]) for label in labels}
    )
    annual_report: dict[str, Any] | None = None
    annual_action = "written"
    if annual_target.exists():
        annual_report = validate_dataset(
            annual_target,
            expected_labels=labels,
            expected_row_counts=annual_row_counts,
            expected_schema=expected_schema,
            required_sort_columns=ANNUAL_SORT_COLUMNS,
            include_row_hash=include_row_hash,
        )
        if annual_report["valid"] and not force:
            annual_action = "reused_existing_valid"
        elif not force:
            raise CompactionValidationError(
                f"existing annual target is invalid: {annual_target}",
                report=annual_report,
            )
    if annual_action == "written":
        _atomic_sink_parquet(
            annual_lazy,
            annual_target,
            force=force,
            row_group_size=row_group_size,
        )
        annual_report = validate_dataset(
            annual_target,
            expected_labels=labels,
            expected_row_counts=annual_row_counts,
            expected_schema=expected_schema,
            required_sort_columns=ANNUAL_SORT_COLUMNS,
            include_row_hash=include_row_hash,
        )
        _require_valid(annual_report, "annual compact target")
    assert annual_report is not None

    # Compare annual rows to each validated compact file through a filtered
    # annual scan.  This keeps the equality check bounded by one variable at a
    # time while proving that the annual artifact came from those 14 files.
    annual_variable_equality: dict[str, Any] = {}
    for label in labels:
        annual_slice = _source_to_lazy(annual_target).filter(pl.col("variable") == label)
        comparison = semantic_equal(
            compact_paths[label],
            annual_slice,
            expected_schema=expected_schema,
            sort_keys=IDENTITY_COLUMNS,
            include_row_hash=include_row_hash,
        )
        annual_variable_equality[label] = {
            "equal": bool(comparison["equal"]),
            "difference_fields": comparison["difference_fields"],
        }
        if not comparison["equal"]:
            raise CompactionValidationError(
                f"compact and annual data differ for {label}",
                report=comparison,
            )

    return {
        "status": "COMPLETED",
        "root": str(root_path),
        "data_dir": str(data_dir),
        "annual_path": str(annual_target),
        "expected_labels": list(labels),
        "source": {
            "part_count": int(source_report["part_count"]),
            "row_count": int(source_report["row_count"]),
            "validation": source_report,
        },
        "variables": variable_results,
        "annual": {
            "path": str(annual_target),
            "action": annual_action,
            "row_count": int(annual_report["fingerprint"]["row_count"]),
            "expected_row_count": int(total_expected_rows),
            "validation": annual_report,
            "variable_semantic_equality": annual_variable_equality,
        },
        "source_fragments_preserved": True,
        "raw_hrrr_arrays_read": False,
    }


__all__ = [
    "ANNUAL_SORT_COLUMNS",
    "CompactionValidationError",
    "EXPECTED_LABELS",
    "EXPECTED_ROW_COUNTS_2024",
    "IDENTITY_COLUMNS",
    "OUTPUT_SCHEMA",
    "VARIABLE_SORT_COLUMNS",
    "compact_existing",
    "discover_source_fragments",
    "semantic_equal",
    "semantic_fingerprint",
    "validate_compact_file",
    "validate_dataset",
    "validate_source_fragments",
]
