#!/usr/bin/env python3
"""Build and download a pinned HRRR-Zarr object inventory.

The source bucket is public and is always accessed with an unsigned S3
client.  The downloader mirrors the local compact-cache layout expected by
``county_statistics.py``.  It deliberately knows
nothing about county geometry or robust statistics: those contracts remain in
the existing materializer and its data modules.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data_pipeline/

import polars as pl

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
    from botocore.exceptions import ClientError, BotoCoreError
except ImportError:  # pragma: no cover - cloud environment installs boto3
    boto3 = None  # type: ignore[assignment]
    UNSIGNED = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment,misc]
    ClientError = Exception  # type: ignore[assignment,misc]
    BotoCoreError = Exception  # type: ignore[assignment,misc]


UTC = dt.timezone.utc
SCHEMA_VERSION = "hrrr-expansion20-cloud-object-inventory-v1"
DATA_OBJECT = "data_chunk"
METADATA_OBJECT = "metadata"
CHECKPOINT_EVERY = 500


def utc_now() -> str:
    return dt.datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_shape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _metadata_key(data_key: str, suffix: str) -> str:
    """Derive the array metadata key from a validated data-object key.

    HRRR-Zarr stores the array metadata beside the repeated level/variable
    chunk directory.  The function only accepts the exact path shape present
    in the frozen plan; malformed keys fail closed instead of being guessed.
    """

    parts = PurePosixPath(data_key).parts
    if len(parts) < 8 or parts[0] != "sfc" or not parts[2].endswith(".zarr"):
        raise ValueError(f"unexpected HRRR-Zarr object key: {data_key}")
    if parts[3] != parts[5] or parts[4] != parts[6]:
        raise ValueError(f"repeated level/variable path contract failed: {data_key}")
    if not parts[7].startswith("0."):
        raise ValueError(f"unexpected Zarr chunk coordinate: {data_key}")
    return "/".join((*parts[:7], suffix))


def _spec_by_variable(variable_inventory: Path) -> dict[str, dict[str, Any]]:
    frame = pl.read_parquet(variable_inventory)
    required = {"variable", "level", "units", "shape", "chunks", "dtype", "fill_value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"variable inventory is missing columns: {missing}")
    result: dict[str, dict[str, Any]] = {}
    for row in frame.to_dicts():
        variable = str(row["variable"])
        shape = row.get("shape")
        chunks = row.get("chunks")
        try:
            shape = json.loads(shape) if isinstance(shape, str) else shape
            chunks = json.loads(chunks) if isinstance(chunks, str) else chunks
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid shape/chunks JSON for {variable}") from exc
        result[variable] = {**row, "shape": shape, "chunks": chunks}
    return result


def build_inventory(
    plan_path: Path,
    variable_inventory_path: Path,
    *,
    source_bucket: str,
    raw_root: Path,
) -> pl.DataFrame:
    """Build the data+metadata inventory without contacting AWS."""

    plan = pl.read_parquet(plan_path)
    required_plan = {
        "year", "model_run_utc", "variable", "level", "chunk_id", "remote_object_key",
        "expected_bytes", "dtype", "fill_value", "forecast_hour_max",
    }
    missing = sorted(required_plan - set(plan.columns))
    if missing:
        raise ValueError(f"download plan is missing columns: {missing}")
    if plan.is_empty():
        raise ValueError("download plan is empty")
    if plan.select("remote_object_key").n_unique() != plan.height:
        raise ValueError("download plan contains duplicate remote_object_key rows")
    specs = _spec_by_variable(variable_inventory_path)
    rows: list[dict[str, Any]] = []
    first_by_year_variable: dict[tuple[int, str], dict[str, Any]] = {}
    for row in plan.sort(["model_run_utc", "variable", "chunk_id"]).to_dicts():
        variable = str(row["variable"])
        if variable not in specs:
            raise ValueError(f"plan variable is absent from variable inventory: {variable}")
        spec = specs[variable]
        key = str(row["remote_object_key"])
        if not key.startswith("sfc/"):
            raise ValueError(f"source object is not in HRRR sfc namespace: {key}")
        local_relative = PurePosixPath("data") / PurePosixPath(key)
        rows.append({
            "year": int(row["year"]),
            "model_run_utc": str(row["model_run_utc"]),
            "variable": variable,
            "level": str(row["level"]),
            "chunk_id": str(row["chunk_id"]),
            "object_type": DATA_OBJECT,
            "remote_bucket": source_bucket,
            "remote_key": key,
            "local_relative_path": local_relative.as_posix(),
            "expected_shape": _json_shape(spec.get("shape")),
            "expected_dtype": str(row.get("dtype") or spec.get("dtype") or ""),
            "fill_value": row.get("fill_value") if row.get("fill_value") is not None else spec.get("fill_value"),
            "content_length": int(row["expected_bytes"]) if row.get("expected_bytes") is not None else None,
            "etag": "",
            "head_status": "NOT_CHECKED",
            "download_status": "PENDING",
            "attempt_count": 0,
            "sha256": "",
            "decode_status": "NOT_CHECKED",
            "quality_reason": "",
            "forecast_hour_min": int(row.get("forecast_hour_min") or 1),
            "forecast_hour_max": int(row["forecast_hour_max"]),
            "units": row.get("units") if row.get("units") is not None else spec.get("units"),
            "native_time_axis": "forecast_period_2" if variable == "ASNOW_acc_fcst" else "forecast_period",
        })
        first_by_year_variable.setdefault((int(row["year"]), variable), row)

    # Metadata is immutable within an annual source layout, but the
    # materializer intentionally keeps each calendar year self-contained.
    # A shared multi-year cache therefore needs a metadata pair for every
    # (year, variable), not merely the first year encountered for a variable.
    for (year, variable), row in sorted(first_by_year_variable.items()):
        spec = specs[variable]
        for suffix in (".zarray", ".zattrs"):
            key = _metadata_key(str(row["remote_object_key"]), suffix)
            local_relative = PurePosixPath("metadata", str(year), str(row["level"]), variable, suffix)
            rows.append({
                "year": int(row["year"]),
                "model_run_utc": str(row["model_run_utc"]),
                "variable": variable,
                "level": str(row["level"]),
                "chunk_id": "",
                "object_type": METADATA_OBJECT,
                "remote_bucket": source_bucket,
                "remote_key": key,
                "local_relative_path": local_relative.as_posix(),
                "expected_shape": _json_shape(spec.get("shape")),
                "expected_dtype": str(spec.get("dtype") or ""),
                "fill_value": spec.get("fill_value"),
                "content_length": None,
                "etag": "",
                "head_status": "NOT_CHECKED",
                "download_status": "PENDING",
                "attempt_count": 0,
                "sha256": "",
                "decode_status": "NOT_APPLICABLE",
                "quality_reason": "array metadata required by existing materializer",
                "forecast_hour_min": 1,
                "forecast_hour_max": int((spec.get("shape") or [0])[0] or 0),
                "units": spec.get("units"),
                "native_time_axis": "forecast_period_2" if variable == "ASNOW_acc_fcst" else "forecast_period",
            })
    inventory = pl.DataFrame(rows, strict=False).sort(["object_type", "model_run_utc", "variable", "chunk_id", "remote_key"])
    local_paths = inventory.get_column("local_relative_path").to_list()
    if len(local_paths) != len(set(local_paths)):
        raise ValueError("inventory contains duplicate local_relative_path rows")
    # Verify the local root is only used for path calculation; no file is
    # created here, which keeps audit/dry-run side-effect free for source data.
    _ = raw_root.resolve()
    return inventory


def inventory_summary(inventory: pl.DataFrame) -> dict[str, Any]:
    data = inventory.filter(pl.col("object_type") == DATA_OBJECT)
    metadata = inventory.filter(pl.col("object_type") == METADATA_OBJECT)
    expected = data.select(pl.col("content_length").fill_null(0).sum()).item()
    largest = data.sort("content_length", descending=True).head(1).to_dicts()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "object_count": inventory.height,
        "data_object_count": data.height,
        "metadata_object_count": metadata.height,
        "variable_count": data.select("variable").n_unique(),
        "run_count": data.select("model_run_utc").n_unique(),
        "chunk_count": data.select("chunk_id").n_unique(),
        "estimated_bytes_from_plan": int(expected or 0),
        "largest_object": largest[0] if largest else None,
        "download_status_counts": inventory.group_by("download_status").len().to_dicts(),
    }


def write_materializer_jsonl(raw_root: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Publish the request ledger consumed by the existing materializer."""

    target = raw_root / "download_objects.jsonl"
    part = target.with_name(target.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    with part.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda value: (str(value.get("model_run_utc")), str(value.get("variable")), str(value.get("chunk_id")))):
            if row.get("object_type") != DATA_OBJECT:
                continue
            status = str(row.get("download_status", ""))
            if status == "REUSED_VALID":
                status = "REUSED_EXISTING"
            elif status not in {"DOWNLOADED", "REUSED_EXISTING", "MISSING_SOURCE_OBJECT", "FAILED", "INTERRUPTED"}:
                status = "FAILED"
            payload = {
                "request_id": f"{row.get('model_run_utc')}|{row.get('variable')}|{row.get('chunk_id')}",
                "kind": "data_chunk",
                "symbol": row.get("variable"),
                "variable": row.get("variable"),
                "level": row.get("level"),
                "model_run_utc": row.get("model_run_utc"),
                "chunk_id": row.get("chunk_id"),
                "object_key": row.get("remote_key"),
                "remote_url": f"https://{row.get('remote_bucket')}.s3.amazonaws.com/{row.get('remote_key')}",
                "local_relative_path": row.get("local_relative_path"),
                "forecast_hour_min": row.get("forecast_hour_min"),
                "forecast_hour_max": row.get("forecast_hour_max"),
                "status": status,
                "sha256": row.get("sha256", ""),
                "etag": row.get("etag", ""),
                "bytes_on_disk": row.get("content_length"),
                "attempts": row.get("attempt_count", 0),
                "quality_reason": row.get("quality_reason", ""),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(part, target)


def _unsigned_client(region: str):
    if boto3 is None or Config is None:
        raise RuntimeError("boto3 is required for cloud object access")
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version=UNSIGNED, retries={"mode": "adaptive", "max_attempts": 8}),
    )


def _error_text(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        error = getattr(exc, "response", {}).get("Error", {})
        return f"{error.get('Code', type(exc).__name__)}: {error.get('Message', str(exc))}"
    return f"{type(exc).__name__}: {exc}"


def _is_missing(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound"}
    return False


def _looks_like_error_page(prefix: bytes) -> bool:
    sample = prefix.lstrip().lower()
    return sample.startswith(b"<html") or sample.startswith(b"<!doctype html") or sample.startswith(b"<?xml")


def _head(client: Any, bucket: str, key: str) -> dict[str, Any]:
    response = client.head_object(Bucket=bucket, Key=key)
    return {
        "content_length": int(response.get("ContentLength", 0)),
        "etag": str(response.get("ETag", "")),
        "last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else None,
    }


def _decode_smoke(path: Path, variable: str, dtype: str, expected_shape: str) -> tuple[str, str]:
    if variable == "" or path.name.startswith("."):
        return "NOT_APPLICABLE", "metadata object"
    try:
        from hrrr.decode import decode_chunk_with_metadata

        array, meta = decode_chunk_with_metadata(path, variable=variable, dtype_hint=dtype or None)
        expected = json.loads(expected_shape) if expected_shape else []
        expected_steps = int(expected[0]) if expected else None
        if expected_steps is not None and int(array.shape[0]) != expected_steps:
            return "FAIL", f"decoded forecast steps {array.shape[0]} != expected {expected_steps}"
        return "PASS", f"decoded shape={list(array.shape)} dtype={meta.get('dtype')}"
    except Exception as exc:  # decode failures are recorded per object
        return "FAIL", _error_text(exc)


def _download_one(
    row: dict[str, Any],
    *,
    client: Any,
    raw_root: Path,
    source_bucket: str,
    retries: int,
    head_only: bool,
    stop_event: threading.Event,
) -> dict[str, Any]:
    result = dict(row)
    result["attempt_count"] = 0
    target = raw_root / str(row["local_relative_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    key = str(row["remote_key"])
    head: dict[str, Any] | None = None
    for attempt in range(1, retries + 1):
        result["attempt_count"] = attempt
        if stop_event.is_set():
            result.update(download_status="INTERRUPTED", quality_reason="SIGTERM received")
            return result
        try:
            head = _head(client, source_bucket, key)
            result.update(content_length=head["content_length"], etag=head["etag"], head_status="OK")
            if int(head["content_length"]) <= 0:
                raise ValueError("remote object has zero Content-Length")
            if target.is_file() and target.stat().st_size == int(head["content_length"]):
                result["sha256"] = sha256_file(target)
                if row["object_type"] == DATA_OBJECT:
                    result["decode_status"], reason = _decode_smoke(target, str(row["variable"]), str(row.get("expected_dtype") or ""), str(row.get("expected_shape") or ""))
                    if result["decode_status"] != "PASS":
                        raise ValueError(reason)
                result.update(download_status="REUSED_VALID", quality_reason="existing file passed size/hash/decode checks")
                return result
            if head_only:
                result.update(download_status="HEAD_OK", quality_reason="HEAD-only inventory")
                return result
            part = target.with_name(target.name + ".part")
            existing = part.stat().st_size if part.is_file() else 0
            remote_size = int(head["content_length"])
            # A decode/rename failure can leave a complete ``.part`` file.
            # Revalidate it on the next attempt instead of issuing the invalid
            # range ``bytes=<remote_size>-`` against S3.
            if existing == remote_size:
                digest = sha256_file(part)
                if row["object_type"] == DATA_OBJECT:
                    decode_status, reason = _decode_smoke(part, str(row["variable"]), str(row.get("expected_dtype") or ""), str(row.get("expected_shape") or ""))
                    result["decode_status"] = decode_status
                    if decode_status != "PASS":
                        raise ValueError(reason)
                else:
                    result["decode_status"] = "NOT_APPLICABLE"
                os.replace(part, target)
                result.update(sha256=digest, download_status="DOWNLOADED", quality_reason="revalidated complete partial file")
                return result
            if existing > remote_size:
                part.unlink()
                existing = 0
            request: dict[str, Any] = {"Bucket": source_bucket, "Key": key}
            if existing:
                request["Range"] = f"bytes={existing}-"
            response = client.get_object(**request)
            body = response["Body"]
            first = b""
            mode = "ab" if existing else "wb"
            with part.open(mode) as handle:
                while True:
                    if stop_event.is_set():
                        result.update(download_status="INTERRUPTED", quality_reason="SIGTERM received")
                        return result
                    block = body.read(1024 * 1024)
                    if not block:
                        break
                    if not first:
                        first = block[:512]
                    handle.write(block)
            body.close()
            if _looks_like_error_page(first):
                raise ValueError("remote response resembles an HTML/XML error page")
            size = part.stat().st_size
            if size != remote_size:
                raise ValueError(f"Content-Length mismatch: local={size} remote={remote_size}")
            digest = sha256_file(part)
            if row["object_type"] == DATA_OBJECT:
                decode_status, reason = _decode_smoke(part, str(row["variable"]), str(row.get("expected_dtype") or ""), str(row.get("expected_shape") or ""))
                result["decode_status"] = decode_status
                if decode_status != "PASS":
                    raise ValueError(reason)
            else:
                result["decode_status"] = "NOT_APPLICABLE"
            os.replace(part, target)
            result.update(sha256=digest, download_status="DOWNLOADED", quality_reason="size/hash/decode checks passed")
            return result
        except (ClientError, BotoCoreError, OSError, ValueError, RuntimeError) as exc:
            result["quality_reason"] = _error_text(exc)
            result["download_status"] = "MISSING_SOURCE_OBJECT" if _is_missing(exc) else "RETRYABLE_FAILURE"
            if _is_missing(exc):
                return result
            if attempt < retries:
                time.sleep(min(30.0, 2.0 ** min(attempt, 5)))
    result["download_status"] = "FAILED"
    return result


def download_inventory(
    inventory: pl.DataFrame,
    *,
    raw_root: Path,
    region: str,
    source_bucket: str,
    state_path: Path,
    max_concurrency: int = 8,
    retries: int = 8,
    head_only: bool = False,
    object_limit: int | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    client = _unsigned_client(region)
    stop_event = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    if object_limit:
        # Keep metadata even when the smoke data limit is applied.  Without
        # this, a lexicographically ordered prefix can contain data chunks
        # but no .zarray/.zattrs records, which is not a valid materializer
        # smoke input.
        metadata_rows = inventory.filter(pl.col("object_type") == METADATA_OBJECT).to_dicts()
        data_rows = inventory.filter(pl.col("object_type") == DATA_OBJECT).head(object_limit).to_dicts()
        rows = data_rows + metadata_rows
    else:
        rows = inventory.to_dicts()
    results: list[dict[str, Any]] = []
    completed = 0
    started = time.monotonic()

    def checkpoint() -> None:
        frame = pl.DataFrame(results, strict=False) if results else pl.DataFrame(schema=inventory.schema)
        summary = inventory_summary(frame) if results else {"object_count": 0}
        write_materializer_jsonl(raw_root, results)
        atomic_json(state_path, {
            "schema_version": SCHEMA_VERSION,
            "updated_at_utc": utc_now(),
            "status": "INTERRUPTED" if stop_event.is_set() else "RUNNING",
            "completed_objects": completed,
            "total_objects": len(rows),
            "elapsed_seconds": time.monotonic() - started,
            "summary": summary,
            "objects": results,
        })

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as pool:
            futures = [pool.submit(_download_one, row, client=client, raw_root=raw_root, source_bucket=source_bucket, retries=retries, head_only=head_only, stop_event=stop_event) for row in rows]
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % CHECKPOINT_EVERY == 0 or completed == len(rows):
                    checkpoint()
                if stop_event.is_set():
                    break
    finally:
        signal.signal(signal.SIGTERM, previous)
        checkpoint()
    result = pl.DataFrame(results, strict=False).sort(["object_type", "remote_key"])
    summary = inventory_summary(result)
    summary.update({
        "status": "INTERRUPTED" if stop_event.is_set() else "COMPLETED",
        "finished_at_utc": utc_now(),
        "actual_bytes": int(result.filter(pl.col("download_status").is_in(["DOWNLOADED", "REUSED_VALID"])).select(pl.col("content_length").fill_null(0).sum()).item() or 0),
        "failed_objects": int(result.filter(pl.col("download_status").is_in(["FAILED", "RETRYABLE_FAILURE"])).height),
        "missing_source_objects": int(result.filter(pl.col("download_status") == "MISSING_SOURCE_OBJECT").height),
        "decode_failures": int(result.filter(pl.col("decode_status") == "FAIL").height),
    })
    write_materializer_jsonl(raw_root, result.to_dicts())
    atomic_json(state_path, {"schema_version": SCHEMA_VERSION, "updated_at_utc": utc_now(), "status": summary["status"], "summary": summary, "objects": result.to_dicts()})
    return result, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reference-variable-inventory", type=Path, required=True)
    parser.add_argument("--source-bucket", default="hrrrzarr")
    parser.add_argument("--region", default="us-west-1")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--inventory-out", type=Path, required=True)
    parser.add_argument("--state-out", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--object-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = build_inventory(args.plan, args.reference_variable_inventory, source_bucket=args.source_bucket, raw_root=args.raw_root)
    args.inventory_out.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_parquet(args.inventory_out, compression="zstd")
    summary = inventory_summary(inventory)
    if args.dry_run:
        atomic_json(args.state_out, {"schema_version": SCHEMA_VERSION, "status": "DRY_RUN", "updated_at_utc": utc_now(), "summary": summary})
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result, summary = download_inventory(
        inventory,
        raw_root=args.raw_root,
        region=args.region,
        source_bucket=args.source_bucket,
        state_path=args.state_out,
        max_concurrency=args.max_concurrency,
        retries=args.retries,
        head_only=args.head_only,
        object_limit=args.object_limit,
    )
    result.write_parquet(args.inventory_out, compression="zstd")
    failures = result.filter(pl.col("download_status").is_in(["FAILED", "RETRYABLE_FAILURE", "MISSING_SOURCE_OBJECT", "INTERRUPTED"]))
    failures.select([
        "remote_key", "variable", "model_run_utc", "chunk_id", "download_status", "quality_reason",
    ]).rename({"download_status": "status", "quality_reason": "reason"}).write_parquet(
        args.state_out.parent / "download_failures.parquet", compression="zstd"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "COMPLETED" and summary["failed_objects"] == 0 and summary["decode_failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
