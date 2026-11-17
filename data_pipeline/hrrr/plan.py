#!/usr/bin/env python3
"""Derive a pinned HRRR-Zarr retrieval plan from an audited county-weather table.

The aggregated training inputs retain the exact NOAA object keys and chunk IDs
used for every weather value.  This utility extracts those identities without
inventing URLs, changing forecast runs, or inspecting labels.  It is intended
for reconstruction of the already declared 2022--2024 weather provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


REQUIRED = {
    "year", "variable", "level", "units", "model_run_utc", "forecast_hour",
    "source_object_keys", "chunk_ids_used",
}

# Values verified from the public HRRR-Zarr ``.zarray`` metadata for the same
# source layout.  They are metadata contracts, not imputation defaults.
ZARR_SPECS = {
    "APCP_1hr_acc_fcst": ("<f2", -10000.0, 48),
    "ASNOW_acc_fcst": ("<f2", -10000.0, 45),
    "CAPE": ("<f2", -10000.0, 48),
    "DPT": ("<f2", -10000.0, 48),
    "GUST": ("<f2", -10000.0, 48),
    "MSLMA": ("<f4", -9999.0, 48),
    "PRATE": ("<f2", -10000.0, 48),
    "PRES": ("<f4", -9999.0, 48),
    "RH": ("<f2", -10000.0, 48),
    "TCDC": ("<f2", -10000.0, 48),
    "TMP": ("<f2", -10000.0, 48),
    "UGRD": ("<f2", -10000.0, 48),
    "VGRD": ("<f2", -10000.0, 48),
    "VIS": ("<f2", -10000.0, 48),
}


def _json_list(value: str | None, field: str) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {field} JSON: {value!r}") from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) and item for item in parsed):
        raise ValueError(f"{field} must be a non-empty JSON list of strings")
    return parsed


def derive(inputs: list[Path], plan_out: Path, inventory_out: Path) -> dict[str, int]:
    if not inputs or any(not path.is_file() for path in inputs):
        raise FileNotFoundError("all weather inputs must exist")
    schema = pl.read_parquet_schema(inputs[0])
    missing = sorted(REQUIRED - set(schema))
    if missing:
        raise ValueError(f"weather input is missing provenance columns: {missing}")
    # A county-weather long table repeats the same provenance once per county
    # and lead.  Collapse that repetition in Polars before Python decodes the
    # small JSON fields; this keeps the planner bounded for multi-year data.
    frame = (
        pl.scan_parquet([str(path) for path in inputs])
        .select(sorted(REQUIRED))
        .group_by([
            "year", "variable", "level", "units", "model_run_utc",
            "source_object_keys", "chunk_ids_used",
        ])
        .agg(pl.col("forecast_hour").max().alias("forecast_hour"))
        .collect(engine="streaming")
    )
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    variables: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame.iter_rows(named=True):
        keys = _json_list(row["source_object_keys"], "source_object_keys")
        chunks = _json_list(row["chunk_ids_used"], "chunk_ids_used")
        if len(keys) != len(chunks):
            raise ValueError("source_object_keys/chunk_ids_used length mismatch")
        max_lead = int(row["forecast_hour"])
        variable = str(row["variable"])
        level = str(row["level"])
        try:
            dtype, fill_value, native_steps = ZARR_SPECS[variable]
        except KeyError as error:
            raise ValueError(f"no verified Zarr metadata contract for {variable}") from error
        variable_key = (variable, level)
        spec = variables.setdefault(variable_key, {
            "variable": variable, "level": level, "units": row["units"] or "unknown",
            "shape": f"[{native_steps},1059,1799]", "chunks": f"[{native_steps},150,150]",
            "dtype": dtype, "fill_value": fill_value,
        })
        if max_lead > 48:
            raise ValueError(f"unsupported native HRRR lead {max_lead}")
        for key, chunk in zip(keys, chunks, strict=True):
            suffix = key.rsplit("/", 1)[-1]
            if suffix != chunk or not key.startswith("sfc/"):
                raise ValueError(f"invalid pinned HRRR object identity: {key}")
            identity = (str(row["model_run_utc"]), variable, level, chunk)
            record = rows.setdefault(identity, {
                "year": int(row["year"]), "model_run_utc": identity[0], "variable": variable,
                "level": level, "chunk_id": chunk, "remote_object_key": key,
                "expected_bytes": None, "dtype": dtype, "fill_value": fill_value,
                "forecast_hour_min": 1, "forecast_hour_max": max_lead,
            })
            record["forecast_hour_max"] = max(int(record["forecast_hour_max"]), max_lead)
    plan = pl.DataFrame(list(rows.values())).sort(["model_run_utc", "variable", "chunk_id"])
    inventory = pl.DataFrame(list(variables.values())).sort(["variable", "level"])
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    inventory_out.parent.mkdir(parents=True, exist_ok=True)
    plan.write_parquet(plan_out, compression="zstd")
    inventory.write_parquet(inventory_out, compression="zstd")
    return {"plan_rows": plan.height, "variables": inventory.height}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--plan-out", required=True, type=Path)
    parser.add_argument("--inventory-out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.input, args.plan_out, args.inventory_out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
