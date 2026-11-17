"""Offline reader for the downloaded HRRR-Zarr compact cache.

The local cache contains Zarr v2 *chunk payloads* rather than a complete
store.  This module deliberately does not try to recreate a remote store or
fetch missing metadata.  It validates the captured layout, decodes the local
Blosc payload with the environment's libblosc, and samples the county grid
cells that were covered by the compact download plan.

The reader is research-only: it exposes the source run, native forecast lead,
and assumed availability time so callers can apply the Phase 1 causal rule.
It never interpolates hourly values to 15 minutes and never reads test truth.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import datetime as dt
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


UTC = dt.timezone.utc
GRID_X0_M = -2697520.142522
GRID_Y0_M = -1587306.152443
GRID_SPACING_M = 3000.0
GRID_CHUNK_SIZE = 150
GRID_X_CELLS = 1799
GRID_Y_CELLS = 1059
LCC_RADIUS_M = 6371229.0
LCC_LAT0_RAD = math.radians(38.5)
LCC_LON0_RAD = math.radians(-97.5)
LCC_N = math.sin(LCC_LAT0_RAD)
LCC_F = (math.cos(LCC_LAT0_RAD) * math.tan(math.pi / 4 + LCC_LAT0_RAD / 2) ** LCC_N) / LCC_N
LCC_RHO0 = LCC_RADIUS_M * LCC_F / math.tan(math.pi / 4 + LCC_LAT0_RAD / 2) ** LCC_N

ZARR_SHAPE = (48, 1059, 1799)
ZARR_CHUNKS = (48, 150, 150)
ZARR_DTYPE = np.dtype("<f4")
# A subset of the archived local HRRR objects was written as float16 even
# though the compact-store convention (and the newer objects) is float32.
# There is no Zarr metadata in the offline payload cache, so decoding must
# inspect both representations and fail closed when neither is physically
# plausible.  The spatial chunk is fixed by the captured download layout;
# only the native time length and scalar dtype are inferred.
LEGACY_DTYPE = np.dtype("<f2")
ZARR_FILL_VALUE = -9999.0
ACTIVE_VARIABLES = {
    "TMP": "temperature_2m_k",
    "DPT": "dew_point_2m_k",
    "UGRD": "wind_u_10m_ms",
    "VGRD": "wind_v_10m_ms",
}
ACTIVE_LEVELS = {
    "TMP": "2m_above_ground",
    "DPT": "2m_above_ground",
    "UGRD": "10m_above_ground",
    "VGRD": "10m_above_ground",
}
LOCAL_VARIABLE_PATHS = {
    "TMP": Path("2m_above_ground/TMP/2m_above_ground/TMP"),
    "DPT": Path("2m_above_ground/DPT/2m_above_ground/DPT"),
    "UGRD": Path("10m_above_ground/UGRD/10m_above_ground/UGRD"),
    "VGRD": Path("10m_above_ground/VGRD/10m_above_ground/VGRD"),
}


class HrrrZarrError(RuntimeError):
    """Raised when a local compact HRRR object is not decodable or aligned."""


@dataclass(frozen=True)
class CountyGridPoint:
    fips_code: str
    county: str
    state: str
    latitude: float
    longitude: float
    grid_y: int
    grid_x: int
    chunk_id: str
    local_y: int
    local_x: int


@dataclass(frozen=True)
class HrrrObject:
    model_run_utc: str
    variable: str
    level: str
    chunk_id: str
    object_key: str
    local_path: Path
    forecast_hour_min: int
    forecast_hour_max: int
    sha256: str


def parse_utc(value: Any) -> dt.datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_utc(value: dt.datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def lcc_xy(latitude: float, longitude: float) -> tuple[float, float]:
    """Convert lon/lat to the documented spherical HRRR LCC metres."""

    lat = math.radians(float(latitude))
    lon = math.radians(float(longitude))
    rho = LCC_RADIUS_M * LCC_F / math.tan(math.pi / 4 + lat / 2) ** LCC_N
    angle = LCC_N * (lon - LCC_LON0_RAD)
    return rho * math.sin(angle), LCC_RHO0 - rho * math.cos(angle)


def point_grid_index(latitude: float, longitude: float) -> tuple[int, int]:
    x, y = lcc_xy(latitude, longitude)
    x_index = math.floor((x - GRID_X0_M) / GRID_SPACING_M)
    y_index = math.floor((y - GRID_Y0_M) / GRID_SPACING_M)
    if not (0 <= x_index < GRID_X_CELLS and 0 <= y_index < GRID_Y_CELLS):
        raise HrrrZarrError(f"county point is outside HRRR grid: lat={latitude}, lon={longitude}")
    return int(y_index), int(x_index)


def chunk_id(grid_y: int, grid_x: int) -> str:
    # Zarr v2 includes the time chunk coordinate as the leading ``0``.
    return f"0.{grid_y // GRID_CHUNK_SIZE}.{grid_x // GRID_CHUNK_SIZE}"


def county_points(selection_path: Path | str) -> list[CountyGridPoint]:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    rows = payload.get("selected_counties", [])
    points: list[CountyGridPoint] = []
    for row in rows:
        fips = str(row.get("fips_code", "")).zfill(5)
        lat = float(row["centroid_lat"])
        lon = float(row["centroid_lon"])
        y_index, x_index = point_grid_index(lat, lon)
        expected_chunk = str(row.get("centroid_chunk_id", ""))
        actual_chunk = chunk_id(y_index, x_index)
        if expected_chunk and expected_chunk != actual_chunk:
            raise HrrrZarrError(
                f"selection chunk mismatch for {fips}: expected {expected_chunk}, got {actual_chunk}"
            )
        points.append(
            CountyGridPoint(
                fips_code=fips,
                county=str(row.get("county", "")),
                state=str(row.get("state", "")),
                latitude=lat,
                longitude=lon,
                grid_y=y_index,
                grid_x=x_index,
                chunk_id=actual_chunk,
                local_y=y_index % GRID_CHUNK_SIZE,
                local_x=x_index % GRID_CHUNK_SIZE,
            )
        )
    if len(points) != 5 or len({p.fips_code for p in points}) != 5:
        raise HrrrZarrError("the compact HRRR reader requires exactly five unique selected counties")
    return points


def _library() -> ctypes.CDLL:
    candidates = [
        ctypes.util.find_library("blosc"),
        os.environ.get("BLOSC_LIBRARY"),
        "libblosc.so.1",
        "libblosc.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        function = library.blosc_decompress
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        return library
    raise HrrrZarrError("libblosc is unavailable; cannot decode the local HRRR-Zarr chunks")


def _infer_variable(path: Path | str) -> str | None:
    """Infer an active variable from a compact object path when possible."""

    text = str(path).upper()
    for variable in ACTIVE_VARIABLES:
        if re.search(rf"(?:^|[/_.]){re.escape(variable)}(?:[/_.]|$)", text):
            return variable
    return None


def _physical_score(array: np.ndarray, variable: str | None) -> tuple[float, float, float]:
    """Return (valid fraction, finite fraction, range margin) for a candidate."""

    values = np.asarray(array, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)
    finite_fraction = float(finite.mean()) if values.size else 0.0
    if variable in {"TMP", "DPT"}:
        lower, upper = 150.0, 350.0
    elif variable in {"UGRD", "VGRD"}:
        lower, upper = -200.0, 200.0
    else:
        # Without a variable token we still reject NaN/inf, but do not make
        # an arbitrary physical assertion.  The caller may supply the token.
        return finite_fraction, finite_fraction, 0.0
    usable = finite & (values != ZARR_FILL_VALUE)
    if not np.any(usable):
        return 0.0, finite_fraction, 0.0
    selected = values[usable]
    in_range = (selected >= lower) & (selected <= upper)
    valid_fraction = float(in_range.mean())
    # A larger margin is a deterministic tie-breaker for two plausible
    # decodings; it is never used to accept an otherwise invalid candidate.
    margin = float(min(selected.max() - lower, upper - selected.min()))
    return valid_fraction, finite_fraction, margin


def decode_chunk_with_metadata(
    path: Path | str,
    variable: str | None = None,
    dtype_hint: str | np.dtype[Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode one C-Blosc Zarr chunk and return [forecast_hour,y,x].

    The compact manifest was built from a 48-step probe, but older local
    archive objects can legitimately contain a shorter time chunk.  The
    decompressed byte count is therefore treated as authoritative for the
    local object; callers record the resulting native lead coverage instead of
    padding it to 48 hours.
    """

    source = Path(path)
    payload = source.read_bytes()
    variable_name = str(variable or _infer_variable(source) or "").upper() or None
    # Allocate enough room for the largest supported representation.  The
    # decompressed payload is at most 48*150*150 float32 values; float16
    # payloads use fewer bytes for the same number of time steps.
    max_bytes = int(ZARR_CHUNKS[0] * ZARR_CHUNKS[1] * ZARR_CHUNKS[2] * ZARR_DTYPE.itemsize)
    output = ctypes.create_string_buffer(max_bytes)
    result = _library().blosc_decompress(payload, output, max_bytes)
    cells_per_time = ZARR_CHUNKS[1] * ZARR_CHUNKS[2]
    if result <= 0:
        raise HrrrZarrError(f"Blosc decompression failed for {source}: result={result}")

    candidates: list[tuple[float, float, float, str, np.ndarray]] = []
    hinted_dtype = np.dtype(dtype_hint) if dtype_hint is not None else None
    candidate_dtypes = [hinted_dtype] if hinted_dtype is not None else []
    candidate_dtypes.extend(dtype for dtype in (ZARR_DTYPE, LEGACY_DTYPE) if dtype not in candidate_dtypes)
    for dtype in candidate_dtypes:
        bytes_per_time = cells_per_time * dtype.itemsize
        if result % bytes_per_time:
            continue
        time_steps = result // bytes_per_time
        if not (0 < time_steps <= ZARR_CHUNKS[0]):
            continue
        count = time_steps * cells_per_time
        array = np.frombuffer(output, dtype=dtype, count=count)
        shaped = array.reshape((time_steps, ZARR_CHUNKS[1], ZARR_CHUNKS[2])).copy()
        valid_fraction, finite_fraction, margin = _physical_score(shaped, variable_name)
        candidates.append((valid_fraction, finite_fraction, margin, dtype.str, shaped))
    if not candidates:
        raise HrrrZarrError(
            f"unexpected Blosc size for {source}: got {result}, not a whole supported spatial slice"
        )
    # Physical plausibility is the authoritative discriminator.  Require a
    # near-complete valid payload for known variables; otherwise an invalid
    # float32 reinterpretation could silently pass through as weather.
    candidates.sort(
        key=lambda item: (
            item[3] == hinted_dtype.str if hinted_dtype is not None else False,
            item[0], item[1], item[2], item[3] == ZARR_DTYPE.str,
        ),
        reverse=True,
    )
    valid_fraction, finite_fraction, margin, dtype_str, selected = candidates[0]
    if variable_name in ACTIVE_VARIABLES and (finite_fraction < 0.999 or valid_fraction < 0.99):
        raise HrrrZarrError(
            f"no physically valid {variable_name} decoding for {source}: "
            f"best_dtype={dtype_str} valid_fraction={valid_fraction:.6f} finite_fraction={finite_fraction:.6f}"
        )
    metadata = {
        "path": str(source),
        "variable": variable_name,
        "dtype_detected": dtype_str,
        "forecast_steps": int(selected.shape[0]),
        "decompressed_bytes": int(result),
        "valid_fraction": valid_fraction,
        "finite_fraction": finite_fraction,
        "physical_margin": margin,
        "candidate_dtypes": [item[3] for item in candidates],
        "decoder_version": "dtype_probe_v4",
    }
    return selected.astype(np.float32, copy=False), metadata


def decode_chunk(path: Path | str, variable: str | None = None) -> np.ndarray:
    """Decode one C-Blosc Zarr chunk and return [forecast_hour,y,x]."""

    return decode_chunk_with_metadata(path, variable=variable)[0]


def load_objects(manifest_path: Path | str, root: Path | str) -> dict[tuple[str, str, str], HrrrObject]:
    """Load and validate the compact manifest using object_key, not stale paths."""

    manifest = Path(manifest_path)
    cache_root = Path(root)
    objects: dict[tuple[str, str, str], HrrrObject] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        variable = str(row.get("variable", "")).upper()
        if variable not in ACTIVE_VARIABLES:
            continue
        model_run = iso_utc(parse_utc(row["model_run_utc"]))
        level = str(row.get("level", ""))
        if level != ACTIVE_LEVELS[variable]:
            raise HrrrZarrError(f"unexpected level at manifest line {line_number}: {level}")
        key = str(row["object_key"])
        path = cache_root / key
        if not path.is_file():
            raise HrrrZarrError(f"manifest object is missing locally: {path}")
        object_key = (model_run, variable, str(row["chunk_id"]))
        if object_key in objects:
            raise HrrrZarrError(f"duplicate compact object: {object_key}")
        objects[object_key] = HrrrObject(
            model_run_utc=model_run,
            variable=variable,
            level=level,
            chunk_id=str(row["chunk_id"]),
            object_key=key,
            local_path=path,
            forecast_hour_min=int(row.get("forecast_hour_min", 1)),
            forecast_hour_max=int(row.get("forecast_hour_max", 48)),
            sha256=str(row.get("sha256", "")),
        )
    if not objects:
        raise HrrrZarrError("no active HRRR objects were found in the local manifest")
    runs = {key[0] for key in objects}
    chunks = {key[2] for key in objects}
    variables = {key[1] for key in objects}
    if variables != set(ACTIVE_VARIABLES) or chunks != {"0.4.9", "0.4.10"}:
        raise HrrrZarrError(
            f"incomplete compact object set: runs={len(runs)}, variables={variables}, chunks={chunks}"
        )
    expected = len(runs) * len(ACTIVE_VARIABLES) * 2
    if len(objects) != expected:
        raise HrrrZarrError(f"compact object set is incomplete: {len(objects)} != {expected}")
    return objects


def discover_local_objects(root: Path | str) -> dict[tuple[str, str, str], HrrrObject]:
    """Discover every core HRRR surface chunk physically present locally.

    The v3 manifest is an analysis-plan subset, not a filesystem boundary.
    This discovery path deliberately walks the downloaded ``sfc`` stores so
    additional complete runs (including the test-window runs) are visible to
    the materializer.  It does not infer missing objects, read a remote store,
    or merge any other weather source.
    """

    cache_root = Path(root)
    sfc_root = cache_root / "sfc"
    if not sfc_root.is_dir():
        raise HrrrZarrError(f"local HRRR sfc root is missing: {sfc_root}")
    objects: dict[tuple[str, str, str], HrrrObject] = {}
    for date_dir in sorted(path for path in sfc_root.iterdir() if path.is_dir()):
        for store in sorted(date_dir.glob("*_fcst.zarr")):
            if not store.is_dir():
                continue
            name = store.name
            match = re.match(r"^(\d{8})_(\d{2})z_fcst\.zarr$", name)
            if match is None:
                continue
            try:
                model_run = iso_utc(parse_utc(f"{match.group(1)}T{match.group(2)}:00:00Z"))
            except ValueError:
                continue
            for variable, relative in LOCAL_VARIABLE_PATHS.items():
                variable_root = store / relative
                for path in sorted(variable_root.glob("0.*")):
                    if not path.is_file():
                        continue
                    chunk = path.name
                    key = (model_run, variable, chunk)
                    if key in objects:
                        raise HrrrZarrError(f"duplicate discovered local HRRR object: {key}")
                    objects[key] = HrrrObject(
                        model_run_utc=model_run,
                        variable=variable,
                        level=ACTIVE_LEVELS[variable],
                        chunk_id=chunk,
                        object_key=str(path.relative_to(cache_root)),
                        local_path=path,
                        forecast_hour_min=1,
                        forecast_hour_max=48,
                        sha256="",
                    )
    if not objects:
        raise HrrrZarrError(f"no local HRRR core chunks discovered under {sfc_root}")
    return objects


def read_run_points(
    objects: Mapping[tuple[str, str, str], HrrrObject],
    model_run_utc: str,
    points: Iterable[CountyGridPoint],
) -> dict[str, list[dict[str, Any]]]:
    """Read all native hourly values for each county in one model run."""

    run = iso_utc(parse_utc(model_run_utc))
    point_rows = list(points)
    by_chunk: dict[str, list[CountyGridPoint]] = {}
    for point in point_rows:
        by_chunk.setdefault(point.chunk_id, []).append(point)
    result: dict[str, list[dict[str, Any]]] = {point.fips_code: [] for point in point_rows}
    for variable in ACTIVE_VARIABLES:
        for selected_chunk, chunk_points in by_chunk.items():
            item = objects.get((run, variable, selected_chunk))
            if item is None:
                raise HrrrZarrError(f"missing {run} {variable} {selected_chunk}")
            array = decode_chunk(item.local_path, variable=variable)
            for point in chunk_points:
                values = array[:, point.local_y, point.local_x]
                for index, value in enumerate(values, start=1):
                    value_float = float(value)
                    row = next((r for r in result[point.fips_code] if r["forecast_hour"] == index), None)
                    if row is None:
                        row = {
                            "model_run_utc": run,
                            "forecast_hour": index,
                            "valid_time_utc": iso_utc(parse_utc(run) + dt.timedelta(hours=index)),
                        }
                        result[point.fips_code].append(row)
                    row[ACTIVE_VARIABLES[variable]] = None if value_float == ZARR_FILL_VALUE else value_float
            del array
    for rows in result.values():
        rows.sort(key=lambda row: int(row["forecast_hour"]))
        for row in rows:
            u = row.get("wind_u_10m_ms")
            v = row.get("wind_v_10m_ms")
            row["wind_speed_10m_ms"] = math.hypot(float(u), float(v)) if u is not None and v is not None else None
            t = row.get("temperature_2m_k")
            d = row.get("dew_point_2m_k")
            row["dew_point_depression_k"] = float(t) - float(d) if t is not None and d is not None else None
    return result
