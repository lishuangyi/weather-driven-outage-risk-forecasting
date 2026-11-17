"""Exact Census/TIGER polygon to HRRR-cell spatial planning helpers.

The helper is intentionally data-layer only.  It reads a supplied vector
source, transforms polygon geometry into the spherical HRRR Lambert
Conformal Conic grid, and computes positive-area polygon/cell intersections.
It does not read weather values, outage labels, or other training tables, and
it has no command-line integration.

The raw ``weight`` is the intersection area in square metres, matching the
existing local HRRR aggregation convention.  Rows also expose
``area_weight`` (fraction of polygon area) and ``normalized_weight``
(fraction of covered area).  Plain row mappings and explicit Polars output
are provided; no dataframe other than Polars is required.
"""

from __future__ import annotations

import math
import os
import sys
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decode import (
    GRID_CHUNK_SIZE,
    GRID_SPACING_M,
    GRID_X0_M,
    GRID_X_CELLS,
    GRID_Y0_M,
    GRID_Y_CELLS,
    LCC_LAT0_RAD,
    LCC_LON0_RAD,
    LCC_RADIUS_M,
    chunk_id as hrrr_chunk_id,
)


DEFAULT_POLYGON_CRS = "EPSG:4326"
DEFAULT_TIGER_CRS = "EPSG:4269"
_LCC_LAT0_DEG = math.degrees(LCC_LAT0_RAD)
_LCC_LON0_DEG = math.degrees(LCC_LON0_RAD)

# These values are derived from decode.py rather than maintained as a
# second grid contract.  One standard parallel is mathematically equivalent
# to the spherical LCC formula in decode.lcc_xy.
HRRR_PROJ4 = (
    f"+proj=lcc +lat_1={_LCC_LAT0_DEG:.12f} "
    f"+lat_0={_LCC_LAT0_DEG:.12f} +lon_0={_LCC_LON0_DEG:.12f} "
    f"+a={LCC_RADIUS_M:.12f} +b={LCC_RADIUS_M:.12f} "
    "+x_0=0 +y_0=0 +units=m +no_defs +type=crs"
)
HRRR_LCC_PROJ4 = HRRR_PROJ4
HRRR_CRS = HRRR_PROJ4

CORE_OVERLAP_FRACTION = 0.25
BOUNDARY_EPSILON = 1e-10
COVERAGE_FULL_THRESHOLD = 0.995
COVERAGE_NEAR_FULL_THRESHOLD = 0.98

_FIPS_ALIASES = (
    "fips_code", "fips", "geoid", "geoid10", "geoid20", "countyfp",
    "GEOID", "GEOID10", "GEOID20",
)
_COUNTY_ALIASES = ("county", "county_name", "name", "namelsad", "NAME")
_STATE_ALIASES = ("state", "state_name", "statename", "STUSPS", "STATE")
_STATE_FIPS_ALIASES = ("state_fips", "statefp", "STATEFP")

STATE_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "60": "AS", "66": "GU", "69": "MP", "72": "PR",
    "78": "VI",
}


class HrrrPolygonError(ValueError):
    """Base error for invalid geometry, grid, or plan inputs."""


class HrrrPolygonDependencyError(HrrrPolygonError):
    """Raised when the optional geometry or Polars runtime is unavailable."""


class HrrrPolygonCoverageError(HrrrPolygonError):
    """Raised when full polygon coverage was explicitly required."""


@dataclass(frozen=True)
class HrrrGridSpec:
    """Projected HRRR grid and Zarr chunk geometry."""

    x0_m: float = GRID_X0_M
    y0_m: float = GRID_Y0_M
    spacing_m: float = GRID_SPACING_M
    x_cells: int = GRID_X_CELLS
    y_cells: int = GRID_Y_CELLS
    chunk_size: int = GRID_CHUNK_SIZE

    def __post_init__(self) -> None:
        for name in ("x0_m", "y0_m", "spacing_m"):
            if not math.isfinite(float(getattr(self, name))):
                raise HrrrPolygonError(f"grid {name} must be finite")
        if float(self.spacing_m) <= 0.0:
            raise HrrrPolygonError("grid spacing must be positive")
        for name in ("x_cells", "y_cells", "chunk_size"):
            if int(getattr(self, name)) <= 0:
                raise HrrrPolygonError(f"grid {name} must be positive")

    @property
    def cell_area_m2(self) -> float:
        return float(self.spacing_m) ** 2

    @property
    def x_chunks(self) -> int:
        return math.ceil(self.x_cells / self.chunk_size)

    @property
    def y_chunks(self) -> int:
        return math.ceil(self.y_cells / self.chunk_size)

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        return (
            self.x0_m,
            self.y0_m,
            self.x0_m + self.x_cells * self.spacing_m,
            self.y0_m + self.y_cells * self.spacing_m,
        )


DEFAULT_HRRR_GRID = HrrrGridSpec()


@dataclass(frozen=True)
class PolygonRecord:
    """A vector feature with normalized FIPS and source CRS metadata."""

    fips_code: str
    geometry: Any
    crs: Any = DEFAULT_TIGER_CRS
    county: str = ""
    state: str = ""
    state_fips: str = ""
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fips_code", _normalise_fips(self.fips_code, allow_empty=False))
        object.__setattr__(self, "state_fips", _normalise_state_fips(self.state_fips))
        if self.geometry is None:
            raise HrrrPolygonError(f"polygon geometry is missing for {self.fips_code}")

    def to_mapping(self, *, include_geometry: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fips_code": self.fips_code,
            "county": self.county,
            "state": self.state,
            "state_fips": self.state_fips,
            "crs": self.crs,
            "properties": dict(self.properties),
        }
        if include_geometry:
            result["geometry"] = self.geometry
        return result


@dataclass(frozen=True)
class HrrrCellIntersection:
    """One unique positive-area polygon/HRRR-cell intersection."""

    polygon_id: str
    chunk_id: str
    grid_y: int
    grid_x: int
    local_y: int
    local_x: int
    flat_index: int
    cell_id: str
    cell_x0_m: float
    cell_y0_m: float
    cell_x1_m: float
    cell_y1_m: float
    center_x_m: float
    center_y_m: float
    intersection_area_m2: float
    polygon_area_m2: float
    covered_area_m2: float
    center_inside_polygon: bool
    boundary_cell: bool

    @property
    def weight(self) -> float:
        """Raw intersection area, in square metres."""

        return self.intersection_area_m2

    @property
    def area_weight(self) -> float:
        """Intersection area divided by the full polygon area."""

        return self.intersection_area_m2 / self.polygon_area_m2 if self.polygon_area_m2 else 0.0

    @property
    def normalized_weight(self) -> float:
        """Intersection area divided by the area covered by the plan."""

        return self.intersection_area_m2 / self.covered_area_m2 if self.covered_area_m2 else 0.0

    def to_row(self, *, coverage_fraction: float, coverage_complete: bool) -> dict[str, Any]:
        return {
            "polygon_id": self.polygon_id,
            "chunk_id": self.chunk_id,
            "grid_y": self.grid_y,
            "grid_x": self.grid_x,
            "local_y": self.local_y,
            "local_x": self.local_x,
            "flat_index": self.flat_index,
            "cell_id": self.cell_id,
            "cell_x0_m": self.cell_x0_m,
            "cell_y0_m": self.cell_y0_m,
            "cell_x1_m": self.cell_x1_m,
            "cell_y1_m": self.cell_y1_m,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "intersection_area_m2": self.intersection_area_m2,
            "weight": self.weight,
            "polygon_area_m2": self.polygon_area_m2,
            "covered_area_m2": self.covered_area_m2,
            "area_weight": self.area_weight,
            "normalized_weight": self.normalized_weight,
            "coverage_fraction": coverage_fraction,
            "coverage_complete": coverage_complete,
            "center_inside_polygon": self.center_inside_polygon,
            "boundary_cell": self.boundary_cell,
        }


@dataclass(frozen=True)
class HrrrCoverageReport(Mapping[str, Any]):
    """Coverage of a polygon by the union of its unique cell intersections."""

    polygon_id: str
    polygon_area_m2: float
    covered_area_m2: float
    uncovered_area_m2: float
    coverage_fraction: float
    intersection_count: int
    unique_cell_count: int
    duplicate_cell_count: int
    requested_chunk_ids: tuple[str, ...]
    complete: bool
    tolerance: float

    @property
    def valid(self) -> bool:
        return self.complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon_id": self.polygon_id,
            "polygon_area_m2": self.polygon_area_m2,
            "covered_area_m2": self.covered_area_m2,
            "uncovered_area_m2": self.uncovered_area_m2,
            "coverage_fraction": self.coverage_fraction,
            "intersection_count": self.intersection_count,
            "unique_cell_count": self.unique_cell_count,
            "duplicate_cell_count": self.duplicate_cell_count,
            "requested_chunk_ids": list(self.requested_chunk_ids),
            "complete": self.complete,
            "valid": self.valid,
            "tolerance": self.tolerance,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __bool__(self) -> bool:
        return self.complete


_PLAN_COLUMNS = (
    "polygon_id", "chunk_id", "grid_y", "grid_x", "local_y", "local_x",
    "flat_index", "cell_id", "cell_x0_m", "cell_y0_m", "cell_x1_m",
    "cell_y1_m", "center_x_m", "center_y_m", "intersection_area_m2",
    "weight", "polygon_area_m2", "covered_area_m2", "area_weight",
    "normalized_weight", "coverage_fraction", "coverage_complete",
    "center_inside_polygon", "boundary_cell",
)


@dataclass(frozen=True)
class HrrrPolygonPlan:
    """Exact spatial plan for one polygon over the selected HRRR chunks."""

    polygon_id: str
    polygon_area_m2: float
    covered_area_m2: float
    coverage_fraction: float
    requested_chunk_ids: tuple[str, ...]
    intersections: tuple[HrrrCellIntersection, ...]
    coverage: HrrrCoverageReport
    grid: HrrrGridSpec = DEFAULT_HRRR_GRID
    polygon_crs: Any = DEFAULT_POLYGON_CRS
    grid_crs: Any = HRRR_CRS

    @property
    def cells(self) -> tuple[HrrrCellIntersection, ...]:
        return self.intersections

    @property
    def cell_count(self) -> int:
        return len(self.intersections)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(sorted({cell.chunk_id for cell in self.intersections}))

    @property
    def indices_by_chunk(self) -> dict[str, list[int]]:
        return {
            chunk: [cell.flat_index for cell in self.intersections if cell.chunk_id == chunk]
            for chunk in self.chunk_ids
        }

    @property
    def weights_by_chunk(self) -> dict[str, list[float]]:
        return {
            chunk: [cell.weight for cell in self.intersections if cell.chunk_id == chunk]
            for chunk in self.chunk_ids
        }

    def to_chunk_mapping(self, *, weight: str = "area") -> dict[str, tuple[list[int], list[float]]]:
        """Return ``chunk -> (flat_indices, weights)`` for Zarr consumers."""

        mode = str(weight).strip().lower()
        if mode in {"area", "raw", "intersection_area_m2", "weight"}:
            getter = _raw_weight
        elif mode in {"polygon", "area_weight", "polygon_fraction"}:
            getter = _polygon_weight
        elif mode in {"covered", "normalized", "normalized_weight"}:
            getter = _covered_weight
        else:
            raise HrrrPolygonError(f"unsupported plan weight mode: {weight}")
        result: dict[str, tuple[list[int], list[float]]] = {}
        for chunk in self.chunk_ids:
            cells = [cell for cell in self.intersections if cell.chunk_id == chunk]
            result[chunk] = (
                [cell.flat_index for cell in cells],
                [float(getter(cell)) for cell in cells],
            )
        return result

    def as_spatial_plan(self, *, weight: str = "area") -> dict[str, tuple[list[int], list[float]]]:
        return self.to_chunk_mapping(weight=weight)

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            cell.to_row(
                coverage_fraction=self.coverage_fraction,
                coverage_complete=self.coverage.complete,
            )
            for cell in self.intersections
        ]

    def to_polars(self) -> Any:
        """Return a Polars DataFrame with deterministic plan columns."""

        try:
            import polars as pl  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency-specific
            raise HrrrPolygonDependencyError("Polars is required for to_polars()") from exc
        rows = self.to_rows()
        if not rows:
            return pl.DataFrame({column: [] for column in _PLAN_COLUMNS})
        return pl.DataFrame(rows)

    def to_dict(self, *, include_rows: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "polygon_id": self.polygon_id,
            "polygon_area_m2": self.polygon_area_m2,
            "covered_area_m2": self.covered_area_m2,
            "coverage_fraction": self.coverage_fraction,
            "requested_chunk_ids": list(self.requested_chunk_ids),
            "coverage": self.coverage.to_dict(),
        }
        if include_rows:
            result["intersections"] = self.to_rows()
        return result

    def require_full_coverage(self, *, tolerance: float | None = None) -> "HrrrPolygonPlan":
        threshold = self.coverage.tolerance if tolerance is None else float(tolerance)
        if not math.isfinite(threshold) or threshold < 0.0 or threshold >= 1.0:
            raise HrrrPolygonError("coverage tolerance must be finite and in [0, 1)")
        if self.coverage_fraction < 1.0 - threshold:
            raise HrrrPolygonCoverageError(
                f"{self.polygon_id} coverage is {self.coverage_fraction:.12f}, "
                f"below {1.0 - threshold:.12f}"
            )
        return self

    def __len__(self) -> int:
        return len(self.intersections)

    def __iter__(self) -> Iterator[HrrrCellIntersection]:
        return iter(self.intersections)


def _raw_weight(cell: HrrrCellIntersection) -> float:
    return cell.weight


def _polygon_weight(cell: HrrrCellIntersection) -> float:
    return cell.area_weight


def _covered_weight(cell: HrrrCellIntersection) -> float:
    return cell.normalized_weight


def _normalise_fips(value: Any, *, allow_empty: bool = True) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise HrrrPolygonError("polygon FIPS/GEOID is missing")
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        if allow_empty:
            return ""
        raise HrrrPolygonError("polygon FIPS/GEOID is empty")
    return text.zfill(5) if text.isdigit() else text


def _normalise_state_fips(value: Any) -> str:
    """Normalize Census STATEFP to its required two-digit code."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        return ""
    if not text.isdigit():
        raise HrrrPolygonError(f"STATEFP must be numeric: {value!r}")
    text = text.zfill(2)
    if len(text) != 2:
        raise HrrrPolygonError(f"STATEFP must be two digits: {value!r}")
    return text


def _first_value(values: Mapping[str, Any], aliases: Sequence[str], default: Any = None) -> Any:
    lowered = {str(key).lower(): value for key, value in values.items()}
    for alias in aliases:
        if alias in values:
            return values[alias]
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return default


def _configure_proj_data() -> None:
    """Select an installed PROJ database before CRS construction."""

    try:
        import pyproj  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise HrrrPolygonDependencyError("polygon planning requires pyproj") from exc
    candidates: list[Path] = []
    for variable in ("PROJ_DATA", "PROJ_LIB"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            Path(sys.prefix) / "share" / "proj",
            Path(pyproj.__file__).resolve().parent / "proj_dir" / "share" / "proj",
        ]
    )
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "share" / "proj")
    try:
        candidates.extend(sorted(Path(sys.prefix).resolve().parents[1].glob("pkgs/proj-*/share/proj")))
    except IndexError:
        pass
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "proj.db").is_file():
            os.environ["PROJ_DATA"] = key
            os.environ["PROJ_LIB"] = key
            pyproj.datadir.set_data_dir(key)
            return
    raise HrrrPolygonDependencyError("no installed PROJ database was found")


def hrrr_crs() -> Any:
    """Return the HRRR spherical LCC CRS as a pyproj CRS object."""

    _configure_proj_data()
    try:
        from pyproj import CRS  # type: ignore[import-not-found]
        return CRS.from_proj4(HRRR_PROJ4)
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise HrrrPolygonDependencyError("polygon planning requires pyproj") from exc


def _require_geometry_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    _configure_proj_data()
    try:
        from pyproj import Transformer  # type: ignore[import-not-found]
        from shapely.geometry import Point, box, shape  # type: ignore[import-not-found]
        from shapely.ops import transform, unary_union  # type: ignore[import-not-found]
        try:
            from shapely import from_wkb  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - Shapely 1.x fallback
            from shapely.wkb import loads as from_wkb  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise HrrrPolygonDependencyError(
            "exact polygon planning requires Shapely and pyproj"
        ) from exc
    return Transformer, Point, box, shape, transform, unary_union, from_wkb


def _geometry_from(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "geom_type") and hasattr(value, "intersection"):
        return value
    runtime = _require_geometry_runtime()
    if isinstance(value, Mapping):
        if str(value.get("type", "")).lower() == "feature":
            return _geometry_from(value.get("geometry"))
        try:
            return runtime[3](value)
        except Exception as exc:
            raise HrrrPolygonError("invalid GeoJSON geometry mapping") from exc
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return runtime[6](bytes(value))
        except Exception as exc:
            raise HrrrPolygonError("invalid WKB geometry") from exc
    interface = getattr(value, "__geo_interface__", None)
    if interface is not None:
        return _geometry_from(interface)
    raise HrrrPolygonError("geometry must be a Shapely object or GeoJSON-like mapping")


def _validate_polygon_geometry(geometry: Any, *, polygon_id: str) -> Any:
    geometry = _geometry_from(geometry)
    if geometry is None or bool(getattr(geometry, "is_empty", False)):
        raise HrrrPolygonError(f"polygon {polygon_id} is empty")
    if str(getattr(geometry, "geom_type", "")) not in {"Polygon", "MultiPolygon"}:
        raise HrrrPolygonError(f"polygon {polygon_id} must be Polygon or MultiPolygon")
    if hasattr(geometry, "is_valid") and not bool(geometry.is_valid):
        raise HrrrPolygonError(f"polygon {polygon_id} is invalid")
    return geometry


def _feature_from_input(
    polygon: Any,
    *,
    polygon_id: str | None,
    polygon_crs: Any | None,
) -> tuple[str, Any, Any, Mapping[str, Any]]:
    if isinstance(polygon, PolygonRecord):
        return (
            polygon_id or polygon.fips_code,
            polygon.geometry,
            polygon.crs if polygon_crs is None else polygon_crs,
            polygon.properties,
        )
    properties: Mapping[str, Any] = {}
    inferred_id = ""
    inferred_crs = polygon_crs
    raw_geometry: Any = polygon
    if isinstance(polygon, Mapping):
        props = polygon.get("properties")
        properties = props if isinstance(props, Mapping) else polygon
        raw_geometry = polygon.get("geometry", polygon) if "geometry" in polygon or str(polygon.get("type", "")).lower() == "feature" else polygon
        if str(polygon.get("type", "")).lower() == "feature":
            raw_geometry = polygon.get("geometry")
        inferred_id = _normalise_fips(
            _first_value(polygon, _FIPS_ALIASES, _first_value(properties, _FIPS_ALIASES, ""))
        )
        inferred_crs = polygon.get("crs", inferred_crs)
        if inferred_crs is None:
            inferred_crs = _first_value(properties, ("crs", "geometry_crs"), None)
    elif hasattr(polygon, "geometry") and not hasattr(polygon, "geom_type"):
        raw_geometry = getattr(polygon, "geometry")
        inferred_id = _normalise_fips(getattr(polygon, "fips_code", ""))
        inferred_crs = getattr(polygon, "crs", inferred_crs)
        properties = getattr(polygon, "properties", {}) or {}
    return str(polygon_id or inferred_id or "polygon"), raw_geometry, inferred_crs or DEFAULT_POLYGON_CRS, properties


def _transform_to_grid(geometry: Any, source_crs: Any, target_crs: Any) -> Any:
    runtime = _require_geometry_runtime()
    try:
        transformer = runtime[0].from_crs(source_crs, target_crs, always_xy=True)
        return runtime[4](transformer.transform, geometry)
    except Exception as exc:
        raise HrrrPolygonError(
            f"unable to transform polygon from {source_crs!r} to {target_crs!r}"
        ) from exc


def _chunk_key(value: Any, *, grid: HrrrGridSpec) -> tuple[int, int]:
    if isinstance(value, Mapping):
        y_value = value.get("chunk_y", value.get("y", value.get("grid_chunk_y")))
        x_value = value.get("chunk_x", value.get("x", value.get("grid_chunk_x")))
        if y_value is None or x_value is None:
            raise HrrrPolygonError(f"chunk mapping is missing y/x: {value!r}")
        y_chunk, x_chunk = int(y_value), int(x_value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise HrrrPolygonError(f"chunk tuple must contain y and x: {value!r}")
        y_chunk, x_chunk = int(value[0]), int(value[1])
    else:
        parts = str(value).strip().split(".")
        if len(parts) == 3 and parts[0] == "0":
            y_chunk, x_chunk = int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            y_chunk, x_chunk = int(parts[0]), int(parts[1])
        else:
            raise HrrrPolygonError(f"unsupported HRRR chunk id: {value!r}")
    if not (0 <= y_chunk < grid.y_chunks and 0 <= x_chunk < grid.x_chunks):
        raise HrrrPolygonError(f"HRRR chunk is outside grid: {y_chunk},{x_chunk}")
    return y_chunk, x_chunk


def _canonical_chunk(y_chunk: int, x_chunk: int) -> str:
    return f"0.{y_chunk}.{x_chunk}"


def _normalise_chunk_ids(
    chunk_ids: Iterable[Any] | Mapping[Any, Any] | str | None,
    *,
    grid: HrrrGridSpec,
) -> tuple[str, ...] | None:
    if chunk_ids is None:
        return None
    if isinstance(chunk_ids, str):
        values: Iterable[Any] = (chunk_ids,)
    elif isinstance(chunk_ids, Mapping):
        values = chunk_ids.keys()
    else:
        values = chunk_ids
    keys = {_chunk_key(value, grid=grid) for value in values}
    return tuple(_canonical_chunk(y, x) for y, x in sorted(keys))


def _cell_box(grid_y: int, grid_x: int, *, grid: HrrrGridSpec, box_factory: Any) -> Any:
    x0 = grid.x0_m + grid_x * grid.spacing_m
    y0 = grid.y0_m + grid_y * grid.spacing_m
    return box_factory(x0, y0, x0 + grid.spacing_m, y0 + grid.spacing_m)


def _cell_id(grid_y: int, grid_x: int) -> str:
    return f"y{grid_y:04d}_x{grid_x:04d}"


def _candidate_index_range(
    lower: float,
    upper: float,
    *,
    origin: float,
    spacing: float,
    count: int,
) -> range:
    start = max(0, math.floor((lower - origin) / spacing) - 1)
    stop = min(count - 1, math.ceil((upper - origin) / spacing) + 1)
    return range(start, stop + 1) if stop >= start else range(0, 0)


def _project_polygon_for_predicate(
    polygon: Any,
    *,
    polygon_crs: Any | None,
    grid_crs: Any,
) -> Any:
    polygon_id, raw_geometry, source_crs, _ = _feature_from_input(
        polygon, polygon_id=None, polygon_crs=polygon_crs
    )
    return _transform_to_grid(
        _validate_polygon_geometry(raw_geometry, polygon_id=polygon_id),
        source_crs,
        grid_crs,
    )


def _cell_argument_geometry(
    cell: Any,
    *,
    grid: HrrrGridSpec,
    cell_crs: Any,
) -> tuple[Any, Any]:
    runtime = _require_geometry_runtime()
    if isinstance(cell, HrrrCellIntersection):
        return runtime[2](cell.cell_x0_m, cell.cell_y0_m, cell.cell_x1_m, cell.cell_y1_m), HRRR_CRS
    if isinstance(cell, Mapping):
        raw = cell.get("cell_geometry", cell.get("geometry"))
        if raw is not None:
            return _geometry_from(raw), cell.get("crs", cell_crs)
        if "grid_y" in cell and "grid_x" in cell:
            return _cell_box(int(cell["grid_y"]), int(cell["grid_x"]), grid=grid, box_factory=runtime[2]), HRRR_CRS
        if "global_y" in cell and "global_x" in cell:
            return _cell_box(int(cell["global_y"]), int(cell["global_x"]), grid=grid, box_factory=runtime[2]), HRRR_CRS
        if "center_x_m" in cell and "center_y_m" in cell:
            half = grid.spacing_m / 2.0
            cx, cy = float(cell["center_x_m"]), float(cell["center_y_m"])
            return runtime[2](cx - half, cy - half, cx + half, cy + half), HRRR_CRS
    if isinstance(cell, Sequence) and not isinstance(cell, (str, bytes, bytearray)):
        if len(cell) != 2:
            raise HrrrPolygonError("cell tuple must contain grid_y and grid_x")
        return _cell_box(int(cell[0]), int(cell[1]), grid=grid, box_factory=runtime[2]), HRRR_CRS
    return _geometry_from(cell), cell_crs


def center_inside_polygon(
    polygon: Any,
    cell: Any,
    grid_x: int | None = None,
    *,
    polygon_crs: Any | None = None,
    cell_crs: Any = HRRR_CRS,
    grid_crs: Any = HRRR_CRS,
    grid: HrrrGridSpec = DEFAULT_HRRR_GRID,
) -> bool:
    """Return whether a cell center is covered by a polygon.

    ``cell`` may be a plan row, intersection object, ``(grid_y, grid_x)``,
    cell geometry, or scalar ``grid_y`` with ``grid_x`` supplied.  Boundary
    centers count as inside through Shapely ``covers`` semantics.
    """

    runtime = _require_geometry_runtime()
    target = _project_polygon_for_predicate(polygon, polygon_crs=polygon_crs, grid_crs=grid_crs)
    if isinstance(cell, int):
        if grid_x is None:
            raise HrrrPolygonError("grid_x is required when cell is a scalar grid_y")
        center = runtime[1](
            grid.x0_m + (int(grid_x) + 0.5) * grid.spacing_m,
            grid.y0_m + (int(cell) + 0.5) * grid.spacing_m,
        )
    elif isinstance(cell, HrrrCellIntersection):
        center = runtime[1](cell.center_x_m, cell.center_y_m)
    else:
        cell_geometry, source_crs = _cell_argument_geometry(cell, grid=grid, cell_crs=cell_crs)
        center = runtime[1](cell_geometry.x, cell_geometry.y) if cell_geometry.geom_type == "Point" else cell_geometry.centroid
        center = _transform_to_grid(center, source_crs, grid_crs)
    return bool(target.covers(center))


def boundary_cell(
    polygon: Any,
    cell: Any,
    grid_x: int | None = None,
    *,
    polygon_crs: Any | None = None,
    cell_crs: Any = HRRR_CRS,
    grid_crs: Any = HRRR_CRS,
    grid: HrrrGridSpec = DEFAULT_HRRR_GRID,
) -> bool:
    """Return whether an intersecting cell touches the polygon boundary."""

    runtime = _require_geometry_runtime()
    target = _project_polygon_for_predicate(polygon, polygon_crs=polygon_crs, grid_crs=grid_crs)
    if isinstance(cell, int):
        if grid_x is None:
            raise HrrrPolygonError("grid_x is required when cell is a scalar grid_y")
        cell_geometry = _cell_box(int(cell), int(grid_x), grid=grid, box_factory=runtime[2])
    else:
        cell_geometry, source_crs = _cell_argument_geometry(cell, grid=grid, cell_crs=cell_crs)
        cell_geometry = _transform_to_grid(cell_geometry, source_crs, grid_crs)
    if not bool(target.intersects(cell_geometry)):
        return False
    return bool(target.boundary.intersects(cell_geometry)) or not bool(target.contains(cell_geometry))


def _coverage_from_geometry(
    *,
    polygon_id: str,
    polygon_area_m2: float,
    intersection_geometries: Sequence[Any],
    requested_chunk_ids: tuple[str, ...],
    intersection_count: int,
    duplicate_cell_count: int,
    tolerance: float,
) -> HrrrCoverageReport:
    runtime = _require_geometry_runtime()
    if intersection_geometries:
        covered_area = float(runtime[5](list(intersection_geometries)).area)
    else:
        covered_area = 0.0
    if not math.isfinite(covered_area) or covered_area < 0.0:
        raise HrrrPolygonError(f"covered area is not finite for {polygon_id}")
    covered_area = min(max(covered_area, 0.0), polygon_area_m2)
    fraction = min(max(covered_area / polygon_area_m2, 0.0), 1.0) if polygon_area_m2 else 0.0
    return HrrrCoverageReport(
        polygon_id=polygon_id,
        polygon_area_m2=polygon_area_m2,
        covered_area_m2=covered_area,
        uncovered_area_m2=max(polygon_area_m2 - covered_area, 0.0),
        coverage_fraction=fraction,
        intersection_count=intersection_count,
        unique_cell_count=intersection_count,
        duplicate_cell_count=duplicate_cell_count,
        requested_chunk_ids=requested_chunk_ids,
        complete=fraction >= 1.0 - tolerance,
        tolerance=tolerance,
    )


def build_hrrr_polygon_plan(
    polygon: Any,
    *,
    polygon_id: str | None = None,
    polygon_crs: Any | None = None,
    chunk_ids: Iterable[Any] | Mapping[Any, Any] | str | None = None,
    grid_crs: Any = HRRR_CRS,
    grid: HrrrGridSpec = DEFAULT_HRRR_GRID,
    coverage_tolerance: float = 1e-9,
    require_full_coverage: bool = False,
) -> HrrrPolygonPlan:
    """Build a deterministic exact polygon/cell plan across any chunks."""

    tolerance = float(coverage_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
        raise HrrrPolygonError("coverage_tolerance must be finite and in [0, 1)")
    final_id, raw_geometry, source_crs, _ = _feature_from_input(
        polygon, polygon_id=polygon_id, polygon_crs=polygon_crs
    )
    projected = _transform_to_grid(
        _validate_polygon_geometry(raw_geometry, polygon_id=final_id),
        source_crs,
        grid_crs,
    )
    projected = _validate_polygon_geometry(projected, polygon_id=final_id)
    polygon_area = float(projected.area)
    if not math.isfinite(polygon_area) or polygon_area <= 0.0:
        raise HrrrPolygonError(f"polygon {final_id} has no positive projected area")

    requested = _normalise_chunk_ids(chunk_ids, grid=grid)
    allowed = None if requested is None else {_chunk_key(value, grid=grid) for value in requested}
    runtime = _require_geometry_runtime()
    min_x, min_y, max_x, max_y = projected.bounds
    x_range = _candidate_index_range(min_x, max_x, origin=grid.x0_m, spacing=grid.spacing_m, count=grid.x_cells)
    y_range = _candidate_index_range(min_y, max_y, origin=grid.y0_m, spacing=grid.spacing_m, count=grid.y_cells)

    seen: set[tuple[int, int]] = set()
    duplicate_count = 0
    candidates: list[tuple[int, int, Any, Any, float, bool, bool]] = []
    for grid_y in y_range:
        for grid_x in x_range:
            chunk_key = (grid_y // grid.chunk_size, grid_x // grid.chunk_size)
            if allowed is not None and chunk_key not in allowed:
                continue
            global_key = (grid_y, grid_x)
            if global_key in seen:
                duplicate_count += 1
                continue
            seen.add(global_key)
            cell_geometry = _cell_box(grid_y, grid_x, grid=grid, box_factory=runtime[2])
            intersection_geometry = projected.intersection(cell_geometry)
            if bool(getattr(intersection_geometry, "is_empty", False)):
                continue
            area = float(intersection_geometry.area)
            if not math.isfinite(area) or area <= 0.0:
                continue
            center = runtime[1](
                grid.x0_m + (grid_x + 0.5) * grid.spacing_m,
                grid.y0_m + (grid_y + 0.5) * grid.spacing_m,
            )
            candidates.append(
                (
                    grid_y,
                    grid_x,
                    cell_geometry,
                    intersection_geometry,
                    area,
                    bool(projected.covers(center)),
                    bool(projected.boundary.intersects(cell_geometry)),
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1]))
    coverage = _coverage_from_geometry(
        polygon_id=final_id,
        polygon_area_m2=polygon_area,
        intersection_geometries=[item[3] for item in candidates],
        requested_chunk_ids=requested or (),
        intersection_count=len(candidates),
        duplicate_cell_count=duplicate_count,
        tolerance=tolerance,
    )

    intersections: list[HrrrCellIntersection] = []
    for grid_y, grid_x, cell_geometry, _, area, center_inside, touches_boundary in candidates:
        min_cell_x, min_cell_y, max_cell_x, max_cell_y = cell_geometry.bounds
        y_chunk, x_chunk = grid_y // grid.chunk_size, grid_x // grid.chunk_size
        chunk = _canonical_chunk(y_chunk, x_chunk)
        intersections.append(
            HrrrCellIntersection(
                polygon_id=final_id,
                chunk_id=chunk,
                grid_y=grid_y,
                grid_x=grid_x,
                local_y=grid_y % grid.chunk_size,
                local_x=grid_x % grid.chunk_size,
                flat_index=(grid_y % grid.chunk_size) * grid.chunk_size + (grid_x % grid.chunk_size),
                cell_id=_cell_id(grid_y, grid_x),
                cell_x0_m=float(min_cell_x),
                cell_y0_m=float(min_cell_y),
                cell_x1_m=float(max_cell_x),
                cell_y1_m=float(max_cell_y),
                center_x_m=float((min_cell_x + max_cell_x) / 2.0),
                center_y_m=float((min_cell_y + max_cell_y) / 2.0),
                intersection_area_m2=area,
                polygon_area_m2=polygon_area,
                covered_area_m2=coverage.covered_area_m2,
                center_inside_polygon=center_inside,
                boundary_cell=touches_boundary,
            )
        )
    plan = HrrrPolygonPlan(
        polygon_id=final_id,
        polygon_area_m2=polygon_area,
        covered_area_m2=coverage.covered_area_m2,
        coverage_fraction=coverage.coverage_fraction,
        requested_chunk_ids=requested or (),
        intersections=tuple(intersections),
        coverage=coverage,
        grid=grid,
        polygon_crs=source_crs,
        grid_crs=grid_crs,
    )
    if require_full_coverage:
        plan.require_full_coverage()
    return plan


def build_hrrr_polygon_plans(
    polygons: Iterable[Any] | Mapping[Any, Any],
    *,
    polygon_crs: Any | None = None,
    chunk_ids: Iterable[Any] | Mapping[Any, Any] | str | None = None,
    chunk_ids_by_polygon: Mapping[str, Iterable[Any] | Mapping[Any, Any] | str | None] | None = None,
    grid_crs: Any = HRRR_CRS,
    grid: HrrrGridSpec = DEFAULT_HRRR_GRID,
    coverage_tolerance: float = 1e-9,
    require_full_coverage: bool = False,
) -> dict[str, HrrrPolygonPlan]:
    """Build plans for an iterable or FIPS-keyed mapping of polygons."""

    if isinstance(polygons, Mapping) and "geometry" not in polygons and "type" not in polygons:
        items = list(polygons.items())
    elif isinstance(polygons, (PolygonRecord, Mapping)):
        items = [(None, polygons)]
    else:
        items = [(None, value) for value in polygons]
    result: dict[str, HrrrPolygonPlan] = {}
    for key, polygon in items:
        inferred_id = None if key is None else str(key)
        final_id, _, _, _ = _feature_from_input(polygon, polygon_id=inferred_id, polygon_crs=polygon_crs)
        selected_chunks = (
            chunk_ids_by_polygon.get(final_id, chunk_ids)
            if chunk_ids_by_polygon is not None else chunk_ids
        )
        plan = build_hrrr_polygon_plan(
            polygon,
            polygon_id=final_id,
            polygon_crs=polygon_crs,
            chunk_ids=selected_chunks,
            grid_crs=grid_crs,
            grid=grid,
            coverage_tolerance=coverage_tolerance,
            require_full_coverage=require_full_coverage,
        )
        if plan.polygon_id in result:
            raise HrrrPolygonError(f"duplicate polygon ID in plan collection: {plan.polygon_id}")
        result[plan.polygon_id] = plan
    return result


def calculate_coverage(plan: HrrrPolygonPlan) -> HrrrCoverageReport:
    """Return the coverage report computed from the plan's unique cells."""

    if not isinstance(plan, HrrrPolygonPlan):
        raise HrrrPolygonError("calculate_coverage expects an HrrrPolygonPlan")
    return plan.coverage


def plans_to_polars(
    plans: HrrrPolygonPlan | Iterable[HrrrPolygonPlan] | Mapping[str, HrrrPolygonPlan],
) -> Any:
    """Combine plans into one Polars DataFrame."""

    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise HrrrPolygonDependencyError("Polars is required for plans_to_polars()") from exc
    if isinstance(plans, HrrrPolygonPlan):
        values = [plans]
    elif isinstance(plans, Mapping):
        values = list(plans.values())
    else:
        values = list(plans)
    rows = [row for plan in values for row in plan.to_rows()]
    return pl.DataFrame({column: [] for column in _PLAN_COLUMNS}) if not rows else pl.DataFrame(rows)


def _source_uris(path: Path) -> list[str]:
    if path.suffix.lower() != ".zip":
        return [str(path)]
    try:
        with zipfile.ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist() if name.lower().endswith(".shp"))
    except (OSError, zipfile.BadZipFile) as exc:
        raise HrrrPolygonError(f"invalid vector ZIP archive: {path}") from exc
    uris: list[str] = []
    for name in names:
        uris.extend((f"zip://{path}!{name}", f"/vsizip/{path.as_posix()}/{name}"))
    uris.append(f"zip://{path}")
    return uris


def _record_from_feature(
    feature: Mapping[str, Any],
    *,
    source_crs: Any,
    from_wkb: Any | None = None,
) -> PolygonRecord:
    properties_value = feature.get("properties")
    if isinstance(properties_value, Mapping):
        properties = dict(properties_value)
    else:
        properties = {
            key: value for key, value in feature.items() if key not in {"geometry", "wkb_geometry"}
        }
    raw_geometry = feature.get("geometry", feature.get("wkb_geometry"))
    if from_wkb is not None and isinstance(raw_geometry, (bytes, bytearray, memoryview)):
        raw_geometry = from_wkb(bytes(raw_geometry))
    geometry = _geometry_from(raw_geometry)
    fips = _normalise_fips(
        _first_value(properties, _FIPS_ALIASES, _first_value(feature, _FIPS_ALIASES, None)),
        allow_empty=False,
    )
    county = str(_first_value(properties, _COUNTY_ALIASES, "") or "")
    state_fips = _normalise_state_fips(_first_value(properties, _STATE_FIPS_ALIASES, ""))
    state = str(_first_value(properties, _STATE_ALIASES, "") or STATE_ABBR.get(state_fips, ""))
    return PolygonRecord(
        fips_code=fips,
        geometry=geometry,
        crs=source_crs or DEFAULT_TIGER_CRS,
        county=county,
        state=state,
        state_fips=state_fips,
        properties=properties,
    )


def _read_with_fiona(uri: str, *, layer: str | None) -> list[PolygonRecord] | None:
    try:
        import fiona  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        kwargs = {} if layer is None else {"layer": layer}
        with fiona.open(uri, **kwargs) as collection:
            source_crs = getattr(collection, "crs_wkt", None) or getattr(collection, "crs", None)
            source_crs = source_crs or DEFAULT_TIGER_CRS
            return [_record_from_feature(feature, source_crs=source_crs) for feature in collection]
    except Exception as exc:
        raise HrrrPolygonError(f"Fiona could not read vector source {uri}") from exc


def _read_with_pyogrio(uri: str, *, layer: str | None) -> list[PolygonRecord]:
    try:
        import pyogrio  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise HrrrPolygonDependencyError("reading polygons requires Fiona or pyogrio") from exc
    try:
        kwargs = {} if layer is None else {"layer": layer}
        metadata, table = pyogrio.read_arrow(uri, read_geometry=True, **kwargs)
        names = list(table.column_names)
        columns = {name: table[name].to_pylist() for name in names}
        geometry_name = str(metadata.get("geometry_name") or "")
        if geometry_name not in columns:
            geometry_name = next((name for name in names if name.lower() in {"geometry", "wkb_geometry"}), "")
        source_crs = metadata.get("crs") or DEFAULT_TIGER_CRS
        from_wkb = _require_geometry_runtime()[6]
        records: list[PolygonRecord] = []
        for index in range(int(table.num_rows)):
            feature = {name: columns[name][index] for name in names if name != geometry_name}
            if geometry_name:
                feature["wkb_geometry"] = columns[geometry_name][index]
            records.append(_record_from_feature(feature, source_crs=source_crs, from_wkb=from_wkb))
        return records
    except HrrrPolygonError:
        raise
    except Exception as exc:
        raise HrrrPolygonError(f"pyogrio could not read vector source {uri}") from exc


def _read_vector_records(path: Path, *, layer: str | None) -> list[PolygonRecord]:
    errors: list[str] = []
    for uri in _source_uris(path):
        try:
            records = _read_with_fiona(uri, layer=layer)
            if records is not None:
                return records
        except HrrrPolygonError as exc:
            errors.append(str(exc))
        try:
            return _read_with_pyogrio(uri, layer=layer)
        except HrrrPolygonError as exc:
            errors.append(str(exc))
    raise HrrrPolygonError(
        f"unable to read vector source {path}: {'; '.join(errors[-3:])}"
    )


def load_tiger_polygons(
    source: Path | str,
    *,
    fips_codes: Iterable[Any] | None = None,
    layer: str | None = None,
) -> dict[str, PolygonRecord]:
    """Read Census/TIGER polygons through Fiona or pyogrio, not a dataframe."""

    path = Path(source)
    if not path.exists():
        raise HrrrPolygonError(f"Census/TIGER source does not exist: {path}")
    wanted = None if fips_codes is None else {
        _normalise_fips(value, allow_empty=False) for value in fips_codes
    }
    records = _read_vector_records(path, layer=layer)
    result: dict[str, PolygonRecord] = {}
    for record in records:
        if wanted is not None and record.fips_code not in wanted:
            continue
        if record.fips_code in result:
            raise HrrrPolygonError(f"duplicate Census/TIGER FIPS/GEOID: {record.fips_code}")
        result[record.fips_code] = record
    if wanted is not None:
        missing = sorted(wanted - set(result))
        if missing:
            raise HrrrPolygonError(f"Census/TIGER source lacks FIPS/GEOID: {missing}")
    return dict(sorted(result.items()))


def chunk_rectangle(chunk_id: str, *, grid: HrrrGridSpec = DEFAULT_HRRR_GRID) -> Any:
    """Return the projected rectangle represented by one HRRR chunk."""

    runtime = _require_geometry_runtime()
    y_chunk, x_chunk = _chunk_key(chunk_id, grid=grid)
    y0 = y_chunk * grid.chunk_size
    x0 = x_chunk * grid.chunk_size
    y1 = min(y0 + grid.chunk_size, grid.y_cells)
    x1 = min(x0 + grid.chunk_size, grid.x_cells)
    return runtime[2](
        grid.x0_m + x0 * grid.spacing_m,
        grid.y0_m + y0 * grid.spacing_m,
        grid.x0_m + x1 * grid.spacing_m,
        grid.y0_m + y1 * grid.spacing_m,
    )


@dataclass(frozen=True)
class ProjectedCounty:
    """Compatibility record for the existing bulk spatial-plan helper."""

    fips_code: str
    county: str
    state: str
    state_fips: str
    geometry: Any
    county_area_m2: float


def load_county_geometries(
    archive: Path | str,
    fips_codes: Sequence[str],
) -> dict[str, ProjectedCounty]:
    """Load and project selected TIGER counties without intermediate tables."""

    requested = [_normalise_fips(value, allow_empty=False) for value in fips_codes]
    if len(requested) != len(set(requested)):
        raise HrrrPolygonError("county FIPS codes must be unique")
    records = load_tiger_polygons(archive, fips_codes=requested)
    result: dict[str, ProjectedCounty] = {}
    for fips in requested:
        record = records[fips]
        geometry = _transform_to_grid(
            _validate_polygon_geometry(record.geometry, polygon_id=fips),
            record.crs,
            HRRR_CRS,
        )
        if geometry.is_empty or not geometry.is_valid or geometry.area <= 0.0:
            raise HrrrPolygonError(f"invalid projected TIGER geometry for {fips}")
        state_fips = record.state_fips or fips[:2]
        state = record.state or STATE_ABBR.get(state_fips, state_fips)
        result[fips] = ProjectedCounty(
            fips_code=fips,
            county=record.county,
            state=state,
            state_fips=state_fips,
            geometry=geometry,
            county_area_m2=float(geometry.area),
        )
    return result


def coverage_status(ratio: float) -> str:
    if ratio >= COVERAGE_FULL_THRESHOLD:
        return "FULL"
    if ratio >= COVERAGE_NEAR_FULL_THRESHOLD:
        return "NEAR_FULL"
    return "PARTIAL_FAIL_FOR_FULL_POLYGON"


def build_spatial_plan(
    counties: Mapping[str, ProjectedCounty],
    chunk_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility wrapper returning plain rows and coverage rows."""

    unique_chunks = tuple(dict.fromkeys(str(value) for value in chunk_ids))
    if not unique_chunks:
        raise HrrrPolygonError("at least one HRRR chunk id is required")
    all_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for key, county in counties.items():
        plan = build_hrrr_polygon_plan(
            county.geometry,
            polygon_id=county.fips_code or str(key),
            polygon_crs=HRRR_CRS,
            chunk_ids=unique_chunks,
        )
        for cell in plan.intersections:
            all_rows.append({
                "fips_code": county.fips_code,
                "county": county.county,
                "state": county.state,
                "chunk_id": cell.chunk_id,
                "local_y": cell.local_y,
                "local_x": cell.local_x,
                "global_y": cell.grid_y,
                "global_x": cell.grid_x,
                "flat_index": cell.flat_index,
                "intersection_area_m2": cell.intersection_area_m2,
                "cell_area_m2": DEFAULT_HRRR_GRID.cell_area_m2,
                "cell_overlap_fraction": cell.intersection_area_m2 / DEFAULT_HRRR_GRID.cell_area_m2,
                "county_area_m2": county.county_area_m2,
                "county_weight": cell.area_weight,
                "center_inside_polygon": cell.center_inside_polygon,
                "boundary_cell": cell.boundary_cell,
            })
        covered_chunks = sorted({row["chunk_id"] for row in all_rows if row["fips_code"] == county.fips_code})
        boundary_area = sum(row["intersection_area_m2"] for row in all_rows if row["fips_code"] == county.fips_code and row["boundary_cell"])
        coverage_rows.append({
            "fips_code": county.fips_code,
            "county": county.county,
            "state": county.state,
            "county_area_m2": county.county_area_m2,
            "downloaded_overlap_area_m2": plan.covered_area_m2,
            "polygon_coverage_ratio": plan.coverage_fraction,
            "coverage_status": coverage_status(plan.coverage_fraction),
            "intersecting_cell_count": plan.cell_count,
            "chunk_ids": covered_chunks,
            "cross_chunk": len(covered_chunks) > 1,
            "boundary_area_ratio": boundary_area / plan.covered_area_m2 if plan.covered_area_m2 else 0.0,
        })
    if not all_rows:
        raise HrrrPolygonError("spatial plan contains no positive-area county/cell intersections")
    all_rows.sort(key=lambda row: (row["fips_code"], row["chunk_id"], row["local_y"], row["local_x"]))
    return all_rows, coverage_rows


def spatial_plan_frame(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Return compatibility rows as a Polars DataFrame."""

    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise HrrrPolygonDependencyError("Polars is required for spatial_plan_frame()") from exc
    columns = [
        "fips_code", "county", "state", "chunk_id", "local_y", "local_x",
        "global_y", "global_x", "flat_index", "intersection_area_m2",
        "cell_area_m2", "cell_overlap_fraction", "county_area_m2",
        "county_weight", "center_inside_polygon", "boundary_cell",
    ]
    return pl.DataFrame(list(rows)).select(columns)


def coverage_frame(rows: Sequence[Mapping[str, Any]]) -> Any:
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise HrrrPolygonDependencyError("Polars is required for coverage_frame()") from exc
    return pl.DataFrame(list(rows)).with_columns(pl.col("chunk_ids").cast(pl.List(pl.Utf8)))


build_polygon_spatial_plan = build_hrrr_polygon_plan
build_polygon_spatial_plans = build_hrrr_polygon_plans
build_exact_polygon_plan = build_hrrr_polygon_plan
read_census_tiger_polygons = load_tiger_polygons
coverage_report = calculate_coverage


def exact_polygon_cell_intersections(polygon: Any, **kwargs: Any) -> tuple[HrrrCellIntersection, ...]:
    """Return only the unique positive-area cells from a polygon plan."""

    return build_hrrr_polygon_plan(polygon, **kwargs).intersections


__all__ = [
    "BOUNDARY_EPSILON", "CORE_OVERLAP_FRACTION", "COVERAGE_FULL_THRESHOLD",
    "COVERAGE_NEAR_FULL_THRESHOLD", "DEFAULT_HRRR_GRID", "DEFAULT_POLYGON_CRS",
    "DEFAULT_TIGER_CRS", "HRRR_CRS", "HRRR_LCC_PROJ4", "HRRR_PROJ4",
    "HrrrCellIntersection", "HrrrCoverageReport", "HrrrGridSpec",
    "HrrrPolygonCoverageError", "HrrrPolygonDependencyError", "HrrrPolygonError",
    "HrrrPolygonPlan", "PolygonRecord", "ProjectedCounty", "STATE_ABBR",
    "build_exact_polygon_plan", "build_hrrr_polygon_plan", "build_hrrr_polygon_plans",
    "build_polygon_spatial_plan", "build_polygon_spatial_plans", "build_spatial_plan",
    "boundary_cell", "calculate_coverage", "center_inside_polygon", "chunk_rectangle",
    "coverage_frame", "coverage_report", "coverage_status", "exact_polygon_cell_intersections",
    "hrrr_crs", "load_county_geometries", "load_tiger_polygons", "plans_to_polars",
    "read_census_tiger_polygons", "spatial_plan_frame",
]
