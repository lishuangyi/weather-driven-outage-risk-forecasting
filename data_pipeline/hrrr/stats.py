"""Deterministic single-variable spatial robust statistics.

The functions in this module deliberately operate on one source variable at a
time.  They accept NumPy arrays because the local HRRR decoder is ndarray
based, while all table construction and Parquet writing remains in Polars in
the materialization CLI.
"""

from __future__ import annotations

from typing import Any

import numpy as np


STAT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
TAIL_FRACTIONS = (0.05, 0.10)


def _aligned_valid(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite values, positive weights, and the aligned validity mask."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size != weights.size:
        raise ValueError(f"values/weights length mismatch: {values.size} != {weights.size}")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if valid is not None:
        valid_array = np.asarray(valid, dtype=bool).reshape(-1)
        if valid_array.size != values.size:
            raise ValueError(f"valid mask length mismatch: {valid_array.size} != {values.size}")
        mask &= valid_array
    return values[mask], weights[mask], mask


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    values, weights, _ = _aligned_valid(values, weights)
    if values.size == 0:
        return None
    return float(np.sum(values * weights, dtype=np.float64) / np.sum(weights, dtype=np.float64))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float | None:
    values, weights, _ = _aligned_valid(values, weights)
    if values.size == 0:
        return None
    total = float(np.sum(weights, dtype=np.float64))
    mean = float(np.sum(values * weights, dtype=np.float64) / total)
    variance = float(np.sum(weights * (values - mean) ** 2, dtype=np.float64) / total)
    return float(np.sqrt(max(0.0, variance)))


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float | None:
    """Return a deterministic area-weighted empirical quantile.

    Values are sorted stably and the first value whose cumulative normalized
    positive area reaches ``quantile`` is selected.  This is a step-function
    weighted empirical distribution, which is deterministic for tied values
    and does not silently revert to equal-cell percentiles.
    """

    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    values, weights, _ = _aligned_valid(values, weights)
    if values.size == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    total = float(cumulative[-1])
    if quantile <= 0.0:
        return float(sorted_values[0])
    if quantile >= 1.0:
        return float(sorted_values[-1])
    index = int(np.searchsorted(cumulative / total, float(quantile), side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def area_tail_mean(
    values: np.ndarray,
    weights: np.ndarray,
    fraction: float,
    *,
    high: bool,
) -> float | None:
    """Mean over the top/bottom represented county-area fraction.

    The last cell crossing the requested cumulative-area boundary contributes
    only the needed fraction of its area, so this is area-tail aggregation and
    not a top-N-cell shortcut.
    """

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    values, weights, _ = _aligned_valid(values, weights)
    if values.size == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    if high:
        order = order[::-1]
    values = values[order]
    weights = weights[order]
    target = float(np.sum(weights, dtype=np.float64)) * float(fraction)
    remaining = target
    selected_weight = 0.0
    selected_value = 0.0
    for value, weight in zip(values, weights, strict=True):
        take = min(float(weight), remaining)
        if take > 0.0:
            selected_value += float(value) * take
            selected_weight += take
            remaining -= take
        if remaining <= 1e-12 * max(1.0, target):
            break
    if selected_weight <= 0.0:
        return None
    return float(selected_value / selected_weight)


def _extreme(values: np.ndarray, weights: np.ndarray, *, minimum: bool) -> float | None:
    values, _, _ = _aligned_valid(values, weights)
    if values.size == 0:
        return None
    return float(np.min(values) if minimum else np.max(values))


def robust_spatial_stats(
    values: np.ndarray,
    intersection_area_m2: np.ndarray,
    *,
    center_inside_polygon: np.ndarray,
    cell_overlap_fraction: np.ndarray,
    boundary_cell: np.ndarray,
    valid_mask: np.ndarray | None = None,
    intersecting_cell_count: int | None = None,
) -> dict[str, Any]:
    """Calculate all required robust statistics for one variable/time row."""

    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    areas = np.asarray(intersection_area_m2, dtype=np.float64).reshape(-1)
    centers = np.asarray(center_inside_polygon, dtype=bool).reshape(-1)
    overlaps = np.asarray(cell_overlap_fraction, dtype=np.float64).reshape(-1)
    boundaries = np.asarray(boundary_cell, dtype=bool).reshape(-1)
    lengths = {values_array.size, areas.size, centers.size, overlaps.size, boundaries.size}
    if valid_mask is not None:
        lengths.add(np.asarray(valid_mask).reshape(-1).size)
    if len(lengths) != 1:
        raise ValueError(f"spatial stat input length mismatch: {sorted(lengths)}")

    base_mask = np.isfinite(areas) & (areas > 0.0)
    total_area = float(np.sum(areas[base_mask], dtype=np.float64))
    if total_area <= 0.0:
        valid = np.zeros(values_array.size, dtype=bool)
    else:
        valid = base_mask & np.isfinite(values_array)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid_area = float(np.sum(areas[valid], dtype=np.float64))
    valid_area_ratio = valid_area / total_area if total_area > 0.0 else 0.0
    effective_n = (
        float(valid_area**2 / np.sum(areas[valid] ** 2, dtype=np.float64))
        if np.any(valid)
        else None
    )

    valid_values = values_array[valid]
    valid_weights = areas[valid]
    core = valid & (centers | (overlaps >= 0.25))
    core_values = values_array[core]
    core_weights = areas[core]
    boundary_area = float(np.sum(areas[base_mask & boundaries], dtype=np.float64))
    result: dict[str, Any] = {
        "area_mean": weighted_mean(valid_values, valid_weights),
        "area_std": weighted_std(valid_values, valid_weights),
        "raw_min": _extreme(valid_values, valid_weights, minimum=True),
        "raw_max": _extreme(valid_values, valid_weights, minimum=False),
        "weighted_p05": weighted_quantile(valid_values, valid_weights, 0.05),
        "weighted_p10": weighted_quantile(valid_values, valid_weights, 0.10),
        "weighted_p25": weighted_quantile(valid_values, valid_weights, 0.25),
        "weighted_p50": weighted_quantile(valid_values, valid_weights, 0.50),
        "weighted_p75": weighted_quantile(valid_values, valid_weights, 0.75),
        "weighted_p90": weighted_quantile(valid_values, valid_weights, 0.90),
        "weighted_p95": weighted_quantile(valid_values, valid_weights, 0.95),
        "weighted_p99": weighted_quantile(valid_values, valid_weights, 0.99),
        "core_min": _extreme(core_values, core_weights, minimum=True),
        "core_max": _extreme(core_values, core_weights, minimum=False),
        "top05_area_mean": area_tail_mean(valid_values, valid_weights, 0.05, high=True),
        "top10_area_mean": area_tail_mean(valid_values, valid_weights, 0.10, high=True),
        "bottom05_area_mean": area_tail_mean(valid_values, valid_weights, 0.05, high=False),
        "bottom10_area_mean": area_tail_mean(valid_values, valid_weights, 0.10, high=False),
        "intersecting_cell_count": int(
            intersecting_cell_count if intersecting_cell_count is not None else int(np.count_nonzero(base_mask))
        ),
        "valid_cell_count": int(np.count_nonzero(valid)),
        "valid_area_ratio": float(valid_area_ratio),
        "effective_cell_count": effective_n,
        "boundary_area_ratio": float(boundary_area / total_area) if total_area > 0.0 else 0.0,
    }
    p25 = result["weighted_p25"]
    p75 = result["weighted_p75"]
    result["iqr"] = float(p75 - p25) if p25 is not None and p75 is not None else None
    raw_min = result["raw_min"]
    raw_max = result["raw_max"]
    result["spatial_range"] = float(raw_max - raw_min) if raw_min is not None and raw_max is not None else None
    core_min = result["core_min"]
    core_max = result["core_max"]
    result["core_range"] = float(core_max - core_min) if core_min is not None and core_max is not None else None
    return result

