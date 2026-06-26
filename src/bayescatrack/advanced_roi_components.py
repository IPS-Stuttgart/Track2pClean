"""Optional advanced ROI evidence components for BayesCaTrack.

The core cost builder already exposes strong overlap, centroid, covariance and
Suite2p-stat evidence.  This module adds zero-default diagnostics and optional
cost terms for the next round of result-improvement experiments:

* rotation/scale-tolerant mask-shape descriptors,
* FOV-border and empty/fragile-mask quality indicators,
* row/column assignment margins and mutual-rank ambiguity costs, and
* top-k candidate graph pruning before the global solver sees impossible edges.

The extension is installed by :mod:`bayescatrack.__init__` and only changes costs
when one of the new weights or pruning knobs is supplied through
``pairwise_cost_kwargs``.  Requesting ``return_components=True`` records the new
components for calibration, error taxonomy and ablations.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
from bayescatrack.core import _bridge_impl
from bayescatrack.core.bridge import CalciumPlaneData


@dataclass(frozen=True)
class CandidatePruningConfig:
    """Top-k candidate pruning configuration for pairwise ROI costs."""

    top_k_per_roi: int | None = None
    include_column_top_k: bool = True
    gate_margin: float | None = None
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        _normalize_optional_positive_int(self.top_k_per_roi, name="top_k_per_roi")
        _normalize_bool(self.include_column_top_k, name="include_column_top_k")
        _normalize_optional_nonnegative_float(self.gate_margin, name="gate_margin")
        _normalize_positive_float(self.large_cost, name="large_cost")


def install_advanced_roi_components() -> None:
    """Install advanced pairwise components on ``CalciumPlaneData``.

    Installation is idempotent.  The patch keeps historical behavior unchanged
    unless advanced weights, component flags or candidate pruning kwargs are set.
    """

    original = CalciumPlaneData.build_pairwise_cost_matrix
    if _pairwise_method_chain_has_patch(original, "_bayescatrack_advanced_roi_patch"):
        return

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    def _build_pairwise_cost_matrix_with_advanced_components(
        self: CalciumPlaneData,
        other: CalciumPlaneData,
        *args: Any,
        shape_descriptor_components: bool = False,
        radial_profile_weight: float = 0.0,
        orientation_weight: float = 0.0,
        eccentricity_weight: float = 0.0,
        compactness_weight: float = 0.0,
        border_proximity_weight: float = 0.0,
        ambiguity_margin_components: bool = False,
        ambiguity_margin_weight: float = 0.0,
        candidate_top_k_per_roi: int | None = None,
        candidate_include_column_top_k: bool = True,
        candidate_gate_margin: float | None = None,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        return_components = bool(kwargs.pop("return_components", False))
        weights = {
            "radial_profile_weight": float(radial_profile_weight),
            "orientation_weight": float(orientation_weight),
            "eccentricity_weight": float(eccentricity_weight),
            "compactness_weight": float(compactness_weight),
            "border_proximity_weight": float(border_proximity_weight),
            "ambiguity_margin_weight": float(ambiguity_margin_weight),
        }
        for weight_name, weight_value in weights.items():
            if weight_value < 0.0:
                raise ValueError(f"{weight_name} must be non-negative")

        large_cost = float(kwargs.get("large_cost", 1.0e6))
        pruning = CandidatePruningConfig(
            top_k_per_roi=candidate_top_k_per_roi,
            include_column_top_k=candidate_include_column_top_k,
            gate_margin=candidate_gate_margin,
            large_cost=large_cost,
        )
        needs_advanced = (
            return_components
            or shape_descriptor_components
            or any(weight_value > 0.0 for weight_value in weights.values())
            or ambiguity_margin_components
            or pruning.top_k_per_roi is not None
            or pruning.gate_margin is not None
        )
        if not needs_advanced:
            return original(
                self,
                other,
                *args,
                return_components=return_components,
                **kwargs,
            )

        base_cost, components = original(
            self,
            other,
            *args,
            return_components=True,
            **kwargs,
        )
        total_cost = np.asarray(base_cost, dtype=float).copy()
        components = dict(components)

        needs_shape = (
            return_components
            or shape_descriptor_components
            or weights["radial_profile_weight"] > 0.0
            or weights["orientation_weight"] > 0.0
            or weights["eccentricity_weight"] > 0.0
            or weights["compactness_weight"] > 0.0
            or weights["border_proximity_weight"] > 0.0
        )
        if needs_shape:
            shape_components = pairwise_shape_descriptor_components(self, other)
            components.update(shape_components)
            if weights["radial_profile_weight"] > 0.0:
                total_cost += (
                    weights["radial_profile_weight"]
                    * shape_components["radial_profile_cost"]
                )
            if weights["orientation_weight"] > 0.0:
                total_cost += (
                    weights["orientation_weight"] * shape_components["orientation_cost"]
                )
            if weights["eccentricity_weight"] > 0.0:
                total_cost += (
                    weights["eccentricity_weight"]
                    * shape_components["eccentricity_cost"]
                )
            if weights["compactness_weight"] > 0.0:
                total_cost += (
                    weights["compactness_weight"] * shape_components["compactness_cost"]
                )
            if weights["border_proximity_weight"] > 0.0:
                total_cost += (
                    weights["border_proximity_weight"]
                    * shape_components["border_proximity_cost"]
                )

        needs_margins = (
            return_components
            or ambiguity_margin_components
            or weights["ambiguity_margin_weight"] > 0.0
            or pruning.top_k_per_roi is not None
            or pruning.gate_margin is not None
        )
        if needs_margins:
            margin_components = pairwise_cost_margin_components(
                total_cost,
                large_cost=large_cost,
            )
            components.update(margin_components)
            if weights["ambiguity_margin_weight"] > 0.0:
                total_cost += (
                    weights["ambiguity_margin_weight"]
                    * margin_components["ambiguity_margin_cost"]
                )

        if pruning.top_k_per_roi is not None or pruning.gate_margin is not None:
            admitted = candidate_mask_from_cost_matrix(
                total_cost,
                top_k=pruning.top_k_per_roi,
                include_columns=pruning.include_column_top_k,
                gate_margin=pruning.gate_margin,
                large_cost=pruning.large_cost,
            )
            total_cost = np.where(admitted, total_cost, pruning.large_cost)
            components["candidate_admitted"] = admitted.astype(float)
            components["candidate_pruned"] = (~admitted).astype(float)

        total_cost = (
            _bridge_impl._ensure_finite_cost_matrix(  # pylint: disable=protected-access
                total_cost,
                large_cost=large_cost,
            )
        )
        components["pairwise_cost_matrix"] = total_cost
        if return_components:
            return total_cost, components
        return total_cost

    setattr(
        _build_pairwise_cost_matrix_with_advanced_components,
        "_bayescatrack_advanced_roi_patch",
        True,
    )
    setattr(
        _build_pairwise_cost_matrix_with_advanced_components,
        "_bayescatrack_original",
        original,
    )
    CalciumPlaneData.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
        _build_pairwise_cost_matrix_with_advanced_components
    )


def _pairwise_method_chain_has_patch(method: Any, marker: str) -> bool:
    """Return whether a pairwise-cost wrapper chain contains ``marker``."""

    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        seen.add(current_id)
        if getattr(current, marker, False):
            return True
        current = getattr(current, "_bayescatrack_original", None)
    return False


def pairwise_shape_descriptor_components(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    radial_bins: int = 6,
) -> dict[str, np.ndarray]:
    """Return pairwise mask-shape descriptor costs for two ROI planes."""

    reference = mask_shape_descriptors(
        reference_plane.roi_masks, radial_bins=radial_bins
    )
    measurement = mask_shape_descriptors(
        measurement_plane.roi_masks,
        radial_bins=radial_bins,
    )
    shape = (reference["area"].shape[0], measurement["area"].shape[0])
    if shape[0] == 0 or shape[1] == 0:
        empty = np.zeros(shape, dtype=float)
        return {
            "radial_profile_cost": empty.copy(),
            "orientation_cost": empty.copy(),
            "eccentricity_cost": empty.copy(),
            "compactness_cost": empty.copy(),
            "border_proximity_cost": empty.copy(),
            "empty_mask_pair_indicator": empty.copy(),
        }

    radial_profile_cost = np.mean(
        np.abs(
            reference["radial_profile"][:, None, :]
            - measurement["radial_profile"][None, :, :]
        ),
        axis=2,
    )
    angle_delta = (
        reference["orientation"][:, None] - measurement["orientation"][None, :]
    )
    orientation_cost = np.abs(np.sin(angle_delta))
    eccentricity_cost = np.abs(
        reference["eccentricity"][:, None] - measurement["eccentricity"][None, :]
    )
    compactness_cost = np.abs(
        reference["compactness"][:, None] - measurement["compactness"][None, :]
    )
    border_proximity_cost = 0.5 * (
        reference["border_proximity"][:, None]
        + measurement["border_proximity"][None, :]
    )
    empty_pair = reference["empty_mask"][:, None] | measurement["empty_mask"][None, :]
    return {
        "radial_profile_cost": _finite_nonnegative(radial_profile_cost),
        "orientation_cost": _finite_nonnegative(orientation_cost),
        "eccentricity_cost": _finite_nonnegative(eccentricity_cost),
        "compactness_cost": _finite_nonnegative(compactness_cost),
        "border_proximity_cost": _finite_nonnegative(border_proximity_cost),
        "empty_mask_pair_indicator": empty_pair.astype(float),
    }


def mask_shape_descriptors(
    masks: np.ndarray, *, radial_bins: int = 6
) -> dict[str, np.ndarray]:
    """Return per-ROI shape descriptors from a mask stack."""

    mask_array = np.asarray(masks, dtype=float)
    if mask_array.ndim != 3:
        raise ValueError("masks must have shape (n_roi, height, width)")
    if radial_bins < 1:
        raise ValueError("radial_bins must be positive")

    n_rois, height, width = mask_array.shape
    area = np.zeros((n_rois,), dtype=float)
    compactness = np.zeros((n_rois,), dtype=float)
    eccentricity = np.zeros((n_rois,), dtype=float)
    orientation = np.zeros((n_rois,), dtype=float)
    border_proximity = np.zeros((n_rois,), dtype=float)
    radial_profile = np.zeros((n_rois, int(radial_bins)), dtype=float)
    empty_mask = np.zeros((n_rois,), dtype=bool)

    for roi_index, mask in enumerate(mask_array):
        support_y, support_x = np.nonzero(mask > 0.0)
        if support_y.size == 0:
            empty_mask[roi_index] = True
            continue
        weights = np.asarray(mask[support_y, support_x], dtype=float)
        weights = np.maximum(np.nan_to_num(weights, nan=0.0), 0.0)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones_like(support_y, dtype=float)
        weight_sum = float(np.sum(weights))
        area[roi_index] = float(support_y.size)
        y_min, y_max = int(np.min(support_y)), int(np.max(support_y))
        x_min, x_max = int(np.min(support_x)), int(np.max(support_x))
        bbox_area = max((y_max - y_min + 1) * (x_max - x_min + 1), 1)
        compactness[roi_index] = float(support_y.size) / float(bbox_area)
        border_pixels = (
            (support_y == 0)
            | (support_y == height - 1)
            | (support_x == 0)
            | (support_x == width - 1)
        )
        border_proximity[roi_index] = float(np.mean(border_pixels))

        centroid_y = float(np.dot(support_y, weights) / weight_sum)
        centroid_x = float(np.dot(support_x, weights) / weight_sum)
        centered = np.vstack((support_x - centroid_x, support_y - centroid_y))
        covariance = (centered * weights[None, :]) @ centered.T / weight_sum
        eigvals, eigvecs = np.linalg.eigh(0.5 * (covariance + covariance.T))
        eigvals = np.maximum(eigvals, 0.0)
        major = float(max(eigvals[-1], 1.0e-12))
        minor = float(max(eigvals[0], 0.0))
        eccentricity[roi_index] = float(np.sqrt(max(0.0, 1.0 - minor / major)))
        major_vector = eigvecs[:, int(np.argmax(eigvals))]
        orientation[roi_index] = float(np.arctan2(major_vector[1], major_vector[0]))

        distances = np.sqrt(
            (support_x - centroid_x) ** 2 + (support_y - centroid_y) ** 2
        )
        max_distance = float(np.max(distances)) if distances.size else 0.0
        if max_distance <= 0.0:
            radial_profile[roi_index, 0] = 1.0
        else:
            normalized = np.clip(distances / max_distance, 0.0, 1.0)
            bin_indices = np.minimum(
                (normalized * int(radial_bins)).astype(int), int(radial_bins) - 1
            )
            histogram = np.bincount(
                bin_indices,
                weights=weights,
                minlength=int(radial_bins),
            ).astype(float)
            radial_profile[roi_index] = histogram / max(
                float(np.sum(histogram)), 1.0e-12
            )

    return {
        "area": area,
        "compactness": np.clip(compactness, 0.0, 1.0),
        "eccentricity": np.clip(eccentricity, 0.0, 1.0),
        "orientation": orientation,
        "border_proximity": np.clip(border_proximity, 0.0, 1.0),
        "radial_profile": radial_profile,
        "empty_mask": empty_mask,
    }


def pairwise_cost_margin_components(
    cost_matrix: np.ndarray,
    *,
    large_cost: float = 1.0e6,
) -> dict[str, np.ndarray]:
    """Return row/column rank and ambiguity-margin components for a cost matrix."""

    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    n_rows, n_cols = costs.shape
    if n_rows == 0 or n_cols == 0:
        empty = np.zeros(costs.shape, dtype=float)
        return {
            "row_rank_cost": empty.copy(),
            "column_rank_cost": empty.copy(),
            "mutual_rank_cost": empty.copy(),
            "row_assignment_margin": empty.copy(),
            "column_assignment_margin": empty.copy(),
            "ambiguity_margin_cost": empty.copy(),
        }

    finite = np.isfinite(costs) & (costs < large_cost)
    safe_costs = np.where(finite, costs, large_cost)
    row_rank = _rank_along_axis(safe_costs, axis=1)
    column_rank = _rank_along_axis(safe_costs, axis=0)
    row_scale = max(n_cols - 1, 1)
    column_scale = max(n_rows - 1, 1)
    row_rank_cost = row_rank / row_scale
    column_rank_cost = column_rank / column_scale

    row_margin = _best_second_margin(safe_costs, axis=1)
    column_margin = _best_second_margin(safe_costs, axis=0)
    row_margin_matrix = np.broadcast_to(row_margin[:, None], costs.shape).astype(float)
    column_margin_matrix = np.broadcast_to(column_margin[None, :], costs.shape).astype(
        float
    )
    combined_margin = np.minimum(row_margin_matrix, column_margin_matrix)
    ambiguity_margin_cost = 1.0 / (1.0 + np.maximum(combined_margin, 0.0))
    ambiguity_margin_cost = np.where(finite, ambiguity_margin_cost, 1.0)

    return {
        "row_rank_cost": _finite_nonnegative(row_rank_cost),
        "column_rank_cost": _finite_nonnegative(column_rank_cost),
        "mutual_rank_cost": _finite_nonnegative(
            0.5 * (row_rank_cost + column_rank_cost)
        ),
        "row_assignment_margin": _finite_nonnegative(row_margin_matrix),
        "column_assignment_margin": _finite_nonnegative(column_margin_matrix),
        "ambiguity_margin_cost": _finite_nonnegative(ambiguity_margin_cost),
    }


def candidate_mask_from_cost_matrix(
    cost_matrix: np.ndarray,
    *,
    top_k: int | None,
    include_columns: bool = True,
    gate_margin: float | None = None,
    large_cost: float = 1.0e6,
) -> np.ndarray:
    """Return a sparse candidate mask using row/column top-k and optional margins."""

    top_k = _normalize_optional_positive_int(top_k, name="top_k")
    include_columns = _normalize_bool(include_columns, name="include_columns")
    gate_margin = _normalize_optional_nonnegative_float(gate_margin, name="gate_margin")
    large_cost = _normalize_positive_float(large_cost, name="large_cost")

    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    admitted = np.isfinite(costs) & (costs < large_cost)
    if top_k is not None:
        top_mask = np.zeros(costs.shape, dtype=bool)
        for row_index in range(costs.shape[0]):
            candidates = np.flatnonzero(admitted[row_index])
            if candidates.size:
                order = np.argsort(costs[row_index, candidates], kind="stable")
                top_mask[row_index, candidates[order[:top_k]]] = True
        if include_columns:
            for column_index in range(costs.shape[1]):
                candidates = np.flatnonzero(admitted[:, column_index])
                if candidates.size:
                    order = np.argsort(costs[candidates, column_index], kind="stable")
                    top_mask[candidates[order[:top_k]], column_index] = True
        admitted &= top_mask
    if gate_margin is not None:
        safe = np.where(np.isfinite(costs), costs, large_cost)
        row_best = np.min(safe, axis=1, keepdims=True)
        col_best = np.min(safe, axis=0, keepdims=True)
        margin_mask = (safe <= row_best + gate_margin) | (
            safe <= col_best + gate_margin
        )
        admitted &= margin_mask
    return admitted


def _rank_along_axis(values: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, axis=axis, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    if axis == 1:
        ranks[np.arange(values.shape[0])[:, None], order] = np.arange(values.shape[1])[
            None, :
        ]
    elif axis == 0:
        ranks[order, np.arange(values.shape[1])[None, :]] = np.arange(values.shape[0])[
            :, None
        ]
    else:
        raise ValueError("axis must be 0 or 1")
    return ranks


def _best_second_margin(values: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if axis == 1:
        sorted_values = np.sort(values, axis=1)
        if values.shape[1] == 1:
            return np.full((values.shape[0],), np.inf, dtype=float)
        return sorted_values[:, 1] - sorted_values[:, 0]
    if axis == 0:
        sorted_values = np.sort(values, axis=0)
        if values.shape[0] == 1:
            return np.full((values.shape[1],), np.inf, dtype=float)
        return sorted_values[1, :] - sorted_values[0, :]
    raise ValueError("axis must be 0 or 1")


def _finite_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.nan_to_num(
            np.asarray(values, dtype=float), nan=0.0, posinf=1.0e6, neginf=0.0
        ),
        0.0,
    )


def _normalize_optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{name} must be a positive integer or None")
    try:
        normalized = operator.index(value)
    except TypeError:
        if not isinstance(value, (float, np.floating)):
            raise ValueError(f"{name} must be a positive integer or None") from None
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer or None")
        normalized = int(numeric)
    normalized = int(normalized)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return normalized


def _normalize_optional_nonnegative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{name} must be a finite non-negative value or None")
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative value or None") from exc
    if array.shape != ():
        raise ValueError(f"{name} must be a finite non-negative value or None")
    try:
        normalized = float(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative value or None") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value or None")
    return normalized


def _normalize_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{name} must be a positive finite value")
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite value") from exc
    if array.shape != ():
        raise ValueError(f"{name} must be a positive finite value")
    try:
        normalized = float(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite value") from exc
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return normalized


def _normalize_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


__all__ = [
    "CandidatePruningConfig",
    "candidate_mask_from_cost_matrix",
    "install_advanced_roi_components",
    "mask_shape_descriptors",
    "pairwise_cost_margin_components",
    "pairwise_shape_descriptor_components",
]
