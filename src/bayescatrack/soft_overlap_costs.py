"""Soft ROI-overlap costs for near-miss registered masks.

This module extends :meth:`CalciumPlaneData.build_pairwise_cost_matrix` with
optional soft overlap terms without changing the public bridge data model.  The
terms are useful when growth-aware registration places a true ROI very near the
reference ROI, but exact-pixel IoU remains zero.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from bayescatrack.core import _bridge_impl
from bayescatrack.core.bridge import CalciumPlaneData


def registered_soft_iou_cost_kwargs(
    *,
    similarity_epsilon: float = 1.0e-6,
    soft_iou_radius: int = 2,
    distance_transform_overlap_radius: int = 3,
    distance_transform_overlap_weight: float = 0.35,
    distance_transform_overlap_scale: float | None = None,
) -> dict[str, float | int | None]:
    """Return kwargs for a registered near-miss-overlap ablation.

    The preset deliberately disables centroid, feature, area and cell-probability
    terms, matching ``registered-iou`` in spirit while replacing exact support
    overlap by two soft overlap terms.
    """

    similarity_epsilon = _finite_positive_float(
        similarity_epsilon, name="similarity_epsilon"
    )
    soft_iou_radius = _nonnegative_int(soft_iou_radius, name="soft_iou_radius")
    distance_transform_overlap_radius = _nonnegative_int(
        distance_transform_overlap_radius,
        name="distance_transform_overlap_radius",
    )
    distance_transform_overlap_weight = _finite_nonnegative_float(
        distance_transform_overlap_weight,
        name="distance_transform_overlap_weight",
    )
    if distance_transform_overlap_scale is not None:
        distance_transform_overlap_scale = _finite_positive_float(
            distance_transform_overlap_scale,
            name="distance_transform_overlap_scale",
        )

    return {
        "centroid_weight": 0.0,
        "iou_weight": 0.0,
        "soft_iou_weight": 1.0,
        "soft_iou_radius": soft_iou_radius,
        "distance_transform_overlap_weight": distance_transform_overlap_weight,
        "distance_transform_overlap_radius": distance_transform_overlap_radius,
        "distance_transform_overlap_scale": distance_transform_overlap_scale,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "similarity_epsilon": similarity_epsilon,
    }


def install_soft_overlap_costs() -> None:
    """Install soft-overlap cost extensions and cost presets."""

    _install_cost_matrix_patch()
    _install_registration_qa_preset()


def _pairwise_method_chain_has_patch(method: Any, marker: str) -> bool:
    """Return whether a pairwise-cost wrapper chain contains ``marker``."""

    seen: set[int] = set()
    current: Any = method
    while True:
        if current is None:
            return False
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, marker, False):
            return True
        seen.add(current_id)
        current = getattr(current, "_bayescatrack_original", None)


def _install_cost_matrix_patch() -> None:
    original = CalciumPlaneData.build_pairwise_cost_matrix
    if _pairwise_method_chain_has_patch(original, "_bayescatrack_soft_overlap_patch"):
        return

    # pylint: disable=too-many-arguments,too-many-locals
    def _build_pairwise_cost_matrix_with_soft_overlap(
        self: CalciumPlaneData,
        other: CalciumPlaneData,
        *args: Any,
        soft_iou_weight: float = 0.0,
        soft_iou_radius: int = 0,
        distance_transform_overlap_weight: float = 0.0,
        distance_transform_overlap_radius: int = 0,
        distance_transform_overlap_scale: float | None = None,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        soft_iou_weight = _finite_nonnegative_float(
            soft_iou_weight,
            name="soft_iou_weight",
        )
        distance_transform_overlap_weight = _finite_nonnegative_float(
            distance_transform_overlap_weight,
            name="distance_transform_overlap_weight",
        )
        soft_iou_radius = _nonnegative_int(soft_iou_radius, name="soft_iou_radius")
        distance_transform_overlap_radius = _nonnegative_int(
            distance_transform_overlap_radius,
            name="distance_transform_overlap_radius",
        )
        if distance_transform_overlap_scale is not None:
            distance_transform_overlap_scale = _finite_positive_float(
                distance_transform_overlap_scale,
                name="distance_transform_overlap_scale",
            )
        similarity_epsilon = _finite_positive_float(
            kwargs.get("similarity_epsilon", 1.0e-6),
            name="similarity_epsilon",
        )
        large_cost = _finite_positive_float(
            kwargs.get("large_cost", 1.0e6),
            name="large_cost",
        )
        kwargs["similarity_epsilon"] = similarity_epsilon
        kwargs["large_cost"] = large_cost

        return_components = bool(kwargs.pop("return_components", False))
        needs_soft_components = return_components and (
            soft_iou_radius > 0 or distance_transform_overlap_radius > 0
        )
        needs_soft_cost = (
            soft_iou_weight > 0.0 or distance_transform_overlap_weight > 0.0
        )
        if not needs_soft_components and not needs_soft_cost:
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

        if soft_iou_weight > 0.0 or (return_components and soft_iou_radius > 0):
            soft_iou = _pairwise_dilated_iou_matrix(
                self.roi_masks,
                other.roi_masks,
                radius=soft_iou_radius,
            )
            soft_iou_cost = -np.log(np.clip(soft_iou, similarity_epsilon, 1.0))
            if soft_iou_weight > 0.0:
                total_cost += soft_iou_weight * soft_iou_cost
        else:
            soft_iou = np.zeros_like(total_cost)
            soft_iou_cost = np.zeros_like(total_cost)

        if distance_transform_overlap_weight > 0.0 or (
            return_components and distance_transform_overlap_radius > 0
        ):
            distance_transform_overlap = _pairwise_distance_transform_overlap_matrix(
                self.roi_masks,
                other.roi_masks,
                radius=distance_transform_overlap_radius,
                distance_scale=distance_transform_overlap_scale,
            )
            distance_transform_overlap_cost = -np.log(
                np.clip(distance_transform_overlap, similarity_epsilon, 1.0)
            )
            if distance_transform_overlap_weight > 0.0:
                total_cost += (
                    distance_transform_overlap_weight * distance_transform_overlap_cost
                )
        else:
            distance_transform_overlap = np.zeros_like(total_cost)
            distance_transform_overlap_cost = np.zeros_like(total_cost)

        total_cost = (
            _bridge_impl._ensure_finite_cost_matrix(  # pylint: disable=protected-access
                total_cost,
                large_cost=large_cost,
            )
        )
        if not return_components:
            return total_cost
        components = dict(components)
        components.update(
            {
                "pairwise_cost_matrix": total_cost,
                "soft_iou": soft_iou,
                "soft_iou_cost": soft_iou_cost,
                "distance_transform_overlap": distance_transform_overlap,
                "distance_transform_overlap_cost": distance_transform_overlap_cost,
            }
        )
        return total_cost, components

    setattr(
        _build_pairwise_cost_matrix_with_soft_overlap,
        "_bayescatrack_soft_overlap_patch",
        True,
    )
    setattr(
        _build_pairwise_cost_matrix_with_soft_overlap,
        "_bayescatrack_original",
        original,
    )
    CalciumPlaneData.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
        _build_pairwise_cost_matrix_with_soft_overlap
    )


def _install_registration_qa_preset() -> None:
    try:
        from bayescatrack.experiments import registration_qa_report
    except ImportError:  # pragma: no cover - optional CLI import path
        return

    original = getattr(registration_qa_report, "_cost_kwargs")
    if getattr(original, "_bayescatrack_soft_overlap_patch", False):
        return

    def _registration_qa_cost_kwargs_with_soft_overlap(config: Any) -> dict[str, Any]:
        if getattr(config, "cost", None) == "registered-soft-iou":
            kwargs = dict(registered_soft_iou_cost_kwargs())
            kwargs.update(getattr(config, "pairwise_cost_kwargs", None) or {})
            return kwargs
        return original(config)

    setattr(
        _registration_qa_cost_kwargs_with_soft_overlap,
        "_bayescatrack_soft_overlap_patch",
        True,
    )
    setattr(
        _registration_qa_cost_kwargs_with_soft_overlap,
        "_bayescatrack_original",
        original,
    )
    setattr(
        registration_qa_report,
        "_cost_kwargs",
        _registration_qa_cost_kwargs_with_soft_overlap,
    )


def _pairwise_dilated_iou_matrix(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    radius = _nonnegative_int(radius, name="radius")
    if radius == 0:
        return _bridge_impl._pairwise_iou_matrix(  # pylint: disable=protected-access
            reference_masks,
            measurement_masks,
        )
    return _bridge_impl._pairwise_iou_matrix(  # pylint: disable=protected-access
        _dilate_binary_mask_stack(reference_masks, radius),
        _dilate_binary_mask_stack(measurement_masks, radius),
    )


def _pairwise_distance_transform_overlap_matrix(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
    distance_scale: float | None,
) -> np.ndarray:
    radius = _nonnegative_int(radius, name="radius")
    if radius == 0:
        return _bridge_impl._pairwise_iou_matrix(  # pylint: disable=protected-access
            reference_masks,
            measurement_masks,
        )
    if distance_scale is None:
        distance_scale = max(float(radius) / 2.0, 1.0)
    else:
        distance_scale = _finite_positive_float(distance_scale, name="distance_scale")
    reference_to_measurement = _pairwise_one_sided_distance_overlap(
        reference_masks,
        measurement_masks,
        radius=radius,
        distance_scale=float(distance_scale),
    )
    measurement_to_reference = _pairwise_one_sided_distance_overlap(
        measurement_masks,
        reference_masks,
        radius=radius,
        distance_scale=float(distance_scale),
    ).T
    return np.clip(
        0.5 * (reference_to_measurement + measurement_to_reference),
        0.0,
        1.0,
    )


def _pairwise_one_sided_distance_overlap(
    source_masks: np.ndarray,
    query_masks: np.ndarray,
    *,
    radius: int,
    distance_scale: float,
) -> np.ndarray:
    source_support = np.asarray(source_masks) > 0
    query_support = np.asarray(query_masks) > 0
    if source_support.shape[1:] != query_support.shape[1:]:
        raise ValueError("Mask stacks must have matching spatial shapes")
    scores = np.zeros((source_support.shape[0], query_support.shape[0]), dtype=float)
    query_areas = np.maximum(
        _bridge_impl._mask_support_areas(  # pylint: disable=protected-access
            query_support
        ),
        1.0,
    )
    covered = np.zeros_like(source_support, dtype=bool)
    for distance in range(radius + 1):
        dilated = _dilate_binary_mask_stack(source_support, distance)
        band = dilated & ~covered
        covered |= dilated
        if not np.any(band):
            continue
        weight = float(np.exp(-0.5 * (float(distance) / float(distance_scale)) ** 2))
        scores += (
            weight
            * _bridge_impl._pairwise_sparse_mask_dot(  # pylint: disable=protected-access
                band,
                query_support,
                binary=True,
            )
        )
    return np.clip(scores / query_areas[None, :], 0.0, 1.0)


def _dilate_binary_mask_stack(masks: np.ndarray, radius: int) -> np.ndarray:
    from bayescatrack.association.soft_overlap import dilate_mask_stack

    radius = _nonnegative_int(radius, name="radius")
    return dilate_mask_stack(masks, radius=radius)


def _unwrap_scalar_array(value: Any, *, message: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(message)
        return value.item()
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    message = f"{name} must be an integer"
    value = _unwrap_scalar_array(value, message=message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    numeric_candidate: Any
    if isinstance(value, str):
        numeric_candidate = value.strip()
        if not numeric_candidate:
            raise ValueError(message)
    elif isinstance(value, (float, np.floating)):
        numeric_candidate = value
    else:
        try:
            return _reject_negative_int(operator.index(value), name=name)
        except TypeError:
            try:
                numeric_candidate = float(value)
            except (TypeError, ValueError, OverflowError) as float_exc:
                raise ValueError(message) from float_exc
    try:
        numeric_value = float(numeric_candidate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(message)
    return _reject_negative_int(int(numeric_value), name=name)


def _reject_negative_int(integer_value: int, *, name: str) -> int:
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(integer_value)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, lower_bound=0.0, positive=False)


def _finite_positive_float(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, lower_bound=0.0, positive=True)


def _finite_float(
    value: Any, *, name: str, lower_bound: float, positive: bool
) -> float:
    qualifier = "positive" if positive else "non-negative"
    message = f"{name} must be a finite {qualifier} value"
    value = _unwrap_scalar_array(value, message=message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    violates_bound = (
        numeric_value <= lower_bound if positive else numeric_value < lower_bound
    )
    if not np.isfinite(numeric_value) or violates_bound:
        raise ValueError(message)
    return numeric_value


__all__ = ["install_soft_overlap_costs", "registered_soft_iou_cost_kwargs"]
