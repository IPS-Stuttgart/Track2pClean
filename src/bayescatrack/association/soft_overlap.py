"""Soft-overlap IoU costs for registered calcium-imaging ROI association.

The standard registered-IoU ablation treats near-miss masks as zero-overlap
candidates. That is brittle when a generally good registration still leaves a
small local residual. This module adds a deliberately simple dilation-based
soft-IoU wrapper. It is less shape-selective than the shifted-overlap module,
so it is mostly useful as an ablation and as a compatibility fix for the
``registered-soft-iou`` benchmark cost option.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from bayescatrack.core import _bridge_impl  # pylint: disable=protected-access
from bayescatrack.core.bridge import CalciumPlaneData

PairwiseCostMethod = Callable[
    ..., np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]
]

SOFT_OVERLAP_KWARG_NAMES = frozenset(
    (
        "soft_iou_radius",
        "use_soft_iou_for_iou_cost",
    )
)


def pairwise_kwargs_use_soft_overlap(
    pairwise_cost_kwargs: Mapping[str, Any] | None,
) -> bool:
    """Return whether pairwise kwargs require soft-overlap support."""

    if not pairwise_cost_kwargs:
        return False
    return any(key in pairwise_cost_kwargs for key in SOFT_OVERLAP_KWARG_NAMES)


def soft_iou_pairwise_cost_matrix(
    original_method: PairwiseCostMethod,
    self: CalciumPlaneData,
    other: CalciumPlaneData,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Wrap ``CalciumPlaneData.build_pairwise_cost_matrix`` with soft IoU.

    Recognized keyword arguments are removed before delegating to the original
    method, so this wrapper can be installed without changing the public core
    method signature.

    Parameters
    ----------
    soft_iou_radius
        Radius, in pixels, used to dilate both ROI mask stacks before computing
        the IoU value used by the IoU cost term.
    use_soft_iou_for_iou_cost
        Replace the standard exact-IoU cost term with ``max(exact_iou,
        dilated_iou)``. Defaults to ``True`` whenever ``soft_iou_radius > 0``.
    """

    soft_iou_radius = _nonnegative_int(
        kwargs.pop("soft_iou_radius", 0), name="soft_iou_radius"
    )
    use_soft_iou_for_iou_cost = kwargs.pop(
        "use_soft_iou_for_iou_cost", soft_iou_radius > 0
    )
    use_soft_iou_for_iou_cost = _strict_bool(
        use_soft_iou_for_iou_cost, name="use_soft_iou_for_iou_cost"
    )

    return_components = _strict_bool(
        kwargs.get("return_components", False), name="return_components"
    )
    uses_soft_iou = soft_iou_radius > 0 and (
        use_soft_iou_for_iou_cost or return_components
    )
    if not uses_soft_iou:
        return original_method(self, other, **kwargs)

    similarity_epsilon = _finite_positive_float(
        kwargs.get("similarity_epsilon", 1.0e-6), name="similarity_epsilon"
    )
    large_cost = _finite_positive_float(kwargs.get("large_cost", 1.0e6), name="large_cost")
    iou_weight = _finite_nonnegative_float(kwargs.get("iou_weight", 6.0), name="iou_weight")

    base_kwargs = dict(kwargs)
    if use_soft_iou_for_iou_cost:
        base_kwargs["iou_weight"] = 0.0
    base_kwargs["return_components"] = True

    base_cost, components = original_method(self, other, **base_kwargs)
    components = dict(components)

    exact_iou = np.asarray(
        components.get(
            "iou",
            _bridge_impl._pairwise_iou_matrix(  # pylint: disable=protected-access
                self.roi_masks,
                other.roi_masks,
            ),
        ),
        dtype=float,
    )
    soft_iou = pairwise_dilated_iou_matrix(
        self.roi_masks,
        other.roi_masks,
        radius=soft_iou_radius,
    )
    effective_iou = np.maximum(exact_iou, soft_iou)
    effective_iou_cost = -np.log(np.clip(effective_iou, similarity_epsilon, 1.0))

    total_cost = np.asarray(base_cost, dtype=float).copy()
    if use_soft_iou_for_iou_cost and iou_weight > 0.0:
        total_cost += iou_weight * effective_iou_cost

    gated = np.asarray(
        components.get("gated", np.zeros_like(total_cost, dtype=bool)), dtype=bool
    )
    if gated.shape == total_cost.shape:
        total_cost = np.where(gated, large_cost, total_cost)
    total_cost = _ensure_finite_cost_matrix(total_cost, large_cost=large_cost)

    components.update(
        {
            "pairwise_cost_matrix": total_cost,
            "iou": exact_iou,
            "soft_iou": soft_iou,
            "effective_iou": effective_iou,
            "iou_for_cost": effective_iou if use_soft_iou_for_iou_cost else exact_iou,
            "soft_iou_cost": effective_iou_cost,
            "soft_iou_radius": np.full_like(total_cost, soft_iou_radius, dtype=float),
        }
    )
    if return_components:
        return total_cost, components
    return total_cost


def pairwise_dilated_iou_matrix(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Return pairwise IoU after circular binary dilation of both mask stacks."""

    reference_dilated = dilate_mask_stack(reference_masks, radius=radius)
    measurement_dilated = dilate_mask_stack(measurement_masks, radius=radius)
    return _bridge_impl._pairwise_iou_matrix(  # pylint: disable=protected-access
        reference_dilated,
        measurement_dilated,
    )


def dilate_mask_stack(masks: np.ndarray, *, radius: int) -> np.ndarray:
    """Dilate a stack of ROI masks with a disk-shaped integer stencil."""

    mask_array = np.asarray(masks) > 0
    if mask_array.ndim != 3:
        raise ValueError("ROI masks must have shape (n_roi, height, width)")
    radius = _nonnegative_int(radius, name="radius")
    if radius == 0 or mask_array.shape[0] == 0:
        return mask_array

    result = np.array(mask_array, copy=True)
    _, height, width = mask_array.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            if dy * dy + dx * dx > radius * radius:
                continue
            src_y = slice(max(0, -dy), min(height, height - dy))
            dst_y = slice(max(0, dy), min(height, height + dy))
            src_x = slice(max(0, -dx), min(width, width - dx))
            dst_x = slice(max(0, dx), min(width, width + dx))
            result[:, dst_y, dst_x] |= mask_array[:, src_y, src_x]
    return result


def install_soft_iou_cost_patch() -> PairwiseCostMethod:
    """Install soft-IoU kwargs support on ``CalciumPlaneData``.

    Returns the previous method so callers can restore it in a ``finally`` block.
    """

    original_method = CalciumPlaneData.build_pairwise_cost_matrix
    if getattr(original_method, "_bayescatrack_soft_iou_patch", False):
        return original_method

    def _patched_pairwise_cost(
        self: CalciumPlaneData,
        other: CalciumPlaneData,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        return soft_iou_pairwise_cost_matrix(original_method, self, other, **kwargs)

    setattr(_patched_pairwise_cost, "_bayescatrack_soft_iou_patch", True)
    setattr(_patched_pairwise_cost, "_bayescatrack_original", original_method)
    CalciumPlaneData.build_pairwise_cost_matrix = _patched_pairwise_cost  # type: ignore[method-assign]
    return original_method


def _ensure_finite_cost_matrix(
    cost_matrix: np.ndarray, *, large_cost: float
) -> np.ndarray:
    sanitized = np.asarray(cost_matrix, dtype=float).copy()
    invalid = ~np.isfinite(sanitized)
    if np.any(invalid):
        sanitized[invalid] = large_cost
    sanitized[sanitized < 0.0] = 0.0
    return sanitized


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer")
        integer_value = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must be an integer")
        try:
            numeric_value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be an integer")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(integer_value)


def _strict_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, lower_bound=0.0, positive=False)


def _finite_positive_float(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, lower_bound=0.0, positive=True)


def _finite_float(
    value: Any, *, name: str, lower_bound: float, positive: bool
) -> float:
    qualifier = "positive" if positive else "non-negative"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite {qualifier} value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {qualifier} value") from exc
    violates_bound = (
        numeric_value <= lower_bound if positive else numeric_value < lower_bound
    )
    if not np.isfinite(numeric_value) or violates_bound:
        raise ValueError(f"{name} must be a finite {qualifier} value")
    return numeric_value
