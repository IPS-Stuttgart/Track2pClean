"""Weak additive activity tie-breaker costs for ROI association."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

_ACTIVITY_AVAILABILITY_COMPONENTS = {
    "activity_tiebreaker_cost": "activity_tiebreaker_available",
    "activity_similarity_cost": "activity_similarity_available",
    "fluorescence_similarity_cost": "fluorescence_similarity_available",
    "spike_similarity_cost": "spike_similarity_available",
    "neuropil_similarity_cost": "neuropil_similarity_available",
    "trace_std_absdiff": "trace_std_available",
    "trace_skew_absdiff": "trace_skew_available",
    "event_rate_absdiff": "event_rate_available",
    "neuropil_ratio_absdiff": "neuropil_ratio_available",
}

_WEIGHT_ERROR = "weight must be a finite non-negative scalar"


def activity_tie_breaker_cost_matrix(
    pairwise_components: Mapping[str, Any],
    *,
    component_name: str = "activity_tiebreaker_cost",
    weight: float = 0.05,
) -> np.ndarray:
    """Return a low-weight activity cost plane for additive tie-breaking.

    The selected component is intentionally scaled outside the calibrated model so
    spatial evidence remains dominant. Undefined numeric values in available
    activity entries are mapped to the neutral cost used by the activity feature
    extractor. Pairs whose corresponding ``*_available`` component is false get
    no additive tie-breaker penalty, because a constant penalty carries no
    discriminative information and can otherwise push valid spatial matches over
    the solver's edge-cost threshold.
    """

    weight = _normalize_weight(weight)
    if component_name not in pairwise_components:
        raise KeyError(f"Pairwise components do not contain {component_name!r}")
    values = np.asarray(pairwise_components[component_name], dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"Pairwise component {component_name!r} must be two-dimensional"
        )
    sanitized_values = np.nan_to_num(
        values,
        nan=0.5,
        posinf=1.0e6,
        neginf=0.0,
    )
    availability = _activity_availability_matrix(
        pairwise_components,
        component_name=component_name,
        shape=values.shape,
    )
    return weight * np.where(availability > 0.0, sanitized_values, 0.0)


def _normalize_weight(weight: Any) -> float:
    if isinstance(weight, (bool, np.bool_, str, bytes)):
        raise ValueError(_WEIGHT_ERROR)

    try:
        weight_array = np.asarray(weight, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_WEIGHT_ERROR) from exc
    if weight_array.shape != ():
        raise ValueError(_WEIGHT_ERROR)

    try:
        normalized_weight = float(weight_array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(_WEIGHT_ERROR) from exc

    if not np.isfinite(normalized_weight) or normalized_weight < 0.0:
        raise ValueError(_WEIGHT_ERROR)
    return normalized_weight


def _activity_availability_matrix(
    pairwise_components: Mapping[str, Any],
    *,
    component_name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    availability_name = _ACTIVITY_AVAILABILITY_COMPONENTS.get(component_name)
    if availability_name is None and component_name.endswith("_cost"):
        availability_name = f"{component_name[:-5]}_available"
    if availability_name is None or availability_name not in pairwise_components:
        return np.ones(shape, dtype=float)

    availability = np.asarray(pairwise_components[availability_name], dtype=float)
    if availability.shape != shape:
        raise ValueError(
            f"Availability component {availability_name!r} must have shape {shape}; "
            f"got {availability.shape}"
        )
    return np.nan_to_num(availability, nan=0.0, posinf=1.0, neginf=0.0)
