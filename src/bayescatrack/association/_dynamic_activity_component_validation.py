"""Shape validation for dynamic activity-missing edge-prior components.

The dynamic activity-missing prior accepts several alternative component names.
Before this patch, a provided component with the wrong shape was skipped and the
next known component, or a zero fallback, was used instead.  That silently
disabled or changed the configured activity cue.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

from . import dynamic_edge_priors as _dynamic_edge_priors

_PATCH_MARKER = "_bayescatrack_dynamic_activity_component_shape_validation_patch"
_ACTIVITY_COMPONENT_NAMES = (
    "activity_tiebreaker_missing",
    "activity_similarity_available",
    "fluorescence_similarity_available",
    "spike_similarity_available",
)


def install_dynamic_activity_component_shape_validation() -> None:
    """Install idempotent validation for configured activity-prior components."""

    original = (
        _dynamic_edge_priors._activity_missing_component
    )  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def activity_missing_component_with_shape_validation(
        pairwise_components: Mapping[str, Any],
        shape: tuple[int, int],
    ) -> np.ndarray:
        _validate_activity_component_shapes(pairwise_components, shape)
        return original(pairwise_components, shape)

    setattr(activity_missing_component_with_shape_validation, _PATCH_MARKER, True)
    setattr(
        activity_missing_component_with_shape_validation,
        "_bayescatrack_original",
        original,
    )
    _dynamic_edge_priors._activity_missing_component = activity_missing_component_with_shape_validation  # pylint: disable=protected-access


def _validate_activity_component_shapes(
    pairwise_components: Mapping[str, Any],
    shape: tuple[int, int],
) -> None:
    expected_shape = tuple(int(dimension) for dimension in shape)
    for component_name in _ACTIVITY_COMPONENT_NAMES:
        if component_name not in pairwise_components:
            continue
        try:
            values = np.asarray(pairwise_components[component_name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Pairwise component {component_name!r} must be numeric"
            ) from exc
        if values.shape != expected_shape:
            raise ValueError(f"Pairwise component {component_name!r} has wrong shape")


__all__ = ["install_dynamic_activity_component_shape_validation"]
