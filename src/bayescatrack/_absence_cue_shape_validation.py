"""Validation for optional ROI cue vector shapes."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_MARKER = "_bayescatrack_absence_cue_shape_validation"


def install_absence_cue_shape_validation(module: ModuleType) -> None:
    """Patch the public vector builder with strict cue-length checks."""

    original = module.absence_cost_vector
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def checked(
        plane: Any,
        *args: Any,
        registered_empty_mask: Any | None = None,
        local_density: Any | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        n_rois = module._validated_roi_count(plane, "plane")
        probabilities = getattr(plane, "cell_probabilities", None)
        if probabilities is not None:
            _check_vector_length(probabilities, "plane.cell_probabilities", n_rois)
        if registered_empty_mask is not None:
            _check_vector_length(registered_empty_mask, "registered_empty_mask", n_rois)
        if local_density is not None:
            _check_vector_length(local_density, "local_density", n_rois)
        return original(
            plane,
            *args,
            registered_empty_mask=registered_empty_mask,
            local_density=local_density,
            **kwargs,
        )

    setattr(checked, _MARKER, True)
    module.absence_cost_vector = checked


def _check_vector_length(values: Any, name: str, n_rois: int) -> None:
    try:
        size = np.asarray(values, dtype=object).reshape(-1).size
    except ValueError as exc:
        raise ValueError(f"{name} must contain one value per ROI") from exc
    if size != n_rois:
        raise ValueError(
            f"{name} must contain one value per ROI; expected {n_rois}, got {size}"
        )


__all__ = ["install_absence_cue_shape_validation"]
