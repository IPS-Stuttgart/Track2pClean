"""Validation for optional absence-model ROI cue vector lengths."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_MARKER = "_track2pclean_absence_cue_shape_validation"


def install_absence_cue_shape_validation(absence_model: ModuleType) -> None:
    """Patch absence-cost construction with fail-fast per-ROI cue checks."""

    original = absence_model.absence_cost_vector
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def checked_absence_cost_vector(
        plane: Any,
        *args: Any,
        registered_empty_mask: Any | None = None,
        local_density: Any | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        n_rois = absence_model._validated_roi_count(plane, "plane")  # noqa: SLF001
        cell_probabilities = getattr(plane, "cell_probabilities", None)
        if cell_probabilities is not None:
            _check_roi_vector_length(
                cell_probabilities,
                "plane.cell_probabilities",
                n_rois,
            )
        if registered_empty_mask is not None:
            _check_roi_vector_length(
                registered_empty_mask,
                "registered_empty_mask",
                n_rois,
            )
        if local_density is not None:
            _check_roi_vector_length(local_density, "local_density", n_rois)
        return original(
            plane,
            *args,
            registered_empty_mask=registered_empty_mask,
            local_density=local_density,
            **kwargs,
        )

    setattr(checked_absence_cost_vector, _MARKER, True)
    absence_model.absence_cost_vector = checked_absence_cost_vector


def _check_roi_vector_length(values: Any, field_name: str, n_rois: int) -> None:
    try:
        size = np.asarray(values, dtype=object).reshape(-1).size
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain one value per ROI") from exc
    if size != n_rois:
        raise ValueError(
            f"{field_name} must contain one value per ROI; expected {n_rois}, got {size}"
        )


__all__ = ["install_absence_cue_shape_validation"]
