"""Shape validation for optional absence-model ROI cue vectors."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_ABSENCE_CUE_VALIDATION_MARKER = "_bayescatrack_absence_cue_shape_validation_patch"


def install_absence_cue_shape_validation(absence_model: ModuleType) -> None:
    """Install fail-fast validation for optional per-ROI absence-model cues."""

    original_absence_cost_vector = absence_model.absence_cost_vector
    if getattr(original_absence_cost_vector, _ABSENCE_CUE_VALIDATION_MARKER, False):
        return

    @wraps(original_absence_cost_vector)
    def _absence_cost_vector_with_cue_shape_validation(
        plane: Any,
        *args: Any,
        registered_empty_mask: Any | None = None,
        local_density: Any | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        n_rois = absence_model._validated_roi_count(plane, "plane")  # noqa: SLF001
        cell_probabilities = getattr(plane, "cell_probabilities", None)
        if cell_probabilities is not None:
            _validate_roi_cue_vector(
                cell_probabilities,
                "plane.cell_probabilities",
                n_rois,
            )
        if registered_empty_mask is not None:
            _validate_roi_cue_vector(
                registered_empty_mask,
                "registered_empty_mask",
                n_rois,
            )
        if local_density is not None:
            _validate_roi_cue_vector(local_density, "local_density", n_rois)
        return original_absence_cost_vector(
            plane,
            *args,
            registered_empty_mask=registered_empty_mask,
            local_density=local_density,
            **kwargs,
        )

    setattr(
        _absence_cost_vector_with_cue_shape_validation,
        _ABSENCE_CUE_VALIDATION_MARKER,
        True,
    )
    absence_model.absence_cost_vector = _absence_cost_vector_with_cue_shape_validation


def _validate_roi_cue_vector(values: Any, field_name: str, n_rois: int) -> None:
    try:
        vector = np.asarray(values, dtype=object).reshape(-1)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must contain one value per ROI; expected {n_rois}"
        ) from exc
    if vector.shape != (n_rois,):
        raise ValueError(
            f"{field_name} must contain one value per ROI; "
            f"expected {n_rois}, got {vector.size}"
        )


__all__ = ["install_absence_cue_shape_validation"]
