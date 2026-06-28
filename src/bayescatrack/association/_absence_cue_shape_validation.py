"""Validation for optional absence-model ROI cue vector lengths and shapes."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_MARKER = "_track2pclean_absence_cue_shape_validation"


def install_absence_cue_shape_validation(absence_model: ModuleType) -> None:
    """Patch absence-cost construction with fail-fast per-ROI cue checks."""

    original_absence_cost_vector = absence_model.absence_cost_vector
    if not getattr(original_absence_cost_vector, _MARKER, False):

        @wraps(original_absence_cost_vector)
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
                _check_roi_vector_shape(
                    cell_probabilities,
                    "plane.cell_probabilities",
                    n_rois,
                )
            if registered_empty_mask is not None:
                _check_roi_vector_shape(
                    registered_empty_mask,
                    "registered_empty_mask",
                    n_rois,
                )
            if local_density is not None:
                _check_roi_vector_shape(local_density, "local_density", n_rois)
            return original_absence_cost_vector(
                plane,
                *args,
                registered_empty_mask=registered_empty_mask,
                local_density=local_density,
                **kwargs,
            )

        setattr(checked_absence_cost_vector, _MARKER, True)
        setattr(
            checked_absence_cost_vector,
            "_bayescatrack_original",
            original_absence_cost_vector,
        )
        absence_model.absence_cost_vector = checked_absence_cost_vector

    original_gap_penalty_matrix = absence_model.gap_penalty_matrix
    if not getattr(original_gap_penalty_matrix, _MARKER, False):

        @wraps(original_gap_penalty_matrix)
        def checked_gap_penalty_matrix(
            reference_plane: Any,
            measurement_plane: Any,
            *args: Any,
            reference_absence_costs: Any | None = None,
            measurement_absence_costs: Any | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            if reference_absence_costs is not None:
                n_reference_rois = absence_model._validated_roi_count(  # noqa: SLF001
                    reference_plane,
                    "reference_plane",
                )
                _check_roi_vector_shape(
                    reference_absence_costs,
                    "reference_absence_costs",
                    n_reference_rois,
                )
            if measurement_absence_costs is not None:
                n_measurement_rois = absence_model._validated_roi_count(  # noqa: SLF001
                    measurement_plane,
                    "measurement_plane",
                )
                _check_roi_vector_shape(
                    measurement_absence_costs,
                    "measurement_absence_costs",
                    n_measurement_rois,
                )
            return original_gap_penalty_matrix(
                reference_plane,
                measurement_plane,
                *args,
                reference_absence_costs=reference_absence_costs,
                measurement_absence_costs=measurement_absence_costs,
                **kwargs,
            )

        setattr(checked_gap_penalty_matrix, _MARKER, True)
        setattr(
            checked_gap_penalty_matrix,
            "_bayescatrack_original",
            original_gap_penalty_matrix,
        )
        absence_model.gap_penalty_matrix = checked_gap_penalty_matrix


def _check_roi_vector_shape(values: Any, field_name: str, n_rois: int) -> None:
    try:
        array = np.asarray(values, dtype=object)
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain one value per ROI") from exc
    if array.ndim != 1:
        raise ValueError(
            f"{field_name} must be a one-dimensional vector with one value per ROI"
        )
    if array.size != n_rois:
        raise ValueError(
            f"{field_name} must contain one value per ROI; expected {n_rois}, got {array.size}"
        )


__all__ = ["install_absence_cue_shape_validation"]
