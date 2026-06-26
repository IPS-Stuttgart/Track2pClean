"""Strict validation for tracking seed ROI controls.

The registered subject tracker accepts optional ``start_roi_indices`` to restrict
reported tracks to seed ROIs in a selected session.  The global and single-session
paths used NumPy integer coercion directly, so malformed values such as booleans,
fractional floats, negatives, or duplicates could silently become different seed
tracks.  These hooks make the public runner and the internal restriction helper
fail fast before that coercion can change the requested benchmark population.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_start_roi_validation_patch"


def install_tracking_start_roi_validation() -> None:
    """Install idempotent validation around tracking seed controls."""

    from . import tracking as _tracking  # pylint: disable=import-outside-toplevel

    original_run = _tracking.run_registered_subject_tracking
    original_restrict = (
        _tracking._restrict_track_rows_to_start_rois
    )  # pylint: disable=protected-access

    if getattr(original_run, _PATCH_MARKER, False) and getattr(
        original_restrict,
        _PATCH_MARKER,
        False,
    ):
        return

    @wraps(original_run)
    def run_registered_subject_tracking_with_start_roi_validation(
        subject_dir: Any,
        *args: Any,
        start_roi_indices: Any | None = None,
        start_session_index: Any = 0,
        **kwargs: Any,
    ) -> Any:
        if start_roi_indices is not None:
            start_roi_indices = _normalize_start_roi_indices(start_roi_indices)
        start_session_index = _normalize_start_session_index(start_session_index)
        return original_run(
            subject_dir,
            *args,
            start_roi_indices=start_roi_indices,
            start_session_index=start_session_index,
            **kwargs,
        )

    @wraps(original_restrict)
    def restrict_track_rows_to_start_rois_with_validation(
        track_rows: Any,
        *,
        start_roi_indices: Any,
        start_session_index: Any,
        fill_value: int,
    ) -> np.ndarray:
        num_sessions = _infer_track_row_session_count(track_rows)
        return original_restrict(
            track_rows,
            start_roi_indices=_normalize_start_roi_indices(start_roi_indices),
            start_session_index=_normalize_start_session_index(
                start_session_index,
                num_sessions=num_sessions,
            ),
            fill_value=fill_value,
        )

    _mark_patch(run_registered_subject_tracking_with_start_roi_validation, original_run)
    _mark_patch(restrict_track_rows_to_start_rois_with_validation, original_restrict)

    _tracking.run_registered_subject_tracking = (
        run_registered_subject_tracking_with_start_roi_validation
    )
    _tracking._restrict_track_rows_to_start_rois = (  # pylint: disable=protected-access
        restrict_track_rows_to_start_rois_with_validation
    )


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _normalize_start_roi_indices(values: Any) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("start_roi_indices must be a sequence of integer ROI indices")
    if not isinstance(values, Sequence) and not isinstance(values, np.ndarray):
        raise ValueError("start_roi_indices must be a sequence of integer ROI indices")

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        raise ValueError("start_roi_indices must be a sequence of integer ROI indices")

    normalized = tuple(
        _normalize_start_roi_index(value) for value in array.reshape(-1).tolist()
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("start_roi_indices must contain unique ROI indices")
    return normalized


def _normalize_start_roi_index(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("start_roi_indices must contain integer ROI indices")

    try:
        integer_value = int(operator.index(value))
    except TypeError:
        if not isinstance(value, (float, np.floating)):
            raise ValueError(
                "start_roi_indices must contain integer ROI indices"
            ) from None
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError("start_roi_indices must contain integer ROI indices")
        integer_value = int(numeric_value)

    if integer_value < 0:
        raise ValueError("start_roi_indices must contain non-negative ROI indices")
    return integer_value


def _infer_track_row_session_count(track_rows: Any) -> int | None:
    try:
        array = np.asarray(track_rows)
    except ValueError:
        return None
    if array.ndim != 2:
        return None
    return int(array.shape[1])


def _normalize_start_session_index(
    value: Any,
    *,
    num_sessions: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("start_session_index must be an integer session index")

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError("start_session_index must be an integer session index")
        normalized = int(numeric_value)
    else:
        try:
            normalized = int(operator.index(value))
        except TypeError as exc:
            raise ValueError(
                "start_session_index must be an integer session index"
            ) from exc

    if normalized < 0:
        raise IndexError(f"start_session_index {normalized} out of bounds")
    if num_sessions is not None and normalized >= num_sessions:
        raise IndexError(
            f"start_session_index {normalized} out of bounds for {num_sessions} sessions"
        )
    return normalized


__all__ = ["install_tracking_start_roi_validation"]
