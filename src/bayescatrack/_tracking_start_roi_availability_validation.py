"""Validate explicit tracking seed ROIs against the loaded start session.

The registered subject tracker lets callers provide ``start_roi_indices`` to
restrict reported tracks to a seed session.  Those values are ROI identifiers,
not row positions, so they must refer to ROIs that were actually loaded for the
selected start session.  Without this check, a valid-looking but unavailable ROI
identifier can produce a phantom one-session track or a missing-only restricted
row instead of failing fast.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_tracking_start_roi_availability_validation_patch"
_MAX_TRACKING_INDEX = int(np.iinfo(np.int_).max)


def install_tracking_start_roi_availability_validation() -> None:
    """Install idempotent validation for explicit tracking seed ROI availability."""

    from . import tracking as _tracking  # pylint: disable=import-outside-toplevel

    original_run = _tracking.run_registered_subject_tracking
    if getattr(original_run, _PATCH_MARKER, False):
        return

    @wraps(original_run)
    def run_registered_subject_tracking_with_start_roi_availability_validation(
        subject_dir: Any,
        *args: Any,
        start_roi_indices: Any | None = None,
        start_session_index: Any = 0,
        **kwargs: Any,
    ) -> Any:
        result = original_run(
            subject_dir,
            *args,
            start_roi_indices=start_roi_indices,
            start_session_index=start_session_index,
            **kwargs,
        )
        if start_roi_indices is not None:
            _validate_start_roi_indices_available(
                start_roi_indices,
                result.sessions,
                start_session_index=start_session_index,
            )
        return result

    setattr(
        run_registered_subject_tracking_with_start_roi_availability_validation,
        _PATCH_MARKER,
        True,
    )
    setattr(
        run_registered_subject_tracking_with_start_roi_availability_validation,
        "_bayescatrack_original",
        original_run,
    )
    _tracking.run_registered_subject_tracking = (
        run_registered_subject_tracking_with_start_roi_availability_validation
    )


def _validate_start_roi_indices_available(
    start_roi_indices: Any,
    sessions: Sequence[Any],
    *,
    start_session_index: Any,
) -> None:
    sessions = tuple(sessions)
    normalized_session_index = _normalize_start_session_index(
        start_session_index,
        num_sessions=len(sessions),
    )
    requested_roi_indices = _normalize_roi_index_sequence(
        start_roi_indices,
        field_name="start_roi_indices",
    )
    if not requested_roi_indices:
        return

    available_roi_indices = _available_roi_indices_for_session(
        sessions[normalized_session_index]
    )
    missing_roi_indices = tuple(
        roi_index
        for roi_index in requested_roi_indices
        if roi_index not in available_roi_indices
    )
    if missing_roi_indices:
        raise ValueError(
            "start_roi_indices must refer to ROI indices present in "
            f"start session {normalized_session_index}; "
            f"missing ROI index {missing_roi_indices[0]}"
        )


def _available_roi_indices_for_session(session: Any) -> set[int]:
    plane = session.plane_data
    if plane.roi_indices is None:
        return set(range(int(plane.n_rois)))
    return set(
        _normalize_roi_index(value, field_name="plane_data.roi_indices")
        for value in np.asarray(plane.roi_indices, dtype=object).reshape(-1).tolist()
    )


def _normalize_roi_index_sequence(values: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of integer ROI indices")
    if not isinstance(values, Sequence) and not isinstance(values, np.ndarray):
        raise ValueError(f"{field_name} must be a sequence of integer ROI indices")

    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(
            f"{field_name} must be a one-dimensional sequence of integer ROI indices"
        )
    return tuple(
        _normalize_roi_index(value, field_name=field_name) for value in array.tolist()
    )


def _normalize_roi_index(value: Any, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain integer ROI indices")

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must contain integer ROI indices")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = int(operator.index(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field_name} must contain integer ROI indices") from exc

    if integer_value < 0:
        raise ValueError(f"{field_name} must contain non-negative ROI indices")
    if integer_value > _MAX_TRACKING_INDEX:
        raise ValueError(f"{field_name} must contain integer ROI indices")
    return int(integer_value)


def _normalize_start_session_index(
    value: Any,
    *,
    num_sessions: int,
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
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "start_session_index must be an integer session index"
            ) from exc

    if normalized < 0 or normalized >= int(num_sessions):
        raise IndexError(
            f"start_session_index {normalized} out of bounds for "
            f"{num_sessions} sessions"
        )
    return normalized


__all__ = ["install_tracking_start_roi_availability_validation"]
