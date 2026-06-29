"""Reject ambiguous duplicate start-session rows during track restriction.

The global assignment path can produce more than one row that contains the same
non-missing ROI in the session selected as the seed population.  Restricting the
result to ``start_roi_indices`` used a first-row-wins dictionary lookup, which
silently discarded later rows for the same seed ROI.  That makes the exported
benchmark population depend on row order and hides an invalid global-track table.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_tracking_duplicate_start_roi_validation_patch"
_ERROR_MESSAGE = "track_rows must not contain duplicate non-missing ROI entries in start_session_index"


def install_tracking_duplicate_start_roi_validation() -> None:
    """Install idempotent duplicate-start checks for track-row restriction."""

    from . import tracking as _tracking  # pylint: disable=import-outside-toplevel

    original_restrict = _tracking._restrict_track_rows_to_start_rois
    if getattr(original_restrict, _PATCH_MARKER, False):
        return

    @wraps(original_restrict)
    def restrict_track_rows_to_start_rois_without_duplicate_starts(
        track_rows: Any,
        *,
        start_roi_indices: Any,
        start_session_index: Any,
        fill_value: Any,
    ) -> np.ndarray:
        _reject_duplicate_start_session_rois(
            track_rows,
            start_session_index=start_session_index,
            fill_value=fill_value,
        )
        return original_restrict(
            track_rows,
            start_roi_indices=start_roi_indices,
            start_session_index=start_session_index,
            fill_value=fill_value,
        )

    setattr(
        restrict_track_rows_to_start_rois_without_duplicate_starts,
        _PATCH_MARKER,
        True,
    )
    setattr(
        restrict_track_rows_to_start_rois_without_duplicate_starts,
        "_bayescatrack_original",
        original_restrict,
    )
    _tracking._restrict_track_rows_to_start_rois = (
        restrict_track_rows_to_start_rois_without_duplicate_starts
    )


def _reject_duplicate_start_session_rois(
    track_rows: Any,
    *,
    start_session_index: Any,
    fill_value: Any,
) -> None:
    try:
        rows = np.asarray(track_rows, dtype=object)
    except (TypeError, ValueError):
        return
    if rows.ndim != 2:
        return

    try:
        session_index = _coerce_integer(start_session_index)
        missing_value = _coerce_integer(fill_value)
    except ValueError:
        return
    if session_index < 0 or session_index >= rows.shape[1]:
        return

    seen: set[int] = set()
    duplicates: list[int] = []
    for raw_roi in rows[:, session_index].tolist():
        try:
            roi_index = _coerce_integer(raw_roi)
        except ValueError:
            continue
        if roi_index == missing_value:
            continue
        if roi_index in seen and roi_index not in duplicates:
            duplicates.append(roi_index)
        seen.add(roi_index)

    if duplicates:
        duplicate_summary = ", ".join(str(value) for value in duplicates)
        raise ValueError(
            f"{_ERROR_MESSAGE}; duplicate ROI indices: {duplicate_summary}"
        )


def _coerce_integer(value: Any) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError("value must be scalar")
        value = value.item()

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("value must be integer-like")
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("value must be integer-like")
        return int(numeric)
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError("value must be integer-like") from exc

    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("value must be integer-like") from exc


__all__ = ["install_tracking_duplicate_start_roi_validation"]
