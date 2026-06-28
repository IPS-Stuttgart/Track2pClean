"""Strict validation for track-refinement missing-value sentinels and issue indices.

Track-refinement helpers use ``fill_value`` as a missing-track sentinel while
walking ROI rows and while materializing split fragments.  Non-negative sentinels
collide with Suite2p ROI identifiers, and permissive ``int(...)`` coercions can
turn booleans, strings, or fractional floats into misleading missing values.

The split helper also consumes externally supplied ``TrackGeometryIssue`` objects.
Their track/session fields select cut points, so malformed values must be rejected
before the underlying implementation can silently coerce them with ``int(...)`` or
ignore out-of-range entries.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_refinement_fill_value_validation_patch"
_ERROR_MESSAGE = "fill_value must be a negative integer sentinel"


def install_track_refinement_fill_value_validation() -> None:
    """Install idempotent validation around track-refinement fill values."""

    from . import track_refinement as module  # pylint: disable=import-outside-toplevel

    original_post_init = module.TrackSmoothingConfig.__post_init__
    if not getattr(original_post_init, _PATCH_MARKER, False):

        @wraps(original_post_init)
        def checked_post_init(self: Any) -> None:
            fill_value = _normalize_fill_value(self.fill_value)
            original_post_init(self)
            object.__setattr__(self, "fill_value", fill_value)

        setattr(checked_post_init, _PATCH_MARKER, True)
        setattr(checked_post_init, "_bayescatrack_original", original_post_init)
        module.TrackSmoothingConfig.__post_init__ = checked_post_init

    original_smoothed = module.smoothed_track_positions
    if not getattr(original_smoothed, _PATCH_MARKER, False):

        @wraps(original_smoothed)
        def smoothed_track_positions_with_fill_value_validation(
            track_rows: Any,
            position_tables: Any,
            *,
            fill_value: Any = -1,
        ) -> Any:
            return original_smoothed(
                track_rows,
                position_tables,
                fill_value=_normalize_fill_value(fill_value),
            )

        setattr(smoothed_track_positions_with_fill_value_validation, _PATCH_MARKER, True)
        setattr(
            smoothed_track_positions_with_fill_value_validation,
            "_bayescatrack_original",
            original_smoothed,
        )
        module.smoothed_track_positions = smoothed_track_positions_with_fill_value_validation

    original_split = module.split_tracks_at_issues
    if not getattr(original_split, _PATCH_MARKER, False):

        @wraps(original_split)
        def split_tracks_at_issues_with_fill_value_validation(
            track_rows: Any,
            issues: Any,
            *,
            fill_value: Any = -1,
        ) -> Any:
            normalized_fill_value = _normalize_fill_value(fill_value)
            rows = module._validated_track_row_matrix(track_rows)  # pylint: disable=protected-access
            issue_tuple = _validated_issue_tuple(
                issues,
                n_tracks=rows.shape[0],
                n_sessions=rows.shape[1],
            )
            return original_split(
                rows,
                issue_tuple,
                fill_value=normalized_fill_value,
            )

        setattr(split_tracks_at_issues_with_fill_value_validation, _PATCH_MARKER, True)
        setattr(
            split_tracks_at_issues_with_fill_value_validation,
            "_bayescatrack_original",
            original_split,
        )
        module.split_tracks_at_issues = split_tracks_at_issues_with_fill_value_validation


def _validated_issue_tuple(
    issues: Any,
    *,
    n_tracks: int,
    n_sessions: int,
) -> tuple[Any, ...]:
    try:
        issue_tuple = tuple(issues)
    except TypeError as exc:
        raise ValueError("issues must be an iterable of TrackGeometryIssue entries") from exc

    for issue in issue_tuple:
        _normalize_bounded_issue_index(
            _issue_field(issue, "track_index"),
            name="issue.track_index",
            upper_bound=n_tracks,
            axis_name="tracks",
        )
        _normalize_bounded_issue_index(
            _issue_field(issue, "session_index"),
            name="issue.session_index",
            upper_bound=n_sessions,
            axis_name="sessions",
        )
    return issue_tuple


def _issue_field(issue: Any, field_name: str) -> Any:
    try:
        return getattr(issue, field_name)
    except AttributeError as exc:
        raise ValueError(
            "issues must contain track_index and session_index fields"
        ) from exc


def _normalize_bounded_issue_index(
    value: Any,
    *,
    name: str,
    upper_bound: int,
    axis_name: str,
) -> int:
    integer_value = _normalize_issue_index(value, name=name)
    if integer_value < 0 or integer_value >= upper_bound:
        raise IndexError(
            f"{name} {integer_value} out of bounds for {upper_bound} {axis_name}"
        )
    return integer_value


def _normalize_issue_index(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (str, bytes, np.bytes_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an integer")

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(numeric_value)

    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _normalize_fill_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_ERROR_MESSAGE)
    if isinstance(value, np.ndarray):
        raise ValueError(_ERROR_MESSAGE)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_ERROR_MESSAGE)
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(_ERROR_MESSAGE) from exc

    integer_value = int(integer_value)
    if integer_value >= 0:
        raise ValueError(
            "fill_value must be a negative integer sentinel that cannot collide "
            "with non-negative ROI indices"
        )
    return integer_value


__all__ = ["install_track_refinement_fill_value_validation"]