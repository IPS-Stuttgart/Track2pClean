"""Strict validation for track-refinement missing-value sentinels.

Track-refinement helpers use ``fill_value`` as a missing-track sentinel while
walking ROI rows and while materializing split fragments.  Non-negative sentinels
collide with Suite2p ROI identifiers, and permissive ``int(...)`` coercions can
turn booleans, strings, or fractional floats into misleading missing values.
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

        setattr(
            smoothed_track_positions_with_fill_value_validation, _PATCH_MARKER, True
        )
        setattr(
            smoothed_track_positions_with_fill_value_validation,
            "_bayescatrack_original",
            original_smoothed,
        )
        module.smoothed_track_positions = (
            smoothed_track_positions_with_fill_value_validation
        )

    original_split = module.split_tracks_at_issues
    if not getattr(original_split, _PATCH_MARKER, False):

        @wraps(original_split)
        def split_tracks_at_issues_with_fill_value_validation(
            track_rows: Any,
            issues: Any,
            *,
            fill_value: Any = -1,
        ) -> Any:
            return original_split(
                track_rows,
                issues,
                fill_value=_normalize_fill_value(fill_value),
            )

        setattr(split_tracks_at_issues_with_fill_value_validation, _PATCH_MARKER, True)
        setattr(
            split_tracks_at_issues_with_fill_value_validation,
            "_bayescatrack_original",
            original_split,
        )
        module.split_tracks_at_issues = (
            split_tracks_at_issues_with_fill_value_validation
        )


def _normalize_fill_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
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
