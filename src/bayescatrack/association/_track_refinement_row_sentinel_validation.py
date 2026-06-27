"""Validate track-refinement rows against the configured missing-value sentinel.

Track-refinement helpers interpret ``fill_value`` as the only missing-track
sentinel.  ROI identifiers are otherwise Suite2p ROI indices and must be
non-negative.  Without this check, a row value such as ``-2`` with the default
``fill_value=-1`` is simply absent from the position table and is silently
ignored, which can hide malformed track rows in geometry diagnostics and split
experiments.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from ._track_refinement_fill_value_validation import _normalize_fill_value

_PATCH_MARKER = "_bayescatrack_track_refinement_row_sentinel_validation_patch"
_ERROR_MESSAGE = "track_rows may only contain ROI indices >= 0 or the configured fill_value"


def install_track_refinement_row_sentinel_validation() -> None:
    """Install idempotent validation for track-refinement row sentinels."""

    from . import track_refinement as module  # pylint: disable=import-outside-toplevel

    original_issues = module.track_geometry_issues
    if not getattr(original_issues, _PATCH_MARKER, False):

        @wraps(original_issues)
        def track_geometry_issues_with_row_sentinel_validation(
            track_rows: Any,
            position_tables: Any,
            *,
            config: Any | None = None,
        ) -> Any:
            rows = _validated_rows_for_fill_value(
                module,
                track_rows,
                fill_value=_fill_value_from_config(config),
            )
            return original_issues(rows, position_tables, config=config)

        setattr(track_geometry_issues_with_row_sentinel_validation, _PATCH_MARKER, True)
        setattr(
            track_geometry_issues_with_row_sentinel_validation,
            "_bayescatrack_original",
            original_issues,
        )
        module.track_geometry_issues = track_geometry_issues_with_row_sentinel_validation

    original_smoothed = module.smoothed_track_positions
    if not getattr(original_smoothed, _PATCH_MARKER, False):

        @wraps(original_smoothed)
        def smoothed_track_positions_with_row_sentinel_validation(
            track_rows: Any,
            position_tables: Any,
            *,
            fill_value: Any = -1,
        ) -> Any:
            normalized_fill_value = _normalize_fill_value(fill_value)
            rows = _validated_rows_for_fill_value(
                module,
                track_rows,
                fill_value=normalized_fill_value,
            )
            return original_smoothed(
                rows,
                position_tables,
                fill_value=normalized_fill_value,
            )

        setattr(smoothed_track_positions_with_row_sentinel_validation, _PATCH_MARKER, True)
        setattr(
            smoothed_track_positions_with_row_sentinel_validation,
            "_bayescatrack_original",
            original_smoothed,
        )
        module.smoothed_track_positions = smoothed_track_positions_with_row_sentinel_validation

    original_split = module.split_tracks_at_issues
    if not getattr(original_split, _PATCH_MARKER, False):

        @wraps(original_split)
        def split_tracks_at_issues_with_row_sentinel_validation(
            track_rows: Any,
            issues: Any,
            *,
            fill_value: Any = -1,
        ) -> Any:
            normalized_fill_value = _normalize_fill_value(fill_value)
            rows = _validated_rows_for_fill_value(
                module,
                track_rows,
                fill_value=normalized_fill_value,
            )
            return original_split(
                rows,
                issues,
                fill_value=normalized_fill_value,
            )

        setattr(split_tracks_at_issues_with_row_sentinel_validation, _PATCH_MARKER, True)
        setattr(
            split_tracks_at_issues_with_row_sentinel_validation,
            "_bayescatrack_original",
            original_split,
        )
        module.split_tracks_at_issues = split_tracks_at_issues_with_row_sentinel_validation


def _fill_value_from_config(config: Any | None) -> int:
    if config is None:
        return _normalize_fill_value(-1)
    try:
        return _normalize_fill_value(config.fill_value)
    except AttributeError as exc:
        raise ValueError("config must provide a fill_value") from exc


def _validated_rows_for_fill_value(module: Any, track_rows: Any, *, fill_value: int) -> np.ndarray:
    rows = module._validated_track_row_matrix(track_rows)  # pylint: disable=protected-access
    invalid = (rows < 0) & (rows != fill_value)
    if np.any(invalid):
        raise ValueError(_ERROR_MESSAGE)
    return rows


__all__ = ["install_track_refinement_row_sentinel_validation"]
