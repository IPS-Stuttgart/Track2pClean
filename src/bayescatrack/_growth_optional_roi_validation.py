"""Strict validation for growth-analysis categorical identifiers.

Growth analysis treats ROI IDs and session indices as categorical labels. Python
booleans are a subclass of ``int``, and the base growth helpers coerce session
indices with ``int(...)``. Without guards, malformed values such as ``True`` or
``1.5`` can silently select session ``1``. Positive ROI IDs that are absent from
the loaded session must also fail fast; otherwise growth summaries silently drop
those tracks and report biased aggregate displacements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_OPTIONAL_ROI_PATCH_MARKER = "_bayescatrack_growth_optional_roi_validation_patch"
_SESSION_INDEX_PATCH_MARKER = "_bayescatrack_growth_session_index_validation_patch"
_TRACK_ROI_LOOKUP_PATCH_MARKER = "_bayescatrack_growth_track_roi_lookup_validation_patch"


def install_growth_optional_roi_validation() -> None:
    """Install idempotent validation around growth-analysis categorical IDs."""

    from .analysis import growth as _growth  # pylint: disable=import-outside-toplevel

    _install_optional_roi_validation(_growth)
    _install_session_index_validation(_growth)
    _install_track_roi_lookup_validation(_growth)


def _install_optional_roi_validation(_growth: Any) -> None:
    original_optional_roi = _growth._optional_roi
    if getattr(original_optional_roi, _OPTIONAL_ROI_PATCH_MARKER, False):
        return

    @wraps(original_optional_roi)
    def _optional_roi_with_validation(value: object) -> int | None:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("ROI index must be integer-like, got boolean")
        return original_optional_roi(value)

    setattr(_optional_roi_with_validation, _OPTIONAL_ROI_PATCH_MARKER, True)
    setattr(_optional_roi_with_validation, "_bayescatrack_original", original_optional_roi)
    _growth._optional_roi = _optional_roi_with_validation


def _install_session_index_validation(_growth: Any) -> None:
    original_validate_session_index = _growth._validate_session_index
    if getattr(original_validate_session_index, _SESSION_INDEX_PATCH_MARKER, False):
        return

    @wraps(original_validate_session_index)
    def _validate_session_index_with_validation(index: object, n_sessions: int) -> int:
        return original_validate_session_index(
            _session_index_to_int(index),
            n_sessions,
        )

    setattr(_validate_session_index_with_validation, _SESSION_INDEX_PATCH_MARKER, True)
    setattr(
        _validate_session_index_with_validation,
        "_bayescatrack_original",
        original_validate_session_index,
    )
    _growth._validate_session_index = _validate_session_index_with_validation


def _install_track_roi_lookup_validation(_growth: Any) -> None:
    original_matched_track_points = _growth._matched_track_points
    if getattr(original_matched_track_points, _TRACK_ROI_LOOKUP_PATCH_MARKER, False):
        return

    @wraps(original_matched_track_points)
    def _matched_track_points_with_roi_lookup_validation(
        matrix: np.ndarray,
        centroid_lookups: Sequence[Mapping[int, np.ndarray]],
        *,
        source_session: int,
        target_session: int,
    ) -> list[tuple[int, int, int, np.ndarray, np.ndarray]]:
        _validate_known_track_rois(
            _growth,
            matrix,
            centroid_lookups,
            source_session=source_session,
            target_session=target_session,
        )
        return original_matched_track_points(
            matrix,
            centroid_lookups,
            source_session=source_session,
            target_session=target_session,
        )

    setattr(
        _matched_track_points_with_roi_lookup_validation,
        _TRACK_ROI_LOOKUP_PATCH_MARKER,
        True,
    )
    setattr(
        _matched_track_points_with_roi_lookup_validation,
        "_bayescatrack_original",
        original_matched_track_points,
    )
    _growth._matched_track_points = _matched_track_points_with_roi_lookup_validation


def _validate_known_track_rois(
    _growth: Any,
    matrix: np.ndarray,
    centroid_lookups: Sequence[Mapping[int, np.ndarray]],
    *,
    source_session: int,
    target_session: int,
) -> None:
    track_matrix = np.asarray(matrix, dtype=object)
    source_lookup = centroid_lookups[source_session]
    target_lookup = centroid_lookups[target_session]
    for track_index, row in enumerate(track_matrix):
        source_roi = _growth._optional_roi(row[source_session])
        target_roi = _growth._optional_roi(row[target_session])
        if source_roi is not None and source_roi not in source_lookup:
            raise ValueError(
                f"track_matrix references ROI {source_roi} in source session "
                f"{source_session} at track {track_index}, but it is not present "
                "in the loaded session"
            )
        if target_roi is not None and target_roi not in target_lookup:
            raise ValueError(
                f"track_matrix references ROI {target_roi} in target session "
                f"{target_session} at track {track_index}, but it is not present "
                "in the loaded session"
            )


def _session_index_to_int(index: object) -> int:
    if isinstance(index, (bool, np.bool_)):
        raise ValueError("session index must be integer-like, got boolean")
    if isinstance(index, bytes):
        index = index.decode("utf-8")
    if isinstance(index, (int, np.integer)):
        return int(index)
    if isinstance(index, (float, np.floating)):
        return _parse_integer_like_session_index(float(index), original=index)
    if isinstance(index, str):
        text = index.strip()
        if not text:
            raise ValueError("session index must be integer-like, got empty string")
        try:
            return int(text)
        except ValueError:
            pass
        try:
            numeric = float(text)
        except ValueError as exc:
            raise ValueError(f"session index must be integer-like, got {index!r}") from exc
        return _parse_integer_like_session_index(numeric, original=index)
    raise ValueError(f"session index must be integer-like, got {type(index).__name__}")


def _parse_integer_like_session_index(value: float, *, original: object) -> int:
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"session index must be integer-like, got {original!r}")
    return int(value)


__all__ = ["install_growth_optional_roi_validation"]
