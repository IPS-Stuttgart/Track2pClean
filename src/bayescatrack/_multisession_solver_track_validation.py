"""Strict validation for multisession solver track indices.

The PyRecEst multisession adapter receives track mappings in solver-coordinate ROI
indices.  Those indices are later converted to Track2p/Suite2p ROI IDs, so
accepting Python's permissive ``int(...)`` coercion can fabricate valid IDs or
hide malformed links as missing detections.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_multisession_solver_track_validation_patch"
_ERROR_SUFFIX = "must be a non-negative integer"


def install_multisession_solver_track_validation() -> None:
    """Install idempotent validation around multisession solver tracks."""

    from . import (
        multisession_tracking as _multisession_tracking,  # pylint: disable=import-outside-toplevel
    )

    original_coerce_solver_tracks = _multisession_tracking._coerce_solver_tracks
    if getattr(original_coerce_solver_tracks, _PATCH_MARKER, False):
        return

    @wraps(original_coerce_solver_tracks)
    def _coerce_solver_tracks_with_validation(
        raw_result: Any,
    ) -> tuple[tuple[dict[int, int], ...], float | None]:
        if isinstance(raw_result, dict):
            tracks = raw_result.get("tracks")
            total_cost = raw_result.get("total_cost")
        else:
            tracks = getattr(raw_result, "tracks", raw_result)
            total_cost = getattr(raw_result, "total_cost", None)

        if tracks is None:
            return original_coerce_solver_tracks(raw_result)

        normalized_tracks: list[dict[int, int]] = []
        for track_number, track in enumerate(tracks):
            if not isinstance(track, Mapping):
                raise TypeError(
                    "Each returned track must be a mapping from session index to detection index"
                )

            normalized_track: dict[int, int] = {}
            for session_index, detection_index in track.items():
                session_integer = _normalize_nonnegative_integer(
                    session_index,
                    label=f"multisession solver track {track_number} session index",
                )
                if session_integer in normalized_track:
                    raise ValueError(
                        "multisession solver tracks must not contain duplicate session indices after normalization"
                    )
                normalized_track[session_integer] = _normalize_nonnegative_integer(
                    detection_index,
                    label=f"multisession solver track {track_number} detection index",
                )
            normalized_tracks.append(normalized_track)

        return tuple(normalized_tracks), (
            None if total_cost is None else float(total_cost)
        )

    setattr(_coerce_solver_tracks_with_validation, _PATCH_MARKER, True)
    setattr(
        _coerce_solver_tracks_with_validation,
        "_bayescatrack_original",
        original_coerce_solver_tracks,
    )
    _multisession_tracking._coerce_solver_tracks = _coerce_solver_tracks_with_validation


def _normalize_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, np.ndarray):
        raise ValueError(f"{label} {_ERROR_SUFFIX}")

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{label} {_ERROR_SUFFIX}")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} {_ERROR_SUFFIX}") from exc

    integer_value = int(integer_value)
    if integer_value < 0:
        raise ValueError(f"{label} {_ERROR_SUFFIX}")
    return integer_value


__all__ = ["install_multisession_solver_track_validation"]
