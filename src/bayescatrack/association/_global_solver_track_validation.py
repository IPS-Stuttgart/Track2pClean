"""Strict validation for PyRecEst global-assignment solver tracks.

The global-assignment adapter receives solver-coordinate track mappings before
converting them to Track2p/Suite2p ROI identifiers. Without validating those raw
indices first, Python's permissive integer coercion can turn booleans, numeric
strings, fractional floats, or out-of-range detections into plausible ROI IDs.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

from . import pyrecest_global_assignment as _global_assignment

_PATCH_MARKER = "_bayescatrack_global_solver_track_validation_patch"
_ERROR_SUFFIX = "must be a non-negative integer"


def install_global_solver_track_validation() -> None:
    """Install an idempotent validator around Suite2p index conversion."""

    original = _global_assignment.tracks_to_suite2p_index_matrix
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def tracks_to_suite2p_index_matrix_with_validation(
        tracks: Sequence[Mapping[int, int]],
        sessions: Sequence[Any],
    ) -> np.ndarray:
        sessions_tuple = tuple(sessions)
        normalized_tracks = _normalize_solver_tracks(tracks, sessions_tuple)
        return original(normalized_tracks, sessions_tuple)

    setattr(tracks_to_suite2p_index_matrix_with_validation, _PATCH_MARKER, True)
    setattr(
        tracks_to_suite2p_index_matrix_with_validation,
        "_bayescatrack_original",
        original,
    )
    _global_assignment.tracks_to_suite2p_index_matrix = (  # type: ignore[assignment]
        tracks_to_suite2p_index_matrix_with_validation
    )


def _normalize_solver_tracks(
    tracks: Sequence[Mapping[int, int]],
    sessions: Sequence[Any],
) -> tuple[dict[int, int], ...]:
    session_sizes = tuple(_session_size(session) for session in sessions)
    normalized_tracks: list[dict[int, int]] = []

    for track_number, track in enumerate(tracks):
        if not isinstance(track, Mapping):
            raise TypeError(
                "Each global assignment track must be a mapping from session index "
                "to detection index"
            )

        normalized_track: dict[int, int] = {}
        for session_index, detection_index in track.items():
            session_integer = _normalize_nonnegative_integer(
                session_index,
                label=f"global assignment track {track_number} session index",
            )
            if session_integer >= len(session_sizes):
                raise ValueError(
                    f"global assignment track {track_number} session index "
                    f"{session_integer} out of bounds for {len(session_sizes)} sessions"
                )
            if session_integer in normalized_track:
                raise ValueError(
                    "global assignment tracks must not contain duplicate session "
                    "indices after normalization"
                )

            detection_integer = _normalize_nonnegative_integer(
                detection_index,
                label=f"global assignment track {track_number} detection index",
            )
            if detection_integer >= session_sizes[session_integer]:
                raise ValueError(
                    f"global assignment track {track_number} detection index "
                    f"{detection_integer} out of bounds for session {session_integer}"
                )
            normalized_track[session_integer] = detection_integer

        normalized_tracks.append(normalized_track)

    return tuple(normalized_tracks)


def _session_size(session: Any) -> int:
    try:
        size = int(session.plane_data.n_rois)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("sessions must expose plane_data.n_rois") from exc
    if size < 0:
        raise ValueError("sessions must expose a non-negative plane_data.n_rois")
    return size


def _normalize_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray, np.ndarray)):
        raise ValueError(f"{label} {_ERROR_SUFFIX}")

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{label} {_ERROR_SUFFIX}")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{label} {_ERROR_SUFFIX}") from exc

    integer_value = int(integer_value)
    if integer_value < 0:
        raise ValueError(f"{label} {_ERROR_SUFFIX}")
    return integer_value


__all__ = ["install_global_solver_track_validation"]
