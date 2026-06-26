"""Validation patch for multisession solver track outputs."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_INSTALLED_FLAG = "_bayescatrack_multisession_solver_validation_installed"


def _coerce_solver_track_index(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be an integer, got boolean {value!r}")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{field_name} must be integer-like, got {value!r}")
        return int(value)
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be integer-like, got {value!r}") from exc
    return int(normalized)


def install_multisession_solver_validation(module: Any | None = None) -> None:
    """Install strict normalization for PyRecEst multisession solver outputs."""

    if module is None:
        from . import multisession_tracking as target_module
    else:
        target_module = module

    if getattr(target_module, _INSTALLED_FLAG, False):
        return

    def _coerce_solver_tracks(
        raw_result: Any,
    ) -> tuple[tuple[dict[int, int], ...], float | None]:
        if isinstance(raw_result, dict):
            tracks = raw_result.get("tracks")
            total_cost = raw_result.get("total_cost")
        else:
            tracks = getattr(raw_result, "tracks", raw_result)
            total_cost = getattr(raw_result, "total_cost", None)

        if tracks is None:
            raise ValueError("Solver result does not contain tracks")

        normalized_tracks: list[dict[int, int]] = []
        for track in tracks:
            if isinstance(track, Mapping):
                normalized_track: dict[int, int] = {}
                for session_index, detection_index in track.items():
                    normalized_track[
                        _coerce_solver_track_index(session_index, "session index")
                    ] = _coerce_solver_track_index(
                        detection_index,
                        "detection index",
                    )
                normalized_tracks.append(normalized_track)
                continue
            raise TypeError(
                "Each returned track must be a mapping from session index to detection index"
            )

        return tuple(normalized_tracks), (
            None if total_cost is None else float(total_cost)
        )

    def _tracks_to_matrix(
        tracks: Sequence[Mapping[int, int]],
        n_sessions: int,
    ) -> np.ndarray:
        n_sessions = int(n_sessions)
        if n_sessions < 0:
            raise ValueError("n_sessions must be non-negative")
        track_matrix = np.full((len(tracks), n_sessions), -1, dtype=int)
        for track_index, track in enumerate(tracks):
            for session_index, detection_index in track.items():
                if session_index < 0 or session_index >= n_sessions:
                    raise ValueError(
                        f"Track references session index {session_index} outside 0..{n_sessions - 1}"
                    )
                if detection_index < 0:
                    raise ValueError(
                        f"Track references negative detection index {detection_index} in session {session_index}"
                    )
                track_matrix[track_index, session_index] = int(detection_index)
        return track_matrix

    def _track_matrix_to_roi_index_matrix(
        track_matrix: np.ndarray,
        sessions: Sequence[Any],
    ) -> np.ndarray:
        track_matrix = np.asarray(track_matrix, dtype=int)
        sessions = list(sessions)
        if track_matrix.ndim != 2:
            raise ValueError("track_matrix must be two-dimensional")
        if track_matrix.shape[1] != len(sessions):
            raise ValueError("track_matrix must have one column per session")

        roi_index_matrix = np.full(track_matrix.shape, -1, dtype=int)
        for session_index, session in enumerate(sessions):
            if session.plane_data.roi_indices is None:
                lookup = np.arange(session.plane_data.n_rois, dtype=int)
            else:
                lookup = np.asarray(session.plane_data.roi_indices, dtype=int)
            if lookup.shape != (int(session.plane_data.n_rois),):
                raise ValueError(
                    "plane_data.roi_indices must have one entry per loaded ROI"
                )

            column = track_matrix[:, session_index]
            invalid_missing = column < -1
            if np.any(invalid_missing):
                raise ValueError(
                    f"track_matrix contains invalid missing marker below -1 in session {session_index}"
                )

            present = column >= 0
            if np.any(present):
                max_detection_index = int(np.max(column[present]))
                if max_detection_index >= lookup.shape[0]:
                    raise ValueError(
                        f"Track references detection index {max_detection_index} outside 0..{lookup.shape[0] - 1} for session {session_index}"
                    )
                roi_index_matrix[present, session_index] = lookup[column[present]]
        return roi_index_matrix

    target_module._coerce_solver_tracks = _coerce_solver_tracks
    target_module._tracks_to_matrix = _tracks_to_matrix
    target_module._track_matrix_to_roi_index_matrix = _track_matrix_to_roi_index_matrix
    setattr(target_module, _INSTALLED_FLAG, True)
