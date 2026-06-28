"""Strict validation for ground-truth track tables and metric row controls."""

from __future__ import annotations

import csv
import operator
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_ground_truth_track_validation_patch"
_ROW_TUPLES_PATCH_MARKER = "_bayescatrack_ground_truth_row_tuples_validation_patch"
_ROI_ERROR = "ROI index must be a non-negative integer or -1 missing sentinel"
_BOOLEAN_ROI_ERROR = "boolean ROI index is not a valid ground-truth ROI value"
_SESSION_NAME_ERROR = "session_names must be unique"
_HORIZON_ERROR = "horizon must be an integer between 1 and the number of sessions"
_REQUIRE_COMPLETE_ERROR = "require_complete must be a boolean"
_MISSING_VALUE_STRINGS = {"", "na", "nan", "none", "null", "-"}
_EMPTY_ROWS_MESSAGE = "contains no data rows"


def install_ground_truth_track_validation() -> None:
    """Install idempotent validation for ground-truth track ROI indices."""

    from . import ground_truth_eval as module

    original_post_init = module.TrackTable.__post_init__
    original_row_tuples = module.TrackTable.row_tuples
    original_rows_from_csv = module._rows_from_csv
    post_init_is_patched = getattr(original_post_init, _PATCH_MARKER, False)
    row_tuples_is_patched = getattr(original_row_tuples, _ROW_TUPLES_PATCH_MARKER, False)
    rows_from_csv_is_patched = getattr(original_rows_from_csv, _PATCH_MARKER, False)
    if post_init_is_patched and row_tuples_is_patched and rows_from_csv_is_patched:
        return

    if not post_init_is_patched:
        original_parse_roi_value = module._parse_roi_value

        @wraps(original_post_init)
        def checked_post_init(self: Any) -> None:
            session_names = tuple(str(name) for name in self.session_names)
            tracks = _normalize_track_matrix(self.tracks)
            if tracks.ndim != 2:
                raise ValueError("tracks must have shape (n_tracks, n_sessions)")
            if tracks.shape[1] != len(session_names):
                raise ValueError(
                    "tracks second dimension must equal the number of session names"
                )
            if len(session_names) == 0:
                raise ValueError("session_names must not be empty")
            _validate_unique_session_names(session_names)
            object.__setattr__(self, "session_names", session_names)
            object.__setattr__(self, "tracks", tracks)

        @wraps(original_parse_roi_value)
        def checked_parse_roi_value(value: Any) -> int:
            return _normalize_roi_index(value)

        _mark_patch(checked_post_init, original_post_init)
        _mark_patch(checked_parse_roi_value, original_parse_roi_value)
        module.TrackTable.__post_init__ = checked_post_init
        module._parse_roi_value = checked_parse_roi_value

    if not row_tuples_is_patched:

        @wraps(original_row_tuples)
        def checked_row_tuples(
            self: Any,
            *,
            horizon: Any | None = None,
            require_complete: Any = False,
        ) -> Any:
            normalized_horizon = _normalize_horizon(horizon, num_sessions=self.n_sessions)
            normalized_require_complete = _normalize_require_complete(require_complete)
            return original_row_tuples(
                self,
                horizon=normalized_horizon,
                require_complete=normalized_require_complete,
            )

        setattr(checked_row_tuples, _ROW_TUPLES_PATCH_MARKER, True)
        setattr(checked_row_tuples, "_bayescatrack_original", original_row_tuples)
        module.TrackTable.row_tuples = checked_row_tuples

    if not rows_from_csv_is_patched:

        @wraps(original_rows_from_csv)
        def checked_rows_from_csv(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
            try:
                return original_rows_from_csv(csv_path)
            except ValueError as exc:
                if _EMPTY_ROWS_MESSAGE not in str(exc):
                    raise
            return _read_header_only_csv(csv_path)

        _mark_patch(checked_rows_from_csv, original_rows_from_csv)
        module._rows_from_csv = checked_rows_from_csv


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _validate_unique_session_names(session_names: tuple[str, ...]) -> None:
    if len(set(session_names)) != len(session_names):
        raise ValueError(_SESSION_NAME_ERROR)


def _normalize_horizon(value: Any | None, *, num_sessions: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_HORIZON_ERROR)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_HORIZON_ERROR)
        normalized = int(numeric_value)
    else:
        try:
            normalized = operator.index(value)
        except TypeError as exc:
            raise ValueError(_HORIZON_ERROR) from exc
    normalized = int(normalized)
    if not 1 <= normalized <= int(num_sessions):
        raise ValueError(_HORIZON_ERROR)
    return normalized


def _normalize_require_complete(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(_REQUIRE_COMPLETE_ERROR)


def _normalize_track_matrix(tracks: Any) -> np.ndarray:
    track_array = np.asarray(tracks, dtype=object)
    if track_array.ndim != 2:
        return track_array
    normalized = np.empty(track_array.shape, dtype=int)
    for index, value in np.ndenumerate(track_array):
        normalized[index] = _normalize_roi_index(value)
    return normalized


def _normalize_roi_index(value: Any) -> int:
    if value is None:
        return -1
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_BOOLEAN_ROI_ERROR)
    try:
        return _validate_roi_integer(operator.index(value))
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if np.isnan(numeric_value):
            return -1
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return _validate_roi_integer(int(numeric_value))
        raise ValueError(_ROI_ERROR)
    if isinstance(value, str):
        return _normalize_roi_index_string(value)
    raise ValueError(_ROI_ERROR)


def _normalize_roi_index_string(value: str) -> int:
    text = value.strip()
    if text.lower().replace(" ", "_") in _MISSING_VALUE_STRINGS:
        return -1
    try:
        numeric_value = float(text)
    except ValueError as exc:
        raise ValueError(_ROI_ERROR) from exc
    if np.isnan(numeric_value):
        return -1
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(_ROI_ERROR)
    return _validate_roi_integer(int(numeric_value))


def _validate_roi_integer(value: int) -> int:
    normalized_value = int(value)
    if normalized_value == -1 or normalized_value >= 0:
        return normalized_value
    raise ValueError(_ROI_ERROR)


def _read_header_only_csv(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        return [str(name) for name in reader.fieldnames], []


__all__ = ["install_ground_truth_track_validation"]
