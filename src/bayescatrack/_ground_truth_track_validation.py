"""Strict validation for ground-truth track ROI indices."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_ground_truth_track_validation_patch"
_ROI_ERROR = "ROI index must be a non-negative integer or -1 missing sentinel"
_MISSING_VALUE_STRINGS = {"", "na", "nan", "none", "null", "-"}


def install_ground_truth_track_validation() -> None:
    """Install idempotent validation for ground-truth track ROI indices."""

    from . import ground_truth_eval as module

    original_post_init = module.TrackTable.__post_init__
    if getattr(original_post_init, _PATCH_MARKER, False):
        return
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
        object.__setattr__(self, "session_names", session_names)
        object.__setattr__(self, "tracks", tracks)

    @wraps(original_parse_roi_value)
    def checked_parse_roi_value(value: Any) -> int:
        return _normalize_roi_index(value)

    _mark_patch(checked_post_init, original_post_init)
    _mark_patch(checked_parse_roi_value, original_parse_roi_value)
    module.TrackTable.__post_init__ = checked_post_init
    module._parse_roi_value = checked_parse_roi_value


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


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
        raise ValueError(_ROI_ERROR)
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


__all__ = ["install_ground_truth_track_validation"]
