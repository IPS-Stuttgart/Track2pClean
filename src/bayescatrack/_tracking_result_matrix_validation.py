"""Strict validation for tracking-result integer matrices."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_tracking_result_matrix_validation_patch"
_FILL_VALUE_ERROR = "fill_value must be a negative integer sentinel"


def install_tracking_result_matrix_validation() -> None:
    """Install idempotent validation around tracking-result matrix inputs."""

    from . import tracking as _tracking  # pylint: disable=import-outside-toplevel

    original_post_init = _tracking.SubjectTrackingResult.__post_init__
    if not getattr(original_post_init, _PATCH_MARKER, False):

        @wraps(original_post_init)
        def subject_tracking_result_post_init_with_matrix_validation(self: Any) -> Any:
            session_names = _normalize_session_names(getattr(self, "session_names", ()))
            object.__setattr__(self, "session_names", session_names)
            fill_value = _normalize_fill_value(getattr(self, "fill_value", -1))
            object.__setattr__(self, "fill_value", fill_value)
            track_rows = _normalize_track_rows(getattr(self, "track_rows"), fill_value=fill_value)
            object.__setattr__(self, "track_rows", track_rows)
            link_target_indices = getattr(self, "link_target_indices", None)
            if link_target_indices is not None:
                object.__setattr__(
                    self,
                    "link_target_indices",
                    _normalize_link_target_indices(
                        link_target_indices,
                        expected_shape=(track_rows.shape[0], max(track_rows.shape[1] - 1, 0)),
                        session_count=_infer_session_count(getattr(self, "session_names", ())),
                        fill_value=fill_value,
                    ),
                )
            return original_post_init(self)

        _mark_patch(subject_tracking_result_post_init_with_matrix_validation, original_post_init)
        _tracking.SubjectTrackingResult.__post_init__ = subject_tracking_result_post_init_with_matrix_validation  # type: ignore[method-assign]

    _patch_track_rows_first_arg(_tracking, "_build_link_cost_matrix")
    _patch_track_rows_first_arg(_tracking, "_restrict_track_rows_to_start_rois")
    _patch_track_rows_third_arg(_tracking, "_build_global_link_cost_matrices")
    _patch_track_rows_third_arg(_tracking, "_global_assignment_edge_match_results")


def _patch_track_rows_first_arg(module: Any, name: str) -> None:
    original = getattr(module, name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def function_with_track_row_validation(track_rows: Any, *args: Any, **kwargs: Any) -> Any:
        if "fill_value" not in kwargs:
            return original(track_rows, *args, **kwargs)
        fill_value = _normalize_fill_value(kwargs["fill_value"])
        kwargs = dict(kwargs)
        kwargs["fill_value"] = fill_value
        return original(_normalize_track_rows(track_rows, fill_value=fill_value), *args, **kwargs)

    _mark_patch(function_with_track_row_validation, original)
    setattr(module, name, function_with_track_row_validation)


def _patch_track_rows_third_arg(module: Any, name: str) -> None:
    original = getattr(module, name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def function_with_track_row_validation(first: Any, second: Any, track_rows: Any, *args: Any, **kwargs: Any) -> Any:
        if "fill_value" not in kwargs:
            return original(first, second, track_rows, *args, **kwargs)
        fill_value = _normalize_fill_value(kwargs["fill_value"])
        kwargs = dict(kwargs)
        kwargs["fill_value"] = fill_value
        return original(first, second, _normalize_track_rows(track_rows, fill_value=fill_value), *args, **kwargs)

    _mark_patch(function_with_track_row_validation, original)
    setattr(module, name, function_with_track_row_validation)


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _infer_session_count(session_names: Any) -> int:
    try:
        return len(_normalize_session_names(session_names))
    except ValueError:
        return 0


def _normalize_session_names(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("session_names must be a sequence of session-name values, not a bare string")
    try:
        return tuple(str(name) for name in values)
    except TypeError as exc:
        raise ValueError("session_names must be a sequence of session-name values") from exc


def _normalize_track_rows(values: Any, *, fill_value: int) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 2:
        raise ValueError("track_rows must be two-dimensional")
    normalized = np.empty(array.shape, dtype=int)
    for index, value in np.ndenumerate(array):
        normalized[index] = _normalize_roi_or_fill_value(value, field_name="track_rows", fill_value=fill_value)
    return normalized


def _normalize_link_target_indices(values: Any, *, expected_shape: tuple[int, int], session_count: int, fill_value: int) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.shape != expected_shape:
        raise ValueError("link_target_indices must have the same shape as link_costs")
    normalized = np.empty(array.shape, dtype=int)
    for index, value in np.ndenumerate(array):
        source_index = int(index[1])
        target_index = _normalize_integer_like(value, field_name="link_target_indices")
        if target_index == fill_value:
            normalized[index] = target_index
            continue
        if target_index < 0:
            raise ValueError("link_target_indices must contain session targets or fill_value")
        if target_index <= source_index or target_index >= session_count:
            raise ValueError("link_target_indices must point to later in-bounds sessions")
        normalized[index] = target_index
    return normalized


def _normalize_roi_or_fill_value(value: Any, *, field_name: str, fill_value: int) -> int:
    integer_value = _normalize_integer_like(value, field_name=field_name)
    if integer_value == fill_value:
        return integer_value
    if integer_value < 0:
        raise ValueError(f"{field_name} must contain non-negative ROI indices or fill_value")
    return integer_value


def _normalize_fill_value(value: Any) -> int:
    if isinstance(value, np.ndarray):
        raise ValueError(_FILL_VALUE_ERROR)
    try:
        integer_value = _normalize_integer_like(value, field_name="fill_value")
    except ValueError as exc:
        raise ValueError(_FILL_VALUE_ERROR) from exc
    if integer_value >= 0:
        raise ValueError(_FILL_VALUE_ERROR)
    return integer_value


def _normalize_integer_like(value: Any, *, field_name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{field_name} must contain integer values")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain integer values")
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must contain integer values")
        return int(numeric_value)
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain integer values") from exc


__all__ = ["install_tracking_result_matrix_validation"]
