"""Strict validation for linear-assignment bundle layouts and options."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_assignment_layout_validation_patch"
_FILL_VALUE_PATCH_MARKER = "_bayescatrack_matching_fill_value_validation_patch"
_SESSION_NAMES_PATCH_MARKER = "_bayescatrack_matching_session_name_validation_patch"
_BUNDLE_SESSION_NAMES_PATCH_MARKER = (
    "_bayescatrack_matching_bundle_session_name_validation_patch"
)
_EXPORT_SESSION_NAMES_PATCH_MARKER = (
    "_bayescatrack_matching_export_session_names_validation_patch"
)
_FILL_VALUE_ERROR_MESSAGE = (
    "fill_value must be an integer; fill_value must be a negative integer sentinel"
)


def install_matching_layout_validation(matching_module: Any) -> None:
    """Install idempotent validators on matching assignment and row-stitching helpers."""

    _patch_assignment_solver(matching_module)
    _patch_fill_value_keyword_function(matching_module, "build_track_rows_from_matches")
    _patch_fill_value_keyword_function(matching_module, "build_track_rows_from_bundles")
    _patch_track_row_session_names(matching_module)
    _patch_bundle_session_names(matching_module)
    _patch_export_session_names(matching_module)


def _patch_assignment_solver(matching_module: Any) -> None:
    original: Callable[..., Any] = matching_module.solve_bundle_linear_assignment
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def solve_bundle_linear_assignment(bundle: Any, *args: Any, **kwargs: Any) -> Any:
        _validate_bundle_assignment_layout(bundle)
        if "max_cost" in kwargs:
            kwargs = dict(kwargs)
            kwargs["max_cost"] = _normalize_max_cost(kwargs["max_cost"])
        return original(bundle, *args, **kwargs)

    setattr(solve_bundle_linear_assignment, _PATCH_MARKER, True)
    setattr(solve_bundle_linear_assignment, "_bayescatrack_original", original)
    matching_module.solve_bundle_linear_assignment = solve_bundle_linear_assignment


def _patch_fill_value_keyword_function(matching_module: Any, name: str) -> None:
    original: Callable[..., Any] = getattr(matching_module, name)
    if getattr(original, _FILL_VALUE_PATCH_MARKER, False):
        return

    @wraps(original)
    def function_with_fill_value_validation(*args: Any, **kwargs: Any) -> Any:
        if "fill_value" in kwargs:
            kwargs = dict(kwargs)
            kwargs["fill_value"] = _normalize_fill_value(kwargs["fill_value"])
        return original(*args, **kwargs)

    setattr(function_with_fill_value_validation, _FILL_VALUE_PATCH_MARKER, True)
    setattr(function_with_fill_value_validation, "_bayescatrack_original", original)
    setattr(matching_module, name, function_with_fill_value_validation)


def _patch_track_row_session_names(matching_module: Any) -> None:
    original: Callable[..., Any] = matching_module.build_track_rows_from_matches
    if getattr(original, _SESSION_NAMES_PATCH_MARKER, False):
        return

    @wraps(original)
    def build_track_rows_from_matches_with_session_name_validation(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if args:
            normalized_session_names = _normalize_unique_session_names(
                args[0],
                field_name="session_names",
            )
            args = (normalized_session_names, *args[1:])
        elif "session_names" in kwargs:
            kwargs = dict(kwargs)
            kwargs["session_names"] = _normalize_unique_session_names(
                kwargs["session_names"],
                field_name="session_names",
            )
        return original(*args, **kwargs)

    setattr(
        build_track_rows_from_matches_with_session_name_validation,
        _SESSION_NAMES_PATCH_MARKER,
        True,
    )
    setattr(
        build_track_rows_from_matches_with_session_name_validation,
        "_bayescatrack_original",
        original,
    )
    matching_module.build_track_rows_from_matches = (
        build_track_rows_from_matches_with_session_name_validation
    )


def _patch_bundle_session_names(matching_module: Any) -> None:
    original: Callable[..., Any] = matching_module._session_names_from_bundles
    if getattr(original, _BUNDLE_SESSION_NAMES_PATCH_MARKER, False):
        return

    @wraps(original)
    def _session_names_from_bundles_with_session_name_validation(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, ...]:
        return _normalize_unique_session_names(
            original(*args, **kwargs),
            field_name="bundle session_names",
        )

    setattr(
        _session_names_from_bundles_with_session_name_validation,
        _BUNDLE_SESSION_NAMES_PATCH_MARKER,
        True,
    )
    setattr(
        _session_names_from_bundles_with_session_name_validation,
        "_bayescatrack_original",
        original,
    )
    matching_module._session_names_from_bundles = (  # pylint: disable=protected-access
        _session_names_from_bundles_with_session_name_validation
    )


def _patch_export_session_names(matching_module: Any) -> None:
    original: Callable[..., Any] = matching_module.export_track_rows_csv
    if getattr(original, _EXPORT_SESSION_NAMES_PATCH_MARKER, False):
        return

    @wraps(original)
    def export_track_rows_csv_with_session_name_validation(
        *args: Any, **kwargs: Any
    ) -> Any:
        if len(args) >= 2:
            args_list = list(args)
            args_list[1] = _normalize_unique_session_names(
                args_list[1],
                field_name="session_names",
            )
            args = tuple(args_list)
        elif "session_names" in kwargs:
            kwargs = dict(kwargs)
            kwargs["session_names"] = _normalize_unique_session_names(
                kwargs["session_names"],
                field_name="session_names",
            )
        return original(*args, **kwargs)

    setattr(
        export_track_rows_csv_with_session_name_validation,
        _EXPORT_SESSION_NAMES_PATCH_MARKER,
        True,
    )
    setattr(
        export_track_rows_csv_with_session_name_validation,
        "_bayescatrack_original",
        original,
    )
    matching_module.export_track_rows_csv = (
        export_track_rows_csv_with_session_name_validation
    )


def _validate_bundle_assignment_layout(bundle: Any) -> None:
    try:
        cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
    except (TypeError, ValueError):
        return
    if cost_matrix.ndim != 2:
        return

    _validate_roi_index_axis(
        "reference_roi_indices",
        bundle.reference_roi_indices,
        expected_len=int(cost_matrix.shape[0]),
        axis_name="row",
    )
    _validate_roi_index_axis(
        "measurement_roi_indices",
        bundle.measurement_roi_indices,
        expected_len=int(cost_matrix.shape[1]),
        axis_name="column",
    )


def _validate_roi_index_axis(
    field_name: str,
    values: Any,
    *,
    expected_len: int,
    axis_name: str,
) -> None:
    roi_indices = np.asarray(values)
    if roi_indices.ndim != 1:
        raise ValueError(f"bundle.{field_name} must be one-dimensional")

    actual_len = int(roi_indices.shape[0])
    if actual_len != int(expected_len):
        raise ValueError(
            f"bundle.{field_name} length ({actual_len}) must match "
            f"pairwise_cost_matrix {axis_name} dimension ({expected_len})"
        )


def _normalize_max_cost(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("max_cost must be a finite non-negative value")
    try:
        max_cost = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_cost must be a finite non-negative value") from exc
    if not np.isfinite(max_cost) or max_cost < 0.0:
        raise ValueError("max_cost must be a finite non-negative value")
    return float(max_cost)


def _normalize_fill_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_FILL_VALUE_ERROR_MESSAGE)
    if isinstance(value, np.ndarray):
        raise ValueError(_FILL_VALUE_ERROR_MESSAGE)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_FILL_VALUE_ERROR_MESSAGE)
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(_FILL_VALUE_ERROR_MESSAGE) from exc

    integer_value = int(integer_value)
    if integer_value >= 0:
        raise ValueError(
            "fill_value must be a negative integer sentinel that cannot collide "
            "with non-negative ROI indices"
        )
    return integer_value


def _normalize_unique_session_names(
    session_names: Any,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(session_names, (str, bytes, bytearray)):
        raise ValueError(
            f"{field_name} must be a sequence of session-name values, not a bare string"
        )
    try:
        normalized_session_names = tuple(str(name) for name in session_names)
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must be a sequence of session-name values"
        ) from exc

    if not normalized_session_names:
        raise ValueError(f"{field_name} must not be empty")

    seen: set[str] = set()
    duplicates: list[str] = []
    for session_name in normalized_session_names:
        if session_name in seen and session_name not in duplicates:
            duplicates.append(session_name)
        seen.add(session_name)
    if duplicates:
        duplicate_summary = ", ".join(repr(name) for name in duplicates)
        raise ValueError(
            f"{field_name} must contain unique session names; "
            f"duplicate values: {duplicate_summary}"
        )
    return normalized_session_names


__all__ = ["install_matching_layout_validation"]
