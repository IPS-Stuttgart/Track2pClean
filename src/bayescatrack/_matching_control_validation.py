"""Strict validation for matching and track-stitching control values.

The matching helpers accept control scalars that determine the assignment gate,
the seed session used for stitching, and the missing-value sentinel written to
track matrices.  The original implementations used Python numeric coercion
(``float(...)``/``int(...)``) directly, which lets malformed values such as
booleans or fractional session indices silently become valid-looking controls.
These hooks fail fast before such coercions can change the requested assignment
or benchmark track population.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

from ._assignment_max_cost_validation import normalize_assignment_max_cost

_PATCH_MARKER = "_bayescatrack_matching_control_validation_patch"


def install_matching_control_validation() -> None:
    """Install idempotent validation around matching control arguments."""

    from . import matching as _matching  # pylint: disable=import-outside-toplevel

    original_solve = _matching.solve_bundle_linear_assignment
    original_build_matches = _matching.build_track_rows_from_matches
    original_bundle_roi_indices = (
        _matching._bundle_roi_indices_for_session
    )  # pylint: disable=protected-access
    original_normalize_roi_index = (
        _matching._normalize_roi_index
    )  # pylint: disable=protected-access
    original_normalize_session_index = (
        _matching._normalize_session_index
    )  # pylint: disable=protected-access

    if (
        getattr(original_solve, _PATCH_MARKER, False)
        and getattr(original_build_matches, _PATCH_MARKER, False)
        and getattr(original_bundle_roi_indices, _PATCH_MARKER, False)
        and getattr(original_normalize_roi_index, _PATCH_MARKER, False)
        and getattr(original_normalize_session_index, _PATCH_MARKER, False)
    ):
        return

    @wraps(original_solve)
    def solve_bundle_linear_assignment_with_control_validation(
        bundle: Any,
        *args: Any,
        max_cost: Any = _matching.DEFAULT_ASSIGNMENT_MAX_COST,
        **kwargs: Any,
    ) -> Any:
        return original_solve(
            bundle,
            *args,
            max_cost=_normalize_assignment_max_cost(max_cost),
            **kwargs,
        )

    @wraps(original_build_matches)
    def build_track_rows_from_matches_with_control_validation(
        session_names: Any,
        matches: Any,
        *args: Any,
        start_roi_indices: Any | None = None,
        start_session_index: Any = 0,
        fill_value: Any = -1,
        **kwargs: Any,
    ) -> np.ndarray:
        normalized_session_names = _normalize_unique_session_names(
            session_names,
            field_name="session_names",
        )
        return original_build_matches(
            normalized_session_names,
            _normalize_empty_match_collections(matches),
            *args,
            start_roi_indices=start_roi_indices,
            start_session_index=_normalize_session_index(
                start_session_index,
                len(normalized_session_names),
            ),
            fill_value=_normalize_integer_control(fill_value, "fill_value"),
            **kwargs,
        )

    @wraps(original_bundle_roi_indices)
    def bundle_roi_indices_for_session_with_control_validation(
        bundles: Any,
        session_index: Any,
    ) -> np.ndarray:
        bundles = list(bundles)
        return original_bundle_roi_indices(
            bundles,
            _normalize_session_index(session_index, len(bundles) + 1),
        )

    @wraps(original_normalize_roi_index)
    def normalize_roi_index_with_array_validation(value: Any, field_name: str) -> int:
        if isinstance(value, np.ndarray):
            raise ValueError(f"{field_name} must contain integer ROI indices")
        return original_normalize_roi_index(value, field_name)

    @wraps(original_normalize_session_index)
    def normalize_session_index_with_array_validation(
        value: Any,
        field_name: str,
        *,
        num_sessions: int | None = None,
    ) -> int:
        if isinstance(value, np.ndarray):
            raise ValueError(f"{field_name} must be an integer session index")
        return original_normalize_session_index(
            value,
            field_name,
            num_sessions=num_sessions,
        )

    _mark_patch(solve_bundle_linear_assignment_with_control_validation, original_solve)
    _mark_patch(
        build_track_rows_from_matches_with_control_validation, original_build_matches
    )
    _mark_patch(
        bundle_roi_indices_for_session_with_control_validation,
        original_bundle_roi_indices,
    )
    _mark_patch(normalize_roi_index_with_array_validation, original_normalize_roi_index)
    _mark_patch(
        normalize_session_index_with_array_validation,
        original_normalize_session_index,
    )

    _matching.solve_bundle_linear_assignment = (
        solve_bundle_linear_assignment_with_control_validation
    )
    _matching.build_track_rows_from_matches = (
        build_track_rows_from_matches_with_control_validation
    )
    _matching._bundle_roi_indices_for_session = bundle_roi_indices_for_session_with_control_validation  # pylint: disable=protected-access
    _matching._normalize_roi_index = (
        normalize_roi_index_with_array_validation  # pylint: disable=protected-access
    )
    _matching._normalize_session_index = normalize_session_index_with_array_validation  # pylint: disable=protected-access


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _normalize_empty_match_collections(matches: Any) -> Any:
    """Normalize explicit empty match collections to an empty pair matrix."""

    if isinstance(matches, (str, bytes, bytearray)):
        return matches
    try:
        match_iterator = iter(matches)
    except TypeError:
        return matches
    return [_normalize_empty_match_collection(match) for match in match_iterator]


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


def _normalize_empty_match_collection(match: Any) -> Any:
    if isinstance(match, Mapping):
        return match
    if isinstance(match, tuple) and len(match) == 2:
        return match

    try:
        match_array = np.asarray(match)
    except ValueError:
        return match

    if match_array.size != 0:
        return match
    if match_array.ndim == 1 or (match_array.ndim == 2 and match_array.shape[1] == 2):
        return np.empty((0, 2), dtype=int)
    raise TypeError("unsupported match representation")


def _normalize_assignment_max_cost(value: Any) -> float | None:
    return normalize_assignment_max_cost(value)


def _normalize_session_index(value: Any, n_sessions: int) -> int:
    session_index = _normalize_integer_control(value, "start_session_index")
    if session_index < 0 or session_index >= n_sessions:
        raise IndexError(
            f"start_session_index {session_index} out of bounds for {n_sessions} sessions"
        )
    return session_index


def _normalize_integer_control(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray, np.ndarray)):
        raise ValueError(f"{field_name} must be an integer")

    try:
        return int(operator.index(value))
    except TypeError:
        pass

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
    else:
        raise ValueError(f"{field_name} must be an integer")

    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{field_name} must be an integer")
    return int(numeric_value)


__all__ = ["install_matching_control_validation"]
