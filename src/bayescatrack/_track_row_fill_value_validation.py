"""Strict validation for track-row missing-value sentinels.

Track-row stitching uses ``fill_value`` both to initialize missing entries and
as the stop sentinel while following pairwise matches.  Passing that value
through ``int(...)`` lets malformed inputs such as booleans or fractional floats
silently change the sentinel.  Non-negative sentinels are also unsafe because
Suite2p ROI identifiers are non-negative and can therefore collide with real
cells.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_row_fill_value_validation_patch"
_ERROR_MESSAGE = (
    "fill_value must be an integer; fill_value must be a negative integer sentinel"
)
_INDEX_PROTOCOL_PASSTHROUGH_TYPES = (
    bool,
    np.bool_,
    int,
    np.integer,
    float,
    np.floating,
    str,
    bytes,
    bytearray,
    np.ndarray,
)


def install_track_row_fill_value_validation() -> None:
    """Install idempotent validation around track-row missing sentinels."""

    from . import matching as _matching  # pylint: disable=import-outside-toplevel

    original_matches = _matching.build_track_rows_from_matches
    if not getattr(original_matches, _PATCH_MARKER, False):

        @wraps(original_matches)
        def build_track_rows_from_matches_with_fill_value_validation(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            kwargs = dict(kwargs)
            kwargs["fill_value"] = _normalize_fill_value(kwargs.get("fill_value", -1))
            _normalize_track_row_index_kwargs(kwargs)
            return original_matches(*args, **kwargs)

        setattr(
            build_track_rows_from_matches_with_fill_value_validation,
            _PATCH_MARKER,
            True,
        )
        setattr(
            build_track_rows_from_matches_with_fill_value_validation,
            "_bayescatrack_original",
            original_matches,
        )
        _matching.build_track_rows_from_matches = (
            build_track_rows_from_matches_with_fill_value_validation
        )

    original_bundles = _matching.build_track_rows_from_bundles
    if not getattr(original_bundles, _PATCH_MARKER, False):

        @wraps(original_bundles)
        def build_track_rows_from_bundles_with_fill_value_validation(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            kwargs = dict(kwargs)
            kwargs["fill_value"] = _normalize_fill_value(kwargs.get("fill_value", -1))
            _normalize_track_row_index_kwargs(kwargs)
            return original_bundles(*args, **kwargs)

        setattr(
            build_track_rows_from_bundles_with_fill_value_validation,
            _PATCH_MARKER,
            True,
        )
        setattr(
            build_track_rows_from_bundles_with_fill_value_validation,
            "_bayescatrack_original",
            original_bundles,
        )
        _matching.build_track_rows_from_bundles = (
            build_track_rows_from_bundles_with_fill_value_validation
        )


def _normalize_track_row_index_kwargs(kwargs: dict[str, Any]) -> None:
    if "start_session_index" in kwargs:
        kwargs["start_session_index"] = _normalize_index_protocol_value(
            kwargs["start_session_index"],
            error_message="start_session_index must be an integer session index",
        )
    if kwargs.get("start_roi_indices") is not None:
        kwargs["start_roi_indices"] = _normalize_start_roi_indices(
            kwargs["start_roi_indices"]
        )


def _normalize_start_roi_indices(values: Any) -> Any:
    if isinstance(values, (str, bytes, bytearray, np.ndarray)):
        return values
    try:
        iterator = iter(values)
    except TypeError:
        return values
    return [
        _normalize_index_protocol_value(
            value,
            error_message="start_roi_indices must contain integer ROI indices",
        )
        for value in iterator
    ]


def _normalize_index_protocol_value(value: Any, *, error_message: str) -> Any:
    if isinstance(value, _INDEX_PROTOCOL_PASSTHROUGH_TYPES):
        return value
    if getattr(type(value), "__index__", None) is None:
        return value
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc


def _normalize_fill_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_ERROR_MESSAGE)
    if isinstance(value, np.ndarray):
        raise ValueError(_ERROR_MESSAGE)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_ERROR_MESSAGE)
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(_ERROR_MESSAGE) from exc

    integer_value = int(integer_value)
    if integer_value >= 0:
        raise ValueError(
            "fill_value must be a negative integer sentinel that cannot collide "
            "with non-negative ROI indices"
        )
    return integer_value


__all__ = ["install_track_row_fill_value_validation"]
