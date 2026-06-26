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
_ERROR_MESSAGE = "fill_value must be a negative integer sentinel"


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


def _normalize_fill_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
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
