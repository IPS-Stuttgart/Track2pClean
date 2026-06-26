"""Strict validation for track-row CSV export options.

``export_track_rows_csv`` uses ``include_track_id`` to choose the exported CSV
schema.  Relying on Python truthiness lets malformed values such as ``"false"``
or ``1`` silently select the wrong schema, which can break downstream readers
that expect a stable column layout.

The exported ``track_rows`` matrix is equally schema-critical: entries are
Suite2p ROI identifiers or negative missing-value sentinels.  Casting it with
``dtype=int`` would silently turn booleans, numeric strings, and fractional
floats into valid-looking ROI identifiers before the CSV is written.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_row_export_option_validation_patch"
_INCLUDE_TRACK_ID_ERROR_MESSAGE = "include_track_id must be a boolean"
_TRACK_ROWS_ERROR_MESSAGE = (
    "track_rows must contain integer ROI indices or negative missing sentinels"
)


def install_track_row_export_option_validation() -> None:
    """Install idempotent validation around track-row CSV export options."""

    from . import matching as _matching  # pylint: disable=import-outside-toplevel

    original_export = _matching.export_track_rows_csv
    if getattr(original_export, _PATCH_MARKER, False):
        return

    @wraps(original_export)
    def export_track_rows_csv_with_option_validation(*args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        track_rows = _track_rows_argument(args, kwargs)
        if track_rows is not None:
            _validate_export_track_rows(track_rows)
        kwargs["include_track_id"] = _normalize_include_track_id(
            kwargs.get("include_track_id", True)
        )
        return original_export(*args, **kwargs)

    setattr(export_track_rows_csv_with_option_validation, _PATCH_MARKER, True)
    setattr(
        export_track_rows_csv_with_option_validation,
        "_bayescatrack_original",
        original_export,
    )
    _matching.export_track_rows_csv = export_track_rows_csv_with_option_validation


def _track_rows_argument(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    if len(args) >= 3:
        if "track_rows" in kwargs:
            return None
        return args[2]
    return kwargs.get("track_rows")


def _validate_export_track_rows(values: Any) -> None:
    try:
        track_array = np.asarray(values, dtype=object)
    except ValueError as exc:
        raise ValueError(_TRACK_ROWS_ERROR_MESSAGE) from exc

    if track_array.ndim != 2:
        return

    for _, value in np.ndenumerate(track_array):
        _validate_export_track_row_value(value)


def _validate_export_track_row_value(value: Any) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_TRACK_ROWS_ERROR_MESSAGE)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_TRACK_ROWS_ERROR_MESSAGE)
        return

    try:
        operator.index(value)
    except TypeError as exc:
        raise ValueError(_TRACK_ROWS_ERROR_MESSAGE) from exc


def _normalize_include_track_id(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(_INCLUDE_TRACK_ID_ERROR_MESSAGE)


__all__ = ["install_track_row_export_option_validation"]
