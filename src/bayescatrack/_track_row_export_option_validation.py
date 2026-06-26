"""Strict validation for track-row CSV export options.

``export_track_rows_csv`` uses ``include_track_id`` to choose the exported CSV
schema.  Relying on Python truthiness lets malformed values such as ``"false"``
or ``1`` silently select the wrong schema, which can break downstream readers
that expect a stable column layout.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_row_export_option_validation_patch"
_ERROR_MESSAGE = "include_track_id must be a boolean"


def install_track_row_export_option_validation() -> None:
    """Install idempotent validation around track-row CSV export options."""

    from . import matching as _matching  # pylint: disable=import-outside-toplevel

    original_export = _matching.export_track_rows_csv
    if getattr(original_export, _PATCH_MARKER, False):
        return

    @wraps(original_export)
    def export_track_rows_csv_with_option_validation(*args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
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


def _normalize_include_track_id(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(_ERROR_MESSAGE)


__all__ = ["install_track_row_export_option_validation"]
