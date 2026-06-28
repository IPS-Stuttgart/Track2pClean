"""Strict scalar validation for growth-prior configuration and indices."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_FINITE_FLOAT_PATCH_MARKER = "_bayescatrack_growth_prior_finite_float_validation_patch"
_TRACK_ROW_PATCH_MARKER = "_bayescatrack_growth_prior_track_row_validation_patch"
_SESSION_COLUMN_PATCH_MARKER = (
    "_bayescatrack_growth_prior_session_column_validation_patch"
)
_TRACK_ROW_MESSAGE = (
    "track_rows must contain integer ROI indices or negative missing sentinels"
)


def install_growth_prior_scalar_validation() -> None:
    """Install idempotent guards for growth-prior scalar controls."""

    from . import (
        growth_priors as _growth_priors,  # pylint: disable=import-outside-toplevel
    )

    _install_finite_float_validation(_growth_priors)
    _install_track_row_entry_validation(_growth_priors)
    _install_session_column_validation(_growth_priors)


def _install_finite_float_validation(_growth_priors: Any) -> None:
    original_finite_float = (
        _growth_priors._finite_float
    )  # pylint: disable=protected-access
    if getattr(original_finite_float, _FINITE_FLOAT_PATCH_MARKER, False):
        return

    @wraps(original_finite_float)
    def _finite_float_with_scalar_validation(value: Any, *, name: str) -> float:
        message = f"{name} must be finite"
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
        if isinstance(value, np.ndarray):
            if value.shape != ():
                raise ValueError(message)
            value = value.item()
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(message)
        try:
            return original_finite_float(value, name=name)
        except OverflowError as exc:
            raise ValueError(message) from exc

    setattr(_finite_float_with_scalar_validation, _FINITE_FLOAT_PATCH_MARKER, True)
    setattr(
        _finite_float_with_scalar_validation,
        "_bayescatrack_original",
        original_finite_float,
    )
    _growth_priors._finite_float = (
        _finite_float_with_scalar_validation  # pylint: disable=protected-access
    )


def _install_track_row_entry_validation(_growth_priors: Any) -> None:
    original_entry = (
        _growth_priors._integer_track_row_entry
    )  # pylint: disable=protected-access
    if getattr(original_entry, _TRACK_ROW_PATCH_MARKER, False):
        return

    @wraps(original_entry)
    def _integer_track_row_entry_with_error_normalization(value: Any) -> int:
        try:
            return original_entry(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(_TRACK_ROW_MESSAGE) from exc

    setattr(
        _integer_track_row_entry_with_error_normalization, _TRACK_ROW_PATCH_MARKER, True
    )
    setattr(
        _integer_track_row_entry_with_error_normalization,
        "_bayescatrack_original",
        original_entry,
    )
    _growth_priors._integer_track_row_entry = _integer_track_row_entry_with_error_normalization  # pylint: disable=protected-access


def _install_session_column_validation(_growth_priors: Any) -> None:
    original_column = (
        _growth_priors._normalize_session_column
    )  # pylint: disable=protected-access
    if getattr(original_column, _SESSION_COLUMN_PATCH_MARKER, False):
        return

    @wraps(original_column)
    def _normalize_session_column_with_error_normalization(
        value: Any, *, name: str, num_sessions: int
    ) -> int:
        try:
            return original_column(value, name=name, num_sessions=num_sessions)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer session column") from exc

    setattr(
        _normalize_session_column_with_error_normalization,
        _SESSION_COLUMN_PATCH_MARKER,
        True,
    )
    setattr(
        _normalize_session_column_with_error_normalization,
        "_bayescatrack_original",
        original_column,
    )
    _growth_priors._normalize_session_column = _normalize_session_column_with_error_normalization  # pylint: disable=protected-access


__all__ = ["install_growth_prior_scalar_validation"]
