"""Normalize advanced ROI pruning configuration values."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_advanced_pruning_normalization_patch"
_ERROR_MESSAGE = "{name} must be a positive integer or None"
_STRICT_CONFIG_MODULE = "bayescatrack._strict_config_validation"


def install_advanced_pruning_normalization() -> None:
    """Install idempotent scalar normalization for advanced ROI pruning controls."""

    from . import (
        advanced_roi_components as advanced,  # pylint: disable=import-outside-toplevel
    )

    original_int_normalizer = (
        advanced._normalize_optional_positive_int
    )  # pylint: disable=protected-access
    if not getattr(original_int_normalizer, _PATCH_MARKER, False):

        @wraps(original_int_normalizer)
        def normalized_optional_positive_int(value: Any, *, name: str) -> int | None:
            return _optional_positive_int(value, name=name)

        setattr(normalized_optional_positive_int, _PATCH_MARKER, True)
        setattr(
            normalized_optional_positive_int,
            "_bayescatrack_original",
            original_int_normalizer,
        )
        advanced._normalize_optional_positive_int = (
            normalized_optional_positive_int  # pylint: disable=protected-access
        )

    if getattr(advanced.CandidatePruningConfig, "__module__", None) == _STRICT_CONFIG_MODULE:
        return

    original_post_init = advanced.CandidatePruningConfig.__post_init__
    if getattr(original_post_init, _PATCH_MARKER, False):
        return

    @wraps(original_post_init)
    def normalized_config_post_init(self: Any) -> None:
        object.__setattr__(
            self,
            "top_k_per_roi",
            advanced._normalize_optional_positive_int(  # pylint: disable=protected-access
                getattr(self, "top_k_per_roi"),
                name="top_k_per_roi",
            ),
        )
        object.__setattr__(
            self,
            "include_column_top_k",
            advanced._normalize_bool(  # pylint: disable=protected-access
                getattr(self, "include_column_top_k"),
                name="include_column_top_k",
            ),
        )
        object.__setattr__(
            self,
            "gate_margin",
            advanced._normalize_optional_nonnegative_float(  # pylint: disable=protected-access
                getattr(self, "gate_margin"),
                name="gate_margin",
            ),
        )
        object.__setattr__(
            self,
            "large_cost",
            advanced._normalize_positive_float(  # pylint: disable=protected-access
                getattr(self, "large_cost"),
                name="large_cost",
            ),
        )

    setattr(normalized_config_post_init, _PATCH_MARKER, True)
    setattr(normalized_config_post_init, "_bayescatrack_original", original_post_init)
    advanced.CandidatePruningConfig.__post_init__ = normalized_config_post_init


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_, bytes)):
        raise ValueError(_ERROR_MESSAGE.format(name=name))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(_ERROR_MESSAGE.format(name=name))
        try:
            normalized = int(text, 10)
        except ValueError as exc:
            raise ValueError(_ERROR_MESSAGE.format(name=name)) from exc
    else:
        try:
            normalized = operator.index(value)
        except TypeError:
            if not isinstance(value, (float, np.floating)):
                raise ValueError(_ERROR_MESSAGE.format(name=name)) from None
            numeric = float(value)
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(_ERROR_MESSAGE.format(name=name))
            normalized = int(numeric)
    normalized = int(normalized)
    if normalized <= 0:
        raise ValueError(_ERROR_MESSAGE.format(name=name))
    return normalized


__all__ = ["install_advanced_pruning_normalization"]
