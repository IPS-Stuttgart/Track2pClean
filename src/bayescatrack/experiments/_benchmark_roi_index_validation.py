"""Strict ROI-index validation for Track2p benchmark helper predicates."""

from __future__ import annotations

import operator
from functools import wraps

import numpy as np

_PATCH_MARKER = "_bayescatrack_benchmark_roi_index_validation_patch"


def install_benchmark_roi_index_validation() -> None:
    """Install idempotent strict ROI-index validation in benchmark helpers."""

    from bayescatrack.experiments import track2p_benchmark as benchmark

    original = benchmark._is_valid_roi_index  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _is_strict_valid_roi_index(value: object) -> bool:
        return _is_nonnegative_integral_roi_index(value)

    setattr(_is_strict_valid_roi_index, _PATCH_MARKER, True)
    setattr(_is_strict_valid_roi_index, "_bayescatrack_original", original)
    benchmark._is_valid_roi_index = _is_strict_valid_roi_index  # type: ignore[assignment]  # pylint: disable=protected-access


def _is_nonnegative_integral_roi_index(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return False

    try:
        return operator.index(value) >= 0  # type: ignore[arg-type]
    except TypeError:
        pass

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        return bool(
            np.isfinite(numeric_value)
            and numeric_value.is_integer()
            and int(numeric_value) >= 0
        )

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            return int(text) >= 0
        except ValueError:
            return False

    return False


__all__ = ["install_benchmark_roi_index_validation"]
