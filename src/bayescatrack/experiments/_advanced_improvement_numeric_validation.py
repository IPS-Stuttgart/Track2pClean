"""Validation for advanced-improvement workbench numeric controls."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_track2pclean_advanced_improvement_numeric_bool_validation_patch"


def install_advanced_improvement_numeric_validation() -> None:
    """Reject boolean-like values for numeric workbench controls."""

    workbench = importlib.import_module(
        "bayescatrack.experiments.advanced_improvement_workbench"
    )
    original = workbench._validated_numeric_float  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _validated_numeric_float_without_bools(
        value: Any,
        *,
        name: str,
    ) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be finite")
        return original(value, name=name)

    setattr(_validated_numeric_float_without_bools, _PATCH_MARKER, True)
    setattr(
        _validated_numeric_float_without_bools,
        "_bayescatrack_original",
        original,
    )
    workbench._validated_numeric_float = (  # pylint: disable=protected-access
        _validated_numeric_float_without_bools
    )


__all__ = ["install_advanced_improvement_numeric_validation"]
