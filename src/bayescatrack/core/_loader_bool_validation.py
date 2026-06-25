"""Boolean-control compatibility patch for loader validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _loader_validation

_PATCH_ATTR = "_bayescatrack_numpy_bool_loader_patch"


def install_numpy_bool_loader_validation() -> None:
    """Allow NumPy boolean scalars for loader boolean controls."""

    original = _loader_validation._strict_bool  # pylint: disable=protected-access
    if getattr(original, _PATCH_ATTR, False):
        return

    def _strict_bool(value: Any, *, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
        return bool(value)

    setattr(_strict_bool, _PATCH_ATTR, True)
    setattr(_strict_bool, "_bayescatrack_original", original)
    _loader_validation._strict_bool = _strict_bool  # pylint: disable=protected-access
