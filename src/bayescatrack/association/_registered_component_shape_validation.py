"""Validate registered pairwise component shape consistency.

Registered-ROI masking mutates every two-dimensional pairwise component over the
same source-target layout.  If one component has a different shape, silently
skipping it leaves invalid target ROI columns unmasked for that component.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_registered_component_shape_validation_patch"


def install_registered_component_shape_validation() -> None:
    """Install idempotent validation for registered pairwise component shapes."""

    from . import registered_masks  # pylint: disable=import-outside-toplevel

    original = registered_masks._infer_pairwise_component_shape  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _infer_pairwise_component_shape_with_validation(
        pairwise_components: Mapping[str, Any],
    ) -> tuple[int, int]:
        inferred_shape: tuple[int, int] | None = None
        inferred_key: str | None = None
        for key, values in pairwise_components.items():
            array_values = np.asarray(values)
            if array_values.ndim != 2:
                continue
            component_shape = (int(array_values.shape[0]), int(array_values.shape[1]))
            if inferred_shape is None:
                inferred_shape = component_shape
                inferred_key = str(key)
                continue
            if component_shape != inferred_shape:
                raise ValueError(
                    "All two-dimensional pairwise components must have the same shape; "
                    f"{key!r} has shape {component_shape}, expected {inferred_shape} "
                    f"from {inferred_key!r}"
                )
        if inferred_shape is None:
            return original(pairwise_components)
        return inferred_shape

    setattr(_infer_pairwise_component_shape_with_validation, _PATCH_MARKER, True)
    setattr(
        _infer_pairwise_component_shape_with_validation,
        "_bayescatrack_original",
        original,
    )
    registered_masks._infer_pairwise_component_shape = (  # pylint: disable=protected-access
        _infer_pairwise_component_shape_with_validation
    )


__all__ = ["install_registered_component_shape_validation"]
