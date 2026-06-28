"""Strict validation for calibrated-training supervised masks."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_PATCH_ATTR = "_track2pclean_supervised_mask_validation"


def install_supervised_mask_validation(calibrated_costs: ModuleType) -> None:
    """Install fail-fast validation for ``ReferencePairwiseExamples.supervised_mask``."""

    original_validated_supervised_mask = (
        calibrated_costs._validated_supervised_mask
    )  # noqa: SLF001
    if getattr(original_validated_supervised_mask, _PATCH_ATTR, False):
        return

    @wraps(original_validated_supervised_mask)
    def _validated_supervised_mask_with_strict_values(
        block: Any,
        label_shape: tuple[int, int],
    ) -> np.ndarray:
        if getattr(block, "supervised_mask", None) is None:
            return original_validated_supervised_mask(block, label_shape)
        return _validated_boolean_mask(block.supervised_mask, label_shape)

    setattr(_validated_supervised_mask_with_strict_values, _PATCH_ATTR, True)
    setattr(
        _validated_supervised_mask_with_strict_values,
        "_track2pclean_original",
        original_validated_supervised_mask,
    )
    calibrated_costs._validated_supervised_mask = (
        _validated_supervised_mask_with_strict_values  # noqa: SLF001
    )


def _validated_boolean_mask(values: Any, label_shape: tuple[int, int]) -> np.ndarray:
    try:
        mask = np.asarray(values, dtype=object)
    except ValueError as exc:
        raise ValueError(
            "supervised_mask must match the pairwise label matrix shape"
        ) from exc
    if mask.shape != tuple(label_shape):
        raise ValueError("supervised_mask must match the pairwise label matrix shape")
    invalid = [
        value for value in mask.reshape(-1) if not isinstance(value, (bool, np.bool_))
    ]
    if invalid:
        raise ValueError("supervised_mask must contain only boolean values")
    return mask.astype(bool, copy=False)


__all__ = ["install_supervised_mask_validation"]
