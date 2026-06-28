"""Feature-name normalization for ROI-feature pairwise costs."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_ROI_FEATURE_NAME_PATCH_ATTR = "_bayescatrack_roi_feature_name_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"


def install_roi_feature_name_validation(bridge_impl: ModuleType) -> None:
    """Install an idempotent wrapper for ROI-feature name handling."""

    original = bridge_impl._pairwise_roi_feature_distance  # pylint: disable=protected-access
    if getattr(original, _ROI_FEATURE_NAME_PATCH_ATTR, False):
        return

    @wraps(original)
    def _pairwise_roi_feature_distance(
        reference_plane: Any,
        measurement_plane: Any,
        *,
        feature_names: Sequence[str] | None = None,
        scale_epsilon: float = 1.0e-6,
    ) -> np.ndarray:
        return original(
            reference_plane,
            measurement_plane,
            feature_names=_normalize_feature_names(feature_names),
            scale_epsilon=scale_epsilon,
        )

    setattr(_pairwise_roi_feature_distance, _ROI_FEATURE_NAME_PATCH_ATTR, True)
    setattr(_pairwise_roi_feature_distance, _ORIGINAL_ATTR, original)
    bridge_impl._pairwise_roi_feature_distance = (  # pylint: disable=protected-access
        _pairwise_roi_feature_distance
    )


def _normalize_feature_names(feature_names: Sequence[str] | None) -> Sequence[str] | None:
    if isinstance(feature_names, str):
        return (feature_names,)
    return feature_names


__all__ = ["install_roi_feature_name_validation"]
