"""Feature-name input normalization for ROI-feature association costs."""

from __future__ import annotations

from types import ModuleType
from typing import Any

_FEATURE_NAME_PATCH_ATTR = "_bayescatrack_feature_name_string_patch"


def install_feature_name_string_normalization(bridge_impl: ModuleType) -> None:
    """Treat a bare feature-name string as one feature instead of characters."""

    original = bridge_impl._pairwise_roi_feature_distance  # pylint: disable=protected-access
    if getattr(original, _FEATURE_NAME_PATCH_ATTR, False):
        return

    def _pairwise_roi_feature_distance(
        reference_plane: Any,
        measurement_plane: Any,
        *,
        feature_names: Any = None,
        scale_epsilon: float = 1.0e-6,
    ) -> Any:
        normalized_feature_names = (
            (feature_names,) if isinstance(feature_names, str) else feature_names
        )
        return original(
            reference_plane,
            measurement_plane,
            feature_names=normalized_feature_names,
            scale_epsilon=scale_epsilon,
        )

    setattr(_pairwise_roi_feature_distance, _FEATURE_NAME_PATCH_ATTR, True)
    setattr(_pairwise_roi_feature_distance, "_bayescatrack_original", original)
    bridge_impl._pairwise_roi_feature_distance = _pairwise_roi_feature_distance  # pylint: disable=protected-access
