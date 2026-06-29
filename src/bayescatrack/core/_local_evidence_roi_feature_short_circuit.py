"""Short-circuit unused ROI-feature diagnostics in local-evidence costs.

The local-evidence pairwise-cost wrapper asks the base cost builder for
components internally so it can preserve gate diagnostics and reuse centroid
terms.  That internal ``return_components=True`` call also asks the base builder
to compute ROI-feature diagnostics, even when the caller explicitly set
``roi_feature_weight=0.0`` and did not request components.

Keep the disabled feature term disabled in that case.  Explicit diagnostic calls
still receive the base builder's normal feature validation and component output.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_roi_feature_zero_weight_short_circuit_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"


def install_roi_feature_zero_weight_short_circuit(calcium_plane_cls: type[Any]) -> None:
    """Install an idempotent guard for disabled ROI-feature costs."""

    original = calcium_plane_cls.build_pairwise_cost_matrix
    if _method_chain_has_patch(original):
        return

    @wraps(original)
    def build_pairwise_cost_matrix_with_roi_feature_zero_weight_short_circuit(
        self: Any,
        other: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if _should_short_circuit_unused_roi_features(kwargs):
            kwargs = dict(kwargs)
            kwargs["feature_names"] = ()
        return original(self, other, *args, **kwargs)

    setattr(
        build_pairwise_cost_matrix_with_roi_feature_zero_weight_short_circuit,
        _PATCH_MARKER,
        True,
    )
    setattr(
        build_pairwise_cost_matrix_with_roi_feature_zero_weight_short_circuit,
        _ORIGINAL_ATTR,
        original,
    )
    calcium_plane_cls.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
        build_pairwise_cost_matrix_with_roi_feature_zero_weight_short_circuit
    )


def _method_chain_has_patch(method: Any) -> bool:
    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


def _should_short_circuit_unused_roi_features(kwargs: dict[str, Any]) -> bool:
    if _bool_control_is_true(kwargs.get("return_components", False)):
        return False
    if _bool_control_is_true(kwargs.get("local_evidence_components", False)):
        return False
    return _explicitly_zero_roi_feature_weight(kwargs.get("roi_feature_weight"))


def _bool_control_is_true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _explicitly_zero_roi_feature_weight(value: Any) -> bool:
    if value is None or isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return False
        value = value.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return np.isfinite(numeric) and numeric == 0.0


__all__ = ["install_roi_feature_zero_weight_short_circuit"]
