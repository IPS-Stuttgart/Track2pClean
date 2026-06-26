"""Validation patches for dynamic edge-prior configuration values."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import dynamic_edge_priors as _dynamic_edge_priors

_PATCH_MARKER = "_bayescatrack_dynamic_edge_prior_bool_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_dynamic_edge_prior_bool_validation_original"

_REQUIRED_NUMERIC_FIELDS = (
    "session_gap_weight",
    "cell_probability_weight",
    "area_ratio_weight",
    "activity_missing_weight",
    "registration_empty_roi_weight",
    "reciprocal_rank_weight",
    "local_margin_weight",
    "local_margin_target",
    "edge_quality_bias",
    "large_cost",
)
_OPTIONAL_NUMERIC_FIELDS = (
    "reciprocal_rank_cap",
    "local_margin_cap",
)


def install_dynamic_edge_prior_bool_validation() -> None:
    """Reject booleans before ``DynamicEdgePriorConfig`` casts values to floats."""

    config_cls = _dynamic_edge_priors.DynamicEdgePriorConfig
    if getattr(config_cls, _PATCH_MARKER, False):
        return

    original_post_init = config_cls.__post_init__

    def validated_post_init(self: Any) -> None:
        for field_name in _REQUIRED_NUMERIC_FIELDS:
            _reject_boolean_numeric_value(getattr(self, field_name), field_name)
        for field_name in _OPTIONAL_NUMERIC_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                _reject_boolean_numeric_value(value, field_name)
        original_post_init(self)

    validated_post_init.__name__ = "__post_init__"
    validated_post_init.__qualname__ = f"{config_cls.__name__}.__post_init__"
    setattr(validated_post_init, _ORIGINAL_ATTR, original_post_init)
    config_cls.__post_init__ = validated_post_init
    setattr(config_cls, _PATCH_MARKER, True)


def _reject_boolean_numeric_value(value: Any, field_name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")


__all__ = ["install_dynamic_edge_prior_bool_validation"]
