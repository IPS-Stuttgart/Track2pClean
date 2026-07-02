"""Validation patches for dynamic edge-prior configuration values."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from . import dynamic_edge_priors as _dynamic_edge_priors

_CONFIG_PATCH_MARKER = "_bayescatrack_dynamic_edge_prior_bool_validation_patch"
_CONFIG_ORIGINAL_ATTR = "_bayescatrack_dynamic_edge_prior_bool_validation_original"
_SESSION_GAP_PATCH_MARKER = (
    "_bayescatrack_dynamic_edge_prior_session_gap_validation_patch"
)
_SESSION_GAP_ORIGINAL_ATTR = (
    "_bayescatrack_dynamic_edge_prior_session_gap_validation_original"
)
_ORIGINAL_CHAIN_ATTRS = (_SESSION_GAP_ORIGINAL_ATTR, "_bayescatrack_original")
_BINARY_TEXT_LIKE_TYPES = (bytes, bytearray, memoryview, np.bytes_)
_TEXT_OR_BINARY_LIKE_TYPES = (str, bytes, bytearray, memoryview, np.str_, np.bytes_)

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
    """Reject ambiguous scalar controls before configs cast them to floats."""

    _install_config_numeric_validation()
    _install_session_gap_binary_validation()


def _install_config_numeric_validation() -> None:
    config_cls = _dynamic_edge_priors.DynamicEdgePriorConfig
    if getattr(config_cls, _CONFIG_PATCH_MARKER, False):
        return

    original_post_init = config_cls.__post_init__

    def validated_post_init(self: Any) -> None:
        for field_name in _REQUIRED_NUMERIC_FIELDS:
            _reject_ambiguous_numeric_value(getattr(self, field_name), field_name)
        for field_name in _OPTIONAL_NUMERIC_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                _reject_ambiguous_numeric_value(value, field_name)
        original_post_init(self)

    validated_post_init.__name__ = "__post_init__"
    validated_post_init.__qualname__ = f"{config_cls.__name__}.__post_init__"
    setattr(validated_post_init, _CONFIG_ORIGINAL_ATTR, original_post_init)
    config_cls.__post_init__ = validated_post_init
    setattr(config_cls, _CONFIG_PATCH_MARKER, True)


def _install_session_gap_binary_validation() -> None:
    original_apply = _dynamic_edge_priors.apply_dynamic_edge_priors
    if _function_chain_has_marker(original_apply, _SESSION_GAP_PATCH_MARKER):
        return

    @wraps(original_apply)
    def apply_dynamic_edge_priors_with_session_gap_validation(
        cost_matrix: Any,
        pairwise_components: Any,
        *,
        session_gap: Any,
        empty_registered_rois: Any | None = None,
        config: Any = None,
    ) -> np.ndarray:
        cfg = _dynamic_edge_priors.dynamic_edge_prior_config_from_mapping(config)
        if cfg is not None and cfg.session_gap_weight:
            _reject_binary_session_gap(session_gap)
        return original_apply(
            cost_matrix,
            pairwise_components,
            session_gap=session_gap,
            empty_registered_rois=empty_registered_rois,
            config=cfg,
        )

    setattr(
        apply_dynamic_edge_priors_with_session_gap_validation,
        _SESSION_GAP_ORIGINAL_ATTR,
        original_apply,
    )
    setattr(
        apply_dynamic_edge_priors_with_session_gap_validation,
        _SESSION_GAP_PATCH_MARKER,
        True,
    )
    _dynamic_edge_priors.apply_dynamic_edge_priors = (
        apply_dynamic_edge_priors_with_session_gap_validation
    )


def _function_chain_has_marker(function: Any, marker: str) -> bool:
    seen: set[int] = set()
    current = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, marker, False):
            return True
        seen.add(current_id)
        current = _next_function_in_chain(current)
    return False


def _next_function_in_chain(function: Any) -> Any | None:
    for attribute_name in _ORIGINAL_CHAIN_ATTRS:
        original = getattr(function, attribute_name, None)
        if original is not None:
            return original
    return None


def _reject_ambiguous_numeric_value(value: Any, field_name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    if isinstance(value, _TEXT_OR_BINARY_LIKE_TYPES):
        raise ValueError(f"{field_name} must be numeric, not text")
    try:
        value_array = np.asarray(value, dtype=object)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a numeric scalar") from exc
    if value_array.shape != ():
        raise ValueError(f"{field_name} must be a numeric scalar")

    scalar_value = value_array.item()
    if isinstance(scalar_value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    if isinstance(scalar_value, _TEXT_OR_BINARY_LIKE_TYPES):
        raise ValueError(f"{field_name} must be numeric, not text")


def _reject_binary_session_gap(value: Any) -> None:
    if isinstance(value, _BINARY_TEXT_LIKE_TYPES):
        raise ValueError("session_gap must be numeric, not binary text")
    try:
        value_array = np.asarray(value, dtype=object)
    except Exception:
        return
    if value_array.shape != ():
        return
    if isinstance(value_array.item(), _BINARY_TEXT_LIKE_TYPES):
        raise ValueError("session_gap must be numeric, not binary text")


__all__ = ["install_dynamic_edge_prior_bool_validation"]
