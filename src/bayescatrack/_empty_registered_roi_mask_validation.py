"""Strict validation for dynamic-prior empty registered ROI masks.

``empty_registered_rois`` is a column mask.  Passing string/object values through
``np.asarray(..., dtype=bool)`` treats every non-empty string as ``True``; for
example, ``"False"`` would mark a column as empty and add an unintended penalty.
Validate active masks before the dynamic edge-prior helper coerces them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .association import dynamic_edge_priors as _dynamic_edge_priors


def install_empty_registered_roi_mask_validation() -> None:
    """Install idempotent validation for active ``empty_registered_rois`` masks."""

    if getattr(
        _dynamic_edge_priors,
        "_bayescatrack_empty_registered_roi_mask_validation_patch",
        False,
    ):
        return

    original = _dynamic_edge_priors.apply_dynamic_edge_priors

    def apply_dynamic_edge_priors_with_empty_mask_validation(
        cost_matrix: Any,
        pairwise_components: Any,
        *,
        session_gap: int | float,
        empty_registered_rois: Any | None = None,
        config: Any = None,
    ) -> np.ndarray:
        cfg = _dynamic_edge_priors.dynamic_edge_prior_config_from_mapping(config)
        if (
            cfg is not None
            and cfg.registration_empty_roi_weight
            and empty_registered_rois is not None
        ):
            empty_registered_rois = _strict_binary_mask(
                empty_registered_rois,
                name="empty_registered_rois",
            )
            config = cfg
        return original(
            cost_matrix,
            pairwise_components,
            session_gap=session_gap,
            empty_registered_rois=empty_registered_rois,
            config=config,
        )

    setattr(
        apply_dynamic_edge_priors_with_empty_mask_validation,
        "_bayescatrack_empty_registered_roi_mask_validation_patch",
        True,
    )
    setattr(
        apply_dynamic_edge_priors_with_empty_mask_validation,
        "_bayescatrack_original",
        original,
    )
    _dynamic_edge_priors.apply_dynamic_edge_priors = (
        apply_dynamic_edge_priors_with_empty_mask_validation
    )
    setattr(
        _dynamic_edge_priors,
        "_bayescatrack_empty_registered_roi_mask_validation_patch",
        True,
    )


def _strict_binary_mask(mask: Any, *, name: str) -> np.ndarray:
    values = np.asarray(mask)
    values = values.reshape(-1)
    if values.dtype.kind == "b":
        return values.astype(bool, copy=False)
    if values.dtype.kind in {"i", "u"}:
        if np.any((values != 0) & (values != 1)):
            raise ValueError(f"{name} must be a boolean or binary numeric mask")
        return values.astype(bool)
    if values.dtype.kind == "f":
        if not np.all(np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
            raise ValueError(f"{name} must be a boolean or binary numeric mask")
        return values.astype(bool)
    raise ValueError(f"{name} must be a boolean or binary numeric mask")


__all__ = ["install_empty_registered_roi_mask_validation"]
