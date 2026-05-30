"""Strict validation for edge-prior session-gap arguments.

Session gaps are integer distances between acquisition sessions. Several public
edge-prior helpers receive them from experiment configuration or workflow glue,
so validate the scalar before a fractional value can be truncated or converted
into a fractional prior penalty.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from .association import dynamic_edge_priors as _dynamic_edge_priors
from .association import track2p_policy_priors as _track2p_policy_priors


def install_session_gap_validation() -> None:
    """Install idempotent validation around edge-prior ``session_gap`` arguments."""

    if getattr(
        _dynamic_edge_priors,
        "_bayescatrack_session_gap_validation_patch",
        False,
    ) and getattr(
        _track2p_policy_priors,
        "_bayescatrack_session_gap_validation_patch",
        False,
    ):
        return

    original_dynamic = _dynamic_edge_priors.apply_dynamic_edge_priors
    original_policy = _track2p_policy_priors.apply_track2p_policy_edge_prior

    def apply_dynamic_edge_priors_with_session_gap_validation(
        cost_matrix: Any,
        pairwise_components: Any,
        *,
        session_gap: Any,
        empty_registered_rois: Any | None = None,
        config: Any = None,
    ) -> np.ndarray:
        return original_dynamic(
            cost_matrix,
            pairwise_components,
            session_gap=_positive_session_gap(session_gap),
            empty_registered_rois=empty_registered_rois,
            config=config,
        )

    def apply_track2p_policy_edge_prior_with_session_gap_validation(
        cost_matrix: np.ndarray,
        pairwise_components: Any,
        *,
        session_gap: Any,
        config: Any,
    ) -> np.ndarray:
        return original_policy(
            cost_matrix,
            pairwise_components,
            session_gap=_positive_session_gap(session_gap),
            config=config,
        )

    setattr(
        apply_dynamic_edge_priors_with_session_gap_validation,
        "_bayescatrack_session_gap_validation_patch",
        True,
    )
    setattr(
        apply_dynamic_edge_priors_with_session_gap_validation,
        "_bayescatrack_original",
        original_dynamic,
    )
    setattr(
        apply_track2p_policy_edge_prior_with_session_gap_validation,
        "_bayescatrack_session_gap_validation_patch",
        True,
    )
    setattr(
        apply_track2p_policy_edge_prior_with_session_gap_validation,
        "_bayescatrack_original",
        original_policy,
    )

    _dynamic_edge_priors.apply_dynamic_edge_priors = (
        apply_dynamic_edge_priors_with_session_gap_validation
    )
    _track2p_policy_priors.apply_track2p_policy_edge_prior = (
        apply_track2p_policy_edge_prior_with_session_gap_validation
    )
    setattr(_dynamic_edge_priors, "_bayescatrack_session_gap_validation_patch", True)
    setattr(_track2p_policy_priors, "_bayescatrack_session_gap_validation_patch", True)


def _positive_session_gap(value: Any) -> int:
    integer_value = _integer_like(value, name="session_gap")
    if integer_value < 1:
        raise ValueError("session_gap must be a finite value")
    return integer_value


def _integer_like(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite value")

    try:
        return int(operator.index(value))
    except TypeError:
        pass

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must be an integer")
        try:
            numeric_value = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    elif isinstance(value, (float, np.floating)):
        numeric_value = float(value)
    else:
        raise ValueError(f"{name} must be an integer")

    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite value")
    if not numeric_value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric_value)


__all__ = ["install_session_gap_validation"]
