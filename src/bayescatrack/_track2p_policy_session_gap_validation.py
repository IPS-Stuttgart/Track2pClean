"""Strict validation for Track2p-policy prior session-gap selectors."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track2p_policy_session_gap_validation_patch"


def install_track2p_policy_session_gap_validation(policy_module: Any) -> None:
    """Install idempotent validation for Track2p-policy session gaps."""

    original_apply = policy_module.apply_track2p_policy_edge_prior
    if getattr(original_apply, _PATCH_MARKER, False):
        return

    @wraps(original_apply)
    def apply_track2p_policy_edge_prior_with_session_gap_validation(
        cost_matrix: np.ndarray,
        pairwise_components: Any,
        *,
        session_gap: Any,
        config: Any,
    ) -> np.ndarray:
        if config is not None:
            session_gap = _positive_integer_like(session_gap, name="session_gap")
        return original_apply(
            cost_matrix,
            pairwise_components,
            session_gap=session_gap,
            config=config,
        )

    setattr(
        apply_track2p_policy_edge_prior_with_session_gap_validation, _PATCH_MARKER, True
    )
    setattr(
        apply_track2p_policy_edge_prior_with_session_gap_validation,
        "_bayescatrack_original",
        original_apply,
    )
    policy_module.apply_track2p_policy_edge_prior = (
        apply_track2p_policy_edge_prior_with_session_gap_validation
    )


def _positive_integer_like(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, not boolean")

    try:
        integer_value = int(operator.index(value))
    except TypeError:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError(f"{name} must be a positive integer")
            try:
                numeric_value = float(text)
            except ValueError as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
        elif isinstance(value, (float, np.floating)):
            numeric_value = float(value)
        else:
            raise ValueError(f"{name} must be a positive integer") from None

        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        integer_value = int(numeric_value)

    if integer_value < 1:
        raise ValueError(f"{name} must be at least 1")
    return integer_value


__all__ = ["install_track2p_policy_session_gap_validation"]
