"""Validation patch for Track2p-policy edge-prior session gaps."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import track2p_policy_priors as _track2p_policy_priors

_PATCH_MARKER = "_bayescatrack_track2p_policy_session_gap_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_track2p_policy_session_gap_validation_original"


def install_track2p_policy_session_gap_validation() -> None:
    """Reject malformed ``session_gap`` values before Track2p-policy gating."""

    if getattr(_track2p_policy_priors, _PATCH_MARKER, False):
        return

    original = _track2p_policy_priors.apply_track2p_policy_edge_prior

    def validated_apply_track2p_policy_edge_prior(
        cost_matrix: np.ndarray,
        pairwise_components: dict[str, np.ndarray],
        *,
        session_gap: Any,
        config: Any,
    ) -> np.ndarray:
        cfg = _track2p_policy_priors.track2p_policy_prior_config_from_mapping(config)
        if cfg is None:
            return original(
                cost_matrix,
                pairwise_components,
                session_gap=session_gap,
                config=None,
            )
        gap = _track2p_policy_priors._integer_like(  # pylint: disable=protected-access
            session_gap,
            name="session_gap",
        )
        if gap < 1:
            raise ValueError("session_gap must be at least 1")
        return original(cost_matrix, pairwise_components, session_gap=gap, config=cfg)

    validated_apply_track2p_policy_edge_prior.__name__ = (
        original.__name__
    )
    validated_apply_track2p_policy_edge_prior.__qualname__ = original.__qualname__
    setattr(validated_apply_track2p_policy_edge_prior, _ORIGINAL_ATTR, original)
    _track2p_policy_priors.apply_track2p_policy_edge_prior = (
        validated_apply_track2p_policy_edge_prior
    )
    setattr(_track2p_policy_priors, _PATCH_MARKER, True)


__all__ = ["install_track2p_policy_session_gap_validation"]
