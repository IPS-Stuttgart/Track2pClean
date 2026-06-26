"""Validation patch for Track2p-policy edge-prior controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import track2p_policy_priors as _track2p_policy_priors

_APPLY_PATCH_MARKER = "_bayescatrack_track2p_policy_session_gap_validation_patch"
_APPLY_ORIGINAL_ATTR = "_bayescatrack_track2p_policy_session_gap_validation_original"
_CONFIG_PATCH_MARKER = "_bayescatrack_track2p_policy_config_bool_validation_patch"
_CONFIG_ORIGINAL_ATTR = "_bayescatrack_track2p_policy_config_bool_validation_original"


def install_track2p_policy_session_gap_validation() -> None:
    """Reject malformed Track2p-policy edge-prior controls."""

    _install_config_bool_validation()
    _install_session_gap_validation()


def _install_config_bool_validation() -> None:
    config_cls = _track2p_policy_priors.Track2pPolicyPriorConfig
    original_post_init = config_cls.__post_init__
    if getattr(original_post_init, _CONFIG_PATCH_MARKER, False):
        return

    def validated_post_init(self: Any) -> None:
        object.__setattr__(
            self,
            "consecutive_only",
            _strict_bool(getattr(self, "consecutive_only"), name="consecutive_only"),
        )
        original_post_init(self)

    validated_post_init.__name__ = original_post_init.__name__
    validated_post_init.__qualname__ = original_post_init.__qualname__
    setattr(validated_post_init, _CONFIG_ORIGINAL_ATTR, original_post_init)
    setattr(validated_post_init, _CONFIG_PATCH_MARKER, True)
    config_cls.__post_init__ = validated_post_init


def _install_session_gap_validation() -> None:
    if getattr(_track2p_policy_priors, _APPLY_PATCH_MARKER, False):
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
        _strict_bool(getattr(cfg, "consecutive_only"), name="consecutive_only")
        gap = _track2p_policy_priors._integer_like(  # pylint: disable=protected-access
            session_gap,
            name="session_gap",
        )
        if gap < 1:
            raise ValueError("session_gap must be at least 1")
        return original(cost_matrix, pairwise_components, session_gap=gap, config=cfg)

    validated_apply_track2p_policy_edge_prior.__name__ = original.__name__
    validated_apply_track2p_policy_edge_prior.__qualname__ = original.__qualname__
    setattr(validated_apply_track2p_policy_edge_prior, _APPLY_ORIGINAL_ATTR, original)
    _track2p_policy_priors.apply_track2p_policy_edge_prior = (
        validated_apply_track2p_policy_edge_prior
    )
    setattr(_track2p_policy_priors, _APPLY_PATCH_MARKER, True)


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


__all__ = ["install_track2p_policy_session_gap_validation"]
