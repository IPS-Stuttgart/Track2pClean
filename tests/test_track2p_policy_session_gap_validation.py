from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.track2p_policy_priors import Track2pPolicyPriorConfig, apply_track2p_policy_edge_prior


def test_track2p_policy_session_gap_rejects_fractional_value():
    with pytest.raises(ValueError, match="session_gap"):
        apply_track2p_policy_edge_prior(
            np.asarray([[0.0]], dtype=float),
            {"iou": np.asarray([[1.0]], dtype=float)},
            session_gap=1.5,
            config=Track2pPolicyPriorConfig(consecutive_only=True),
        )


def test_track2p_policy_session_gap_rejects_boolean_value():
    with pytest.raises(ValueError, match="session_gap"):
        apply_track2p_policy_edge_prior(
            np.asarray([[0.0]], dtype=float),
            {"iou": np.asarray([[1.0]], dtype=float)},
            session_gap=True,
            config=Track2pPolicyPriorConfig(consecutive_only=True),
        )


def test_track2p_policy_session_gap_accepts_integer_like_value():
    adjusted = apply_track2p_policy_edge_prior(
        np.asarray([[0.0]], dtype=float),
        {"iou": np.asarray([[1.0]], dtype=float)},
        session_gap="1",
        config=Track2pPolicyPriorConfig(consecutive_only=True, relief=0.25),
    )

    assert adjusted[0, 0] == pytest.approx(-0.25)
