from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)
from bayescatrack.association.track2p_policy_priors import (
    Track2pPolicyPriorConfig,
    apply_track2p_policy_edge_prior,
)


class _OverflowingIndex:
    def __index__(self) -> int:
        raise OverflowError


@pytest.mark.parametrize("session_gap", [True, 0, 1.5, str(2.5), float("nan")])
def test_dynamic_edge_prior_rejects_non_integer_session_gaps(
    session_gap: object,
) -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_dynamic_edge_priors(
            np.asarray([[1.0]], dtype=float),
            {},
            session_gap=session_gap,
            config=DynamicEdgePriorConfig(session_gap_weight=1.0),
        )


def test_dynamic_edge_prior_normalizes_overflowing_index_session_gap() -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_dynamic_edge_priors(
            np.asarray([[1.0]], dtype=float),
            {},
            session_gap=_OverflowingIndex(),
            config=DynamicEdgePriorConfig(session_gap_weight=1.0),
        )


@pytest.mark.parametrize("session_gap", [str(3), np.str_(str(3)), np.asarray(str(3))])
def test_dynamic_edge_prior_rejects_integer_like_text_session_gaps(
    session_gap: object,
) -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_dynamic_edge_priors(
            np.asarray([[1.0]], dtype=float),
            {},
            session_gap=session_gap,
            config=DynamicEdgePriorConfig(session_gap_weight=0.5),
        )


@pytest.mark.parametrize("session_gap", [True, 0, 1.5, str(2.5), float("inf")])
def test_track2p_policy_prior_rejects_non_integer_session_gaps(
    session_gap: object,
) -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_track2p_policy_edge_prior(
            np.asarray([[5.0]], dtype=float),
            {"iou": np.asarray([[0.95]], dtype=float)},
            session_gap=session_gap,
            config=Track2pPolicyPriorConfig(consecutive_only=True, relief=1.0),
        )


def test_track2p_policy_prior_normalizes_overflowing_index_session_gap() -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_track2p_policy_edge_prior(
            np.asarray([[5.0]], dtype=float),
            {"iou": np.asarray([[0.95]], dtype=float)},
            session_gap=_OverflowingIndex(),
            config=Track2pPolicyPriorConfig(consecutive_only=True, relief=1.0),
        )


def test_track2p_policy_prior_does_not_truncate_fractional_gap_to_consecutive() -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_track2p_policy_edge_prior(
            np.asarray([[5.0]], dtype=float),
            {"iou": np.asarray([[0.95]], dtype=float)},
            session_gap=1.5,
            config=Track2pPolicyPriorConfig(consecutive_only=True, relief=1.0),
        )


def test_track2p_policy_prior_accepts_integer_like_session_gap_strings() -> None:
    costs = np.asarray([[5.0]], dtype=float)
    adjusted = apply_track2p_policy_edge_prior(
        costs,
        {"iou": np.asarray([[0.95]], dtype=float)},
        session_gap=str(2),
        config=Track2pPolicyPriorConfig(consecutive_only=True, relief=1.0),
    )

    np.testing.assert_allclose(adjusted, costs)
