"""Tests for independent Track2p-policy edge priors."""

from __future__ import annotations

import numpy as np
from bayescatrack.association.track2p_policy_priors import (
    Track2pPolicyPriorConfig,
    apply_track2p_policy_edge_prior,
    track2p_policy_edge_mask,
)


def test_policy_mask_keeps_high_thresholded_hungarian_edges() -> None:
    iou = np.asarray(
        [
            [0.90, 0.10, 0.00],
            [0.20, 0.80, 0.00],
            [0.00, 0.10, 0.05],
        ],
        dtype=float,
    )

    mask = track2p_policy_edge_mask(
        iou,
        config=Track2pPolicyPriorConfig(threshold_method="otsu"),
    )

    assert mask.shape == iou.shape
    assert bool(mask[0, 0])
    assert bool(mask[1, 1])
    assert not bool(mask[2, 2])


def test_policy_mask_keeps_degenerate_positive_hungarian_edges() -> None:
    iou = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    mask = track2p_policy_edge_mask(
        iou,
        config=Track2pPolicyPriorConfig(threshold_method="otsu"),
    )

    assert mask.tolist() == [[True, False], [False, True]]


def test_policy_mask_rejects_degenerate_zero_iou_edges() -> None:
    mask = track2p_policy_edge_mask(
        np.zeros((2, 2), dtype=float),
        config=Track2pPolicyPriorConfig(threshold_method="otsu"),
    )

    assert not bool(np.any(mask))


def test_policy_prior_caps_reliefs_and_penalizes_costs() -> None:
    costs = np.asarray(
        [
            [5.0, 4.0, 4.0],
            [3.0, 5.0, 4.0],
            [4.0, 4.0, 5.0],
        ],
        dtype=float,
    )
    components = {
        "iou": np.asarray(
            [
                [0.90, 0.00, 0.00],
                [0.05, 0.85, 0.00],
                [0.00, 0.00, 0.05],
            ],
            dtype=float,
        )
    }

    adjusted = apply_track2p_policy_edge_prior(
        costs,
        components,
        session_gap=1,
        config=Track2pPolicyPriorConfig(
            threshold_method="otsu",
            relief=0.5,
            accepted_cost_cap=1.0,
            non_policy_penalty=0.25,
            min_cost=0.0,
        ),
    )

    np.testing.assert_allclose(adjusted[0, 0], 0.5)
    np.testing.assert_allclose(adjusted[1, 1], 0.5)
    np.testing.assert_allclose(adjusted[2, 2], 5.25)
    np.testing.assert_allclose(adjusted[0, 1], 4.25)
    np.testing.assert_allclose(adjusted[1, 0], 3.25)


def test_policy_prior_respects_gap_restrictions() -> None:
    costs = np.asarray([[5.0]], dtype=float)
    components = {"iou": np.asarray([[0.95]], dtype=float)}

    adjusted = apply_track2p_policy_edge_prior(
        costs,
        components,
        session_gap=2,
        config=Track2pPolicyPriorConfig(consecutive_only=True, relief=1.0),
    )

    np.testing.assert_allclose(adjusted, costs)


def test_policy_prior_can_rescue_row_top_k_edges() -> None:
    iou = np.asarray([[0.20, 0.45, 0.40]], dtype=float)

    mask = track2p_policy_edge_mask(
        iou,
        config=Track2pPolicyPriorConfig(
            row_top_k=2, rescue_min_iou=0.40, rescue_margin=0.10
        ),
    )

    assert bool(mask[0, 1])
    assert bool(mask[0, 2])
    assert not bool(mask[0, 0])
