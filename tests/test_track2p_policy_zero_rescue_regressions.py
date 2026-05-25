"""Regression tests for Track2p-policy rescue edge filtering."""

from __future__ import annotations

import numpy as np

from bayescatrack.association.track2p_policy_priors import (
    Track2pPolicyPriorConfig,
    track2p_policy_edge_mask,
)


def test_policy_prior_row_rescue_does_not_add_zero_iou_edges() -> None:
    mask = track2p_policy_edge_mask(
        np.zeros((2, 3), dtype=float),
        config=Track2pPolicyPriorConfig(
            threshold_method="otsu",
            row_top_k=2,
            rescue_min_iou=0.0,
            rescue_margin=1.0,
        ),
    )

    assert not bool(np.any(mask))
