"""Tests for mutual-top-k Track2p policy rescue edges."""

from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.track2p_policy_priors import (
    Track2pPolicyPriorConfig,
    track2p_policy_edge_mask,
)


def test_policy_prior_can_rescue_mutual_top_k_edges_without_one_sided_edges() -> None:
    iou = np.asarray(
        [
            [0.95, 0.80, 0.05],
            [0.02, 0.94, 0.04],
            [0.01, 0.85, 0.93],
        ],
        dtype=float,
    )

    mask = track2p_policy_edge_mask(
        iou,
        config=Track2pPolicyPriorConfig(
            threshold_method="otsu",
            mutual_top_k=2,
            rescue_min_iou=0.70,
            rescue_margin=0.30,
        ),
    )

    assert bool(mask[0, 0])
    assert bool(mask[1, 1])
    assert bool(mask[2, 1])
    assert bool(mask[2, 2])
    assert not bool(mask[0, 1])
    assert not bool(mask[1, 2])


def test_policy_config_rejects_negative_mutual_top_k() -> None:
    with pytest.raises(ValueError, match="mutual_top_k"):
        Track2pPolicyPriorConfig(mutual_top_k=-1)
