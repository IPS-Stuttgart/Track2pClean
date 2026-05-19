from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.edge_thresholds import (
    apply_edge_cost_thresholds,
    compute_manual_oracle_edge_cost_thresholds,
    manual_oracle_cost_threshold,
    otsu_cost_threshold,
)


def test_manual_oracle_threshold_keeps_strictest_f1_optimum():
    costs = np.asarray(
        [
            [0.1, 0.2, 1.5],
            [0.3, 0.4, 2.0],
        ],
        dtype=float,
    )
    true_matches = np.asarray(
        [
            [True, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    threshold = manual_oracle_cost_threshold(costs, true_matches)

    assert threshold == pytest.approx(0.2)


def test_manual_oracle_threshold_rejects_all_when_edge_has_no_positives():
    costs = np.asarray([[0.5, 1.0]], dtype=float)
    true_matches = np.zeros_like(costs, dtype=bool)

    threshold = manual_oracle_cost_threshold(costs, true_matches)

    assert threshold < 0.5
    filtered = apply_edge_cost_thresholds({(0, 1): costs}, {(0, 1): threshold})
    assert np.all(np.isinf(filtered[(0, 1)]))


def test_compute_manual_oracle_thresholds_defaults_missing_edge_to_no_positives():
    pairwise_costs = {
        (0, 1): np.asarray([[0.1, 0.2]], dtype=float),
        (1, 2): np.asarray([[0.3, 0.4]], dtype=float),
    }
    true_masks = {(0, 1): np.asarray([[False, True]], dtype=bool)}

    thresholds = compute_manual_oracle_edge_cost_thresholds(pairwise_costs, true_masks)

    assert thresholds[(0, 1)] == pytest.approx(0.2)
    assert thresholds[(1, 2)] < 0.3


def test_otsu_threshold_splits_separated_cost_modes():
    costs = np.asarray([[0.05, 0.10, 0.15, 3.0, 3.5, 4.0]], dtype=float)

    threshold = otsu_cost_threshold(costs, bins=8)

    assert threshold is not None
    assert 0.15 < threshold < 3.0


def test_otsu_threshold_ignores_optional_huge_sentinel_costs():
    costs = np.asarray([[0.1, 0.2, 0.3, 1.0e6]], dtype=float)

    threshold = otsu_cost_threshold(costs, bins=4, max_cost=10.0)

    assert threshold is not None
    assert threshold < 0.3


def test_manual_oracle_threshold_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        manual_oracle_cost_threshold(
            np.asarray([[0.1, 0.2]], dtype=float),
            np.asarray([[True], [False]], dtype=bool),
        )
