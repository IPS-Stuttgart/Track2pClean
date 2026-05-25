import numpy as np
import pytest
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyPruneConfig,
    policy_link_diagnostics_from_iou_matrix,
    should_prune_policy_edge,
)


def test_should_prune_policy_edge_requires_all_weak_signals() -> None:
    config = Track2pPolicyPruneConfig(
        threshold_margin=0.02,
        competition_margin=0.02,
        min_area_ratio=0.45,
        centroid_distance=10.0,
    )

    common = {
        "assigned_iou": 0.21,
        "threshold": 0.20,
        "row_margin": 0.01,
        "column_margin": 0.01,
        "area_ratio": 0.30,
        "centroid_distance": 11.0,
        "config": config,
    }
    assert should_prune_policy_edge(**common)
    assert not should_prune_policy_edge(**{**common, "assigned_iou": 0.25})
    assert not should_prune_policy_edge(**{**common, "row_margin": 0.05})
    assert not should_prune_policy_edge(**{**common, "column_margin": 0.05})
    assert not should_prune_policy_edge(
        **{**common, "area_ratio": 0.80, "centroid_distance": 3.0}
    )


def test_policy_link_diagnostics_from_iou_matrix_prunes_only_weak_links() -> None:
    config = Track2pPolicyPruneConfig(
        threshold_margin=0.02,
        competition_margin=0.02,
        min_area_ratio=0.45,
        centroid_distance=10.0,
    )
    iou = np.asarray(
        [
            [0.200, 0.195],
            [0.195, 0.200],
        ]
    )
    weak_distances = np.full_like(iou, 11.0)
    weak_area_ratios = np.full_like(iou, 0.30)

    links, diagnostics = policy_link_diagnostics_from_iou_matrix(
        iou,
        threshold_method="otsu",
        threshold_override=0.19,
        prune_config=config,
        distances=weak_distances,
        area_ratios=weak_area_ratios,
    )

    assert links.size == 0
    assert len(diagnostics) == 2
    assert all(diagnostic.pruned for diagnostic in diagnostics)
    assert {diagnostic.prune_reason for diagnostic in diagnostics} == {
        "weak-threshold-margin;weak-row-margin;weak-column-margin;"
        "weak-area-ratio;large-centroid-distance"
    }

    kept_links, kept_diagnostics = policy_link_diagnostics_from_iou_matrix(
        iou,
        threshold_method="otsu",
        threshold_override=0.19,
        prune_config=config,
        distances=np.zeros_like(iou),
        area_ratios=np.ones_like(iou),
    )

    assert kept_links.tolist() == [[0, 0], [1, 1]]
    assert len(kept_diagnostics) == 2
    assert all(not diagnostic.pruned for diagnostic in kept_diagnostics)


def test_policy_link_diagnostics_keep_degenerate_positive_iou_links() -> None:
    iou = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    links, diagnostics = policy_link_diagnostics_from_iou_matrix(
        iou,
        threshold_method="otsu",
        prune_config=Track2pPolicyPruneConfig(),
        distances=np.zeros_like(iou),
        area_ratios=np.ones_like(iou),
    )

    assert links.tolist() == [[0, 0], [1, 1]]
    assert len(diagnostics) == 2
    assert all(not diagnostic.pruned for diagnostic in diagnostics)


def test_policy_link_diagnostics_reject_degenerate_zero_iou_links() -> None:
    links, diagnostics = policy_link_diagnostics_from_iou_matrix(
        np.zeros((2, 2), dtype=float), threshold_method="otsu"
    )

    assert links.size == 0
    assert diagnostics == ()


def test_prune_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="threshold_margin"):
        Track2pPolicyPruneConfig(threshold_margin=-0.1)
    with pytest.raises(ValueError, match="min_area_ratio"):
        Track2pPolicyPruneConfig(min_area_ratio=1.5)
