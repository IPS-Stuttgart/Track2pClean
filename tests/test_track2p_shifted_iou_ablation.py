from pathlib import Path

import pytest

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_shifted_iou_ablation import (
    _aggregate_edge_ranking_summary_rows,
    _config_for_radius,
    _parse_radii,
)


def test_parse_radii_trims_deduplicates_and_preserves_order() -> None:
    assert _parse_radii("0, 2,4,2, 8") == (0, 2, 4, 8)


def test_parse_radii_rejects_empty_and_negative_values() -> None:
    with pytest.raises(ValueError, match="At least one"):
        _parse_radii(" , ")
    with pytest.raises(ValueError, match="non-negative"):
        _parse_radii("0,-1")


def test_config_for_radius_zero_removes_stale_shifted_kwargs() -> None:
    config = Track2pBenchmarkConfig(
        data=Path("data"),
        method="global-assignment",
        cost="registered-iou",
        pairwise_cost_kwargs={"shifted_iou_radius": 99, "custom_weight": 3.0},
    )

    updated = _config_for_radius(
        config,
        0,
        shifted_iou_additive_weight=0.0,
        shifted_mask_cosine_weight=0.0,
        shifted_iou_shift_penalty_weight=0.25,
        shifted_iou_shift_penalty_scale=None,
    )

    assert updated.cost == "registered-iou"
    assert updated.pairwise_cost_kwargs == {"custom_weight": 3.0}


def test_config_for_radius_enables_registered_shifted_iou() -> None:
    config = Track2pBenchmarkConfig(
        data=Path("data"),
        method="global-assignment",
        cost="registered-iou",
    )

    updated = _config_for_radius(
        config,
        4,
        shifted_iou_additive_weight=0.1,
        shifted_mask_cosine_weight=0.2,
        shifted_iou_shift_penalty_weight=0.25,
        shifted_iou_shift_penalty_scale=4.0,
    )

    assert updated.cost == "registered-shifted-iou"
    assert updated.pairwise_cost_kwargs is not None
    assert updated.pairwise_cost_kwargs["shifted_iou_radius"] == 4
    assert updated.pairwise_cost_kwargs["use_shifted_iou_for_iou_cost"] is True
    assert updated.pairwise_cost_kwargs["shifted_iou_weight"] == 0.1
    assert updated.pairwise_cost_kwargs["shifted_mask_cosine_weight"] == 0.2
    assert updated.pairwise_cost_kwargs["shifted_iou_shift_penalty_weight"] == 0.25
    assert updated.pairwise_cost_kwargs["shifted_iou_shift_penalty_scale"] == 4.0


def test_edge_ranking_overview_uses_gt_weighted_rates() -> None:
    rows = [
        {
            "approach": "r0-exact",
            "shifted_iou_radius": "0",
            "score_name": "pairwise_cost_matrix",
            "gt_edges": "10",
            "present_edges": "9",
            "missing_edges": "1",
            "finite_true_edges": "9",
            "row_hit_at_1": "0.2",
            "row_hit_at_3": "0.4",
            "mutual_top1_rate": "0.1",
            "mean_row_margin": "1.0",
            "mean_column_margin": "2.0",
            "median_row_rank": "3",
            "median_column_rank": "4",
        },
        {
            "approach": "r0-exact",
            "shifted_iou_radius": "0",
            "score_name": "pairwise_cost_matrix",
            "gt_edges": "30",
            "present_edges": "30",
            "missing_edges": "0",
            "finite_true_edges": "30",
            "row_hit_at_1": "0.6",
            "row_hit_at_3": "0.8",
            "mutual_top1_rate": "0.5",
            "mean_row_margin": "2.0",
            "mean_column_margin": "4.0",
            "median_row_rank": "1",
            "median_column_rank": "2",
        },
    ]

    overview = _aggregate_edge_ranking_summary_rows(rows)

    assert len(overview) == 1
    assert overview[0]["gt_edges"] == 40
    assert overview[0]["missing_edges"] == 1
    assert overview[0]["row_hit_at_1"] == pytest.approx(0.5)
    assert overview[0]["row_hit_at_3"] == pytest.approx(0.7)
    assert overview[0]["mutual_top1_rate"] == pytest.approx(0.4)
    assert overview[0]["mean_row_margin"] == pytest.approx((9.0 + 60.0) / 39.0)
