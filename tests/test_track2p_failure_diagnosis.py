from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.experiments.track2p_failure_diagnosis import (
    _aggregate_edge_ranking_rows,
    classify_failure_mode,
    format_diagnosis_table,
)


def test_classify_failure_mode_prioritizes_oracle_failures() -> None:
    mode, recommendation, next_diagnostic = classify_failure_mode(
        oracle_complete_track_f1=0.50,
        method_pairwise_f1=0.90,
        method_complete_track_f1=0.40,
        edge_mutual_top1_rate=1.00,
        edge_missing_rate=0.00,
    )

    assert mode == "row-assembly-or-scoring"
    assert "Oracle GT" in recommendation
    assert "oracle-gt-links" in next_diagnostic


def test_classify_failure_mode_flags_registration_or_cost_ranking() -> None:
    mode, _, next_diagnostic = classify_failure_mode(
        oracle_complete_track_f1=1.00,
        method_pairwise_f1=0.40,
        method_complete_track_f1=0.35,
        edge_mutual_top1_rate=0.25,
        edge_missing_rate=0.00,
    )

    assert mode == "registration-or-cost-ranking"
    assert "edge-ranking" in next_diagnostic


def test_classify_failure_mode_flags_solver_prior_fragmentation() -> None:
    mode, recommendation, next_diagnostic = classify_failure_mode(
        oracle_complete_track_f1=1.00,
        method_pairwise_f1=0.88,
        method_complete_track_f1=0.55,
        edge_mutual_top1_rate=0.95,
        edge_missing_rate=0.00,
    )

    assert mode == "solver-priors-or-track-stitching"
    assert "start/end/gap/threshold" in recommendation
    assert "solver-prior" in next_diagnostic


def test_aggregate_edge_ranking_rows_counts_missing_edges() -> None:
    summary = _aggregate_edge_ranking_rows(
        [
            {"edge_present": 1, "true_is_finite": 1, "row_rank": 1, "column_rank": 1},
            {"edge_present": 1, "true_is_finite": 1, "row_rank": 2, "column_rank": 1},
            {"edge_present": 0, "true_is_finite": 0, "row_rank": -1, "column_rank": -1},
        ]
    )

    assert summary["edge_gt_edges"] == 3
    assert summary["edge_present_edges"] == 2
    assert summary["edge_missing_edges"] == 1
    assert summary["edge_missing_rate"] == pytest.approx(1 / 3)
    assert summary["edge_row_hit_at_1"] == pytest.approx(1 / 3)
    assert summary["edge_column_hit_at_1"] == pytest.approx(2 / 3)
    assert summary["edge_mutual_top1_rate"] == pytest.approx(1 / 3)
    assert summary["edge_median_row_rank"] == pytest.approx(1.5)
    assert summary["edge_median_column_rank"] == pytest.approx(1.0)


def test_format_diagnosis_table_formats_nan() -> None:
    table = format_diagnosis_table(
        [
            {
                "subject": "jm001",
                "failure_mode": "no-dominant-failure",
                "method_pairwise_f1": 1.0,
                "method_complete_track_f1": 1.0,
                "oracle_complete_track_f1": 1.0,
                "edge_mutual_top1_rate": np.nan,
                "edge_missing_rate": 0.0,
                "next_diagnostic": "benchmark track2p-teacher-debug",
            }
        ]
    )

    assert "jm001" in table
    assert "1.000" in table
    assert "nan" in table
