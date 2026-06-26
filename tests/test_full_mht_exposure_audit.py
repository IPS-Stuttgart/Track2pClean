from __future__ import annotations

import numpy as np
import pytest


def _audit_module():
    pytest.importorskip("pyrecest")
    from bayescatrack.experiments import track2p_policy_full_mht_exposure_audit

    return track2p_policy_full_mht_exposure_audit


def test_missing_observation_count_ignores_empty_rows() -> None:
    audit = _audit_module()
    tracks = np.asarray(
        [
            [1, 2, 3, 4],
            [10, -1, 30, 40],
            [100, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=int,
    )

    assert audit._missing_observation_count(tracks) == 4


def test_history_totals_sums_scan_history() -> None:
    audit = _audit_module()

    totals = audit._history_totals(
        (
            {
                "assigned_edges": 3,
                "missed_tracks": 1,
                "selected_non_prior_edges": 2,
                "scan_candidates": 8,
            },
            {
                "assigned_edges": 4,
                "missed_tracks": 0,
                "selected_non_prior_edges": 1,
                "scan_candidates": 7,
            },
        )
    )

    assert totals["history_assigned_edges"] == 7
    assert totals["history_missed_tracks"] == 1
    assert totals["history_selected_non_prior_edges"] == 3
    assert totals["history_scan_candidates"] == 15


def test_history_totals_reports_growth_prediction_exposure() -> None:
    audit = _audit_module()

    totals = audit._history_totals(
        (
            {
                "selected_edge_summaries": (
                    "0:1->2|score=2.0|growth_pred=0|growth_pred_weighted=0;"
                    "0:2->3|score=1.1|growth_pred=1.5|growth_pred_weighted=0.75"
                )
            },
            {"selected_edge_summaries": "0:3->4|score=0.9"},
        )
    )

    assert totals["history_growth_prediction_evaluated_edges"] == 2
    assert totals["history_growth_prediction_penalized_edges"] == 1
    assert totals["history_growth_prediction_penalty"] == pytest.approx(1.5)
    assert totals["history_growth_prediction_weighted_penalty"] == pytest.approx(0.75)


def test_all_subjects_row_reports_exposure_maxima() -> None:
    audit = _audit_module()

    all_row = audit._all_subjects_row(
        (
            {
                "n_sessions": 7,
                "n_seed_tracks": 10,
                "final_hypotheses": 8,
                "n_output_tracks": 10,
                "n_selected_edges": 42,
                "n_selected_prior_edges": 41,
                "n_selected_non_prior_edges": 1,
                "n_missing_observations": 3,
                "history_assigned_edges": 42,
                "history_missed_tracks": 2,
                "history_selected_prior_edges": 41,
                "history_selected_non_prior_edges": 1,
                "history_missed_prior_successors": 0,
                "history_switched_prior_successors": 0,
                "history_no_prior_successor_continuations": 1,
                "history_gap_reactivated_tracks": 0,
                "history_scan_candidates": 120,
                "history_growth_prediction_evaluated_edges": 10,
                "history_growth_prediction_penalized_edges": 1,
                "history_growth_prediction_penalty": 1.25,
                "history_growth_prediction_weighted_penalty": 0.5,
            },
            {
                "n_sessions": 7,
                "n_seed_tracks": 12,
                "final_hypotheses": 8,
                "n_output_tracks": 12,
                "n_selected_edges": 50,
                "n_selected_prior_edges": 47,
                "n_selected_non_prior_edges": 3,
                "n_missing_observations": 5,
                "history_assigned_edges": 50,
                "history_missed_tracks": 4,
                "history_selected_prior_edges": 47,
                "history_selected_non_prior_edges": 3,
                "history_missed_prior_successors": 1,
                "history_switched_prior_successors": 1,
                "history_no_prior_successor_continuations": 2,
                "history_gap_reactivated_tracks": 1,
                "history_scan_candidates": 140,
                "history_growth_prediction_evaluated_edges": 12,
                "history_growth_prediction_penalized_edges": 3,
                "history_growth_prediction_penalty": 2.0,
                "history_growth_prediction_weighted_penalty": 1.25,
            },
        )
    )

    assert all_row["subject"] == "ALL"
    assert all_row["n_seed_tracks"] == 22
    assert all_row["n_selected_non_prior_edges"] == 4
    assert all_row["history_scan_candidates"] == 260
    assert all_row["history_growth_prediction_evaluated_edges"] == 22
    assert all_row["history_growth_prediction_penalized_edges"] == 4
    assert all_row["history_growth_prediction_penalty"] == pytest.approx(3.25)
    assert all_row["history_growth_prediction_weighted_penalty"] == pytest.approx(1.75)
    assert all_row["max_selected_non_prior_edges_per_subject"] == 3
    assert all_row["max_missing_observations_per_subject"] == 5
    assert all_row["max_growth_prediction_penalized_edges_per_subject"] == 3
    assert all_row["max_growth_prediction_weighted_penalty_per_subject"] == pytest.approx(1.25)
