from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.edge_ranking import (
    missing_reference_edge_rows,
    rank_labeled_edges,
    summarize_edge_ranking_rows,
)
from bayescatrack.experiments.track2p_edge_ranking import (
    _edge_ranking_base_cost,
    _effective_learned_score_model,
    _learned_score_matrices,
)


def test_rank_labeled_edges_reports_row_column_ranks_and_margins():
    labels = np.array([[0, 1, 0], [0, 0, 1]])
    costs = np.array([[0.0, 2.0, 1.0], [5.0, 4.0, 3.0]])

    rows = rank_labeled_edges(
        labels,
        {"cost": costs},
        reference_roi_indices=np.array([10, 11]),
        measurement_roi_indices=np.array([20, 21, 22]),
        metadata={
            "subject": "jm_test",
            "session_a": 0,
            "session_b": 1,
            "session_gap": 1,
        },
    )

    first = next(row for row in rows if row["reference_roi_index"] == 10)
    assert first["measurement_roi_index"] == 21
    assert first["score_direction"] == "cost"
    assert first["row_rank"] == 3
    assert first["column_rank"] == 1
    assert first["best_false_row_roi_index"] == 20
    assert first["row_margin"] == pytest.approx(-2.0)

    second = next(row for row in rows if row["reference_roi_index"] == 11)
    assert second["row_rank"] == 1
    assert second["column_rank"] == 2
    assert second["row_margin"] == pytest.approx(1.0)
    assert second["column_margin"] == pytest.approx(-2.0)


def test_rank_labeled_edges_supports_similarity_scores():
    labels = np.array([[1, 0, 0]])
    iou = np.array([[0.8, 0.6, 0.1]])

    rows = rank_labeled_edges(
        labels,
        {"iou": iou},
        reference_roi_indices=np.array([7]),
        measurement_roi_indices=np.array([1, 2, 3]),
    )

    assert rows[0]["score_direction"] == "similarity"
    assert rows[0]["row_rank"] == 1
    assert rows[0]["row_margin"] == pytest.approx(0.2)


def test_missing_reference_edges_are_counted_in_summary_denominator():
    present_rows = rank_labeled_edges(
        np.array([[1]]),
        {"cost": np.array([[0.0]])},
        reference_roi_indices=np.array([1]),
        measurement_roi_indices=np.array([2]),
        metadata={
            "subject": "jm_test",
            "session_a": 0,
            "session_b": 1,
            "session_gap": 1,
        },
    )
    missing_rows = missing_reference_edge_rows(
        [(1, 2), (3, 4)],
        reference_roi_indices=np.array([1]),
        measurement_roi_indices=np.array([2]),
        score_names=("cost",),
        metadata={
            "subject": "jm_test",
            "session_a": 0,
            "session_b": 1,
            "session_gap": 1,
        },
    )

    summary = summarize_edge_ranking_rows(present_rows + missing_rows)

    assert len(summary) == 1
    assert summary[0]["gt_edges"] == 2
    assert summary[0]["present_edges"] == 1
    assert summary[0]["missing_edges"] == 1
    assert summary[0]["row_hit_at_1"] == pytest.approx(0.5)
    assert summary[0]["row_hit_at_1_present"] == pytest.approx(1.0)


class _FakeLearnedModel:
    def predict_match_probability(self, features):
        return np.clip(1.0 - np.asarray(features, dtype=float)[..., 0], 0.0, 1.0)

    def predict_score(self, features):
        return np.asarray(features, dtype=float)[..., 0]


def test_learned_score_matrices_are_rankable_cost_and_probability_planes():
    features = np.array([[[0.1], [0.9]], [[0.8], [0.2]]], dtype=float)

    matrices = _learned_score_matrices(
        _FakeLearnedModel(), features, learned_score_model="monotone"
    )

    assert set(matrices) == {
        "monotone_match_probability",
        "monotone_cost",
        "monotone_raw_score",
    }
    assert matrices["monotone_match_probability"][0, 0] > matrices[
        "monotone_match_probability"
    ][0, 1]
    assert matrices["monotone_cost"][0, 0] < matrices["monotone_cost"][0, 1]

    rows = rank_labeled_edges(
        np.eye(2, dtype=int),
        {"monotone_cost": matrices["monotone_cost"]},
        reference_roi_indices=np.arange(2),
        measurement_roi_indices=np.arange(2),
    )
    assert [row["row_rank"] for row in rows] == [1, 1]


def test_edge_ranking_cost_aliases_select_learned_score_models():
    assert _effective_learned_score_model("none", "registered-iou") == "none"
    assert _effective_learned_score_model("none", "calibrated") == "logistic"
    assert _effective_learned_score_model("none", "monotone") == "monotone"
    assert _effective_learned_score_model("monotone", "registered-iou") == "monotone"
    assert _edge_ranking_base_cost("calibrated") == "registered-iou"
    assert _edge_ranking_base_cost("monotone") == "registered-iou"
