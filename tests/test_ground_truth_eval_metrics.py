from __future__ import annotations

from bayescatrack.ground_truth_eval import (
    TrackTable,
    complete_tracks_score,
    evaluate_track_table_prediction,
    proportion_correct_by_horizon,
)


def test_complete_tracks_score_ignores_incomplete_rows():
    ground_truth = TrackTable(("s1", "s2", "s3"), [[10, -1, 30]])
    prediction = TrackTable(("s1", "s2", "s3"), [[10, -1, 30]])

    assert complete_tracks_score(ground_truth, prediction) == 0.0


def test_proportion_correct_by_horizon_uses_complete_ground_truth_denominator():
    ground_truth = TrackTable(
        ("s1", "s2", "s3"),
        [
            [10, 20, 30],
            [40, -1, 60],
        ],
    )
    prediction = TrackTable(("s1", "s2", "s3"), [[10, 20, 30]])

    scores = proportion_correct_by_horizon(ground_truth, prediction)

    assert scores[2] == 1.0
    assert scores[3] == 1.0


def test_evaluate_track_table_prediction_counts_only_complete_exact_matches():
    ground_truth = TrackTable(
        ("s1", "s2", "s3"),
        [
            [10, 20, 30],
            [40, -1, 60],
        ],
    )
    prediction = TrackTable(
        ("s1", "s2", "s3"),
        [
            [10, 20, 30],
            [40, -1, 60],
        ],
    )

    evaluation = evaluate_track_table_prediction(ground_truth, prediction)

    assert evaluation.n_exact_full_track_matches == 1
