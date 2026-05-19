import numpy as np
import pytest

from bayescatrack.ground_truth_eval import (
    TrackTable,
    complete_tracks_score,
    evaluate_track_table_prediction,
)


SESSION_NAMES = ("day0", "day1", "day2")


def _table(rows: list[list[int]]) -> TrackTable:
    return TrackTable(session_names=SESSION_NAMES, tracks=np.asarray(rows, dtype=int))


def test_complete_tracks_score_ignores_incomplete_ground_truth_rows():
    ground_truth = _table([[0, 10, 20], [1, -1, 21]])
    prediction = _table([[0, 10, 20]])

    assert complete_tracks_score(ground_truth, prediction) == pytest.approx(1.0)


def test_complete_tracks_score_ignores_incomplete_prediction_rows():
    ground_truth = _table([[0, 10, 20]])
    prediction = _table([[0, 10, 20], [1, -1, 21]])

    assert complete_tracks_score(ground_truth, prediction) == pytest.approx(1.0)


def test_evaluation_exact_full_track_matches_exclude_incomplete_rows():
    ground_truth = _table([[0, 10, 20], [1, -1, 21]])
    prediction = _table([[0, 10, 20], [1, -1, 21]])

    evaluation = evaluate_track_table_prediction(ground_truth, prediction)

    assert evaluation.n_exact_full_track_matches == 1
    assert evaluation.complete_tracks == pytest.approx(1.0)
