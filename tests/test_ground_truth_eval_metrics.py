from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.ground_truth_eval import (
    TrackTable,
    complete_tracks_score,
    evaluate_track_table_prediction,
    load_track_table_csv,
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


def test_track_table_normalizes_missing_roi_values():
    table = TrackTable(("s1", "s2", "s3"), [[None, np.nan, -1]])

    assert table.tracks.tolist() == [[-1, -1, -1]]


@pytest.mark.parametrize(
    "tracks",
    [
        [[True, 2]],
        [[np.bool_(False), 2]],
        [[1.5, 2]],
        [[-2, 2]],
    ],
)
def test_track_table_rejects_invalid_roi_values(tracks):
    with pytest.raises(ValueError, match="ROI index"):
        TrackTable(("s1", "s2"), tracks)


@pytest.mark.parametrize(
    ("roi_text", "message"),
    [
        ("true", "integer-like"),
        ("1.5", "integer-like"),
        ("-2", "non-negative"),
    ],
)
def test_wide_csv_rejects_invalid_roi_values(tmp_path, roi_text, message):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(f"s1,s2\n{roi_text},2\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_track_table_csv(csv_path)


def test_long_format_rejects_missing_track_ids(tmp_path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(
        "track_id,session,roi\n" ",s1,10\n" "nan,s2,20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing track id"):
        load_track_table_csv(csv_path)


def test_long_format_rejects_missing_session_names(tmp_path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(
        "track_id,session,roi\n" "track_1,,10\n" "track_2,nan,20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing session name"):
        load_track_table_csv(csv_path)
