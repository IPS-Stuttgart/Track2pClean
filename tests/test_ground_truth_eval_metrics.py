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


@pytest.mark.parametrize(
    "csv_body",
    [
        "track_id,session,roi\n" "track_1,s1,10\n" "track_1,s1,\n" "track_1,s2,20\n",
        "track_id,session,roi\n" "track_1,s1,\n" "track_1,s1,10\n" "track_1,s2,20\n",
    ],
)
def test_long_format_duplicate_missing_rows_preserve_nonmissing_roi(tmp_path, csv_body):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_body, encoding="utf-8")

    table = load_track_table_csv(csv_path, session_names=("s1", "s2"))

    assert table.session_names == ("s1", "s2")
    np.testing.assert_array_equal(table.tracks, [[10, 20]])


def test_long_format_rejects_conflicting_nonmissing_duplicate_rows(tmp_path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(
        "track_id,session,roi\n" "track_1,s1,10\n" "track_1,s1,11\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting ROI entries"):
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
