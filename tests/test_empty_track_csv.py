from __future__ import annotations

import numpy as np
from bayescatrack.ground_truth_eval import (
    TrackTable,
    evaluate_track_table_prediction,
    load_track_table_csv,
)


def test_header_only_wide_track_csv_loads_as_zero_tracks(tmp_path):
    csv_path = tmp_path / "predicted_tracks.csv"
    csv_path.write_text("day0,day1\n", encoding="utf-8")

    table = load_track_table_csv(csv_path)

    assert table.session_names == ("day0", "day1")
    np.testing.assert_array_equal(table.tracks, np.zeros((0, 2), dtype=int))


def test_header_only_prediction_csv_scores_as_zero_predicted_tracks(tmp_path):
    csv_path = tmp_path / "predicted_tracks.csv"
    csv_path.write_text("day0,day1\n", encoding="utf-8")
    ground_truth = TrackTable(
        session_names=("day0", "day1"),
        tracks=np.asarray([[3, 7]], dtype=int),
    )

    prediction = load_track_table_csv(
        csv_path, session_names=ground_truth.session_names
    )
    evaluation = evaluate_track_table_prediction(ground_truth, prediction)

    assert evaluation.n_predicted_tracks == 0
    assert evaluation.n_ground_truth_tracks == 1
    assert evaluation.complete_tracks == 0.0
    assert evaluation.proportion_correct_by_horizon == {2: 0.0}


def test_header_only_long_track_csv_uses_supplied_session_names(tmp_path):
    csv_path = tmp_path / "predicted_tracks_long.csv"
    csv_path.write_text("track_id,session,roi\n", encoding="utf-8")

    table = load_track_table_csv(csv_path, session_names=("day0", "day1"))

    assert table.session_names == ("day0", "day1")
    np.testing.assert_array_equal(table.tracks, np.zeros((0, 2), dtype=int))
