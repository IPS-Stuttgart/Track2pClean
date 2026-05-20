from __future__ import annotations

import pytest
from bayescatrack.ground_truth_eval import load_track_table_csv


def test_long_format_duplicate_track_session_conflict_raises(tmp_path):
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text(
        "track_id,session,roi\n" "track_a,session_1,4\n" "track_a,session_1,5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting ROI entries"):
        load_track_table_csv(csv_path)


def test_long_format_duplicate_track_session_same_roi_is_allowed(tmp_path):
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text(
        "track_id,session,roi\n" "track_a,session_1,4\n" "track_a,session_1,4\n",
        encoding="utf-8",
    )

    table = load_track_table_csv(csv_path)

    assert table.tracks.tolist() == [[4]]
