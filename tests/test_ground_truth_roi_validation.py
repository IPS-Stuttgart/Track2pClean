from __future__ import annotations

import pytest
from bayescatrack.ground_truth_eval import TrackTable, load_track_table_csv


def test_track_table_rejects_invalid_negative_roi_index() -> None:
    with pytest.raises(ValueError, match="ROI index"):
        TrackTable(("s1", "s2"), [[0, -2]])


def test_wide_ground_truth_csv_rejects_invalid_negative_roi_index(tmp_path) -> None:
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text("track_id,s1,s2\ntrack_a,0,-2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ROI index"):
        load_track_table_csv(csv_path)
