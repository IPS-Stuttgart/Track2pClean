from __future__ import annotations

import pytest

from bayescatrack.ground_truth_eval import (
    TrackTable,
    load_track2p_ground_truth_csv,
    load_track_table_csv,
    tracks_from_consecutive_matches,
)


def test_track_table_rejects_duplicate_session_names() -> None:
    with pytest.raises(ValueError, match="unique session names"):
        TrackTable(("day0", "day0"), [[1, 2]])


def test_track_table_rejects_duplicate_session_names_after_string_normalization() -> None:
    with pytest.raises(ValueError, match="duplicate values: '1'"):
        TrackTable((1, "1"), [[1, 2]])


def test_track_table_rejects_bare_string_session_names() -> None:
    with pytest.raises(ValueError, match="bare string"):
        TrackTable("day0", [[1, 2, 3, 4]])


def test_track_table_alignment_rejects_duplicate_target_session_names() -> None:
    table = TrackTable(("day0", "day1"), [[1, 2]])

    with pytest.raises(ValueError, match="target session_names must contain unique session names"):
        table.aligned_to(("day0", "day0"))


def test_track_table_alignment_rejects_bare_string_target_session_names() -> None:
    table = TrackTable(("day0", "day1"), [[1, 2]])

    with pytest.raises(ValueError, match="bare string"):
        table.aligned_to("day1")


def test_track_table_csv_loader_rejects_bare_string_session_names(tmp_path) -> None:
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text("track_id,session,roi\ncell0,day0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bare string"):
        load_track_table_csv(csv_path, session_names="day0")


def test_track2p_ground_truth_loader_rejects_bare_string_session_names(tmp_path) -> None:
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text("track_id,session,roi\ncell0,day0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bare string"):
        load_track2p_ground_truth_csv(csv_path, session_names="day0")


def test_tracks_from_consecutive_matches_rejects_bare_string_session_names() -> None:
    with pytest.raises(ValueError, match="bare string"):
        tracks_from_consecutive_matches("ab", [{0: 1}])
