from __future__ import annotations

import pytest
from bayescatrack._ground_truth_track_validation import (
    install_ground_truth_track_validation,
)
from bayescatrack.ground_truth_eval import (
    TrackTable,
    load_track2p_ground_truth_csv,
    load_track_table_csv,
    tracks_from_consecutive_matches,
)

_STRING_LIKE_SESSION_NAMES = ["day0", b"day0", bytearray(b"day0")]
_TWO_SESSION_STRING_LIKE_VALUES = ["ab", b"ab", bytearray(b"ab")]


def test_track_table_rejects_duplicate_session_names() -> None:
    with pytest.raises(ValueError, match="unique session names"):
        TrackTable(("day0", "day0"), [[1, 2]])


def test_track_table_rejects_duplicate_session_names_after_string_normalization() -> (
    None
):
    with pytest.raises(ValueError, match="duplicate values: '1'"):
        TrackTable((1, "1"), [[1, 2]])


@pytest.mark.parametrize("session_names", _STRING_LIKE_SESSION_NAMES)
def test_track_table_rejects_bare_string_like_session_names(session_names) -> None:
    with pytest.raises(ValueError, match="bare string"):
        TrackTable(session_names, [[1, 2, 3, 4]])


@pytest.mark.parametrize("session_names", _TWO_SESSION_STRING_LIKE_VALUES)
def test_track_table_rejects_bare_string_like_session_names_after_ground_truth_reinstall(
    session_names,
) -> None:
    install_ground_truth_track_validation()

    with pytest.raises(ValueError, match="bare string"):
        TrackTable(session_names, [[1, 2]])


def test_track_table_alignment_rejects_duplicate_target_session_names() -> None:
    table = TrackTable(("day0", "day1"), [[1, 2]])

    with pytest.raises(
        ValueError, match="target session_names must contain unique session names"
    ):
        table.aligned_to(("day0", "day0"))


@pytest.mark.parametrize("session_names", _STRING_LIKE_SESSION_NAMES)
def test_track_table_alignment_rejects_bare_string_like_target_session_names(
    session_names,
) -> None:
    table = TrackTable(("day0", "day1"), [[1, 2]])

    with pytest.raises(ValueError, match="bare string"):
        table.aligned_to(session_names)


@pytest.mark.parametrize("session_names", _STRING_LIKE_SESSION_NAMES)
def test_track_table_csv_loader_rejects_bare_string_like_session_names(
    tmp_path, session_names
) -> None:
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text("track_id,session,roi\ncell0,day0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bare string"):
        load_track_table_csv(csv_path, session_names=session_names)


@pytest.mark.parametrize("session_names", _STRING_LIKE_SESSION_NAMES)
def test_track2p_ground_truth_loader_rejects_bare_string_like_session_names(
    tmp_path, session_names
) -> None:
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text("track_id,session,roi\ncell0,day0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bare string"):
        load_track2p_ground_truth_csv(csv_path, session_names=session_names)


@pytest.mark.parametrize("session_names", _TWO_SESSION_STRING_LIKE_VALUES)
def test_tracks_from_consecutive_matches_rejects_bare_string_like_session_names(
    session_names,
) -> None:
    with pytest.raises(ValueError, match="bare string"):
        tracks_from_consecutive_matches(session_names, [{0: 1}])
