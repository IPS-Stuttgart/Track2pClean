from __future__ import annotations

import pytest
from bayescatrack.ground_truth_eval import TrackTable


def test_track_table_rejects_duplicate_session_names() -> None:
    with pytest.raises(ValueError, match="unique session names"):
        TrackTable(("day0", "day0"), [[1, 2]])


def test_track_table_rejects_duplicate_session_names_after_string_normalization() -> (
    None
):
    with pytest.raises(ValueError, match="duplicate values: '1'"):
        TrackTable((1, "1"), [[1, 2]])


def test_track_table_alignment_rejects_duplicate_target_session_names() -> None:
    table = TrackTable(("day0", "day1"), [[1, 2]])

    with pytest.raises(
        ValueError, match="target session_names must contain unique session names"
    ):
        table.aligned_to(("day0", "day0"))
