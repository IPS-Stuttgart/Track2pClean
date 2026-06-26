from __future__ import annotations

import pytest

from tests import _support  # noqa: F401
from bayescatrack.matching import build_track_rows_from_bundles, build_track_rows_from_matches


_ERROR_MATCH = "negative integer sentinel"


def test_build_track_rows_from_matches_rejects_nonnegative_fill_value():
    with pytest.raises(ValueError, match=_ERROR_MATCH):
        build_track_rows_from_matches(
            ["session0", "session1"],
            [{0: 1}],
            start_roi_indices=[0],
            fill_value=0,
        )


def test_build_track_rows_from_matches_rejects_boolean_fill_value():
    with pytest.raises(ValueError, match=_ERROR_MATCH):
        build_track_rows_from_matches(
            ["session0", "session1"],
            [{0: 1}],
            start_roi_indices=[0],
            fill_value=False,
        )


def test_build_track_rows_from_matches_normalizes_negative_integer_like_fill_value():
    rows = build_track_rows_from_matches(
        ["session0", "session1"],
        [{}],
        start_roi_indices=[3],
        fill_value=-7.0,
    )

    assert rows.tolist() == [[3, -7]]


def test_build_track_rows_from_bundles_rejects_nonnegative_fill_value_before_stitching():
    with pytest.raises(ValueError, match=_ERROR_MATCH):
        build_track_rows_from_bundles([], fill_value=0)
