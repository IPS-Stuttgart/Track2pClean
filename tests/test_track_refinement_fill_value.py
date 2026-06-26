from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.track_refinement import (
    TrackGeometryIssue,
    TrackSmoothingConfig,
    smoothed_track_positions,
    split_tracks_at_issues,
)


@pytest.mark.parametrize("bad_fill_value", [0, 1, True, np.bool_(False), 1.5, np.nan, ""])
def test_track_smoothing_config_rejects_non_negative_or_malformed_fill_value(bad_fill_value):
    with pytest.raises(ValueError, match="fill_value"):
        TrackSmoothingConfig(fill_value=bad_fill_value)


def test_smoothed_track_positions_rejects_non_negative_fill_value():
    track_rows = np.array([[0, 1]], dtype=int)
    position_tables = ({0: np.array([0.0, 0.0])}, {1: np.array([1.0, 1.0])})

    with pytest.raises(ValueError, match="negative integer sentinel"):
        smoothed_track_positions(track_rows, position_tables, fill_value=0)


def test_split_tracks_at_issues_rejects_non_negative_fill_value():
    track_rows = np.array([[0, 1]], dtype=int)

    with pytest.raises(ValueError, match="negative integer sentinel"):
        split_tracks_at_issues(track_rows, (), fill_value=0)


def test_split_tracks_at_issues_preserves_roi_zero_with_negative_fill_value():
    track_rows = np.array([[0, 1, 2]], dtype=int)
    issues = [
        TrackGeometryIssue(
            track_index=0,
            session_index=1,
            roi_index=1,
            residual=10.0,
            robust_z=4.0,
            suggested_action="split_or_relink",
        )
    ]

    split_rows = split_tracks_at_issues(track_rows, issues, fill_value=-1)

    npt.assert_array_equal(split_rows, np.array([[0, -1, -1], [-1, 1, 2]], dtype=int))
