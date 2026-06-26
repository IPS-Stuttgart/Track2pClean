import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.track_refinement import (
    TrackGeometryIssue,
    TrackSmoothingConfig,
    smoothed_track_positions,
    split_tracks_at_issues,
)


@pytest.mark.parametrize(
    "bad_fill_value",
    [
        True,
        np.bool_(False),
        0,
        1,
        0.5,
        -1.5,
        np.nan,
        np.inf,
        "-1",
    ],
)
def test_track_smoothing_config_rejects_malformed_or_colliding_fill_value(
    bad_fill_value,
):
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        TrackSmoothingConfig(fill_value=bad_fill_value)


@pytest.mark.parametrize(
    "bad_fill_value",
    [True, np.bool_(False), 0, 1, 0.5, -1.5, np.nan, np.inf, "-1"],
)
def test_smoothed_track_positions_rejects_malformed_or_colliding_fill_value(
    bad_fill_value,
):
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        smoothed_track_positions(
            np.array([[0, 1]], dtype=int),
            ({0: np.array([0.0, 0.0])}, {1: np.array([1.0, 1.0])}),
            fill_value=bad_fill_value,
        )


@pytest.mark.parametrize(
    "bad_fill_value",
    [True, np.bool_(False), 0, 1, 0.5, -1.5, np.nan, np.inf, "-1"],
)
def test_split_tracks_at_issues_rejects_malformed_or_colliding_fill_value(
    bad_fill_value,
):
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        split_tracks_at_issues(
            np.array([[0, 1]], dtype=int),
            [],
            fill_value=bad_fill_value,
        )


def test_split_tracks_at_issues_preserves_negative_missing_sentinel():
    rows = split_tracks_at_issues(
        np.array([[1, 2, 3]], dtype=int),
        [
            TrackGeometryIssue(
                track_index=0,
                session_index=1,
                roi_index=2,
                residual=9.0,
                robust_z=4.0,
                suggested_action="split_or_relink",
            )
        ],
        fill_value=np.int64(-9),
    )

    npt.assert_array_equal(rows, np.array([[1, -9, -9], [-9, 2, 3]], dtype=int))
